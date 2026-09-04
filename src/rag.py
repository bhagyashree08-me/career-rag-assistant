#rag.py
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from src.prompts import build_prompt
from src.document_processor import load_pdfs, split_documents


load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHROMA_DIR = str(PROJECT_ROOT / "chroma_db")
COLLECTION_NAME = "career_documents"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-3.6-flash"

# Chroma's normalized relevance score is roughly 0-1 (higher =
# more relevant). Chunks below this are treated as noise, not
# real evidence -- this is also the signal used to decide
# whether local documents are strong enough to skip live web
# search (see generate_answer()).
RELEVANCE_THRESHOLD = 0.15

# Hard ceiling on how long we wait for a single Gemini call.
# ChatGoogleGenerativeAI's own `timeout` kwarg is passed below,
# but there is a known upstream issue where it isn't always
# honored (langchain-google-genai #731 / #1180), so a real
# wall-clock timeout is also enforced here via a worker thread.
# This turns "the app hangs for a long, unpredictable time" into
# "the app fails clearly after LLM_TIMEOUT_SECONDS".
LLM_TIMEOUT_SECONDS = 45

# Cap on how much retrieved-context text gets sent to the model.
# A very large local knowledge base (or a bad chunking pass)
# could otherwise balloon the prompt and slow every request.
MAX_CONTEXT_CHARS = 6000


# ============================================================
# EMBEDDINGS
# ============================================================
#
# PERFORMANCE FIX:
# HuggingFaceEmbeddings loads the sentence-transformers model
# weights into memory. The original version rebuilt this from
# scratch on every single question (get_vector_store() called
# get_embeddings() fresh each time retrieve_documents() ran) --
# that repeated model load was almost certainly the dominant
# source of the slow responses, on top of actual retrieval and
# generation time.
#
# @lru_cache(maxsize=1) turns this into a singleton: the model
# loads once per running process (i.e. once per Streamlit
# session, not once per question), and every later call reuses
# the same in-memory object.
# ============================================================

@lru_cache(maxsize=1)
def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


# ============================================================
# VECTOR DATABASE
# ============================================================

@lru_cache(maxsize=1)
def get_vector_store():

    embeddings = get_embeddings()

    # Use the existing Chroma database if it already exists.
    chroma_path = Path(CHROMA_DIR)

    if chroma_path.exists():
        return Chroma(
            persist_directory=CHROMA_DIR,
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
        )

    # If Chroma does not exist, build it automatically
    # from the PDFs currently present in data/.
    documents = load_pdfs()

    if not documents:
        raise RuntimeError(
            "No PDF documents were found in the data directory."
        )

    chunks = split_documents(documents)

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )


# ============================================================
# GEMINI
# ============================================================

@lru_cache(maxsize=1)
def get_llm():
    """
    Plain Gemini client, no tools bound. Cached as a singleton
    for the same reason as get_embeddings() -- avoids rebuilding
    the client on every question.

    timeout / max_retries are set explicitly so a slow or
    transient-error API call fails predictably instead of
    silently retrying several times with backoff (which can
    itself look like "the app is just slow").
    """

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is missing from the .env file."
        )

    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=api_key,
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=1,
    )


@lru_cache(maxsize=1)
def get_grounded_llm():
    """
    Same Gemini client as get_llm(), with Google Search
    grounding bound as a tool.

    When bound, Gemini can perform a live web search itself
    server-side and fold the results directly into its response
    -- no separate search API key, and no manual search/tool-
    execution loop needed on our side.

    This is what lets the analysis draw on more than just the
    two local PDFs (resume + Future of Jobs report) -- e.g.
    current role expectations, company context, or terminology
    trends that aren't present in either PDF. It is only used
    when the local knowledge base doesn't have a relevant match
    for the question (see generate_answer()), since search adds
    real network latency and shouldn't be paid on every request.
    """

    return get_llm().bind_tools([{"google_search": {}}])


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(question, k=5):
    """
    Retrieve locally relevant chunks from Chroma.

    EDGE CASE HANDLING:
    - If the Chroma DB/collection is missing, empty, or fails to
      open, this returns [] instead of crashing the whole
      analysis -- generate_answer() already treats an empty
      document list as "no local evidence" and proceeds using
      the resume/job text plus (if needed) live web search.
    - Chunks below RELEVANCE_THRESHOLD are dropped. Previously,
      similarity_search() always returned its top-k chunks
      regardless of how weak the match actually was, which could
      inject barely-related text as if it were solid evidence.
      Filtering weak matches out also means a genuinely
      off-topic/unsupported question naturally ends up with no
      local context, which is the trigger used below to bring in
      web search instead of forcing a doc-only answer.
    """

    try:
        vector_store = get_vector_store()
    except Exception:
        return []

    try:
        scored = vector_store.similarity_search_with_relevance_scores(
            question,
            k=k,
        )

        return [
            doc for doc, score in scored
            if score >= RELEVANCE_THRESHOLD
        ]

    except Exception:
        # Older langchain-chroma versions, or a store that
        # doesn't support relevance scoring, fall back to a
        # plain similarity search rather than failing outright.
        try:
            return vector_store.similarity_search(
                question,
                k=k,
            )
        except Exception:
            return []


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(documents):

    if not documents:
        return ""

    context_parts = []

    for doc in documents:

        source = doc.metadata.get(
            "source",
            "Unknown",
        )

        page = doc.metadata.get(
            "page",
            "Unknown",
        )

        context_parts.append(
            f"""
SOURCE: {source}
PAGE: {page}

{doc.page_content}
"""
        )

    context = "\n\n====================\n\n".join(
        context_parts
    )

    if len(context) > MAX_CONTEXT_CHARS:

        context = (
            context[:MAX_CONTEXT_CHARS]
            + "\n\n[Additional retrieved context truncated for length.]"
        )

    return context


# ============================================================
# INVOKE WITH A HARD TIMEOUT
# ============================================================

def _invoke_with_timeout(llm, prompt, timeout_seconds=LLM_TIMEOUT_SECONDS):
    """
    Run llm.invoke(prompt) with a real wall-clock timeout.

    ChatGoogleGenerativeAI accepts a `timeout` kwarg, but there
    is a known upstream bug where it isn't always respected
    (langchain-google-genai issues #731 / #1180), which can let
    a stalled request hang indefinitely. Running the call in a
    worker thread and bounding it with .result(timeout=...)
    guarantees a real upper bound regardless of whether the SDK
    itself honors its own setting.
    """

    executor = ThreadPoolExecutor(max_workers=1)

    future = executor.submit(llm.invoke, prompt)

    try:
        result = future.result(timeout=timeout_seconds)

    except FutureTimeoutError:
        # Don't block here waiting for the runaway call to finish
        # -- that would defeat the point of the timeout. The
        # thread is left to finish (or fail) in the background;
        # shutdown(wait=False) cleans up the executor without
        # waiting for it.
        executor.shutdown(wait=False)

        raise TimeoutError(
            f"The analysis took longer than {timeout_seconds} "
            "seconds and was stopped. Please try again -- if "
            "this keeps happening, try a shorter resume/job "
            "description or check your network connection."
        )

    else:
        executor.shutdown(wait=False)
        return result


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(question, documents):

    context = build_context(documents)

    # An empty local context means the local knowledge base had
    # nothing relevant for this specific question -- this is the
    # signal used to bring in live web search instead of just
    # giving up or being limited to the two local PDFs.
    need_web_search = not context

    if not context:

        context = (
            "No matching content was found in the local "
            "knowledge base (my_resume.pdf / future_of_jobs.pdf) "
            "for this question. Base the analysis on the resume "
            "and job description supplied directly in the "
            "question below, plus any relevant web search "
            "results, instead."
        )

    prompt = build_prompt(
        question,
        context,
    )

    # PERFORMANCE: only pay the extra latency of live web search
    # when the local documents didn't actually have a relevant
    # match. Most resume/job-fit questions match the local
    # knowledge base directly and can be answered from the fast,
    # non-grounded model; search is reserved for questions the
    # local documents genuinely can't answer, which is also
    # exactly when it's most useful.

    llm_attempts = (
        [get_grounded_llm, get_llm]
        if need_web_search
        else [get_llm]
    )

    last_error = None
    response = None

    for get_client in llm_attempts:

        try:
            llm = get_client()
            response = _invoke_with_timeout(
                llm,
                prompt,
                timeout_seconds=LLM_TIMEOUT_SECONDS,
            )
            break

        except TimeoutError:
            # A real timeout is the same regardless of which
            # client we tried -- no point retrying with the
            # other client, it will just time out again.
            raise

        except Exception as exc:
            last_error = exc
            continue

    if response is None:

        raise RuntimeError(
            "The analysis could not be generated. This is "
            "usually a temporary API or network issue -- please "
            "try again in a moment."
        ) from last_error

    content = response.content

    # Gemini may return a string or structured content.
    if isinstance(content, str):
        result = content.strip()

    elif isinstance(content, list):

        text_parts = []

        for block in content:

            if isinstance(block, dict):

                text = block.get("text")

                if text:
                    text_parts.append(str(text))

            elif isinstance(block, str):

                text_parts.append(block)

        result = "\n".join(text_parts).strip()

    else:
        result = str(content).strip()

    # EDGE CASE: Gemini can return an empty/blank response (e.g.
    # a safety block with no visible text). Surface a clear
    # message instead of silently showing a blank analysis card.
    if not result:

        result = (
            "The model did not return a usable answer for this "
            "question. This can happen if the question is out of "
            "scope for this assistant, or if the response was "
            "blocked. Please rephrase the question or try again."
        )

    return result