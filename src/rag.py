import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from src.prompts import build_prompt


load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHROMA_DIR = str(PROJECT_ROOT / "chroma_db")
COLLECTION_NAME = "career_documents"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-3.6-flash"


# ============================================================
# EMBEDDINGS
# ============================================================
#
# PERFORMANCE FIX:
# HuggingFaceEmbeddings loads the sentence-transformers model
# weights into memory. The previous version rebuilt this from
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

    return Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )


# ============================================================
# GEMINI
# ============================================================

@lru_cache(maxsize=1)
def get_llm():
    """
    Plain Gemini client, no tools bound. Cached as a singleton
    for the same reason as get_embeddings() -- avoids rebuilding
    the client on every question. Also used as a fallback if
    grounded generation (below) fails for any reason.
    """

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is missing from the .env file."
        )

    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=api_key,
    )


@lru_cache(maxsize=1)
def get_grounded_llm():
    """
    Same Gemini client as get_llm(), with Google Search
    grounding bound as a tool.

    When Gemini judges that live web context would help answer
    the question, it performs the search itself server-side and
    folds the results directly into its response -- no separate
    search API key, and no manual search/tool-execution loop
    needed on our side.

    This is what lets the analysis draw on more than just the
    two local PDFs (resume + Future of Jobs report) -- e.g.
    current role expectations, company context, or terminology
    trends that aren't present in either PDF.
    """

    return get_llm().bind_tools([{"google_search": {}}])


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(question, k=5):

    vector_store = get_vector_store()

    documents = vector_store.similarity_search(
        question,
        k=k,
    )

    return documents


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

    return "\n\n====================\n\n".join(
        context_parts
    )


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(question, documents):

    context = build_context(documents)

    # PREVIOUSLY: if the local Chroma knowledge base had no
    # relevant chunks, this function returned a canned refusal
    # string immediately and never even called the LLM -- so
    # the analysis was capped by whatever happened to be in
    # my_resume.pdf / future_of_jobs.pdf, with no fallback.
    #
    # NOW: an empty local knowledge base no longer blocks the
    # analysis. The model still has the resume + job description
    # text (supplied directly in the question) and, via grounded
    # generation below, live web search -- so it can keep going
    # instead of giving up.

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

    # Try grounded generation first so the analysis can draw on
    # current web information, not just the two local PDFs.
    # Falls back to the plain (non-grounded) model if grounding
    # fails for any reason -- e.g. the tool isn't supported by
    # the current API/model version, or a transient network
    # issue -- so a search hiccup never breaks the analysis
    # outright.

    try:

        llm = get_grounded_llm()

        response = llm.invoke(prompt)

    except Exception:

        llm = get_llm()

        response = llm.invoke(prompt)

    content = response.content

    # Gemini may return a string or structured content.
    if isinstance(content, str):

        return content.strip()

    if isinstance(content, list):

        text_parts = []

        for block in content:

            if isinstance(block, dict):

                text = block.get("text")

                if text:
                    text_parts.append(str(text))

            elif isinstance(block, str):

                text_parts.append(block)

        result = "\n".join(text_parts).strip()

        if result:
            return result

    return str(content).strip()