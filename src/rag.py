#rag.py
import os
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

def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


# ============================================================
# VECTOR DATABASE
# ============================================================

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

def get_llm():

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is missing from the .env file."
        )

    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=api_key,
    )


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

    if not context:

        return (
            "I could not find relevant information in the "
            "current knowledge base."
        )

    prompt = build_prompt(
        question,
        context,
    )

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