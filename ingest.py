from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.document_processor import (
    load_pdfs,
    split_documents,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

CHROMA_DIR = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "career_documents"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("CAREER RAG — DOCUMENT INGESTION")
    print("=" * 60)

    # --------------------------------------------------------
    # LOAD DOCUMENTS
    # --------------------------------------------------------

    print("\n[1/4] Loading PDF documents...")

    documents = load_pdfs()

    print(
        f"Loaded {len(documents)} pages."
    )

    if not documents:

        raise RuntimeError(
            "No readable PDF pages were found."
        )

    # --------------------------------------------------------
    # SPLIT DOCUMENTS
    # --------------------------------------------------------

    print("\n[2/4] Splitting documents...")

    chunks = split_documents(documents)

    print(
        f"Created {len(chunks)} document chunks."
    )

    # --------------------------------------------------------
    # EMBEDDINGS
    # --------------------------------------------------------

    print("\n[3/4] Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    print("Embedding model loaded.")

    # --------------------------------------------------------
    # CHROMA
    # --------------------------------------------------------

    print("\n[4/4] Creating Chroma vector database...")

    # Delete/recreate behavior is intentionally avoided here.
    # The database should be deleted manually when a completely
    # fresh rebuild is required.

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
    )

    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)

    print(f"Vector database: {CHROMA_DIR}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Vectors created: {len(chunks)}")

    # --------------------------------------------------------
    # TEST RETRIEVAL
    # --------------------------------------------------------

    print("\nTesting retrieval...")

    results = vectorstore.similarity_search(
        "What skills are important for future jobs?",
        k=3,
    )

    print("\nTop results:")
    print("-" * 60)

    for index, result in enumerate(results, start=1):

        print(f"\nResult {index}")

        print(
            f"Source: "
            f"{result.metadata.get('source', 'Unknown')}"
        )

        print(
            f"Page: "
            f"{result.metadata.get('page', 'Unknown')}"
        )

        print(
            result.page_content[:500]
        )

        print("-" * 60)


if __name__ == "__main__":
    main()