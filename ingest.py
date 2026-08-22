from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


DATA_DIR = Path("data")
CHROMA_DIR = Path("chroma_db")

COLLECTION_NAME = "career_documents"

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


def load_pdfs():

    documents = []

    for pdf_file in DATA_DIR.glob("*.pdf"):

        print(f"Loading: {pdf_file.name}")

        reader = PdfReader(pdf_file)

        for page_number, page in enumerate(reader.pages):

            text = page.extract_text() or ""

            if text.strip():

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": pdf_file.name,
                            "page": page_number + 1,
                        },
                    )
                )

    return documents


def main():

    # ========================================================
    # 1. LOAD DOCUMENTS
    # ========================================================

    documents = load_pdfs()

    print(
        f"\nTotal pages loaded: {len(documents)}"
    )

    if not documents:

        raise RuntimeError(
            "No PDF files were found in the data directory."
        )


    # ========================================================
    # 2. SPLIT DOCUMENTS
    # ========================================================

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(
        documents
    )

    print(
        f"Total chunks created: {len(chunks)}"
    )


    # ========================================================
    # 3. EMBEDDINGS
    # ========================================================

    print(
        "\nLoading embedding model..."
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    print(
        "Embedding model loaded."
    )


    # ========================================================
    # 4. CREATE VECTOR DATABASE
    # ========================================================

    print(
        "\nCreating Chroma vector database..."
    )

    # Delete previous local database before rebuilding.
    if CHROMA_DIR.exists():

        import shutil

        shutil.rmtree(CHROMA_DIR)

        print(
            "Previous Chroma database removed."
        )


    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
    )


    print(
        f"Vector database created at: {CHROMA_DIR}"
    )

    print(
        f"Total vectors: {len(chunks)}"
    )


    # ========================================================
    # 5. TEST RETRIEVAL
    # ========================================================

    print(
        "\nTesting retrieval..."
    )

    results = vectorstore.similarity_search(
        "What skills are important for future jobs?",
        k=3,
    )


    print(
        "\nTop retrieved results:"
    )

    print(
        "=" * 60
    )


    for i, result in enumerate(
        results,
        1
    ):

        print(
            f"\nResult {i}"
        )

        print(
            f"Source: {result.metadata.get('source')}"
        )

        print(
            f"Page: {result.metadata.get('page')}"
        )

        print(
            result.page_content[:500]
        )

        print(
            "-" * 60
        )


if __name__ == "__main__":
    main()