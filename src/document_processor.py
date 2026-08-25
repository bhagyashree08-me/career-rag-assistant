#document_processor.py
from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def load_pdfs():
    """
    Load all PDFs from the data/ directory.

    Each PDF page becomes a LangChain Document with:
    - source: PDF filename
    - page: 1-based page number
    """

    documents = []

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Data directory not found: {DATA_DIR}"
        )

    pdf_files = list(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in: {DATA_DIR}"
        )

    for pdf_file in pdf_files:

        print(f"Loading: {pdf_file.name}")

        reader = PdfReader(str(pdf_file))

        for page_number, page in enumerate(reader.pages, start=1):

            text = page.extract_text() or ""

            text = text.strip()

            if not text:
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_file.name,
                        "page": page_number,
                    },
                )
            )

    return documents


def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(documents)

    return chunks