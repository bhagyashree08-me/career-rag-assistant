from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


DATA_DIR = Path("data")


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


def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )

    return splitter.split_documents(documents)