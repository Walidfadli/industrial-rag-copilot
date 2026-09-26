from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


DATA_DIR = Path("data/documents")


def load_documents():
    documents = []

    for pdf_file in DATA_DIR.glob("*.pdf"):
        print(f"Loading: {pdf_file.name}")

        loader = PyPDFLoader(str(pdf_file))
        pages = loader.load()

        for page in pages:
            page.metadata["source_file"] = pdf_file.name

        documents.extend(pages)

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(documents)

    return chunks


if __name__ == "__main__":
    documents = load_documents()

    print(f"\nPages loaded: {len(documents)}")

    chunks = split_documents(documents)

    print(f"Chunks created: {len(chunks)}")

    print("\nExample chunk:")
    print(chunks[0].page_content[:1000])

    print("\nMetadata:")
    print(chunks[0].metadata)