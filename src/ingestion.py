
from pathlib import Path
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_DIR = Path("data/documents")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


# --------------------------------------------------
# Load PDF documents
# --------------------------------------------------

def load_documents():
    documents = []

    for pdf_file in sorted(DATA_DIR.glob("*.pdf")):

        print(f"Loading: {pdf_file.name}")

        loader = PyPDFLoader(str(pdf_file))
        pages = loader.load()

        for page in pages:

            page.metadata["source_file"] = pdf_file.name

            # PyPDFLoader uses zero-based page indexes.
            # Convert to human-readable page numbers.
            page.metadata["page"] = (
                page.metadata.get("page", 0) + 1
            )

        documents.extend(pages)

    return documents


# --------------------------------------------------
# Clean extracted PDF line
# --------------------------------------------------

def clean_line(line):
    line = line.replace("\t", " ")
    line = re.sub(r"\s+", " ", line)

    return line.strip()


# --------------------------------------------------
# Detect useful section headings
# --------------------------------------------------

def is_section_heading(line):

    line = clean_line(line)

    if not line:
        return False

    # --------------------------------------------------
    # Basic filters
    # --------------------------------------------------

    if len(line) < 4:
        return False

    if len(line) > 100:
        return False

    # Page numbers
    if re.fullmatch(r"\d+", line):
        return False

    # Common non-section labels
    ignored_labels = {
        "WARNING",
        "CAUTION",
        "DANGER",
        "ATTENTION",
        "NOTE",
        "NOTICE",
        "CONTENTS",
        "TABLE OF CONTENTS",
    }

    if line.upper() in ignored_labels:
        return False

    # --------------------------------------------------
    # Ignore obvious document titles
    # --------------------------------------------------

    document_title_patterns = [
        "MANUAL FOR",
        "APPLICATIONS MANUAL",
        "THREE SCREW",
        "CIG SERIES",
    ]

    upper_line = line.upper()

    if any(
        pattern in upper_line
        for pattern in document_title_patterns
    ):
        return False

    # --------------------------------------------------
    # Numbered sections
    #
    # Examples:
    # 1. Installation
    # 2. Operation
    # 4.1 Maintenance
    # --------------------------------------------------

    if re.match(
        r"^\d+(\.\d+)*[\s\-:]+[A-Za-z]",
        line
    ):
        return True

    # --------------------------------------------------
    # Lettered sections
    #
    # Examples:
    # A. GENERAL
    # B. INSTALLATION
    # F. MAINTENANCE
    # G. TROUBLESHOOTING
    # --------------------------------------------------

    if re.match(
        r"^[A-Z]\.\s+[A-Za-z]",
        line
    ):
        return True

    # --------------------------------------------------
    # Uppercase headings
    #
    # Example:
    # PUMP INSTALLATION
    # TROUBLESHOOTING
    # MAINTENANCE
    # --------------------------------------------------

    letters = [
        char
        for char in line
        if char.isalpha()
    ]

    if letters:

        uppercase_ratio = sum(
            char.isupper()
            for char in letters
        ) / len(letters)

        if uppercase_ratio >= 0.90:

            # Avoid treating long sentences as headings
            if len(line.split()) <= 10:

                return True

    return False


# --------------------------------------------------
# Detect section on a page
# --------------------------------------------------

def detect_section_from_page(
    text,
    current_section="GENERAL",
):

    lines = [
        clean_line(line)
        for line in text.splitlines()
        if clean_line(line)
    ]

    for line in lines:

        if is_section_heading(line):

            return line

    return current_section


# --------------------------------------------------
# Assign sections across pages
# --------------------------------------------------

def assign_sections(documents):

    current_section = "GENERAL"

    for document in documents:

        detected_section = detect_section_from_page(
            document.page_content,
            current_section,
        )

        current_section = detected_section

        document.metadata["section"] = current_section

    return documents


# --------------------------------------------------
# Split documents into chunks
# --------------------------------------------------

def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = []

    for document in documents:

        page_chunks = splitter.split_documents(
            [document]
        )

        source_file = document.metadata.get(
            "source_file",
            "unknown",
        )

        page_number = document.metadata.get(
            "page",
            "unknown",
        )

        section = document.metadata.get(
            "section",
            "GENERAL",
        )

        for chunk_index, chunk in enumerate(
            page_chunks
        ):

            chunk.metadata.update(
                {
                    "source_file": source_file,
                    "page": page_number,
                    "section": section,
                    "chunk_index": chunk_index,
                    "chunk_id": (
                        f"{Path(source_file).stem}"
                        f"_p{page_number}"
                        f"_c{chunk_index}"
                    ),
                }
            )

            chunks.append(chunk)

    return chunks


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    print("=" * 70)
    print("INDUSTRIAL MAINTENANCE DOCUMENT INGESTION")
    print("=" * 70)

    # --------------------------------------------------
    # Load PDFs
    # --------------------------------------------------

    documents = load_documents()

    print(
        f"\nPages loaded: {len(documents)}"
    )

    # --------------------------------------------------
    # Assign sections
    # --------------------------------------------------

    documents = assign_sections(
        documents
    )

    # --------------------------------------------------
    # Create chunks
    # --------------------------------------------------

    chunks = split_documents(
        documents
    )

    print(
        f"Chunks created: {len(chunks)}"
    )

    # --------------------------------------------------
    # Display first chunk
    # --------------------------------------------------

    if chunks:

        print("\n" + "=" * 70)
        print("EXAMPLE CHUNK")
        print("=" * 70)

        print(
            chunks[0].page_content[:1000]
        )

        print("\n" + "=" * 70)
        print("METADATA")
        print("=" * 70)

        print(
            chunks[0].metadata
        )

    # --------------------------------------------------
    # Display examples
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("CHUNK EXAMPLES")
    print("=" * 70)

    for i, chunk in enumerate(
        chunks[:10],
        start=1,
    ):

        print(
            f"\n--- Chunk {i} ---"
        )

        print(
            f"Source: "
            f"{chunk.metadata.get('source_file')}"
        )

        print(
            f"Page: "
            f"{chunk.metadata.get('page')}"
        )

        print(
            f"Section: "
            f"{chunk.metadata.get('section')}"
        )

        print(
            f"Chunk ID: "
            f"{chunk.metadata.get('chunk_id')}"
        )

        print(
            f"Text preview: "
            f"{chunk.page_content[:250]}"
        )

