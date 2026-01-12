#!/usr/bin/env python3
"""
Ask questions about a PDF book using RAG (Retrieval-Augmented Generation).

Usage:
    python ask_book.py load path/to/book.pdf    # Load a PDF into the database
    python ask_book.py ask "Your question here" # Ask a question about the book

Requires ANTHROPIC_API_KEY to be set (via .env file or environment variable).
"""

import sys
import os
from dotenv import load_dotenv

load_dotenv()
import chromadb
import chromadb.errors
import fitz  # pymupdf
import anthropic


CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "book"
CHUNK_SIZE = 1000  # characters per chunk
CHUNK_OVERLAP = 200


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():  # Skip empty chunks
            chunks.append(chunk)
        start = end - overlap
    return chunks


def load_pdf(pdf_path: str):
    """Load a PDF into ChromaDB."""
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    print(f"Extracting text from {pdf_path}...")
    text = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(text):,} characters")

    print("Chunking text...")
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks")

    print("Storing in ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Delete existing collection if it exists
    try:
        client.delete_collection(COLLECTION_NAME)
    except chromadb.errors.NotFoundError:
        pass

    collection = client.create_collection(COLLECTION_NAME)

    # Add chunks in batches
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        ids = [f"chunk_{i + j}" for j in range(len(batch))]
        collection.add(documents=batch, ids=ids)
        print(f"  Added {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")

    print(
        'Done! You can now ask questions with: python ask_book.py ask "your question"'
    )


def ask_question(question: str):
    """Ask a question about the loaded book."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Run: export ANTHROPIC_API_KEY='your-key-here'")
        sys.exit(1)

    # Get relevant chunks from ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except chromadb.errors.NotFoundError:
        print(
            "Error: No book loaded. First run: python ask_book.py load path/to/book.pdf"
        )
        sys.exit(1)

    results = collection.query(query_texts=[question], n_results=5)
    context = "\n\n---\n\n".join(results["documents"][0])

    # Ask Claude
    claude = anthropic.Anthropic(api_key=api_key)

    message = claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"""Based on the following excerpts from a book, answer the question.
If the answer isn't in the excerpts, say so.

EXCERPTS:
{context}

QUESTION: {question}""",
            }
        ],
    )

    print(message.content[0].text)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "load":
        if len(sys.argv) < 3:
            print("Usage: python ask_book.py load path/to/book.pdf")
            sys.exit(1)
        load_pdf(sys.argv[2])

    elif command == "ask":
        if len(sys.argv) < 3:
            print('Usage: python ask_book.py ask "Your question here"')
            sys.exit(1)
        ask_question(" ".join(sys.argv[2:]))

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
