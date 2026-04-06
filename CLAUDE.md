# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install chromadb pymupdf anthropic

# Load a PDF into the vector database
python ask_book.py load path/to/book.pdf

# Ask a question about the loaded book
python ask_book.py ask "Your question here"
```

Requires `ANTHROPIC_API_KEY` environment variable.

## Architecture

This is a RAG (Retrieval-Augmented Generation) pipeline in a single file:

1. **PDF extraction** — `pymupdf` extracts text from PDFs
2. **Chunking** — Text is split into 1000-char chunks with 200-char overlap
3. **Vector storage** — ChromaDB stores chunks with embeddings (uses `all-MiniLM-L6-v2`)
4. **Retrieval** — Questions are embedded and matched to the 5 most similar chunks
5. **Generation** — Claude receives the chunks and generates an answer

Data persists in `./chroma_db/`. Loading a new book replaces the previous one.
