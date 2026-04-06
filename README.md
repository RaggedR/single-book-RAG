# Ask Book

A RAG (Retrieval-Augmented Generation) tool that lets you ask questions about PDF books using natural language.

## How It Works

1. **Load a PDF** — Text is extracted and split into ~1000-character chunks
2. **Store in ChromaDB** — Chunks are converted to embeddings and stored in a vector database
3. **Ask questions** — Your question is matched against the chunks to find relevant passages
4. **Get answers** — The relevant passages are sent to Claude, which generates an answer

## Setup

### Install dependencies

```bash
pip install chromadb pymupdf anthropic
```

### Set your API key

```bash
export ANTHROPIC_API_KEY="your-key-here"
```

Get a key at https://console.anthropic.com/settings/keys

## Usage

### Load a book

```bash
python ask_book.py load path/to/book.pdf
```

This only needs to be done once per book. The data is stored in `./chroma_db/`.

### Ask questions

```bash
python ask_book.py ask "Who is the main character?"
python ask_book.py ask "What happens at the end?"
python ask_book.py ask "Describe the relationship between Alice and Bob"
```

## Technology

### ChromaDB (Vector Database)

ChromaDB stores text as high-dimensional vectors (embeddings). When you ask a question, it converts your question to a vector and finds the most similar chunks using cosine similarity. This is much more powerful than keyword search — it understands meaning, not just words.

- Runs locally, no external service needed
- Uses `all-MiniLM-L6-v2` for embeddings by default
- Data persists in `./chroma_db/`

### Claude API

Claude (by Anthropic) is the LLM that reads the retrieved passages and generates answers. The app uses `claude-sonnet-4-20250514` for a good balance of speed and quality.

The API is pay-per-use. Typical costs for this app:
- ~$0.003 per question (input tokens for context)
- ~$0.015 per answer (output tokens)

## Limitations

- Only stores one book at a time (loading a new book replaces the old one)
- Very long books may have less accurate retrieval
- Answers are only as good as the relevant chunks found
