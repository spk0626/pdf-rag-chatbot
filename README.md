# PDF RAG Chatbot

A simple Retrieval-Augmented Generation (RAG) chatbot that lets you upload a PDF and ask questions grounded in that document.

## What it does

- Uploads a PDF and extracts text
- Splits text into chunks
- Creates embeddings with all-MiniLM-L6-v2
- Stores and retrieves relevant chunks with ChromaDB
- Generates answers with Groq LLMs using retrieved context

## Tech stack

- Python
- Gradio (UI)
- PyMuPDF (PDF parsing)
- sentence-transformers (embeddings)
- ChromaDB (vector store)
- Groq API (LLM inference)

## Project structure

- app.py: Gradio app and UI handlers
- pdf_loader.py: PDF text extraction and chunking
- rag.py: indexing, retrieval, and LLM response generation
- requirements.txt: Python dependencies
- .env.example: example environment variables

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

   pip install -r requirements.txt

3. Create a local environment file from .env.example and set your key:

   GROQ_API_KEY=your_groq_api_key_here
   LLM_MODEL=llama-3.1-8b-instant

Notes:
- LLM_MODEL is optional. If omitted, the app uses llama-3.1-8b-instant.
- The app has fallback models configured in rag.py.

## Run

Start the app:

python app.py

Open the local URL shown in the terminal (usually http://127.0.0.1:7860).

## How to use

1. Upload a PDF.
2. Click Load PDF.
3. Ask questions in the chat box.

## Troubleshooting

- Model decommission error from Groq:
  Update LLM_MODEL in your .env to a currently supported Groq model.

- Hugging Face unauthenticated warning:
  Optional. Set HF_TOKEN to improve download rate limits.

- First run is slow:
  Expected, because embedding model files are downloaded and loaded.
