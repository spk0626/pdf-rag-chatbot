# PDF RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot with an API-first architecture:
- FastAPI backend for indexing, retrieval, and response generation
- TypeScript (React + Vite) frontend that consumes FastAPI endpoints

## What it does

- Uploads a PDF and extracts text
- Splits text into overlapping chunks
- Creates embeddings with all-MiniLM-L6-v2
- Stores and retrieves relevant chunks with ChromaDB
- Generates grounded answers with Groq LLMs
- Isolates each uploaded document in a per-session index

## Tech stack

- Python
- FastAPI (backend API)
- TypeScript + React + Vite (frontend client)
- PyMuPDF (PDF parsing)
- sentence-transformers (embeddings, Hugging Face ecosystem)
- ChromaDB (vector store)
- Groq API (LLM inference)

## Project structure

- api/main.py: FastAPI app and routes
- core/pdf_loader.py: PDF text extraction and chunking utilities
- core/rag_service.py: session-aware RAG service
- src/app: frontend app UI
- src/app/api/client.ts: typed API client for backend endpoints
- scripts/smoke_test.ps1: local backend/frontend smoke test script
- legacy/gradio/app.py: legacy Gradio client
- legacy/gradio/requirements.txt: legacy Gradio dependencies
- tests/test_api.py: backend API tests
- requirements.txt: Python dependencies
- package.json: frontend dependencies and scripts
- .env.example: example backend/frontend environment variables

## Migration notes

- The legacy root files pdf_loader.py and rag.py were replaced by the modular code under core/ and api/.
- ChromaDB now persists to the path in CHROMA_PERSIST_DIR, which defaults to .chroma.
- If REDIS_URL is set, session metadata is stored in Redis for multi-instance deployments.
- TypeScript frontend is now the primary UI and talks to FastAPI over HTTP.
- legacy/gradio/app.py is an optional legacy Gradio client.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

   pip install -r requirements.txt

3. Install frontend dependencies:

  npm install

4. Create a local environment file from .env.example and set values:

   GROQ_API_KEY=your_groq_api_key_here
   LLM_MODEL=llama-3.1-8b-instant
  API_BASE_URL=http://127.0.0.1:8000
  VITE_API_BASE_URL=http://127.0.0.1:8000
  CHROMA_PERSIST_DIR=.chroma
  REDIS_URL=
  CORS_ALLOW_ORIGINS=http://127.0.0.1:5173,http://localhost:5173

Notes:
- LLM_MODEL is optional. If omitted, the app uses llama-3.1-8b-instant.
- VITE_API_BASE_URL is used by the TypeScript frontend.
- CHROMA_PERSIST_DIR enables disk-backed vector storage.
- REDIS_URL is optional and enables shared session metadata across app instances.
- CORS_ALLOW_ORIGINS controls which frontend origins can call the API.

## Run

1. Start the FastAPI backend:

   uvicorn api.main:app --reload

2. In a second terminal, start the TypeScript frontend:

  npm run dev

3. Open the local frontend URL shown by Vite (usually http://127.0.0.1:5173).

## Smoke test

Run this from the project root to verify backend and frontend startup:

  npm run smoke:e2e

The smoke test starts both processes, checks API and frontend reachability, and shuts both down.

## Legacy Gradio client (optional)

If you want to run the old Gradio client:

1. Install legacy dependencies:

  pip install -r legacy/gradio/requirements.txt

2. Start the backend:

  uvicorn api.main:app --reload

3. Start Gradio client:

  python legacy/gradio/app.py

## How to use

1. Upload a PDF.
2. Click Load PDF.
3. Ask questions in the chat box.

## Troubleshooting

- FastAPI is not running:
  If frontend upload or chat fails, ensure the backend is running at API_BASE_URL.

- CORS error in browser console:
  Add your frontend origin to CORS_ALLOW_ORIGINS.

- Persisted data not updating:
  Delete the CHROMA_PERSIST_DIR folder to reset the local ChromaDB store.

- Model decommission error from Groq:
  Update LLM_MODEL in your .env to a currently supported Groq model.

- Hugging Face unauthenticated warning:
  Optional. Set HF_TOKEN to improve download rate limits.

- First run is slow:
  Expected, because embedding model files are downloaded and loaded.
