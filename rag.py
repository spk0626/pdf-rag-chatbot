import os
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Model initialisation ────────────────────────────────────────────────────────
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()          # chroma is used because its in-memory, fast, no disk setup needed
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

COLLECTION_NAME = "pdf_chunks"
LLM_MODEL = "llama3-8b-8192"


# ── Indexing ────────────────────────────────────────────────────────────────────
def index_chunks(chunks: list[str]):
    """Embed chunks and load them into a fresh ChromaDB collection."""
    # Always start with a clean slate when a new PDF is uploaded
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = chroma_client.create_collection(COLLECTION_NAME)

    embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
    )

    return collection


# ── Retrieval ───────────────────────────────────────────────────────────────────
def retrieve(query: str, collection, top_k: int = 3) -> list[str]:
    """Return the top_k most semantically similar chunks for a query."""
    query_embedding = embedder.encode([query], show_progress_bar=False).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    return results["documents"][0]   # list of chunk strings


# ── Generation ──────────────────────────────────────────────────────────────────
def generate_response(
    query: str,
    chunks: list[str],
    chat_history: list[tuple[str, str]],
) -> str:
    """Build a RAG prompt and call Groq to get a grounded answer."""

    # Format retrieved context with source labels
    context = "\n\n".join(
        f"[Source {i + 1}]: {chunk}" for i, chunk in enumerate(chunks)
    )

    # Include last 3 turns of conversation so the bot has memory
    history_text = ""
    for user_msg, bot_msg in chat_history[-3:]:
        history_text += f"User: {user_msg}\nAssistant: {bot_msg}\n"

    system_prompt = (
        "You are a helpful assistant that answers questions strictly based on "
        "the document context provided.\n"
        "- Always cite which source your answer comes from, e.g. [Source 1].\n"
        "- If the answer cannot be found in the context, say so clearly.\n"
        "- Be concise, accurate, and friendly."
    )

    user_prompt = (
        f"Document Context:\n{context}\n\n"
        f"Previous Conversation:\n{history_text}\n"
        f"User Question: {query}\n\n"
        "Answer based only on the context above:"
    )

    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=1024,
    )

    content = response.choices[0].message.content
    if content is None:
        return "I could not generate a response from the model output."

    return content