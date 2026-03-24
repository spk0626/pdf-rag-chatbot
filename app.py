import gradio as gr
from gradio.themes import Soft
from pdf_loader import load_and_chunk_pdf
from rag import index_chunks, retrieve, generate_response

# ── Global state ────────────────────────────────────────────────────────────────
collection = None
chat_history: list[tuple[str, str]] = []


# ── Handlers ────────────────────────────────────────────────────────────────────
def upload_pdf(file):
    """Load a PDF, chunk it, and index it into ChromaDB."""
    global collection, chat_history

    if file is None:
        return "⚠️  Please select a PDF file first.", []

    # Gradio 4.x passes a file path string; 3.x passes an object with .name
    file_path = file if isinstance(file, str) else file.name

    try:
        chunks = load_and_chunk_pdf(file_path)

        if not chunks:
            return "❌  Could not extract text from this PDF. Is it a scanned image?", []

        collection = index_chunks(chunks)
        chat_history = []

        return (
            f"✅  PDF loaded! {len(chunks)} chunks indexed.\n"
            "You can now ask questions in the chat →"
        ), []

    except Exception as e:
        return f"❌  Error loading PDF: {e}", []


def respond(message: str, history: list[dict] | None):
    """Handle a chat message: retrieve relevant chunks and generate a response."""
    global collection, chat_history

    if history is None:
        history = []

    message = message.strip()
    if not message:
        return history, ""

    if collection is None:
        history.append({"role": "user", "content": message})
        history.append(
            {
                "role": "assistant",
                "content": "⚠️  Please upload a PDF first using the panel on the left.",
            }
        )
        return history, ""

    try:
        chunks = retrieve(message, collection)
        answer = generate_response(message, chunks, chat_history)
    except Exception as e:
        answer = f"❌  Error generating response: {e}"

    chat_history.append((message, answer))
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})

    return history, ""


def clear_chat():
    """Reset conversation history."""
    global chat_history
    chat_history = []
    return []


# ── UI ──────────────────────────────────────────────────────────────────────────
CHAT_CSS = """
#chatbot .message {
    max-width: 78% !important;
}
"""


with gr.Blocks(title="PDF Q&A Chatbot") as demo:

    gr.Markdown(
        """
        # 📄 PDF Q & A Chatbot
        Upload any PDF and ask questions about it.  
        Powered by **RAG** · **all-MiniLM-L6-v2** embeddings · **Llama 3 via Groq**
        """
    )

    with gr.Row():
        # ── Left panel: upload ──────────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📁 1. Upload Your Document")
            pdf_upload = gr.File(
                label="Select a PDF",
                file_types=[".pdf"],
            )
            upload_btn = gr.Button("🔄  Load PDF", variant="primary")
            upload_status = gr.Textbox(
                label="Status",
                interactive=False,
                lines=3,
                placeholder="Upload a PDF to get started...",
            )

        # ── Right panel: chat ───────────────────────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### 💬 2. Ask Questions")
            chatbot = gr.Chatbot(
                height=420,
                label="Conversation",
                elem_id="chatbot",
            )
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask something about your PDF...",
                    label="",
                    scale=5,
                    container=False,
                )
                send_btn = gr.Button("Send ➤", variant="primary", scale=1)
            clear_btn = gr.Button("🗑️  Clear Chat", variant="secondary")

    # ── Event wiring ────────────────────────────────────────────────────────────
    upload_btn.click(
        fn=upload_pdf,
        inputs=pdf_upload,
        outputs=[upload_status, chatbot],
    )

    send_btn.click(
        fn=respond,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, msg_input],
    )

    msg_input.submit(
        fn=respond,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, msg_input],
    )

    clear_btn.click(fn=clear_chat, outputs=chatbot)


if __name__ == "__main__":
    demo.launch(theme=Soft(), css=CHAT_CSS)