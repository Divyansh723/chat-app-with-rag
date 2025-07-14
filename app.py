from flask import Flask, render_template, request, send_file, session, redirect, url_for, flash
from chat_logic import ask_tempest, save_chat
import os
import uuid
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  

# Load env variables
load_dotenv()

# Embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Flask app setup
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY") or "your-fallback-secret"
SESSION_ID = str(uuid.uuid4())
chat_history = []
summary = ""
show_options = False

# ========= File Readers =========
def read_pdf(file_path):
    reader = PdfReader(file_path)
    return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# ========= Chunking & Embedding =========
def chunk_text(text, chunk_size=500, overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)

def embed_chunks_docs(chunks):
    return embedding_model.embed_documents(chunks)

def store_chunks_in_chroma(text, collection_name="default"):
    chunks = chunk_text(text)
    persist_directory = os.path.join("chroma_db", collection_name)
    os.makedirs(persist_directory, exist_ok=True)

    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    return len(chunks)

# ========= Routes =========
@app.route("/", methods=["GET", "POST"])
def chat():
    global chat_history, summary, show_options
    is_typing = False

    if request.method == "POST":
        user_msg = request.form.get("message", "").strip()
        if not user_msg:
            return render_template("chat.html", conversation=chat_history, summary=summary)

        if user_msg.lower() == "show summary":
            _, summary = ask_tempest("dummy")
        elif user_msg.lower() == "show options":
            show_options = True
        else:
            is_typing = True
            response, summary = ask_tempest(user_msg)
            chat_history.append(("You", user_msg))
            chat_history.append(("Tempest", response))
            save_chat(chat_history, SESSION_ID)

    return render_template(
        "chat.html",
        conversation=chat_history,
        summary=summary,
        is_typing=is_typing,
        show_options=show_options
    )

@app.route("/upload")
def upload_page():
    return render_template("upload.html")

@app.route("/process_upload", methods=["POST"])
def process_upload():
    uploaded_file = request.files.get("file")

    if not uploaded_file:
        flash("❌ No file uploaded!", "error")
        return redirect(url_for("upload_page"))

    filename = secure_filename(uploaded_file.filename)
    ext = os.path.splitext(filename)[1].lower()

    if ext not in [".pdf", ".txt"]:
        flash("❌ Invalid file type. Only PDF and TXT allowed.", "error")
        return redirect(url_for("upload_page"))

    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", filename)
    uploaded_file.save(file_path)

    # Read & process file
    try:
        if ext == ".pdf":
            full_text = read_pdf(file_path)
        else:
            full_text = read_txt(file_path)
    except Exception as e:
        flash(f"❌ Error reading file: {str(e)}", "error")
        return redirect(url_for("upload_page"))

    # Chunk, embed, store
    try:
        collection_name = os.path.splitext(filename)[0]
        num_chunks = store_chunks_in_chroma(full_text, collection_name=collection_name)
        flash(f"✅ File processed, embedded & stored. Chunks: {num_chunks}", "success")
    except Exception as e:
        flash(f"❌ Error storing in vector DB: {str(e)}", "error")

    return redirect(url_for("upload_page"))

@app.route("/download/<format>")
def download(format):
    filepath = f"chat_logs/{SESSION_ID}.{format}"
    if format == "txt":
        save_chat(chat_history, SESSION_ID, as_pdf=False)
    elif format == "pdf":
        save_chat(chat_history, SESSION_ID, as_pdf=True)

    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return "File not found", 404

@app.route("/download/summary")
def download_summary():
    global summary
    if summary:
        path = f"chat_logs/summary_{SESSION_ID}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(summary)
        return send_file(path, as_attachment=True)
    return "No summary available", 404

# ========= Main =========
if __name__ == "__main__":
    os.makedirs("chat_logs", exist_ok=True)
    app.run(debug=True)
