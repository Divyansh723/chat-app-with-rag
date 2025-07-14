
# 🧠 Chat App with RAG (Retrieval-Augmented Generation)

A modern context-aware chatbot built with LangChain, Gemini Pro, and ChromaDB that supports semantic document search via vector database tools. It can answer user queries using uploaded files and fallback to general knowledge if needed.

## 🚀 Features

- 💬 Natural language chat with memory and summarization
- 🔎 Document-aware question answering using vector search (ChromaDB + HuggingFace embeddings)
- 🛠️ Tool calling with `db_search` and support for custom tools like `add`, `web_search`, etc.
- 🧠 Gemini 2.0 Flash LLM with LangGraph for stateful conversation flows
- 📁 Upload & parse documents (PDF, TXT)
- 📄 Chat history saving as `.txt` or `.pdf`
- 🌐 Ready for local or server deployment (Flask compatible)

## 📦 Tech Stack

- Python 3.12+
- [LangChain](https://www.langchain.com/)
- [Gemini Pro (via LangChain Google)](https://ai.google.dev/)
- [ChromaDB](https://www.trychroma.com/)
- [HuggingFace Sentence Transformers](https://huggingface.co/sentence-transformers)
- Flask (for frontend)

## 📁 Folder Structure

```
chat_app/
│
├── app.py                    # Flask entrypoint
├── chat_logic.py             # Core LangGraph + RAG logic
├── requirements.txt          # Dependencies
├── .env                      # Environment config
├── templates/
│   └── upload.html, chat.html
├── static/
│   └── styles.css, upload.css
├── chroma_db/                # Vector DB persistence
├── uploads/                  # Uploaded files
└── chat_logs/                # Saved chat transcripts
```

## ⚙️ Usage

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Set environment variables in `.env`:**

```env
GOOGLE_API_KEY=your_api_key
LANGSMITH_API_KEY=your_api_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=your project name
FLASK_SECRET_KEY=your_api_key
MONGO_URI=your_api_key
TAVILY_API_KEY=your_api_key
```

3. **Run the app:**

```bash
python app.py
```

Visit `http://localhost:5000` in your browser.

---

**Built with ❤️ using LangChain + Gemini + ChromaDB**