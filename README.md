# Document Q&A System — LangChain + ChromaDB + Groq + Ollama

A production-style **Retrieval-Augmented Generation (RAG)** backend API that allows users to upload PDF documents and ask natural language questions — powered by **LangChain**, **ChromaDB**, and dual LLM support via **Groq API (cloud)** and **Ollama (local)**.

Built with a clean modular architecture using **FastAPI** — with session-based conversation memory, persistent vector storage, and **LLM-agnostic design** using LangChain's unified interface.

---

## Features

- Upload any PDF document via REST API
- Ask natural language questions about the document
- Session-based conversation memory (multi-turn Q&A)
- Semantic search using **ChromaDB** vector database (production-ready)
- Cloud LLM inference via **Groq API** (llama-3.1-8b-instant) — fast, free
- Local LLM inference via **Ollama** (LLaMA3) — privacy-first, offline
- **LLM-agnostic design** — switch between Groq and Ollama with one parameter
- Automatic ChromaDB persistence — no manual save/load needed
- Clean modular architecture (services layer pattern)
- Auto-generated API docs via Swagger UI (`/docs`)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI (Python) |
| LLM (Cloud) | LLaMA3.1 via Groq API (llama-3.1-8b-instant) |
| LLM (Local) | LLaMA3 via Ollama |
| Embeddings | nomic-embed-text via OllamaEmbeddings |
| Vector Database | ChromaDB (persistent, SQLite backend) |
| Document Loader | LangChain PyPDFLoader |
| Text Splitting | RecursiveCharacterTextSplitter |
| RAG Chain | LangChain LCEL (Runnable pipeline) |
| Memory | In-memory session-based chat history |
| Data Validation | Pydantic |
| Environment | python-dotenv |

---

## Project Structure

```
DOCUMENT_QNA_LANGCHAIN/
│
├── app/
│   ├── services/
│   │   ├── __init__.py
│   │   ├── memory.py          # Session-based conversation memory
│   │   ├── pdf_loader.py      # PDF loading and text chunking
│   │   ├── rag_chain.py       # LangChain RAG pipeline (LCEL) — Groq + Ollama
│   │   └── vector_store.py    # ChromaDB vector store creation & loading
│   ├── main.py                # FastAPI app & route handlers
│   └── models.py              # Pydantic request/response models
│
├── chroma_db/                 # Auto-persisted ChromaDB vector store
│   └── chroma.sqlite3         # ChromaDB SQLite backend
│
├── .env                       # Environment variables (API keys)
├── .gitignore
├── requirements.txt
├── temp.pdf                   # Temporary uploaded file
└── README.md
```

---

## Working of RAG 

```
User uploads PDF
      │
      ▼
PyPDFLoader loads & extracts text
      │
      ▼
RecursiveCharacterTextSplitter
chunks text (size=1000, overlap=200)
      │
      ▼
OllamaEmbeddings (nomic-embed-text)
converts chunks → vectors
      │
      ▼
ChromaDB Vector Store
automatically persists to chroma_db/
      │
      ▼
User asks a question
      │
      ▼
Question embedded → ChromaDB similarity search
retrieves top-K relevant chunks
      │
      ▼
Session memory appends past conversation
      │
      ▼
LangChain LCEL RAG Chain
{context: retriever, question: passthrough}
→ ChatPromptTemplate
→ ChatGroq (llama-3.1-8b-instant) OR ChatOllama (llama3)
→ StrOutputParser
      │
      ▼
Answer returned to user
```

---

## Dual LLM Architecture

This project supports two LLM backends — switchable via a single parameter:

```python
# Use Groq (cloud — fast, free API)
rag_chain = build_rag_chain(retriever, use_groq=True)

# Use Ollama (local — privacy-first, offline)
rag_chain = build_rag_chain(retriever, use_groq=False)
```

---

## API Endpoints

### `POST /upload`
Upload a PDF document to be processed and indexed into ChromaDB.

**Request:** `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| file | File | PDF document to upload |

**Response:**
```json
{
  "message": "PDF uploaded and processed successfully."
}
```

---

### `POST /ask`
Ask a natural language question about the uploaded document.

**Request Body:**
```json
{
  "session_id": "user_123",
  "question": "What is the main topic of this document?"
}
```

**Response:**
```json
{
  "answer": "The document discusses...",
  "llm_used": "Groq (llama-3.1-8b-instant)"
}
```

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally
- Groq API key (free at [console.groq.com](https://console.groq.com))

### 1. Pull required Ollama models
```bash
ollama pull llama3
ollama pull nomic-embed-text
```

### 2. Clone the repository
```bash
git clone https://github.com/yourgithub/document-qna-langchain.git
cd document-qna-langchain
```

### 3. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Add your Groq API key to `.env`
```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
```

### 6. Run the application
```bash
uvicorn app.main:app --reload
```

### 7. Open Swagger UI
```
http://localhost:8000/docs
```

---

