# Document Q&A System — LangChain + ChromaDB + Groq + Ollama + HuggingFace

A production-style **Retrieval-Augmented Generation (RAG)** backend API that allows users to upload PDF documents and ask natural language questions — powered by **LangChain**, **ChromaDB**, and **triple LLM support** via Groq API (cloud), Ollama (local), and HuggingFace Inference API.

Built with a clean modular architecture using **FastAPI** — with session-based conversation memory, persistent vector storage, and **LLM-agnostic design** using LangChain's unified interface.

---

## Features

- Upload any PDF document via REST API
- Ask natural language questions about the document
- Session-based conversation memory (multi-turn Q&A)
- Semantic search using **ChromaDB** vector database (production-ready)
- Cloud LLM inference via **Groq API** (llama-3.1-8b-instant) — fast, free
- Local LLM inference via **Ollama** (LLaMA3) — privacy-first, offline
- HuggingFace LLM inference via **Mistral-7B-Instruct** — open-source cloud inference
- **HuggingFace Embeddings** (sentence-transformers/all-MiniLM-L6-v2) — no Ollama dependency for embeddings
- **LLM-agnostic design** — switch between Groq, Ollama, and HuggingFace with one parameter
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
| LLM (HuggingFace) | Mistral-7B-Instruct via HuggingFace Inference API |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 via HuggingFaceEmbeddings |
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
│   │   ├── rag_chain.py       # LangChain RAG pipeline — Groq + Ollama + HuggingFace
│   │   └── vector_store.py    # ChromaDB + HuggingFace Embeddings
│   ├── main.py                # FastAPI app & route handlers
│   └── models.py              # Pydantic request/response models
│
├── chroma_db/                 # Auto-persisted ChromaDB vector store
│   └── chroma.sqlite3         # ChromaDB SQLite backend
│
├── .env                       # Environment variables (API keys)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## RAG Pipeline
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
HuggingFaceEmbeddings (all-MiniLM-L6-v2)
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
→ ChatGroq OR ChatOllama OR HuggingFaceEndpoint
→ StrOutputParser
      │
      ▼
Answer returned to user
```

---

## Triple LLM Architecture

This project supports three LLM backends — switchable via a single API parameter:
```python
# Use Groq (cloud — fast, free API)
llm_provider = "groq"

# Use Ollama (local — privacy-first, offline)
llm_provider = "ollama"

# Use HuggingFace (open-source cloud inference)
llm_provider = "huggingface"
```

---

## API Endpoints

### `POST /upload`
Upload a PDF document to be processed and indexed into ChromaDB.

**Query Parameter:**

| Parameter | Type | Default | Options |
|---|---|---|---|
| llm_provider | string | groq | groq, ollama, huggingface |

**Request:** `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| file | File | PDF document to upload |

**Response:**
```json
{
  "message": "PDF uploaded and processed successfully.",
  "llm_used": "Groq (llama-3.1-8b-instant)"
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
- Groq API key — free at [console.groq.com](https://console.groq.com)
- HuggingFace API token — free at [huggingface.co](https://huggingface.co)
- Ollama (optional) — only needed if using `llm_provider=ollama`

### 1. Clone the repository
```bash
git clone https://github.com/yourgithub/document-qna-langchain.git
cd document-qna-langchain
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add API keys to `.env`
```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
HUGGINGFACEHUB_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

### 5. Run the application
```bash
uvicorn app.main:app --reload
```

### 6. Open Swagger UI
```
http://localhost:8000/docs
```

### 7. (Optional) For Ollama local inference only
```bash
ollama pull llama3
ollama serve
```
