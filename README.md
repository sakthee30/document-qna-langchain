# Document Q&A System — LangChain + FAISS + Ollama

A production-style **Retrieval-Augmented Generation (RAG)** backend API that allows users to upload PDF documents and ask natural language questions about them — powered by **LangChain**, **FAISS**, and a locally running **LLaMA3 LLM via Ollama**.

Built with a clean modular architecture using **FastAPI** — with session-based conversation memory, persistent vector storage, and fully local LLM inference.

---

## Features

- Upload any PDF document via REST API
- Ask natural language questions about the document
- Session-based conversation memory (multi-turn Q&A)
- Semantic search using FAISS vector database
- Local LLM inference via Ollama (LLaMA3) — no external API calls
- Persistent FAISS index — no need to re-process documents on restart
- Clean modular architecture (services layer pattern)
- Auto-generated API docs via Swagger UI (`/docs`)
- Docker support

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI (Python) |
| LLM | LLaMA3 via Ollama (local) |
| Embeddings | nomic-embed-text via OllamaEmbeddings |
| Vector Database | FAISS (persistent local index) |
| Document Loader | LangChain PyPDFLoader |
| Text Splitting | RecursiveCharacterTextSplitter |
| RAG Chain | LangChain LCEL (Runnable pipeline) |
| Memory | In-memory session-based chat history |
| Data Validation | Pydantic |

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
│   │   ├── rag_chain.py       # LangChain RAG pipeline (LCEL)
│   │   └── vector_store.py    # FAISS vector store creation & loading
│   ├── main.py                # FastAPI app & route handlers
│   └── models.py              # Pydantic request/response models
│
├── data/
│   ├── sample.pdf
│   └── faiss_index/           # Persisted FAISS vector index
│
├── .env                       # Environment variables
├── .dockerignore
├── Dockerfile
├── requirements.txt
├── temp.pdf                   # Temporary uploaded file
└── README.md
```

---

## How RAG Works in This Project

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
FAISS Vector Store
saves index locally (faiss_index/)
      │
      ▼
User asks a question
      │
      ▼
Question embedded → FAISS similarity search
retrieves top-K relevant chunks
      │
      ▼
Session memory appends past conversation
      │
      ▼
LangChain LCEL RAG Chain
{context: retriever, question: passthrough}
→ ChatPromptTemplate
→ ChatOllama (llama3)
→ StrOutputParser
      │
      ▼
Answer returned to user
```

---

## API Endpoints

### `POST /upload`
Upload a PDF document to be processed and indexed.

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

| Field | Type | Description |
|---|---|---|
| session_id | string | Unique ID to maintain conversation history per user |
| question | string | Natural language question about the document |

**Response:**
```json
{
  "answer": "The document discusses..."
}
```

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally
- LLaMA3 and nomic-embed-text models pulled in Ollama

### 1. Pull required Ollama models
```bash
ollama pull llama3
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

### 5. Run the application
```bash
uvicorn app.main:app --reload
```

### 6. Open Swagger UI
```
http://localhost:8000/docs
```

