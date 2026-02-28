from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

def create_vector_store(documents):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore

def load_vector_store():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )