from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import os

CHROMA_DIR = "chroma_db"  

def get_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text")

def create_vector_store(documents):
    embeddings = get_embeddings()

    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    return vectorstore

def load_vector_store():
    embeddings = get_embeddings()

    
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
