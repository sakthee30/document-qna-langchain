from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

def build_rag_chain(retriever, llm_provider="groq"):
    
    if llm_provider == "groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.2
        )
        llm_label = "Groq (llama-3.1-8b-instant)"

    elif llm_provider == "ollama":
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model="llama3")
        llm_label = "Ollama (LLaMA3 - local)"

    elif llm_provider == "huggingface":
        from langchain_huggingface import HuggingFaceEndpoint
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            temperature=0.2,
            max_new_tokens=512
        )
        llm_label = "HuggingFace (Mistral-7B-Instruct)"

    else:
        raise ValueError(f"Unknown llm_provider: {llm_provider}. Use 'groq', 'ollama', or 'huggingface'.")

    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful assistant. Answer the question based ONLY on the context below.
        If the answer is not in the context, say "I don't know based on the provided document."

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
    )

    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    ), llm_label