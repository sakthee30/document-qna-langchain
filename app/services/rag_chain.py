from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

def build_rag_chain(retriever, use_groq=True):

    if use_groq:

        llm = ChatGroq(
            model="llama-3.1-8b-instant",   
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.2
        )

        
    else:
        #local Ollama
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model="llama3")

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
    )