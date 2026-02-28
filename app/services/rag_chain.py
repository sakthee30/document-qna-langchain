from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

def build_rag_chain(retriever):

    llm = ChatOllama(model="llama3")


    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question based only on the context below.

        Context:
        {context}

        Question:
        {question}
        """
    )

    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )