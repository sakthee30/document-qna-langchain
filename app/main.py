from fastapi import FastAPI, UploadFile, File
from app.models import AskRequest

from app.services.pdf_loader import load_and_split_pdf
from app.services.vector_store import create_vector_store, load_vector_store
from app.services.rag_chain import build_rag_chain
from app.services.memory import get_session_history, add_to_history

app = FastAPI()

vectorstore = None
rag_chain = None

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global vectorstore, rag_chain

    contents = await file.read()

    with open("temp.pdf", "wb") as f:
        f.write(contents)

    documents = load_and_split_pdf("temp.pdf")
    vectorstore = create_vector_store(documents)

    retriever = vectorstore.as_retriever()
    rag_chain = build_rag_chain(retriever)

    return {"message": "PDF uploaded and processed successfully."}


@app.post("/ask")
def ask(req: AskRequest):
    global rag_chain, vectorstore

    if rag_chain is None:
        vectorstore = load_vector_store()
        retriever = vectorstore.as_retriever()
        rag_chain = build_rag_chain(retriever)

    history = get_session_history(req.session_id)
    history_text = "\n".join(history)

    full_question = f"""
    Previous conversation:
    {history_text}

    Current question:
    {req.question}
    """

    answer = rag_chain.invoke(full_question)

    add_to_history(req.session_id, req.question, answer)

    return {"answer": answer}