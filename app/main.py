from fastapi import FastAPI
from pydantic import BaseModel

from app.ragflow_client import RAGFlowClient

app = FastAPI(title="NYP Engineering Diploma Chatbot Backend")

ragflow = RAGFlowClient()


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {
        "message": "NYP Engineering Diploma Chatbot Backend is running."
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):
    answer = ragflow.ask(request.question)

    return {
        "question": request.question,
        "answer": answer
    }