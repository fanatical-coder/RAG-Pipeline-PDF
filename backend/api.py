from fastapi import FastAPI
from pydantic import BaseModel
from rag_query_lancedb import answer_question
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,

    
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/query")
def query(req: QueryRequest):
    return answer_question(req.question)
