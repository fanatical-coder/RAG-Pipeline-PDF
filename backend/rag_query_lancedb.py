import lancedb
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_DIR = PROJECT_ROOT / "ingestion" / "lancedb"


EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GENERATION_MODEL = "google/flan-t5-large"

TOP_K = 4
MAX_CONTEXT_CHARS = 1000
MAX_NEW_TOKENS = 200



_db = None
_table = None
_embedder = None
_generator = None


def _load_db():
    global _db, _table
    if _table is None:
        _db = lancedb.connect(DB_DIR)
        _table = _db.open_table("pdf_chunks")
    return _table


def _load_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def _load_generator():
    global _generator
    if _generator is None:
        _generator = pipeline(
            "text2text-generation",
            model=GENERATION_MODEL,
            device_map="cpu",
        )
    return _generator




def _retrieve_context(question: str) -> Dict[str, List[str]]:
    table = _load_db()
    embedder = _load_embedder()

    query_vector = embedder.encode(question).tolist()

    results = (
        table.search(query_vector)
        .limit(TOP_K)
        .to_pandas()
    )

    chunks = []
    sources = []

    for i, row in results.iterrows():
        chunks.append(
            f"[Source {i+1} | {row['source']}]\n{row['text']}"
        )
        sources.append(row["source"])

    context = "\n\n".join(chunks)[:MAX_CONTEXT_CHARS]

    return {
        "context": context,
        "sources": sources,
    }


def _generate_answer(question: str, context: str) -> str:
    generator = _load_generator()

    prompt = f"""Answer using the context below.
Cite sources like [Source 1], [Source 2].

Context:
{context}

Question:
{question}

Answer:
"""

    out = generator(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.7,
        do_sample=True,
    )

    return out[0]["generated_text"].strip()



def answer_question(question: str) -> Dict[str, object]:
    """
    Main entry point for RAG.
    This is the function FastAPI / Angular will call.
    """
    try:
        retrieved = _retrieve_context(question)
        answer = _generate_answer(question, retrieved["context"])
        return {
            "answer": answer,
            "sources": retrieved["sources"],
        }
    except Exception as e:
        print("RAG ERROR:", repr(e))
        raise


   
