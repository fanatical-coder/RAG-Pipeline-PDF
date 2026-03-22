from rag_query_lancedb import answer_question, answer_question_stream
from groq import Groq
import os
import json

_groq_client = None

def _load_groq():
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq_client


def combined_search(query: str, user_id: str) -> dict:
    try:
        text_result = answer_question(query, user_id)
    except ValueError as e:
        return {"answer": str(e), "sources": [], "images": []}
    except Exception as e:
        print(f"[Text search error: {e}]")
        text_result = {"answer": "An error occurred while retrieving the answer.", "sources": [], "images": []}

    return {
        "answer": text_result.get("answer", ""),
        "sources": text_result.get("sources", []),
        "images": text_result.get("images", []),
    }


def combined_search_stream(query: str, user_id: str):
    sources = []
    images = []

    try:
        for chunk in answer_question_stream(query, user_id):
            if "__METADATA__" in chunk:
                try:
                    metadata = json.loads(chunk.split("__METADATA__")[1])
                    sources = metadata.get("sources", [])
                    images = metadata.get("images", [])
                except Exception:
                    pass
                break
            else:
                yield chunk
    except ValueError as e:
        yield str(e)

    yield f"\n\n__METADATA__{json.dumps({'sources': sources, 'images': images})}"