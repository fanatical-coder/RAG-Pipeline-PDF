import lancedb
from pathlib import Path
from groq import Groq
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import torch
import os
import json
from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL = "llama-3.1-8b-instant"
LANCEDB_URI = os.getenv("LANCEDB_URI")
LANCEDB_API_KEY = os.getenv("LANCEDB_API_KEY")

TOP_K = 4
TOP_K_IMAGES = 3
MAX_CONTEXT_CHARS = 2000

_db = None
_groq_client = None
_embedding_model = None
_clip_model = None
_clip_processor = None


def get_embedding(text):
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model.encode(text).tolist()


def get_clip_text_embedding(text: str) -> list:
    global _clip_model, _clip_processor
    if _clip_model is None:
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model.eval()
    inputs = _clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        features = _clip_model.get_text_features(**inputs)
    features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features.squeeze().cpu().numpy().tolist()


def _get_db():
    global _db
    if _db is None:
        _db = lancedb.connect(LANCEDB_URI, api_key=LANCEDB_API_KEY)
    return _db


def _load_groq():
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq_client


# ── Per-user table loaders ────────────────────────────────────────────────────
def _get_text_table(user_id: str):
    table_name = f"pdf_chunks_{user_id}"
    try:
        return _get_db().open_table(table_name)
    except Exception:
        raise ValueError(f"No indexed PDFs found for user. Please upload a PDF first.")


def _get_image_table(user_id: str):
    table_name = f"image_embeddings_{user_id}"
    try:
        return _get_db().open_table(table_name)
    except Exception:
        return None  # images are optional


# ── Image retrieval ───────────────────────────────────────────────────────────
def retrieve_images(question: str, user_id: str) -> list:
    try:
        table = _get_image_table(user_id)
        if table is None:
            return []
        clip_vector = get_clip_text_embedding(question)
        results = table.search(clip_vector).limit(TOP_K_IMAGES).to_pandas()
        return [
            {
                "image_path": row["image_path"],
                "source_pdf": row["source_pdf"],
                "page_number": int(row.get("page_number", 0)),
            }
            for _, row in results.iterrows()
            if Path(row["image_path"]).exists()
        ]
    except Exception as e:
        print(f"Image retrieval error: {e}")
        return []


# ── Text retrieval + prompt ───────────────────────────────────────────────────
def _build_prompt_and_sources(question: str, user_id: str):
    table = _get_text_table(user_id)
    query_vector = get_embedding(question)

    results = table.search(query_vector).limit(TOP_K).to_pandas()

    context = "\n\n".join(
        f"[Source {i+1} | {row['source']}]\n{row['text']}"
        for i, row in results.iterrows()
    )[:MAX_CONTEXT_CHARS]

    prompt = f"""Answer using the context below.
Cite sources like [Source 1].

Context:
{context}

Question:
{question}

Answer:"""

    return prompt, results["source"].tolist()


# ── Public API ────────────────────────────────────────────────────────────────
def answer_question(question: str, user_id: str):
    client = _load_groq()
    prompt, sources = _build_prompt_and_sources(question, user_id)
    images = retrieve_images(question, user_id)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context. Always cite your sources."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.2,
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": sources,
        "images": images,
    }


def answer_question_stream(question: str, user_id: str):
    client = _load_groq()
    prompt, sources = _build_prompt_and_sources(question, user_id)
    images = retrieve_images(question, user_id)

    stream = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context. Always cite your sources."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.2,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta

    yield f"\n\n__METADATA__{json.dumps({'sources': sources, 'images': images})}"