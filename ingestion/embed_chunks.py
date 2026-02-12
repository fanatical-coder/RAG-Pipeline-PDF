from pathlib import Path
import pickle
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
CHUNK_DIR = BASE_DIR / "data" / "chunks"
OUT_DIR = BASE_DIR / "data" / "embeddings"

OUT_DIR.mkdir(exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

all_embeddings = []

for chunk_file in CHUNK_DIR.glob("*.txt"):
    text = chunk_file.read_text(encoding="utf-8")
    chunks = [c.strip() for c in text.split("\n\n---\n\n") if c.strip()]

    embeddings = model.encode(chunks, show_progress_bar=True)

    for chunk, emb in zip(chunks, embeddings):
        all_embeddings.append({
            "source": chunk_file.name,
            "text": chunk,
            "embedding": emb
        })

with open(OUT_DIR / "embeddings.pkl", "wb") as f:
    pickle.dump(all_embeddings, f)

print(f"Saved {len(all_embeddings)} embeddings")
