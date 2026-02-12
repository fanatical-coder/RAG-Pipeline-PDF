from pathlib import Path
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
EMB_DIR = BASE_DIR / "data" / "embeddings"


with open(EMB_DIR / "embeddings.pkl", "rb") as f:
    data = pickle.load(f)

index = faiss.read_index(str(EMB_DIR / "faiss.index"))

model = SentenceTransformer("all-MiniLM-L6-v2")


query = "What is the main theme of the book?"
q_emb = model.encode([query]).astype("float32")

D, I = index.search(q_emb, k=3)

print("Top results:\n")
for idx in I[0]:
    print("----")
    print(data[idx]["text"][:500])
