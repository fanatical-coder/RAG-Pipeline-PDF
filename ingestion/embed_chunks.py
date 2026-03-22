from pathlib import Path
import pickle
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
CHUNK_DIR = BASE_DIR / "chunks"
OUT_DIR = BASE_DIR / "embeddings"
OUT_DIR.mkdir(exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

# Load existing progress if resuming
progress_file = OUT_DIR / "embeddings.pkl"
if progress_file.exists():
    with open(progress_file, "rb") as f:
        all_embeddings = pickle.load(f)
    done_files = set(e["source"] for e in all_embeddings)
    print(f"Resuming — {len(all_embeddings)} embeddings already done")
    print(f"Done files: {done_files}\n")
else:
    all_embeddings = []
    done_files = set()

chunk_files = sorted(CHUNK_DIR.glob("*.txt"))

for file_idx, chunk_file in enumerate(chunk_files, 1):
    # Skip already processed files
    if chunk_file.name in done_files:
        print(f"[{file_idx}/{len(chunk_files)}] SKIPPED (already done) — {chunk_file.name}")
        continue

    text = chunk_file.read_text(encoding="utf-8")
    chunks = [c.strip() for c in text.split("\n\n---\n\n") if c.strip()]

    if not chunks:
        print(f"[{file_idx}/{len(chunk_files)}] SKIPPED (empty) — {chunk_file.name}")
        continue

    print(f"[{file_idx}/{len(chunk_files)}] {chunk_file.name} — {len(chunks)} chunks", flush=True)

    embeddings = model.encode(chunks, batch_size=64, show_progress_bar=True)

    for chunk, emb in zip(chunks, embeddings):
        all_embeddings.append({
            "source": chunk_file.name,
            "text": chunk,
            "embedding": emb.tolist()
        })

    # Save after every file — so progress is never lost
    with open(progress_file, "wb") as f:
        pickle.dump(all_embeddings, f)

    print(f"  ✔ Saved progress — {len(all_embeddings)} total embeddings so far\n")

print(f"Done. Total: {len(all_embeddings)} embeddings")