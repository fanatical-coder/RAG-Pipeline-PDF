from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CHUNK_DIR = BASE_DIR / "chunks"

total = 0
for chunk_file in sorted(CHUNK_DIR.glob("*.txt")):
    text = chunk_file.read_text(encoding="utf-8")
    chunks = [c.strip() for c in text.split("\n\n---\n\n") if c.strip()]
    total += len(chunks)
    print(f"{len(chunks):>6} chunks — {chunk_file.name}")

print(f"\nTotal: {total} chunks")