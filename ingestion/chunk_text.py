from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
TEXT_DIR = BASE_DIR / "data" / "extracted_text"
CHUNK_DIR = BASE_DIR / "data" / "chunks"

chunk_size = 700
overlap = 100

for txt_file in TEXT_DIR.glob("*.txt"):
    text = txt_file.read_text(encoding="utf-8")
    words = text.split()

    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap

    out_file = CHUNK_DIR / f"{txt_file.stem}_chunks.txt"
    out_file.write_text("\n\n---\n\n".join(chunks), encoding="utf-8")

    print(f"Chunked: {txt_file.name} -> {len(chunks)} chunks")
