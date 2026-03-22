import os
import pickle
import lancedb
import pandas as pd
import pyarrow as pa
import torch
import fitz
import pytesseract
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel

pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env")

LANCEDB_URI = os.getenv("LANCEDB_URI")
LANCEDB_API_KEY = os.getenv("LANCEDB_API_KEY")

MIN_NATIVE_TEXT_CHARS = 50
CHUNK_SIZE = 200
OVERLAP = 100


def run_pipeline(pdf_path: Path, user_id: str):
    print(f"\n===== PIPELINE STARTED for {pdf_path.name} =====")

    # ── Directories ───────────────────────────────────────────────────────────
    user_dir = BASE_DIR / "users" / user_id
    image_dir = user_dir / "images" / pdf_path.stem
    text_dir = user_dir / "extracted_text"
    chunk_dir = user_dir / "chunks"
    emb_dir = user_dir / "embeddings"

    for d in [image_dir, text_dir, chunk_dir, emb_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── STEP 1: Extract text + images ────────────────────────────────────────
    print("\n[1/5] Extracting text and images...")
    collected_text = []
    total_images = 0
    ocr_count = 0

    doc = fitz.open(pdf_path)

    for page_number in range(len(doc)):
        page = doc[page_number]
        page_label = page_number + 1

        # Text
        raw_text = page.get_text()
        native_text = raw_text.strip() if isinstance(raw_text, str) else ""

        if len(native_text) >= MIN_NATIVE_TEXT_CHARS:
            page_text = native_text
        else:
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            page_text = pytesseract.image_to_string(img).strip()
            ocr_count += 1

        if page_text:
            collected_text.append(f"--- Page {page_label} ---\n{page_text}")

        # Images
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                filename = f"page{page_label}_img{img_index + 1}.{image_ext}"
                image_path = image_dir / filename
                if not image_path.exists():
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)
                    total_images += 1
            except Exception as e:
                print(f"  Warning: image extraction error on page {page_label}: {e}")

    doc.close()

    # Save text
    text_path = text_dir / f"{pdf_path.stem}.txt"
    text_path.write_text("\n\n".join(collected_text), encoding="utf-8")
    print(f"  ✔ {total_images} images | {ocr_count} OCR pages | text saved")

    # ── STEP 2: Chunk text ────────────────────────────────────────────────────
    print("\n[2/5] Chunking text...")
    text = text_path.read_text(encoding="utf-8")
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + CHUNK_SIZE]))
        i += CHUNK_SIZE - OVERLAP

    chunk_path = chunk_dir / f"{pdf_path.stem}_chunks.txt"
    chunk_path.write_text("\n\n---\n\n".join(chunks), encoding="utf-8")
    print(f"  ✔ {len(chunks)} chunks created")

    # ── STEP 3: Embed text ────────────────────────────────────────────────────
    print("\n[3/5] Embedding text chunks...")
    text_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings_list = text_model.encode(chunks, batch_size=64, show_progress_bar=True)

    text_embeddings = []
    for chunk, emb in zip(chunks, embeddings_list):
        text_embeddings.append({
            "source": chunk_path.name,
            "text": chunk,
            "embedding": emb.tolist()
        })

    emb_path = emb_dir / "embeddings.pkl"
    with open(emb_path, "wb") as f:
        pickle.dump(text_embeddings, f)
    print(f"  ✔ {len(text_embeddings)} text embeddings saved")

    # ── STEP 4: Embed images ──────────────────────────────────────────────────
    print("\n[4/5] Embedding images...")
    device = "cpu"
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.to(device)

    image_records = []
    for img_path in image_dir.glob("*.*"):
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = clip_processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                features = clip_model.get_image_features(**inputs)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            image_records.append({
                "vector": features.squeeze().cpu().numpy().tolist(),
                "image_path": str(img_path),
                "source_pdf": pdf_path.stem
            })
        except Exception as e:
            print(f"  Warning: image embedding error {img_path}: {e}")

    print(f"  ✔ {len(image_records)} image embeddings done")

    # ── STEP 5: Build LanceDB indexes ─────────────────────────────────────────
    print("\n[5/5] Building LanceDB indexes...")
    db = lancedb.connect(LANCEDB_URI, api_key=LANCEDB_API_KEY)

    # Text index — per user table
    text_table_name = f"pdf_chunks_{user_id}"
    rows = [{"id": i, "vector": e["embedding"],
             "text": e["text"], "source": e["source"]}
            for i, e in enumerate(text_embeddings)]
    df = pd.DataFrame(rows)
    db.create_table(text_table_name, data=df, mode="overwrite")
    print(f"  ✔ Text index '{text_table_name}' built with {len(df)} vectors")

    # Image index — per user table
    image_table_name = f"image_embeddings_{user_id}"
    schema = pa.schema([
        pa.field("vector", pa.list_(pa.float32(), 512)),
        pa.field("image_path", pa.string()),
        pa.field("source_pdf", pa.string())
    ])
    db.create_table(image_table_name, data=image_records,
                    schema=schema, mode="overwrite")
    print(f"  ✔ Image index '{image_table_name}' built with {len(image_records)} vectors")

    print(f"\n===== PIPELINE COMPLETE for {pdf_path.name} =====")
    return {
        "chunks": len(chunks),
        "images": total_images,
        "text_embeddings": len(text_embeddings),
        "image_embeddings": len(image_records)
    }