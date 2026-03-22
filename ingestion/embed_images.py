import os
from pathlib import Path
import lancedb
import pyarrow as pa
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "images"

device = "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.to(device)

load_dotenv(BASE_DIR.parent / ".env")
# LanceDB Cloud connection
LANCEDB_URI = os.getenv("LANCEDB_URI")
LANCEDB_API_KEY = os.getenv("LANCEDB_API_KEY")

db = lancedb.connect(
    LANCEDB_URI,
    api_key=LANCEDB_API_KEY,
)

records = []

for pdf_folder in IMAGE_DIR.iterdir():
    if not pdf_folder.is_dir():
        continue

    for img_path in pdf_folder.glob("*.*"):
        try:
            image = Image.open(img_path).convert("RGB")

            inputs = processor(images=image, return_tensors="pt").to(device)

            with torch.no_grad():
                image_features = model.get_image_features(**inputs)

            # Normalize
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

            embedding = image_features.squeeze().cpu().numpy().tolist()

            records.append({
                "vector": embedding,
                "image_path": str(img_path),
                "source_pdf": pdf_folder.name
            })

        except Exception as e:
            print("Error:", img_path, e)

schema = pa.schema([
    pa.field("vector", pa.list_(pa.float32(), 512)),
    pa.field("image_path", pa.string()),
    pa.field("source_pdf", pa.string())
])

db.create_table(
    "image_embeddings",
    data=records,
    schema=schema,
    mode="overwrite"
)

print("Ingestion complete.")