import os
from pathlib import Path
import lancedb
from transformers import CLIPProcessor, CLIPModel
import torch
from dotenv import load_dotenv
BASE_DIR = Path(__file__).resolve().parent
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
table = db.open_table("image_embeddings")

query = "a man standing in darkness"

# Convert text → CLIP embedding
inputs = processor(text=query, return_tensors="pt").to(device)

with torch.no_grad():
    text_features = model.get_text_features(**inputs)

query_vector = text_features[0].cpu().numpy().tolist()

# Search
results = (
    table.search(query_vector)
    .limit(5)
    .to_pandas()
)

print("Top Matching Images:\n")

for i, row in results.iterrows():
    print(f"{i+1}. {row['image_path']} (from {row['source_pdf']})")