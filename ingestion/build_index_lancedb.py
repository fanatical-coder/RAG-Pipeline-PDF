import os
import lancedb
import pandas as pd
import pickle
from pathlib import Path
from dotenv import load_dotenv
BASE_DIR = Path(__file__).resolve().parent
EMB_DIR = BASE_DIR / "embeddings"

load_dotenv(BASE_DIR.parent / ".env")
# Load embeddings
with open(EMB_DIR / "embeddings.pkl", "rb") as f:
    data = pickle.load(f)

rows = []
for i, item in enumerate(data):
    rows.append({
        "id": i,
        "vector": item["embedding"],
        "text": item["text"],
        "source": item["source"],
    })

df = pd.DataFrame(rows)

# LanceDB Cloud connection
LANCEDB_URI = os.getenv("LANCEDB_URI")
LANCEDB_API_KEY = os.getenv("LANCEDB_API_KEY")

db = lancedb.connect(
    LANCEDB_URI,
    api_key=LANCEDB_API_KEY,
)
table = db.create_table("pdf_chunks", data=df, mode="overwrite")

print(f"LanceDB index built with {len(df)} vectors")