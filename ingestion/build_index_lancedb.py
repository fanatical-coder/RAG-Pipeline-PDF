import lancedb
import pandas as pd
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
EMB_DIR = BASE_DIR / "data" / "embeddings"
DB_DIR = BASE_DIR / "lancedb"

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

# Create LanceDB
db = lancedb.connect(DB_DIR)
table = db.create_table("pdf_chunks", data=df, mode="overwrite")

print(f"LanceDB index built with {len(df)} vectors")
