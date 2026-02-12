import lancedb
from sentence_transformers import SentenceTransformer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "lancedb"

db = lancedb.connect(DB_DIR)
table = db.open_table("pdf_chunks")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

question = "What dilemma is the main character facing according to the text?"
query_vector = embedder.encode(question).tolist()

results = (
    table.search(query_vector)
    .limit(5)
    .to_pandas()
)

for i, row in results.iterrows():
    print("\n---")
    print(row["source"])
    print(row["text"][:300])
