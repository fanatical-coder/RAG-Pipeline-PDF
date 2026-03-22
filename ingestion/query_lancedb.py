import lancedb
import requests
import os
from dotenv import load_dotenv
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env")
# LanceDB Cloud credentials — set these in your environment
LANCEDB_URI = os.getenv("LANCEDB_URI")       # e.g. "db://your-project-slug"
LANCEDB_API_KEY = os.getenv("LANCEDB_API_KEY")

db = lancedb.connect(
    LANCEDB_URI,
    api_key=LANCEDB_API_KEY,
)
table = db.open_table("pdf_chunks")

HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"

def get_embedding(text):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(API_URL, headers=headers, json={"inputs": text})
    return response.json()

question = "What dilemma is the main character facing according to the text?"
query_vector = get_embedding(question)


results = (
    table.search(query_vector)
    .limit(5)
    .to_pandas()
)

for i, row in results.iterrows():
    print("\n---")
    print(row["source"])
    print(row["text"][:300])