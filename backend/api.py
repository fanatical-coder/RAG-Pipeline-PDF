import sys
import os
import shutil
import asyncio
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT / "ingestion"))

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from intent_search import combined_search, combined_search_stream
from pipeline import run_pipeline
import firebase_admin
from firebase_admin import credentials, auth

IMAGE_DIR = PROJECT_ROOT / "ingestion" / "images"
DIST_DIR = PROJECT_ROOT / "frontend" / "dist" / "frontend"
INGESTION_DIR = PROJECT_ROOT / "ingestion"

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        decoded_token = auth.verify_id_token(credentials.credentials)
        return decoded_token
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), user=Depends(verify_token)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    user_id = user["uid"]
    user_pdf_dir = INGESTION_DIR / "users" / user_id / "raw_pdfs"
    user_pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = user_pdf_dir / file.filename

    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    print(f"PDF saved: {pdf_path}")
    asyncio.create_task(run_pipeline_async(pdf_path, user_id))

    return JSONResponse({
        "message": f"'{file.filename}' uploaded successfully. Processing started.",
        "filename": file.filename,
        "user_id": user_id
    })

async def run_pipeline_async(pdf_path: Path, user_id: str):
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, run_pipeline, pdf_path, user_id)
        print(f"✔ Pipeline complete for {pdf_path.name}")
    except Exception as e:
        print(f"Pipeline error: {e}")

@app.get("/upload/status/{filename}")
async def upload_status(filename: str, user=Depends(verify_token)):
    user_id = user["uid"]
    stem = Path(filename).stem
    chunk_path = INGESTION_DIR / "users" / user_id / "chunks" / f"{stem}_chunks.txt"
    emb_path = INGESTION_DIR / "users" / user_id / "embeddings" / "embeddings.pkl"
    if chunk_path.exists() and emb_path.exists():
        return {"status": "complete", "filename": filename}
    return {"status": "processing", "filename": filename}

@app.post("/search")
def search(req: QueryRequest, user=Depends(verify_token)):
    return combined_search(req.query, user["uid"])  # 👈 user_id passed

@app.post("/search/stream")
def search_stream(req: QueryRequest, user=Depends(verify_token)):
    return StreamingResponse(
        combined_search_stream(req.query, user["uid"]),  # 👈 user_id passed
        media_type="text/plain",
    )
@app.get("/pdfs")
async def get_user_pdfs(user=Depends(verify_token)):
    user_id = user["uid"]
    user_pdf_dir = INGESTION_DIR / "users" / user_id / "raw_pdfs"
    chunk_dir = INGESTION_DIR / "users" / user_id / "chunks"
    emb_path = INGESTION_DIR / "users" / user_id / "embeddings" / "embeddings.pkl"

    if not user_pdf_dir.exists():
        return {"pdfs": []}

    pdfs = []
    for pdf in user_pdf_dir.glob("*.pdf"):
        chunk_file = chunk_dir / f"{pdf.stem}_chunks.txt"
        status = "ready" if chunk_file.exists() and emb_path.exists() else "processing"
        pdfs.append({
            "filename": pdf.name,
            "size_kb": round(pdf.stat().st_size / 1024, 1),
            "status": status
        })

    return {"pdfs": pdfs}

@app.get("/image-file")
def get_image(path: str):
    file_path = Path(path)
    if not file_path.exists():
        return {"error": "Image not found"}
    return FileResponse(str(file_path))

app.mount("/images", StaticFiles(directory=str(IMAGE_DIR)), name="images")
app.mount("/", StaticFiles(directory=str(DIST_DIR), html=True), name="static")