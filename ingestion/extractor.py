from pathlib import Path
import fitz

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "data" / "raw_pdfs"
OUT_DIR = BASE_DIR / "data" / "extracted_text"

OUT_DIR.mkdir(exist_ok=True)

for pdf_file in PDF_DIR.glob("*.pdf"):
    doc = fitz.open(pdf_file)
    collected_text = []

    for page in doc:
        text = page.get_text().strip()
        if text:
            collected_text.append(text)

    output_path = OUT_DIR / f"{pdf_file.stem}.txt"
    output_path.write_text("\n\n".join(collected_text), encoding="utf-8")

    print(f"Extracted: {pdf_file.name}")
