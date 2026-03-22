from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
# ---------------- CONFIG ----------------

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

RAW_PDF_DIR = PROJECT_ROOT / "ingestion" / "raw_pdfs"
IMAGE_OUTPUT_DIR = PROJECT_ROOT / "ingestion" / "images"
TEXT_OUTPUT_DIR = PROJECT_ROOT / "ingestion" / "extracted_text"

# Pages with fewer than this many characters trigger Tesseract OCR
MIN_NATIVE_TEXT_CHARS = 50

# ----------------------------------------


def ocr_page(page: fitz.Page) -> str:
    """Render a PDF page and run Tesseract OCR on it."""
    pix = page.get_pixmap(dpi=200)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return pytesseract.image_to_string(img).strip()


def extract_images_from_pdf(pdf_path: Path):
    print(f"\nProcessing: {pdf_path.name}")

    pdf_name = pdf_path.stem
    image_out_dir = IMAGE_OUTPUT_DIR / pdf_name
    image_out_dir.mkdir(parents=True, exist_ok=True)
    TEXT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"❌ Failed to open {pdf_path.name}: {e}")
        return

    total_images = 0
    ocr_count = 0
    collected_text = []

    for page_number in range(len(doc)):
        page = doc[page_number]
        page_label = page_number + 1  # 1-based


        # ── Text extraction ───────────────────────────────────────────────────
        raw_text = page.get_text()
        if isinstance(raw_text, str):
            native_text = raw_text.strip()
        elif isinstance(raw_text, list):
            native_text = " ".join(str(item) for item in raw_text).strip()
        elif isinstance(raw_text, dict):
            native_text = str(raw_text).strip()
        else:
            native_text = ""

        if len(native_text) >= MIN_NATIVE_TEXT_CHARS:
            page_text = native_text
        else:
            # Scanned page — fall back to Tesseract
            page_text = ocr_page(page)
            ocr_count += 1
            print(f"  Page {page_label}: OCR used")

        if page_text:
            collected_text.append(f"--- Page {page_label} ---\n{page_text}")

        # ── Image extraction ──────────────────────────────────────────────────
        images = page.get_images(full=True)

        if not images:
            continue

        for img_index, img in enumerate(images):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                filename = f"page{page_label}_img{img_index + 1}.{image_ext}"
                image_path = image_out_dir / filename

                if not image_path.exists():
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)
                    total_images += 1

            except Exception as e:
                print(f"  ⚠ Error extracting image on page {page_label}: {e}")

    doc.close()

    # ── Save extracted text ───────────────────────────────────────────────────
    text_output_path = TEXT_OUTPUT_DIR / f"{pdf_name}.txt"
    text_output_path.write_text("\n\n".join(collected_text), encoding="utf-8")

    print(f"✔ {pdf_name}: {total_images} images extracted | OCR on {ocr_count} pages | text saved")


def main():
    print("===== EXTRACTION STARTED =====")
    print("RAW_PDF_DIR:", RAW_PDF_DIR)
    print("Exists:", RAW_PDF_DIR.exists())

    if not RAW_PDF_DIR.exists():
        print("❌ RAW_PDF_DIR does not exist")
        return

    IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = list(RAW_PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print("❌ No PDFs found in raw_pdfs/")
        return

    print(f"Found {len(pdf_files)} PDFs")

    for pdf in pdf_files:
        extract_images_from_pdf(pdf)

    print("\n===== EXTRACTION FINISHED =====")


if __name__ == "__main__":
    main()