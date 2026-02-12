import pytesseract
from PIL import Image
from pathlib import Path

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

BASE_DIR = Path(__file__).resolve().parent
IMG_DIR = BASE_DIR / "data" / "raw_images"
OUT_DIR = BASE_DIR / "data" / "extracted_text" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

for img_path in IMG_DIR.glob("*"):
    if img_path.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
        continue

    img = Image.open(img_path)
    text = pytesseract.image_to_string(img, config="--psm 6")

    out_file = OUT_DIR / f"{img_path.stem}.txt"
    out_file.write_text(text, encoding="utf-8")

    print(f"OCR done: {img_path.name}")
