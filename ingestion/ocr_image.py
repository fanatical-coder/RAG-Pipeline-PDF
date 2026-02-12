import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

IMAGE_PATH = r"G:\VScodes\AI\pdf_rag(lancedb)\test-image2.png"  # CHANGE THIS

img = cv2.imread(IMAGE_PATH)

if img is None:
    print("ERROR: Image not loaded. Path is wrong.")
    exit()

print("Image loaded successfully")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
text = pytesseract.image_to_string(gray, config="--psm 6")

print("OCR OUTPUT:")
print(text)