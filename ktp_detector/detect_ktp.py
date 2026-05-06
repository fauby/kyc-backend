import cv2
import pytesseract
from PIL import Image
import numpy as np
import os

def detect_ktp(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    h, w = image.shape[:2]
    # Downscale if image is very large for faster processing
    max_dim = max(h, w)
    if max_dim > 2000:
        scale = 2000.0 / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Preprocessing pipeline to improve OCR robustness
    # 1) Convert to gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2) Denoise with bilateral filter to preserve edges
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # 3) Contrast enhancement with CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # 4) Try Otsu thresholding (typically more stable than small-block adaptive)
    try:
        _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Small morphological closing to join character fragments
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
    except Exception:
        cleaned = enhanced

    # Save processed image next to original for debugging and to ensure the
    # image used for OCR matches the processed preview returned to client.
    proc_path = os.path.splitext(image_path)[0] + '_processed.png'
    try:
        cv2.imwrite(proc_path, cleaned)
    except Exception:
        # ignore write failures
        pass

    # OCR using pytesseract with a restricted character set tuned for KTP
    tess_cfg = "--oem 1 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,.-/"

    def try_ocr(img_array, config=None, lang=None):
        try:
            pil = Image.fromarray(img_array)
            if lang:
                return pytesseract.image_to_string(pil, lang=lang, config=config or "")
            return pytesseract.image_to_string(pil, config=config or "")
        except Exception:
            return ''

    text = try_ocr(cleaned, config=tess_cfg, lang='ind')

    # Fallbacks if output is too short
    if not text or len(text.strip()) < 10:
        # Try without specifying language
        text = try_ocr(cleaned, config=tess_cfg, lang=None)

    if not text or len(text.strip()) < 10:
        # Try different engine/psm
        alt_cfg = "--oem 3 --psm 3"
        text = try_ocr(cleaned, config=alt_cfg, lang=None)

    if not text or len(text.strip()) < 10:
        # Upscale and try again
        try:
            scale_img = cv2.resize(cleaned, (0, 0), fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
            text = try_ocr(scale_img, config=tess_cfg, lang=None)
        except Exception:
            pass

    if not text or len(text.strip()) < 10:
        # Try dilation to thicken strokes
        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            dil = cv2.dilate(cleaned, kernel, iterations=1)
            text = try_ocr(dil, config=tess_cfg, lang=None)
        except Exception:
            pass

    # Debug print: length and preview
    print(f'[detect_ktp] OCR length={len(text or "")}')
    if text:
        print(text[:400])

    return text

if __name__ == "__main__":
    img_path = input("Enter path to KTP image: ")
    result = detect_ktp(img_path)
    print("Detected Text:\n", result)
