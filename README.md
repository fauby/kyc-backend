# KTP OCR (Tesseract)

This repository contains a small Flask API to extract data from Indonesian KTP images using Tesseract OCR.

Windows setup (quick):

1. Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Install Tesseract for Windows:
   - Download the installer from the Tesseract repo or the UB Mannheim builds: https://github.com/tesseract-ocr/tesseract
   - Install to the default location (e.g. `C:\Program Files\Tesseract-OCR`).
   - Ensure `tessdata\ind.traineddata` exists. If not, download `ind.traineddata` and copy it to `C:\Program Files\Tesseract-OCR\tessdata`.

3. (Optional) If Tesseract is installed in a custom location, set the environment variables before running the server:

Windows (PowerShell):

```powershell
$env:TESSERACT_CMD = 'C:\Program Files\Tesseract-OCR\tesseract.exe'
$env:TESSDATA_PREFIX = 'C:\Program Files\Tesseract-OCR\tessdata'
```

4. Run the server:

```bash
python ktp_ocr_server.py
```

5. Test the endpoint with `curl` (multipart upload):

```bash
curl -X POST -F "file=@/path/to/ktp.jpg" http://127.0.0.1:5000/ocr/ktp
```

Notes:

- If the server prints a warning about Tesseract not being found, set `pytesseract.pytesseract.tesseract_cmd` or the `TESSERACT_CMD` env var to point to your `tesseract.exe`.
- The code expects Indonesian language data (`ind`). Make sure `ind.traineddata` is present in `tessdata`.
