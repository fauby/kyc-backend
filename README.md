# OCR Backend — Setup & Startup Checklist

This README lists the minimal steps to get the OCR / Liveness / Matching backend running on a developer machine (WSL or Linux) and avoid common startup errors. Follow each step before attempting to start the Node proxy or mobile frontend.

## Prerequisites

- Python 3.8+ installed (use system Python or a virtualenv).
- pip available.
- On Windows: use WSL for running the Python backend, and run the Node proxy on Windows (index.js) as provided.
- Ensure `node` is installed on Windows for the proxy.

## Install Python dependencies

From the backend root (WSL or Linux):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If there is no `requirements.txt`, install at least these packages:

```bash
pip install opencv-python-headless numpy dlib flask Pillow
```

## Models (critical)

The backend requires pretrained dlib models. The repository may include `.bz2` archives (compressed). You MUST decompress them into `.dat` files before starting the services.

Example commands (PowerShell):

```powershell
cd C:\Data\Mobile\ocr-backend
python -c "import bz2, pathlib; p=pathlib.Path('models/dlib_face_recognition_resnet_model_v1.dat.bz2'); pathlib.Path(str(p)[:-4]).write_bytes(bz2.decompress(p.read_bytes()))"
python -c "import bz2, pathlib; p=pathlib.Path('models/shape_predictor_68_face_landmarks.dat.bz2'); pathlib.Path(str(p)[:-4]).write_bytes(bz2.decompress(p.read_bytes()))"
python -c "import bz2, pathlib; p=pathlib.Path('models/shape_predictor_5_face_landmarks.dat.bz2'); pathlib.Path(str(p)[:-4]).write_bytes(bz2.decompress(p.read_bytes()))"
```

Or on WSL / Linux:

```bash
cd /path/to/ocr-backend
python - <<'PY'
import bz2, pathlib
for f in ['models/dlib_face_recognition_resnet_model_v1.dat.bz2', \
          'models/shape_predictor_68_face_landmarks.dat.bz2', \
          'models/shape_predictor_5_face_landmarks.dat.bz2']:
    p = pathlib.Path(f)
    if p.exists():
        out = pathlib.Path(str(p)[:-4])
        out.write_bytes(bz2.decompress(p.read_bytes()))
        print('Decompressed', out)
    else:
        print('Archive not found:', p)
PY
```

Verify files are present:

PowerShell:

```powershell
Get-ChildItem .\models\*.dat | Select Name, Length
```

WSL / Linux:

```bash
ls -l models/*.dat
```

If the `.dat` files are missing, the face matcher and liveness detector will fail at startup with messages about missing models.

## Environment variables

Set `DLIB_MODEL_DIR` if you keep models in a non-default location. Example:

```bash
export DLIB_MODEL_DIR=/path/to/ocr-backend/models
```

## Start the Python backend (WSL / Linux)

From the backend root in WSL:

```bash
source .venv/bin/activate
# Replace the next command with the project entrypoint if different (check your repo for the main script)
python -m kyc_flow_api  # or: python main_backend.py | python app.py
```

If the project exposes a Flask app, you can also run using the `flask` CLI after exporting `FLASK_APP`.

## Start the Node proxy (Windows)

On Windows, from the `ocr-backend` root:

```powershell
node index.js
```

The Node proxy expects the Flask backend to be reachable (default localhost:5000) and will forward `/liveness` to a liveness service on port 5001 if configured.

## Quick health checks

- After starting the Python backend, verify the server responds (example endpoints may vary):

```bash
curl http://127.0.0.1:5000/health || curl http://127.0.0.1:5000/
```

- For liveness service (if separate):

```bash
curl http://127.0.0.1:5001/health
```

If those endpoints are not present, look for log lines indicating the server binding and listen ports.

## Common errors & fixes

- "Face recognition model not available": decompress the `.bz2` model archives and ensure `models/*.dat` exists (see above).
- "Backend timeout" from the frontend: the Flask app may not be running or is listening on a different address/port. Check `index.js` proxy settings and ensure Flask is bound to 0.0.0.0 or 127.0.0.1 and correct port.
- Permission or file-not-found errors: confirm the process has read access to `models/` and other project files.

## Logging & diagnostics

- Check the Python service logs for lines prefixed with `[liveness_detector]` or `[face_matcher]` to debug face detection and model loading.
- Enable extra logging by setting environment variables in code or editing `kyc_flow_api/config.py` if present.

## After starting successfully

- Start the Node proxy (`node index.js`) on Windows.
- Open the mobile frontend and perform a test run.

If you want, I can add a small `start.sh` and `start.ps1` to automate these steps — tell me which platform to target and I'll create them.

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
