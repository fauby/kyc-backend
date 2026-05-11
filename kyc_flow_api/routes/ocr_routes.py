"""OCR detection routes (FastAPI)."""

import json
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..services import OCRService

router = APIRouter()
ocr_service = OCRService()


class _UploadBytesAdapter:
    """Small adapter to satisfy OCRService(FileStorage-like)."""

    def __init__(self, filename: str, content: bytes):
        self.filename = filename
        self._content = content or b""

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            f.write(self._content)


@router.post("/ocr/ktp")
async def ocr_ktp(
    file: UploadFile = File(...),
    box: Optional[str] = Form(None),
    photoBox: Optional[str] = Form(None),
    signatureBox: Optional[str] = Form(None),
    photoWidth: Optional[str] = Form(None),
    photoHeight: Optional[str] = Form(None),
):
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail={"error": "No selected file"})

    print(f"[ocr_routes] Processing file: {file.filename}")

    try:
        box_obj = json.loads(box) if box else None
    except Exception as e:
        print(f"[ocr_routes] box parsing failed: {e}")
        box_obj = None

    try:
        photo_box_obj = json.loads(photoBox) if photoBox else None
    except Exception as e:
        print(f"[ocr_routes] photoBox parsing failed: {e}")
        photo_box_obj = None

    try:
        signature_box_obj = json.loads(signatureBox) if signatureBox else None
    except Exception as e:
        print(f"[ocr_routes] signatureBox parsing failed: {e}")
        signature_box_obj = None

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail={"error": "Empty file"})

    # OCRService expects a Flask/Werkzeug FileStorage-like object with .save(path).
    adapter = _UploadBytesAdapter(file.filename, content)
    return ocr_service.process_ktp_image(adapter, box_obj, photo_box_obj, signature_box_obj)
