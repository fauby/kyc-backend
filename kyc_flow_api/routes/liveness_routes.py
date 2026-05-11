"""Liveness detection and face matching routes (FastAPI)."""

import os
import tempfile
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from ..services import LivenessService, MatchingService

router = APIRouter()

liveness_service = LivenessService()
matching_service = MatchingService()


@router.post("/liveness")
async def liveness_video(
    video: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
):
    if not video:
        raise HTTPException(status_code=400, detail={"error": "No video file provided"})

    suffix = os.path.splitext(video.filename or "liveness.mp4")[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        content = await video.read()
        if not content:
            raise HTTPException(status_code=400, detail={"error": "Empty video"})
        tmp.write(content)
        tmp.close()

        result = liveness_service.detect_liveness_from_video(tmp.name)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result)
        return result
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


class MatchRequest(BaseModel):
    ktp_face_image: Optional[str] = None
    ktp_image: Optional[str] = None
    processed_ktp_image: Optional[str] = None
    ktp_photo_crop_path: Optional[str] = None

    liveness_image: Optional[str] = None
    liveness_frames: Optional[List[str]] = None


@router.post("/match/faces")
def match_faces(req: MatchRequest):
    ktp_b64 = req.ktp_face_image or req.ktp_image or req.processed_ktp_image
    ktp_frame = matching_service.load_ktp_image(ktp_b64, req.ktp_photo_crop_path)
    if ktp_frame is None:
        raise HTTPException(
            status_code=400,
            detail={"error": "KTP photo crop required. Please rescan the KTP."},
        )

    if req.liveness_image:
        liveness_frame = matching_service.decode_image(req.liveness_image)
        if liveness_frame is None:
            raise HTTPException(
                status_code=400,
                detail={"error": "Could not decode liveness image"},
            )
        return matching_service.match_faces_single(ktp_frame, liveness_frame)

    if req.liveness_frames:
        liveness_frames = matching_service.decode_frames(req.liveness_frames)
        if not liveness_frames:
            raise HTTPException(
                status_code=400,
                detail={"error": "Could not decode liveness frames"},
            )
        return matching_service.match_faces_sequence(ktp_frame, liveness_frames)

    raise HTTPException(
        status_code=400,
        detail={"error": "liveness_image or liveness_frames required"},
    )


class SignatureMatchRequest(BaseModel):
    ktp_signature_base64: str
    user_signature_base64: str


@router.post("/match/signature")
def match_signature(req: SignatureMatchRequest):
    if not req.ktp_signature_base64 or not req.user_signature_base64:
        raise HTTPException(
            status_code=400,
            detail={"error": "ktp_signature_base64 and user_signature_base64 are required"},
        )

    result = matching_service.match_signatures(
        req.ktp_signature_base64,
        req.user_signature_base64,
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result)
    return result
