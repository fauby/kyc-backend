"""Face matching service."""

import os
import sys
import base64
import logging

import cv2
import numpy as np

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from face_matcher import (
    match_faces,
    match_faces_with_best_liveness,
    extract_face_image,
)

from ..config import LOG_PREFIX_MATCHER, UPLOAD_DIR
from ..utils import load_image_correct_orientation

logger = logging.getLogger(__name__)


class MatchingService:
    """Service for handling face matching."""
    
    def __init__(self):
        """Initialize matching service."""
        self.log_prefix = LOG_PREFIX_MATCHER
        self.upload_dir = UPLOAD_DIR
        self.max_frame_side = 640

    def _resize_for_speed(self, frame):
        """Resize large frames before face extraction for faster matching."""
        if frame is None or frame.size == 0:
            return frame

        h, w = frame.shape[:2]
        max_side = max(h, w)
        if max_side <= self.max_frame_side:
            return frame

        scale = float(self.max_frame_side) / float(max_side)
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    
    def decode_image(self, image_b64):
        """Decode base64-encoded image to BGR frame."""
        try:
            data = base64.b64decode(image_b64)
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return self._resize_for_speed(frame)
        except Exception as e:
            print(f'{self.log_prefix} failed to decode image: {e}')
            return None

    def decode_canvas_image(self, image_b64):
        """
        Decode a drawn-signature PNG to BGR, compositing transparent areas onto
        a white background. react-native-signature-canvas outputs transparent-
        background PNGs: loading with IMREAD_COLOR collapses alpha=0 to black,
        making the entire canvas look like ink. IMREAD_UNCHANGED preserves alpha
        so we can composite correctly before binarisation.
        """
        try:
            data = base64.b64decode(image_b64)
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            if frame is None:
                return None

            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                # BGRA → composite onto white
                bgra = frame.astype(np.float32)
                alpha = bgra[:, :, 3:4] / 255.0
                ink = bgra[:, :, :3]
                white = np.full_like(ink, 255.0)
                composited = (ink * alpha + white * (1.0 - alpha))
                frame = np.clip(composited, 0, 255).astype(np.uint8)
            # already BGR (3 channels) — use as-is

            return self._resize_for_speed(frame)
        except Exception as e:
            print(f'{self.log_prefix} failed to decode canvas image: {e}')
            return None
    
    def decode_frames(self, frames_b64):
        """Decode list of base64-encoded frames."""
        frames = []
        for frame_b64 in frames_b64:
            try:
                frame_data = base64.b64decode(frame_b64)
                nparr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is not None:
                    frames.append(self._resize_for_speed(frame))
            except Exception as e:
                print(f'{self.log_prefix} failed to decode frame: {e}')
                continue
        return frames
    
    def load_ktp_image(self, ktp_b64=None, ktp_photo_path=None):
        """
        Load KTP image from base64 or file path.
        
        Args:
            ktp_b64: Base64-encoded image (preferred)
            ktp_photo_path: File path (fallback)
            
        Returns:
            BGR image or None if loading fails
        """
        if ktp_b64:
            return self.decode_image(ktp_b64)
        
        if ktp_photo_path:
            # Validate path is within upload directory
            abs_path = os.path.abspath(ktp_photo_path)
            uploads_abs = os.path.abspath(self.upload_dir)
            
            if not abs_path.startswith(uploads_abs):
                print(f'{self.log_prefix} path traversal blocked')
                return None
            
            if not os.path.exists(abs_path):
                print(f'{self.log_prefix} KTP photo path does not exist')
                return None
            
            try:
                frame = load_image_correct_orientation(abs_path)
                return self._resize_for_speed(frame)
            except Exception as e:
                print(f'{self.log_prefix} failed to load KTP image: {e}')
                return None
        
        return None
    
    def match_faces_single(self, ktp_frame, liveness_frame):
        """
        Match single KTP frame with single liveness frame.
        
        Args:
            ktp_frame: KTP BGR image
            liveness_frame: Liveness BGR image
            
        Returns:
            Match result dictionary
        """
        result = match_faces(ktp_frame, liveness_frame)
        self._add_face_encodings(result, ktp_frame, liveness_frame)
        return result
    
    def match_faces_sequence(self, ktp_frame, liveness_frames):
        """
        Match KTP frame with best frame from liveness sequence.
        
        Args:
            ktp_frame: KTP BGR image
            liveness_frames: List of liveness BGR images
            
        Returns:
            Match result dictionary with best frame index
        """
        result = match_faces_with_best_liveness(ktp_frame, liveness_frames)
        
        # Add face encodings
        if result.get('best_frame_idx', -1) >= 0 and liveness_frames:
            idx = result['best_frame_idx']
            self._add_face_encodings(result, ktp_frame, liveness_frames[idx])
        
        return result
    
    def _add_face_encodings(self, result, ktp_frame, liveness_frame):
        """Add encoded face images to result."""
        ktp_face = extract_face_image(ktp_frame)
        if ktp_face is not None:
            _, ktp_face_buf = cv2.imencode('.jpg', ktp_face)
            result['ktp_face_base64'] = base64.b64encode(ktp_face_buf).decode('utf-8')
        
        live_face = extract_face_image(liveness_frame)
        if live_face is not None:
            _, live_face_buf = cv2.imencode('.jpg', live_face)
            result['liveness_face_base64'] = base64.b64encode(live_face_buf).decode('utf-8')

    def _preprocess_signature(self, img_bgr, trim_top_ratio=0.0):
        """
        Preprocess a signature image for matching:
        - optional top trim (removes date text from KTP signature crops)
        - grayscale + Otsu binarise (ink = white)
        - crop to ink bounding box
        - resize to fixed canvas (320×140)
        Returns a uint8 binary image, or None if too little ink found.
        """
        if img_bgr is None or img_bgr.size == 0:
            return None

        # Trim top rows to remove printed text (e.g. "02-12-2012") that sits
        # above the handwritten signature in the KTP crop.
        if trim_top_ratio > 0:
            h_full = img_bgr.shape[0]
            trim = int(h_full * trim_top_ratio)
            img_bgr = img_bgr[trim:, :, :]
            if img_bgr.shape[0] < 10:
                return None

        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        except Exception:
            return None

        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Remove small noise
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)

        ys, xs = np.where(th > 0)
        if len(xs) < 30 or len(ys) < 30:
            return None

        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        pad = 6
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(th.shape[1] - 1, x2 + pad)
        y2 = min(th.shape[0] - 1, y2 + pad)
        crop = th[y1 : y2 + 1, x1 : x2 + 1]

        target_w, target_h = 320, 140
        crop = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)
        return crop

    def _grid_histogram_similarity(self, a, b, cols=8, rows=4):
        """
        Compare ink density in a cols×rows grid. Each cell's ink fraction is
        one dimension of a feature vector; we return the cosine similarity of
        those vectors. This is tolerant of small positional shifts that would
        kill pixel-level IoU, while still penalising completely different patterns.
        """
        cell_h = a.shape[0] // rows
        cell_w = a.shape[1] // cols
        if cell_h == 0 or cell_w == 0:
            return 0.0

        def hist(img):
            vec = []
            for gy in range(rows):
                for gx in range(cols):
                    y1 = gy * cell_h
                    y2 = (gy + 1) * cell_h if gy < rows - 1 else img.shape[0]
                    x1 = gx * cell_w
                    x2 = (gx + 1) * cell_w if gx < cols - 1 else img.shape[1]
                    cell = img[y1:y2, x1:x2]
                    vec.append(float(np.sum(cell > 128)) / max(cell.size, 1))
            return np.array(vec, dtype=np.float32)

        va = hist(a)
        vb = hist(b)
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        if denom < 1e-6:
            return 0.0
        return float(np.dot(va, vb) / denom)

    def match_signatures(self, ktp_signature_b64, user_signature_b64):
        """
        Match KTP signature crop against user's drawn signature.
        Returns score 0-100 and is_match.

        Uses IoU + cosine similarity on binarised ink masks instead of
        TM_CCOEFF_NORMED, which falsely returns ~1.0 when both same-size
        binary images have similar ink density regardless of stroke pattern.
        """
        print(f"{self.log_prefix} ktp_b64 len={len(ktp_signature_b64)}, user_b64 len={len(user_signature_b64)}")
        ktp_img = self.decode_image(ktp_signature_b64)
        # User's drawn signature is a transparent-background PNG from the canvas —
        # use the alpha-aware decoder so transparent areas become white, not black.
        user_img = self.decode_canvas_image(user_signature_b64)
        if ktp_img is None or user_img is None:
            return {"error": "Could not decode signature image(s)"}

        print(f"{self.log_prefix} ktp_img shape={ktp_img.shape}, user_img shape={user_img.shape}")
        # Trim the top 35% of the KTP image to remove printed date text
        # (e.g. "02-12-2012") that sits above the handwritten signature.
        # The user only draws the signature, not the printed text.
        a = self._preprocess_signature(ktp_img, trim_top_ratio=0.35)
        b = self._preprocess_signature(user_img, trim_top_ratio=0.0)
        if a is None or b is None:
            print(f"{self.log_prefix} preprocess returned None: a={a is None}, b={b is None}")
            return {"error": "Signature too faint or empty"}

        count_a = int(np.sum(a > 128))
        count_b = int(np.sum(b > 128))
        print(f"{self.log_prefix} preprocessed: count_a={count_a}, count_b={count_b}")

        # Grid histogram similarity: tolerant of small positional shifts.
        grid_sim = self._grid_histogram_similarity(a, b, cols=8, rows=4)

        # Pixel-level IoU: strict spatial overlap bonus.
        ink_a = (a > 128).astype(np.float32)
        ink_b = (b > 128).astype(np.float32)
        intersection = float(np.sum(ink_a * ink_b))
        union = float(np.sum(np.clip(ink_a + ink_b, 0.0, 1.0)))
        iou = intersection / max(union, 1.0)

        # Combined: grid histogram is primary (position-tolerant),
        # IoU is secondary (rewards exact overlap).
        sim = grid_sim * 0.7 + iou * 0.3
        score = max(0.0, min(100.0, sim * 100.0))
        is_match = score >= 50.0

        print(
            f"{self.log_prefix} signature match: grid_sim={grid_sim:.3f} iou={iou:.3f} "
            f"score={score:.1f} is_match={is_match}"
        )

        return {
            "is_match": bool(is_match),
            "match_score": float(score),
            "diagnostics": {
                "grid_sim": float(grid_sim),
                "iou": float(iou),
                "method": "grid_histogram+iou",
            },
        }
