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
