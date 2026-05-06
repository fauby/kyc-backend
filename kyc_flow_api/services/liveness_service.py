"""Liveness detection service."""

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

from liveness_detector import (
    detect_liveness_sequence,
    get_best_liveness_frame,
)

from ..config import LOG_PREFIX_LIVENESS, DEFAULT_REQUIRED_POSES

logger = logging.getLogger(__name__)


class LivenessService:
    """Service for handling liveness detection."""
    
    def __init__(self):
        """Initialize liveness service."""
        self.log_prefix = LOG_PREFIX_LIVENESS
        self.max_frame_side = 640

    def _resize_for_speed(self, frame):
        """Resize large frames to speed up decode, pose, and matching."""
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
    
    def decode_frames(self, frames_b64):
        """
        Decode base64-encoded frames to numpy arrays.
        
        Args:
            frames_b64: List of base64-encoded frame strings
            
        Returns:
            List of BGR frames, empty list if decoding fails
        """
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
    
    def detect_liveness(self, frames_b64, required_poses=None):
        """
        Detect liveness from frame sequence.
        
        Args:
            frames_b64: List of base64-encoded frames
            required_poses: List of required poses, defaults to all poses
            
        Returns:
            Dictionary with liveness detection result and diagnostics
        """
        if required_poses is None:
            required_poses = DEFAULT_REQUIRED_POSES
        
        # Decode frames
        frames = self.decode_frames(frames_b64)
        if not frames:
            return {'error': 'Could not decode frames'}
        
        # Detect liveness
        result = detect_liveness_sequence(frames, required_poses)

        # Reuse decoded frames to provide best frame and avoid a second API round-trip.
        best_frame, yaw, pitch, roll = get_best_liveness_frame(frames)
        if best_frame is not None:
            ok, buffer = cv2.imencode(
                '.jpg', best_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 62]
            )
            if ok:
                result['best_frame'] = base64.b64encode(buffer).decode('utf-8')
                result['best_frame_head_pose'] = {
                    'yaw': float(yaw) if yaw is not None else None,
                    'pitch': float(pitch) if pitch is not None else None,
                    'roll': float(roll) if roll is not None else None,
                }
        
        # Add diagnostics
        faces_count = int(round(float(result.get('face_coverage', 0.0)) * len(frames)))
        result['diagnostics'] = {
            'frames_total': len(frames),
            'faces_detected': faces_count,
            'required_poses': required_poses,
            'downscaled_max_side': self.max_frame_side,
        }
        
        return result
    
    def get_best_frame(self, frames_b64):
        """
        Find the best (most neutral) frame from sequence.
        
        Args:
            frames_b64: List of base64-encoded frames
            
        Returns:
            Dictionary with best frame and head pose, or error message
        """
        frames = self.decode_frames(frames_b64)
        if not frames:
            return {'error': 'Could not decode frames'}
        
        best_frame, yaw, pitch, roll = get_best_liveness_frame(frames)
        
        if best_frame is None:
            return {'error': 'Could not find face in frames'}
        
        _, buffer = cv2.imencode('.jpg', best_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 62])
        best_frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            'best_frame': best_frame_b64,
            'head_pose': {
                'yaw': float(yaw) if yaw is not None else None,
                'pitch': float(pitch) if pitch is not None else None,
                'roll': float(roll) if roll is not None else None,
            }
        }
