"""
Face matching module for KYC flow.
Compares face from KTP with face from liveness detection.
"""
import os
import cv2
import numpy as np
import dlib
from scipy.spatial import distance

MODEL_DIR = os.environ.get(
    'DLIB_MODEL_DIR',
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models'),
)
SHAPE_PREDICTOR_5_PATH = os.environ.get(
    'DLIB_SHAPE_PREDICTOR_5_PATH',
    os.path.join(MODEL_DIR, 'shape_predictor_5_face_landmarks.dat'),
)
FACE_RECOGNITION_MODEL_PATH = os.environ.get(
    'DLIB_FACE_RECOGNITION_MODEL_PATH',
    os.path.join(MODEL_DIR, 'dlib_face_recognition_resnet_model_v1.dat'),
)

FACE_MATCHER_INIT_ERROR = None

try:
    detector = dlib.get_frontal_face_detector()
    try:
        sp = dlib.shape_predictor(SHAPE_PREDICTOR_5_PATH)
        facerec = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL_PATH)
    except Exception as e:
        facerec = None
        sp = None
        FACE_MATCHER_INIT_ERROR = (
            'Face model files could not be loaded. '
            f'shape5={SHAPE_PREDICTOR_5_PATH} (exists={os.path.exists(SHAPE_PREDICTOR_5_PATH)}), '
            f'resnet={FACE_RECOGNITION_MODEL_PATH} (exists={os.path.exists(FACE_RECOGNITION_MODEL_PATH)}), '
            f'error={e}'
        )
except:
    detector = None
    facerec = None
    sp = None
    FACE_MATCHER_INIT_ERROR = 'dlib face detector is not available.'


def get_face_descriptor(frame, return_face_rect=False):
    """
    Extract face descriptor (embedding) from a frame.
    
    Args:
        frame: BGR image
        return_face_rect: if True, also return face bounding box
    
    Returns:
        descriptor (128-D vector) or None if no face detected
    """
    if detector is None or facerec is None or sp is None:
        return None if not return_face_rect else (None, None)

    # Resize for faster detection if very large
    h, w = frame.shape[:2]
    max_dim = max(h, w)
    scale = 1.0
    if max_dim > 640:
        scale = 640.0 / max_dim
    if scale < 1.0:
        small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = frame

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)

    if len(faces) == 0:
        return None if not return_face_rect else (None, None)

    face = faces[0]
    # If we processed a scaled image, map face rect to original coordinates approximately
    if scale < 1.0:
        # scale up rect coordinates
        left = int(face.left() / scale)
        top = int(face.top() / scale)
        right = int(face.right() / scale)
        bottom = int(face.bottom() / scale)
        rect_for_shape = dlib.rectangle(left, top, right, bottom)
        shape = sp(frame, rect_for_shape)
        face_descriptor = np.array(facerec.compute_face_descriptor(frame, shape))
    else:
        shape = sp(frame, face)
        face_descriptor = np.array(facerec.compute_face_descriptor(frame, shape))

    if return_face_rect:
        rect = (face.left(), face.top(), face.width(), face.height())
        return face_descriptor, rect
    return face_descriptor


def extract_face_image(frame):
    """
    Extract and crop the face region from an image.
    Returns: cropped face image or None if no face detected
    """
    if detector is None:
        return None

    h, w = frame.shape[:2]
    max_dim = max(h, w)
    scale = 1.0
    if max_dim > 640:
        scale = 640.0 / max_dim
    if scale < 1.0:
        small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = frame

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)

    if len(faces) == 0:
        return None

    face = faces[0]
    # Map coordinates back to original image if we scaled
    if scale < 1.0:
        x = int(face.left() / scale)
        y = int(face.top() / scale)
        w = int(face.width() / scale)
        h = int(face.height() / scale)
    else:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

    # Add 10% padding
    pad = int(0.1 * max(w, h))
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(frame.shape[1] - x, w + 2 * pad)
    h = min(frame.shape[0] - y, h + 2 * pad)

    return frame[y : y + h, x : x + w]


def match_faces(ktp_frame, liveness_frame):
    """
    Compare KTP face with liveness face.
    
    Args:
        ktp_frame: BGR image from KTP card
        liveness_frame: BGR image from liveness detection
    
    Returns:
        dict with:
        - 'match_score': float 0-100 (similarity percentage)
        - 'ktp_face': cropped face from KTP
        - 'liveness_face': cropped face from liveness
        - 'is_match': bool (True if score > 50)
    """
    if facerec is None or sp is None:
        return {
            "match_score": 0,
            "ktp_face": None,
            "liveness_face": None,
            "is_match": False,
            "error": FACE_MATCHER_INIT_ERROR or "Face recognition model not available",
        }

    ktp_desc = get_face_descriptor(ktp_frame)
    liveness_desc = get_face_descriptor(liveness_frame)

    if ktp_desc is None or liveness_desc is None:
        return {
            "match_score": 0,
            "ktp_face": None,
            "liveness_face": None,
            "is_match": False,
            "error": "Could not extract face from one or both images",
        }

    # Calculate Euclidean distance between descriptors
    # dlib face recognition uses threshold ~0.6 for same person
    # Distance < 0.6 = likely same person
    dist = distance.euclidean(ktp_desc, liveness_desc)

    # Convert distance to similarity score (0-100)
    # dist=0 -> score=100, dist=0.6 -> score=50, dist=1.2 -> score=0
    score = max(0, min(100, (1 - dist / 1.2) * 100))

    ktp_face = extract_face_image(ktp_frame)
    liveness_face = extract_face_image(liveness_frame)

    return {
        "match_score": score,
        "ktp_face_base64": None,  # Will be set by caller if needed
        "liveness_face_base64": None,  # Will be set by caller if needed
        "is_match": score >= 50,
        "distance": dist,
    }


def match_faces_with_best_liveness(ktp_frame, liveness_frames):
    """
    Match KTP face with the best frame from liveness video sequence.
    
    Args:
        ktp_frame: BGR image from KTP card
        liveness_frames: list of BGR frames from liveness video
    
    Returns:
        dict with match results and best liveness frame index
    """
    if not liveness_frames or len(liveness_frames) == 0:
        return {
            "match_score": 0,
            "is_match": False,
            "error": "No liveness frames provided",
            "best_frame_idx": -1,
        }

    best_score = 0
    best_result = None
    best_idx = -1

    # Try to find the best matching frame from liveness video
    for i, liveness_frame in enumerate(liveness_frames):
        result = match_faces(ktp_frame, liveness_frame)
        if result["match_score"] > best_score:
            best_score = result["match_score"]
            best_result = result
            best_idx = i

    if best_result is None:
        return {
            "match_score": 0,
            "is_match": False,
            "error": "Could not match faces",
            "best_frame_idx": -1,
        }

    best_result["best_frame_idx"] = best_idx
    return best_result
