"""
Liveness detection module for KYC flow.
Detects head pose changes (front, left, right, up, down).
"""

import os

import cv2
import numpy as np
import dlib

MODEL_DIR = os.environ.get(
    'DLIB_MODEL_DIR',
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models'),
)
SHAPE_PREDICTOR_68_PATH = os.environ.get(
    'DLIB_SHAPE_PREDICTOR_68_PATH',
    os.path.join(MODEL_DIR, 'shape_predictor_68_face_landmarks.dat'),
)

# Initialize face detector and shape predictor
try:
    detector = dlib.get_frontal_face_detector()
    # Try to load shape predictor; if not available, use basic eye/face detection
    try:
        predictor = dlib.shape_predictor(SHAPE_PREDICTOR_68_PATH)
    except:
        predictor = None
except:
    detector = None
    predictor = None


def estimate_head_pose(frame):
    """
    Estimate head pose from a frame.
    Returns: (yaw, pitch, roll) angles in degrees, or None if no face detected.
    """
    if detector is None:
        return None, None, None

    # Work on a smaller copy for speed while preserving pose estimation accuracy
    h, w = frame.shape[:2]
    max_dim = max(h, w)
    scale = 1.0
    if max_dim > 800:
        scale = 800.0 / max_dim
    if scale < 1.0:
        small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = frame

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)

    if len(faces) == 0:
        return None, None, None

    face = faces[0]

    if predictor is None:
        # Basic face center estimation if predictor not available
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        center_x = x + w / 2
        center_y = y + h / 2
        frame_center_x = gray.shape[1] / 2
        frame_center_y = gray.shape[0] / 2

        yaw = (center_x - frame_center_x) / max(frame_center_x, 1.0) * 45
        pitch = (center_y - frame_center_y) / max(frame_center_y, 1.0) * 45
        roll = 0
        return yaw, pitch, roll

    try:
        # Use 68-point landmarks for head pose estimation
        landmarks = predictor(gray, face)

        # Convert landmarks to array
        points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)], dtype=np.float32)

        # Use eyes and nose for head pose
        left_eye = points[36:42].mean(axis=0)
        right_eye = points[42:48].mean(axis=0)
        nose = points[30]

        # Calculate yaw (left-right tilt)
        eye_center = (left_eye + right_eye) / 2
        yaw = np.arctan2(
            float(nose[0] - eye_center[0]),
            float(nose[1] - eye_center[1]) + 1e-6,
        ) * 180 / np.pi

        # Calculate pitch (up-down tilt) using the mouth center.
        mouth_center = points[48:68].mean(axis=0)
        pitch = np.arctan2(
            float(mouth_center[1] - nose[1]),
            float(np.linalg.norm(mouth_center - nose)) + 1e-6,
        ) * 180 / np.pi

        # Roll is harder to estimate with landmarks
        roll = 0

        # Since we ran on a scaled image, return approximate angles (scale invariant)
        return yaw, pitch, roll
    except Exception:
        # If landmark math fails on a noisy frame, fall back to the face box center.
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        center_x = x + w / 2
        center_y = y + h / 2
        frame_center_x = frame.shape[1] / 2
        frame_center_y = frame.shape[0] / 2

        yaw = (center_x - frame_center_x) / max(frame_center_x, 1.0) * 45
        pitch = (center_y - frame_center_y) / max(frame_center_y, 1.0) * 45
        roll = 0
        return yaw, pitch, roll


def detect_liveness_sequence(frames, required_poses=None):
    """
    Detect if a sequence of frames shows liveness (head movements).
    
    Args:
        frames: List of consecutive video frames (BGR format)
        required_poses: List of required poses ['front', 'left', 'right', 'up', 'down']
                       If None, detects any movement.
    
    Returns:
        dict with keys:
        - 'is_live': bool indicating liveness
        - 'poses_detected': list of poses detected
        - 'scores': dict with confidence scores for each pose
        - 'confidence': float 0-100 for overall liveness confidence
    """
    if required_poses is None:
        required_poses = ["front", "left", "right", "up", "down"]

    poses_detected = set()
    pose_scores = {pose: 0.0 for pose in required_poses}
    pose_counts = {pose: 0 for pose in required_poses}

    # Collect only frames where a face was actually detected.
    measured = []
    for idx, frame in enumerate(frames):
        yaw, pitch, roll = estimate_head_pose(frame)
        if yaw is None or pitch is None:
            continue
        measured.append(
            {
                "index": idx,
                "yaw": float(yaw),
                "pitch": float(pitch),
                "roll": float(roll or 0.0),
            }
        )

    if not measured:
        return {
            "is_live": False,
            "poses_detected": [],
            "scores": pose_scores,
            "confidence": 0.0,
            "face_coverage": 0.0,
            "calibrated_baseline": None,
        }

    # Use the most neutral measured frame as baseline (usually FRONT frame).
    baseline = min(measured, key=lambda p: abs(p["yaw"]) + abs(p["pitch"]))
    baseline_yaw = baseline["yaw"]
    baseline_pitch = baseline["pitch"]

    # Delta thresholds relative to baseline to make detection robust across devices.
    yaw_delta_threshold = 10.0
    pitch_delta_threshold = 8.0
    front_delta_yaw = 10.0
    front_delta_pitch = 8.0

    for m in measured:
        dyaw = m["yaw"] - baseline_yaw
        dpitch = m["pitch"] - baseline_pitch

        # FRONT: near baseline and stable.
        if abs(dyaw) <= front_delta_yaw and abs(dpitch) <= front_delta_pitch:
            conf = max(0.0, 100.0 - (abs(dyaw) * 4.0 + abs(dpitch) * 4.0))
            poses_detected.add("front")
            pose_counts["front"] += 1
            pose_scores["front"] = max(pose_scores["front"], conf)

        # LEFT / RIGHT from yaw deltas.
        if dyaw <= -yaw_delta_threshold:
            conf = min((abs(dyaw) - yaw_delta_threshold) / 18.0, 1.0) * 100.0
            poses_detected.add("left")
            pose_counts["left"] += 1
            pose_scores["left"] = max(pose_scores["left"], conf)
        elif dyaw >= yaw_delta_threshold:
            conf = min((abs(dyaw) - yaw_delta_threshold) / 18.0, 1.0) * 100.0
            poses_detected.add("right")
            pose_counts["right"] += 1
            pose_scores["right"] = max(pose_scores["right"], conf)

        # UP / DOWN from pitch deltas.
        if dpitch <= -pitch_delta_threshold:
            conf = min((abs(dpitch) - pitch_delta_threshold) / 14.0, 1.0) * 100.0
            poses_detected.add("down")
            pose_counts["down"] += 1
            pose_scores["down"] = max(pose_scores["down"], conf)
        elif dpitch >= pitch_delta_threshold:
            conf = min((abs(dpitch) - pitch_delta_threshold) / 14.0, 1.0) * 100.0
            poses_detected.add("up")
            pose_counts["up"] += 1
            pose_scores["up"] = max(pose_scores["up"], conf)

    # Check required poses only (supports custom required_poses input).
    required_set = set(required_poses)
    detected_required = sorted(list(required_set.intersection(poses_detected)))
    detected_ratio = len(detected_required) / max(len(required_set), 1)
    all_detected = len(detected_required) == len(required_set)

    # Penalize if face cannot be detected in enough frames.
    face_coverage = len(measured) / max(len(frames), 1)

    required_scores = [pose_scores.get(p, 0.0) for p in required_poses]
    avg_score = sum(required_scores) / max(len(required_scores), 1)

    confidence = (
        0.65 * avg_score
        + 25.0 * detected_ratio
        + 10.0 * face_coverage
    )
    confidence = max(0.0, min(100.0, confidence))

    # Accuracy-focused decision: require strong pose coverage + face coverage.
    is_live = (
        all_detected and face_coverage >= 0.8 and confidence >= 45.0
    ) or (
        detected_ratio >= 0.8 and face_coverage >= 0.9 and confidence >= 55.0
    )

    return {
        "is_live": bool(is_live),
        "poses_detected": detected_required,
        "scores": {p: pose_scores.get(p, 0.0) for p in required_poses},
        "confidence": float(confidence),
        "face_coverage": float(face_coverage),
        "calibrated_baseline": {
            "frame_index": int(baseline["index"]),
            "yaw": float(baseline_yaw),
            "pitch": float(baseline_pitch),
        },
    }


def get_best_liveness_frame(frames):
    """
    Find the best frame showing most neutral head pose (for face extraction).
    Returns: (best_frame, yaw, pitch, roll) or (None, None, None, None)
    """
    best_frame = None
    best_score = float("inf")

    for frame in frames:
        yaw, pitch, roll = estimate_head_pose(frame)
        if yaw is None:
            continue

        # Score: lower is better (neutral is 0,0,0)
        score = abs(yaw) + abs(pitch) + abs(roll)
        if score < best_score:
            best_score = score
            best_frame = frame

    if best_frame is None:
        return None, None, None, None

    yaw, pitch, roll = estimate_head_pose(best_frame)
    return best_frame, yaw, pitch, roll
