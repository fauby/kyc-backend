"""Configuration and constants for KYC Flow API."""

import os

# Directories
MODULE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(MODULE_DIR, 'uploads')

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Image processing
# Card contour can be relatively small in mobile captures (card not filling frame),
# so keep this low to avoid falling back to full-image cropping.
MIN_CONTOUR_AREA_RATIO = 0.05
CARD_RATIO_MIN = 1.35
CARD_RATIO_MAX = 1.9
MIN_DIMENSION = 50
CARD_EXPANSION_RATIO = 0.03
IMAGE_PADDING_RATIO = 0.02
# Additional vertical trim ratio applied after card refinement.
# Example: 0.2 removes 20% from top and 20% from bottom.
OCR_TOP_BOTTOM_TRIM_RATIO = 0.2

# KTP Photo extraction (Indonesian KTP layout)
KTP_PHOTO_X_START = 0.73
KTP_PHOTO_Y_START = 0.24
KTP_PHOTO_X_END = 0.94
KTP_PHOTO_Y_END = 0.76

# KTP Signature extraction (card-relative, same system as KTP photo crop)
# Adjust these 4 values to reposition the signature crop on the warped card.
KTP_SIGNATURE_X_START = 0.73
KTP_SIGNATURE_Y_START = 0.78
KTP_SIGNATURE_X_END   = 0.94
KTP_SIGNATURE_Y_END   = 0.96

# Flask
FLASK_HOST = '0.0.0.0'
OCR_PORT = 5000
LIVENESS_PORT = 5001

# Logging
LOG_PREFIX_OCR = '[ktp_detector]'
LOG_PREFIX_LIVENESS = '[liveness_detector]'
LOG_PREFIX_MATCHER = '[face_matcher]'

# Default poses for liveness detection
DEFAULT_REQUIRED_POSES = ['front', 'left', 'right', 'up', 'down']

# Canny edge detection thresholds
CANNY_THRESHOLD_LOW = 50
CANNY_THRESHOLD_HIGH = 150

# Head pose thresholds
POSE_THRESHOLD_YAW = 20
POSE_THRESHOLD_PITCH = 15
POSE_FRONT_THRESHOLD = 10

# Image dimensions for face detection scaling
FACE_DETECTION_MAX_DIM = 1000
HEAD_POSE_MAX_DIM = 800

# OCR Configuration
OCR_TESSERACT_CONFIG = "--oem 1 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,.-/"
OCR_LANG = 'ind'
