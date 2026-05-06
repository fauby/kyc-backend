"""Image processing utilities for KYC Flow."""

import cv2
import numpy as np
from PIL import Image, ExifTags
from ..config import (
    CANNY_THRESHOLD_LOW, CANNY_THRESHOLD_HIGH,
    CARD_EXPANSION_RATIO, MIN_DIMENSION, CARD_RATIO_MIN, CARD_RATIO_MAX,
    MIN_CONTOUR_AREA_RATIO, KTP_PHOTO_X_START, KTP_PHOTO_Y_START,
    KTP_PHOTO_X_END, KTP_PHOTO_Y_END
)


def load_image_correct_orientation(image_path):
    """
    Load image and apply EXIF orientation before converting to OpenCV BGR.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        BGR image as numpy array
        
    Raises:
        ValueError: If image cannot be read
    """
    try:
        pil_image = Image.open(image_path)
        exif = pil_image.getexif()
        if exif:
            orientation_key = next(
                (k for k, v in ExifTags.TAGS.items() if v == "Orientation"),
                None,
            )
            orientation = exif.get(orientation_key) if orientation_key else None
            if orientation == 3:
                pil_image = pil_image.rotate(180, expand=True)
            elif orientation == 6:
                pil_image = pil_image.rotate(270, expand=True)
            elif orientation == 8:
                pil_image = pil_image.rotate(90, expand=True)

        rgb = np.array(pil_image.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Unable to read image: {image_path}")
        return img


def order_points(pts):
    """Order 4 points in clockwise order starting from top-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    """Apply 4-point perspective transform."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    if max_width < MIN_DIMENSION or max_height < MIN_DIMENSION:
        return image

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def extract_ktp_card_region(image):
    """
    Try to isolate the KTP card area from the capture image using contour detection.
    
    Args:
        image: BGR image
        
    Returns:
        Warped KTP card region or original image if no card detected
    """
    h, w = image.shape[:2]
    image_area = float(h * w)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, CANNY_THRESHOLD_LOW, CANNY_THRESHOLD_HIGH)
    edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours[:25]:
        area = cv2.contourArea(cnt)
        if area < image_area * MIN_CONTOUR_AREA_RATIO:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue

        rect = cv2.minAreaRect(approx)
        rw, rh = rect[1]
        if rw == 0 or rh == 0:
            continue
        ratio = max(rw, rh) / min(rw, rh)
        if ratio < CARD_RATIO_MIN or ratio > CARD_RATIO_MAX:
            continue

        # Expand the quad slightly to avoid cutting off card edges
        pts = approx.reshape(4, 2).astype("float32")
        center = pts.mean(axis=0)
        pad_px = int(CARD_EXPANSION_RATIO * max(w, h))
        vecs = pts - center
        norms = np.linalg.norm(vecs, axis=1).reshape(-1, 1)
        norms[norms == 0] = 1.0
        expanded = pts + (vecs / norms) * pad_px
        
        # Clip to image bounds
        expanded[:, 0] = np.clip(expanded[:, 0], 0, w - 1)
        expanded[:, 1] = np.clip(expanded[:, 1], 0, h - 1)

        warped = four_point_transform(image, expanded.astype("float32"))
        if warped.shape[0] > warped.shape[1]:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

        return warped

    return image


def extract_ktp_photo_region(card_image):
    """
    Crop the KTP photo area from a warped KTP card using fixed relative bounds.
    Tuned to Indonesian KTP layout.
    
    Args:
        card_image: Warped KTP card image
        
    Returns:
        Cropped photo region or None if extraction fails
    """
    h, w = card_image.shape[:2]
    if h < MIN_DIMENSION or w < MIN_DIMENSION:
        return None

    x1 = int(w * KTP_PHOTO_X_START)
    y1 = int(h * KTP_PHOTO_Y_START)
    x2 = int(w * KTP_PHOTO_X_END)
    y2 = int(h * KTP_PHOTO_Y_END)

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))

    photo = card_image[y1:y2, x1:x2]
    if photo.size == 0:
        return None
    return photo


def encode_image_to_base64(image, format='jpg'):
    """Encode image to base64 string."""
    success, buffer = cv2.imencode(f'.{format}', image)
    if not success:
        return None
    return buffer


def save_image(image, filepath):
    """Save image to file."""
    return cv2.imwrite(filepath, image)


def load_image_from_file(filepath):
    """Load image from file."""
    return cv2.imread(filepath)
