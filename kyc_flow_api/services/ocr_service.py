"""OCR service for KTP detection and processing."""

import os
import sys
import base64
import json
import uuid
import logging

import cv2

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from ktp_detector import detect_ktp
from face_matcher import extract_face_image

from ..config import (
    UPLOAD_DIR,
    LOG_PREFIX_OCR,
    IMAGE_PADDING_RATIO,
    OCR_TOP_BOTTOM_TRIM_RATIO,
    CARD_RATIO_MIN,
    CARD_RATIO_MAX,
    KTP_PHOTO_X_START,
    KTP_PHOTO_Y_START,
    KTP_PHOTO_X_END,
    KTP_PHOTO_Y_END,
    KTP_SIGNATURE_X_START,
    KTP_SIGNATURE_Y_START,
    KTP_SIGNATURE_X_END,
    KTP_SIGNATURE_Y_END,
)
from ..utils import (
    load_image_correct_orientation,
    extract_ktp_card_region,
    extract_ktp_photo_region,
    extract_ktp_signature_region,
    encode_image_to_base64,
    save_image,
    parse_ktp_text,
)

logger = logging.getLogger(__name__)

OCR_SERVICE_BUILD = "2026-05-10-signature-fe-box-v3"


class OCRService:
    """Service for handling KTP OCR detection and processing."""
    
    def __init__(self):
        """Initialize OCR service."""
        self.upload_dir = UPLOAD_DIR
        self.log_prefix = LOG_PREFIX_OCR
        print(f"{self.log_prefix} OCRService build={OCR_SERVICE_BUILD}")

    def _is_card_like(self, img) -> bool:
        """Heuristic: treat an image as a KTP card if its aspect ratio matches KTP layout."""
        try:
            if img is None or img.size == 0:
                return False
            h, w = img.shape[:2]
            if h <= 0 or w <= 0:
                return False
            ratio = (w / h) if h else 0.0
            return CARD_RATIO_MIN <= ratio <= CARD_RATIO_MAX
        except Exception:
            return False
    
    def process_ktp_image(self, file, box=None, photo_box=None, signature_box=None):
        """
        Process uploaded KTP image and extract data.
        
        Args:
            file: File object from Flask request
            box: Optional bounding box dict with x, y, w, h (normalized 0-1)
            photo_box: Optional red-box photo crop from FE with x, y, w, h (normalized 0-1)
            signature_box: Optional signature crop from FE with x, y, w, h (normalized 0-1)
            
        Returns:
            Dictionary with OCR results and image paths
        """
        request_id = uuid.uuid4().hex
        tmp_path = os.path.join(self.upload_dir, f'ktp_{request_id}_original.png')
        
        try:
            # Save uploaded file
            file.save(tmp_path)
            print(f'{self.log_prefix} saved upload to {tmp_path}')
            
            # Load and orient image
            img = load_image_correct_orientation(tmp_path)
            h, w = img.shape[:2]
            
            # Apply bounding box crop if provided
            cropped = self._apply_crop(img, box, w, h)
            cropped_path = os.path.join(self.upload_dir, f'ktp_{request_id}_card_crop.png')
            save_image(cropped, cropped_path)
            print(f'{self.log_prefix} cropped image saved to {cropped_path}')
            
            # Refine card region
            refined = extract_ktp_card_region(cropped)
            refined_for_signature = refined.copy() if refined is not None else None

            # Extract KTP photo region
            photo_path = None
            ktp_photo = None
            print(f'{self.log_prefix} photo_box received: {photo_box is not None}')
            print(f'{self.log_prefix} signature_box received: {signature_box is not None}')
            try:
                # Prefer explicit FE red-box crop for KTP person photo.
                ktp_photo = self._apply_photo_crop_box(img, photo_box, w, h)
                print(f'{self.log_prefix} after photoBox crop: ktp_photo is not None: {ktp_photo is not None}')
                if ktp_photo is not None:
                    photo_path = os.path.join(self.upload_dir, f'ktp_{request_id}_photo_box_crop.png')
                    save_image(ktp_photo, photo_path)
                    print(f'{self.log_prefix} ktp photo (photoBox) saved to {photo_path}, shape={ktp_photo.shape}')
                else:
                    print(f'{self.log_prefix} photoBox crop returned None, trying fallback...')
                    # Fallback: fixed layout crop from refined card
                    ktp_photo = extract_ktp_photo_region(refined)
                    print(f'{self.log_prefix} after fallback crop: ktp_photo is not None: {ktp_photo is not None}')
                if ktp_photo is not None:
                    if photo_path is None:
                        photo_path = os.path.join(self.upload_dir, f'ktp_{request_id}_photo_crop.png')
                    save_image(ktp_photo, photo_path)
                    print(f'{self.log_prefix} ktp photo crop saved to {photo_path}, shape={ktp_photo.shape}')
                else:
                    print(f'{self.log_prefix} WARNING: ktp_photo is still None after both attempts!')
            except Exception as e:
                print(f'{self.log_prefix} photo crop failed: {e}', exc_info=True)

            # Extract signature region — same 3-tier system as ktp_photo_crop:
            # 1. signatureBox from FE (direct coords, identical to photoBox)
            # 2. Derived from photoBox using KTP layout math
            # 3. Fixed coords on warped card (fallback)
            signature_path = None
            ktp_signature_crop = None
            try:
                ktp_signature_crop = self._apply_signature_crop_box(img, signature_box, w, h)
                if ktp_signature_crop is not None:
                    print(f'{self.log_prefix} signature crop from signatureBox OK, shape={ktp_signature_crop.shape}')
                else:
                    print(f'{self.log_prefix} signatureBox not provided, deriving from photoBox...')
                    ktp_signature_crop = self._apply_signature_crop_from_photo_box(img, photo_box, w, h)
                    if ktp_signature_crop is not None:
                        print(f'{self.log_prefix} signature crop derived from photoBox OK, shape={ktp_signature_crop.shape}')
                    else:
                        print(f'{self.log_prefix} photoBox derive failed, trying warped card fallback...')
                        ktp_signature_crop = extract_ktp_signature_region(refined_for_signature)
                        if ktp_signature_crop is not None:
                            print(f'{self.log_prefix} signature crop from warped card OK, shape={ktp_signature_crop.shape}')

                if ktp_signature_crop is not None:
                    signature_path = os.path.join(
                        self.upload_dir, f'ktp_signature_crop_{request_id}.png'
                    )
                    save_image(ktp_signature_crop, signature_path)
                    print(f'{self.log_prefix} ktp signature saved to {signature_path}')
                else:
                    print(f'{self.log_prefix} WARNING: signature crop returned None from all attempts')
            except Exception as e:
                print(f'{self.log_prefix} signature crop failed: {e}', exc_info=True)

            # Optional vertical trim to remove noisy margins above/below card text area
            refined = self._apply_top_bottom_trim(refined)

            refined_path = os.path.join(self.upload_dir, f'ktp_{request_id}_refined_card.png')
            save_image(refined, refined_path)
            print(f'{self.log_prefix} refined image saved to {refined_path}')
            
            # Perform OCR
            text = detect_ktp(refined_path)
            print(f'{self.log_prefix} raw OCR text extracted')
            
            # Parse KTP data
            ktp_data = parse_ktp_text(text)
            print(f'{self.log_prefix} parsed OCR data')
            
            # Prepare response
            response = self._prepare_response(
                refined,
                refined_path,
                photo_path,
                ktp_data,
                ktp_photo,
                refined_for_signature,
                request_id,
                ktp_signature_crop,
                signature_path,
            )
            
            return response
        
        finally:
            # Clean up temporary file
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    
    def _apply_crop(self, img, box, w, h):
        """Apply bounding box crop if provided."""
        if not box:
            print(f'{self.log_prefix} using full image without crop')
            return img
        
        try:
            x = max(0, int(box.get('x', 0) * w))
            y = max(0, int(box.get('y', 0) * h))
            cw = max(1, int(box.get('w', 1) * w))
            ch = max(1, int(box.get('h', 1) * h))
            
            # Add padding
            pad = int(IMAGE_PADDING_RATIO * min(w, h))
            x = max(0, x - pad)
            y = max(0, y - pad)
            cw = min(w - x, cw + 2 * pad)
            ch = min(h - y, ch + 2 * pad)
            
            print(f'{self.log_prefix} Image size: {w}x{h}, Crop: x={x}, y={y}, w={cw}, h={ch}')
            return img[y:y+ch, x:x+cw]
        
        except Exception as e:
            print(f'{self.log_prefix} box parsing failed: {e}, using full image')
            return img

    def _apply_photo_crop_box(self, img, photo_box, w, h):
        """Apply frontend red-box crop for KTP person photo extraction."""
        if not photo_box:
            return None

        try:
            x = max(0, int(photo_box.get('x', 0) * w))
            y = max(0, int(photo_box.get('y', 0) * h))
            cw = max(1, int(photo_box.get('w', 0) * w))
            ch = max(1, int(photo_box.get('h', 0) * h))

            if cw <= 1 or ch <= 1:
                return None

            x2 = min(w, x + cw)
            y2 = min(h, y + ch)
            if x >= x2 or y >= y2:
                return None

            print(
                f'{self.log_prefix} photoBox crop from FE: '
                f'x={x}, y={y}, w={x2 - x}, h={y2 - y}'
            )
            crop = img[y:y2, x:x2]
            if crop is None or crop.size == 0:
                return None
            
            # Apply top/bottom trim (0.2 ratio) to remove margins from photo
            cropped_h, cropped_w = crop.shape[:2]
            trim = int(cropped_h * 0.2)
            if trim > 0 and (cropped_h - (2 * trim)) >= 20:
                trimmed_crop = crop[trim : cropped_h - trim, 0:cropped_w]
                print(
                    f'{self.log_prefix} applied 0.2 top/bottom trim to photo: '
                    f'{cropped_w}x{cropped_h} -> {trimmed_crop.shape[1]}x{trimmed_crop.shape[0]}'
                )
                return trimmed_crop
            return crop
        except Exception as e:
            print(f'{self.log_prefix} photoBox crop parse failed: {e}')
            return None

    def _apply_signature_crop_box(self, img, signature_box, w, h):
        """Crop KTP signature from full frame using explicit signatureBox coords (mirrors _apply_photo_crop_box)."""
        if not signature_box:
            return None

        try:
            x = max(0, int(signature_box.get('x', 0) * w))
            y = max(0, int(signature_box.get('y', 0) * h))
            cw = max(1, int(signature_box.get('w', 0) * w))
            ch = max(1, int(signature_box.get('h', 0) * h))

            if cw <= 1 or ch <= 1:
                return None

            x2 = min(w, x + cw)
            y2 = min(h, y + ch)
            if x >= x2 or y >= y2:
                return None

            print(
                f'{self.log_prefix} signatureBox crop from FE: '
                f'x={x}, y={y}, w={x2 - x}, h={y2 - y}'
            )
            crop = img[y:y2, x:x2]
            if crop is None or crop.size == 0:
                return None
            return crop
        except Exception as e:
            print(f'{self.log_prefix} signatureBox crop parse failed: {e}')
            return None

    def _apply_signature_crop_from_photo_box(self, img, photo_box, w, h):
        """
        Derive the signature position in the full frame from photo_box,
        then crop — identical logic to _apply_photo_crop_box but for the signature area.

        Since we know where the KTP photo sits inside the card (KTP_PHOTO_* ratios),
        we can compute where the signature sits in the full frame.
        """
        if not photo_box:
            return None

        try:
            # Photo pixel coords in the full frame
            px = photo_box.get('x', 0) * w
            py = photo_box.get('y', 0) * h
            pw = photo_box.get('w', 0) * w
            ph = photo_box.get('h', 0) * h

            if pw <= 1 or ph <= 1:
                return None

            # Estimate card pixel size from photo size + known KTP layout ratios
            photo_ratio_w = float(KTP_PHOTO_X_END) - float(KTP_PHOTO_X_START)
            photo_ratio_h = float(KTP_PHOTO_Y_END) - float(KTP_PHOTO_Y_START)
            card_w = pw / photo_ratio_w
            card_h = ph / photo_ratio_h

            # Card top-left corner in the full frame
            card_x = px - float(KTP_PHOTO_X_START) * card_w
            card_y = py - float(KTP_PHOTO_Y_START) * card_h

            # Signature corners in the full frame
            sig_x1 = int(card_x + float(KTP_SIGNATURE_X_START) * card_w)
            sig_y1 = int(card_y + float(KTP_SIGNATURE_Y_START) * card_h)
            sig_x2 = int(card_x + float(KTP_SIGNATURE_X_END)   * card_w)
            sig_y2 = int(card_y + float(KTP_SIGNATURE_Y_END)   * card_h)

            sig_x1 = max(0, min(w - 1, sig_x1))
            sig_y1 = max(0, min(h - 1, sig_y1))
            sig_x2 = max(sig_x1 + 1, min(w, sig_x2))
            sig_y2 = max(sig_y1 + 1, min(h, sig_y2))

            if (sig_x2 - sig_x1) <= 1 or (sig_y2 - sig_y1) <= 1:
                return None

            print(
                f'{self.log_prefix} signatureBox derived from photoBox: '
                f'x={sig_x1}, y={sig_y1}, w={sig_x2 - sig_x1}, h={sig_y2 - sig_y1}'
            )
            crop = img[sig_y1:sig_y2, sig_x1:sig_x2]
            if crop is None or crop.size == 0:
                return None
            return crop
        except Exception as e:
            print(f'{self.log_prefix} signature crop from photoBox failed: {e}')
            return None

    def _apply_top_bottom_trim(self, img):
        """Trim top and bottom margins by configured ratio."""
        if img is None or img.size == 0:
            return img

        try:
            ratio = float(OCR_TOP_BOTTOM_TRIM_RATIO)
        except Exception:
            ratio = 0.0

        if ratio <= 0:
            return img

        h, w = img.shape[:2]
        trim = int(h * ratio)
        if trim <= 0 or (h - (2 * trim)) < 50:
            return img

        trimmed = img[trim:h - trim, 0:w]
        print(
            f'{self.log_prefix} applied top/bottom trim ratio={ratio}, '
            f'from {w}x{h} to {trimmed.shape[1]}x{trimmed.shape[0]}'
        )
        return trimmed
    
    def _prepare_response(
        self,
        refined,
        refined_path,
        photo_path,
        ktp_data,
        ktp_photo,
        refined_for_signature=None,
        request_id=None,
        ktp_signature_crop=None,
        ktp_signature_crop_path=None,
    ):
        """Prepare OCR response with encoded images."""
        print(f'{self.log_prefix} _prepare_response: ktp_photo is not None: {ktp_photo is not None}')
        if ktp_photo is not None:
            print(f'{self.log_prefix} _prepare_response: ktp_photo shape: {ktp_photo.shape}')
        
        # Prefer the exact processed image generated by detect_ktp preprocessing
        processed_path = os.path.splitext(refined_path)[0] + '_processed.png'
        preview_image = None
        if os.path.exists(processed_path):
            preview_image = cv2.imread(processed_path)
        if preview_image is None:
            preview_image = refined

        # Keep preview payload compact for React Native bridge stability
        ph, pw = preview_image.shape[:2]
        max_side = max(ph, pw)
        if max_side > 1200:
            scale = 1200.0 / max_side
            nw = int(pw * scale)
            nh = int(ph * scale)
            preview_image = cv2.resize(preview_image, (nw, nh), interpolation=cv2.INTER_AREA)

        ok, buffer = cv2.imencode('.jpg', preview_image, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
        processed_image_b64 = base64.b64encode(buffer).decode('utf-8') if ok else None
        
        ktp_photo_base64 = None
        # Prefer encoding ktp_photo directly if already extracted (faster, avoids file I/O)
        if ktp_photo is not None and ktp_photo.size > 0:
            print(f'{self.log_prefix} encoding ktp_photo directly (shape: {ktp_photo.shape})...')
            # Keep mobile payload smaller to reduce transfer and JSON parsing time.
            ph, pw = ktp_photo.shape[:2]
            max_side = max(ph, pw)
            if max_side > 700:
                scale = 700.0 / max_side
                nw = max(1, int(pw * scale))
                nh = max(1, int(ph * scale))
                ktp_photo = cv2.resize(ktp_photo, (nw, nh), interpolation=cv2.INTER_AREA)
                print(f'{self.log_prefix} downscaled ktp_photo to {nw}x{nh} for response payload')

            _, photo_buf = cv2.imencode('.png', ktp_photo)
            ktp_photo_base64 = base64.b64encode(photo_buf).decode('utf-8')
            print(f'{self.log_prefix} SUCCESS: encoded ktp_photo to base64, size={len(ktp_photo_base64)} chars')
        elif photo_path and os.path.exists(photo_path):
            # Fallback: read from saved file if ktp_photo not available
            print(f'{self.log_prefix} reading photo from file: {photo_path}...')
            photo_img = cv2.imread(photo_path)
            if photo_img is not None:
                ph, pw = photo_img.shape[:2]
                max_side = max(ph, pw)
                if max_side > 700:
                    scale = 700.0 / max_side
                    nw = max(1, int(pw * scale))
                    nh = max(1, int(ph * scale))
                    photo_img = cv2.resize(photo_img, (nw, nh), interpolation=cv2.INTER_AREA)
                    print(f'{self.log_prefix} downscaled fallback photo to {nw}x{nh} for response payload')

                _, photo_buf = cv2.imencode('.png', photo_img)
                ktp_photo_base64 = base64.b64encode(photo_buf).decode('utf-8')
                print(f'{self.log_prefix} encoded photo from file to base64, size={len(ktp_photo_base64)} chars')
            else:
                print(f'{self.log_prefix} ERROR: could not read photo from file: {photo_path}')
        else:
            print(f'{self.log_prefix} WARNING: ktp_photo is None and photo_path missing/invalid: {photo_path}')
        
        # Extract face from KTP for liveness matching
        ktp_face = extract_face_image(refined)
        ktp_face_base64 = None
        if ktp_face is not None:
            _, face_buf = cv2.imencode('.jpg', ktp_face)
            ktp_face_base64 = base64.b64encode(face_buf).decode('utf-8')
            print(f'{self.log_prefix} extracted KTP face, size={ktp_face.shape[1]}x{ktp_face.shape[0]}')
        else:
            print(f'{self.log_prefix} could not extract face from KTP')

        # Extract signature crop for signature matching step.
        # Prefer a precomputed crop (signatureBox or fallback) from process_ktp_image.
        ktp_signature_base64 = None
        ktp_signature_crop_path_out = ktp_signature_crop_path
        try:
            sig_crop = ktp_signature_crop

            if sig_crop is not None:
                sh, sw = sig_crop.shape[:2]
                max_side = max(sh, sw)
                if max_side > 700:
                    scale = 700.0 / max_side
                    nw = max(1, int(sw * scale))
                    nh = max(1, int(sh * scale))
                    sig_crop = cv2.resize(sig_crop, (nw, nh), interpolation=cv2.INTER_AREA)

                ok2, sig_buf = cv2.imencode('.png', sig_crop)
                if ok2:
                    ktp_signature_base64 = base64.b64encode(sig_buf).decode('utf-8')
                    if ktp_signature_crop_path_out is None:
                        try:
                            suffix = request_id or str(int(uuid.uuid4().int % 10**12))
                            ktp_signature_crop_path_out = os.path.join(
                                self.upload_dir,
                                f'ktp_signature_crop_{suffix}.png',
                            )
                            save_image(sig_crop, ktp_signature_crop_path_out)
                        except Exception:
                            ktp_signature_crop_path_out = None
        except Exception as e:
            print(f'{self.log_prefix} signature crop failed: {e}')
        
        print(f'{self.log_prefix} FINAL RESPONSE: ktp_photo_base64 is not None: {ktp_photo_base64 is not None}')
        if ktp_photo_base64:
            print(f'{self.log_prefix} FINAL RESPONSE: ktp_photo_base64 size: {len(ktp_photo_base64)} chars')
        else:
            print(f'{self.log_prefix} ERROR: ktp_photo_base64 is NONE in final response!')
        
        return {
            **ktp_data,
            'processed_image': processed_image_b64,
            'ktp_face_base64': ktp_face_base64,
            'ktp_photo_base64': ktp_photo_base64,
            'ktp_signature_base64': ktp_signature_base64,
            'ktp_crop_path': None,  # Paths are server-side only
            'ktp_refined_path': refined_path,
            'ktp_photo_crop_path': photo_path,
            'ktp_signature_crop_path': ktp_signature_crop_path_out,
        }
