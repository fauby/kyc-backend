"""OCR detection routes."""

import json
from flask import request, jsonify
from ..services import OCRService

ocr_service = OCRService()


def register_ocr_routes(app):
    """Register OCR routes to Flask app."""
    
    @app.route('/ocr/ktp', methods=['POST'])
    def ocr_ktp():
        """
        Detect and extract KTP card data from image.
        
        Request:
            - file: KTP image file
            - box (optional): JSON with bounding box {x, y, w, h} (0-1 normalized)
        
        Response:
            - KTP data fields (nik, nama, alamat, etc.)
            - processed_image: base64 encoded processed image
            - ktp_face_base64: extracted face from KTP
            - ktp_photo_base64: extracted photo region
            - ktp_refined_path: server path to refined card
            - ktp_photo_crop_path: server path to photo crop
        """
        try:
            # Validate file upload
            if 'file' not in request.files:
                return jsonify({'error': 'No file part'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400
            
            print(f'[ocr_routes] Processing file: {file.filename}')
            
            # Parse bounding boxes if provided
            box = None
            box_raw = request.form.get('box')
            if box_raw:
                try:
                    box = json.loads(box_raw)
                except Exception as e:
                    print(f'[ocr_routes] box parsing failed: {e}')

            photo_box = None
            photo_box_raw = request.form.get('photoBox')
            if photo_box_raw:
                try:
                    photo_box = json.loads(photo_box_raw)
                except Exception as e:
                    print(f'[ocr_routes] photoBox parsing failed: {e}')
            
            # Process KTP image
            response = ocr_service.process_ktp_image(file, box, photo_box)
            return jsonify(response)
        
        except Exception as e:
            print(f'[ocr_routes] Error: {e}')
            return jsonify({'error': str(e)}), 500
