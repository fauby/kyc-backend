"""Liveness detection and face matching routes."""

from flask import request, jsonify
from ..services import LivenessService, MatchingService

liveness_service = LivenessService()
matching_service = MatchingService()


def register_liveness_routes(app):
    """Register liveness and matching routes to Flask app."""
    
    @app.route('/liveness/detect', methods=['POST'])
    def detect_liveness():
        """
        Detect liveness from frame sequence.
        
        Request (JSON):
            - frames: list of base64-encoded frames
            - required_poses (optional): list of poses to detect
        
        Response:
            - is_live: bool indicating liveness
            - poses_detected: list of detected poses
            - scores: confidence scores per pose
            - confidence: overall confidence 0-100
            - diagnostics: frame analysis details
        """
        try:
            data = request.get_json()
            frames_b64 = data.get('frames', [])
            required_poses = data.get('required_poses')
            
            if not frames_b64:
                return jsonify({'error': 'No frames provided'}), 400
            
            print('[liveness_routes] Detecting liveness from frames')
            result = liveness_service.detect_liveness(frames_b64, required_poses)
            
            if 'error' in result:
                return jsonify(result), 400
            
            return jsonify(result)
        
        except Exception as e:
            print(f'[liveness_routes] Error: {e}')
            return jsonify({'error': str(e)}), 500
    
    @app.route('/liveness/best-frame', methods=['POST'])
    def get_best_frame():
        """
        Find best (most neutral) frame from liveness sequence.
        
        Request (JSON):
            - frames: list of base64-encoded frames
        
        Response:
            - best_frame: base64-encoded best frame
            - head_pose: {yaw, pitch, roll} angles in degrees
        """
        try:
            data = request.get_json()
            frames_b64 = data.get('frames', [])
            
            if not frames_b64:
                return jsonify({'error': 'No frames provided'}), 400
            
            print('[liveness_routes] Finding best frame')
            result = liveness_service.get_best_frame(frames_b64)
            
            if 'error' in result:
                return jsonify(result), 400
            
            return jsonify(result)
        
        except Exception as e:
            print(f'[liveness_routes] Error: {e}')
            return jsonify({'error': str(e)}), 500
    
    @app.route('/match/faces', methods=['POST'])
    def match_ktp_liveness():
        """
        Match KTP face with liveness face.
        
        Request (JSON or multipart):
            - ktp_face_image: base64 KTP face (preferred) OR
            - ktp_photo_crop_path: path to KTP photo on server
            - liveness_image: base64 single liveness frame OR
            - liveness_frames: list of base64 liveness frames
        
        Response:
            - match_score: 0-100 similarity
            - is_match: bool (True if score >= 50)
            - ktp_face_base64: extracted KTP face
            - liveness_face_base64: extracted liveness face
            - distance: euclidean distance between descriptors
            - best_frame_idx: index of best matching frame (if multiple frames provided)
        """
        try:
            # Parse request (supports both JSON and multipart)
            if request.is_json:
                data = request.get_json()
                ktp_b64 = data.get('ktp_face_image')
                liveness_b64 = data.get('liveness_image')
                liveness_frames_b64 = data.get('liveness_frames')
                ktp_photo_path = data.get('ktp_photo_crop_path')
            else:
                data = request.form
                ktp_b64 = data.get('ktp_face_image')
                liveness_b64 = data.get('liveness_image')
                liveness_frames_b64 = request.form.getlist('liveness_frames')
                ktp_photo_path = data.get('ktp_photo_crop_path')
            
            print('[match_routes] Matching faces')
            
            # Load KTP image
            ktp_frame = matching_service.load_ktp_image(ktp_b64, ktp_photo_path)
            if ktp_frame is None:
                return jsonify({
                    'error': 'KTP photo crop required. Please rescan the KTP.'
                }), 400
            
            # Match single liveness frame
            if liveness_b64:
                liveness_frame = matching_service.decode_image(liveness_b64)
                if liveness_frame is None:
                    return jsonify({'error': 'Could not decode liveness image'}), 400
                
                result = matching_service.match_faces_single(ktp_frame, liveness_frame)
            
            # Match against sequence of liveness frames
            elif liveness_frames_b64:
                liveness_frames = matching_service.decode_frames(liveness_frames_b64)
                if not liveness_frames:
                    return jsonify({'error': 'Could not decode liveness frames'}), 400
                
                result = matching_service.match_faces_sequence(ktp_frame, liveness_frames)
            
            else:
                return jsonify({'error': 'Liveness image or frames required'}), 400
            
            return jsonify(result)
        
        except Exception as e:
            print(f'[match_routes] Error: {e}')
            return jsonify({'error': str(e)}), 500
