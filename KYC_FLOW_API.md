# KTP OCR & KYC Flow Backend API

Complete KYC (Know Your Customer) flow with KTP OCR detection, liveness detection, and face matching.

## Features

- **KTP OCR Detection** — Extract and parse KTP card data (NIK, nama, alamat, etc.)
- **Liveness Detection** — Detect head movements (left, right, up, down) for anti-spoofing
- **Face Matching** — Compare KTP photo with liveness face and calculate match score
- **Structured KYC Flow** — Complete end-to-end verification pipeline

## Installation

### Requirements

```bash
pip install flask opencv-python pillow numpy pytesseract dlib scipy
```

### Tesseract Installation

**Linux (Debian/Ubuntu):**

```bash
apt-get install tesseract-ocr libtesseract-dev
apt-get install libdlib-dev
```

**macOS:**

```bash
brew install tesseract
brew install dlib
```

**Windows:**
Download from: https://github.com/UB-Mannheim/tesseract/wiki

### dlib Face Recognition Models

Download the models actually used by the current code:

```bash
wget http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2
bunzip2 shape_predictor_5_face_landmarks.dat.bz2
sudo mv shape_predictor_5_face_landmarks.dat /usr/share/dlib/models/

wget http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
bunzip2 dlib_face_recognition_resnet_model_v1.dat.bz2
sudo mv dlib_face_recognition_resnet_model_v1.dat /usr/share/dlib/models/

wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
sudo mv shape_predictor_68_face_landmarks.dat /usr/share/dlib/models/
```

### Why these models are needed

- `shape_predictor_5_face_landmarks.dat` is used by face matching to align the face before building the descriptor.
- `dlib_face_recognition_resnet_model_v1.dat` is the actual face embedding model used for matching.
- `shape_predictor_68_face_landmarks.dat` is used by liveness detection to estimate head pose.
- `mmod_human_face_detector.dat` is not required by the current code.

## API Endpoints

### 1. KTP OCR Detection

**Endpoint:** `POST /ocr/ktp`

Extract and parse KTP card from uploaded image.

**Request:**

```
multipart/form-data:
- file: image file (PNG/JPG)
- box: (optional) JSON crop box {"x": 0.0-1.0, "y": 0.0-1.0, "w": 0.0-1.0, "h": 0.0-1.0}
```

**Response:**

```json
{
  "nik": "3171234567890123",
  "nama": "MIRA SETIAWAN",
  "tempat_tgl_lahir": "JAKARTA, 18-02-1986",
  "jenis_kelamin": "PEREMPUAN",
  "alamat": "JL. PASTI CEPAT A7/66",
  "agama": "ISLAM",
  "pekerjaan": "PEGAWAI SWASTA",
  "status_perkawinan": "KAWIN",
  "kewarganegaraan": "WNI",
  "berlaku_hingga": "22-02-2017",
  "rt_rw": "007/008",
  "kel_desa": "PEGADUNGAN",
  "kecamatan": "KALIDERES",
  "processed_image": "base64_encoded_ktp_image",
  "ktp_face_base64": "base64_encoded_face_crop"
}
```

### 2. Liveness Detection

**Endpoint:** `POST /liveness/detect`

Detect head movements for liveness verification.

**Request (JSON):**

```json
{
  "frames": ["base64_frame1", "base64_frame2", ...],
  "required_poses": ["left", "right", "up", "down"]
}
```

**Response:**

```json
{
  "is_live": true,
  "poses_detected": ["left", "right", "up", "down"],
  "scores": {
    "left": 85.5,
    "right": 92.3,
    "up": 78.2,
    "down": 88.9
  },
  "confidence": 86.2
}
```

### 3. Get Best Frame

**Endpoint:** `POST /liveness/best-frame`

Extract the most neutral face from liveness video sequence.

**Request (JSON):**

```json
{
  "frames": ["base64_frame1", "base64_frame2", ...]
}
```

**Response:**

```json
{
  "best_frame": "base64_encoded_best_frame",
  "head_pose": {
    "yaw": 2.5,
    "pitch": -1.2,
    "roll": 0.0
  }
}
```

### 4. Face Matching

**Endpoint:** `POST /match/faces`

Compare KTP face with liveness face and calculate match score.

**Request (JSON):**

```json
{
  "ktp_image": "base64_ktp_with_face",
  "liveness_frames": ["base64_frame1", "base64_frame2", ...]
}
```

Or with single image:

```json
{
  "ktp_image": "base64_ktp_with_face",
  "liveness_image": "base64_liveness_frame"
}
```

**Response:**

```json
{
  "match_score": 87.5,
  "is_match": true,
  "distance": 0.42,
  "ktp_face_base64": "base64_ktp_face",
  "liveness_face_base64": "base64_liveness_face",
  "best_frame_idx": 5
}
```

## Complete KYC Flow

### Frontend Implementation Steps

1. **Capture KTP**
   - Show camera with alignment box
   - Call `/ocr/ktp` endpoint
   - Receive KTP data and `ktp_face_base64`
   - Show OCR results to user
   - User presses "Continue"

2. **Liveness Detection**
   - Show instruction: "Turn your head LEFT"
   - Capture video frames while user moves
   - Show instruction: "Turn your head RIGHT"
   - Capture more frames
   - Show instruction: "Look UP"
   - Capture more frames
   - Show instruction: "Look DOWN"
   - Capture remaining frames
   - Call `/liveness/detect` to verify liveness
   - Call `/liveness/best-frame` to get best neutral frame

3. **Face Matching**
   - Call `/match/faces` with `ktp_image` (KTP face) and `liveness_frames` (video frames)
   - Receive `match_score` (0-100)
   - Show match score to user

4. **Decision**
   - If `match_score >= 50`: ✅ KYC Passed → Button: "Complete" (go to home)
   - If `match_score < 50`: ❌ KYC Failed → Buttons: "Re-scan KTP" or "Try Liveness Again"

## Run Backend

```bash
python main_backend.py
```

Server will start on `http://0.0.0.0:5000`

## Testing

### Test KTP OCR

```bash
curl -X POST -F "file=@ktp_image.png" http://127.0.0.1:5000/ocr/ktp
```

### Test Liveness (with video frames)

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "frames": ["base64_frame1", "base64_frame2"],
    "required_poses": ["left", "right", "up", "down"]
  }' \
  http://127.0.0.1:5000/liveness/detect
```

### Test Face Matching

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "ktp_image": "base64_ktp_face",
    "liveness_image": "base64_liveness_face"
  }' \
  http://127.0.0.1:5000/match/faces
```

## Performance Notes

- **Image Downscaling** — Images > 2000px are automatically downscaled for faster OCR
- **Fast Tesseract Mode** — Uses `--oem 1 --psm 6` for faster text recognition
- **Face Detection** — Uses dlib for reliable face detection and landmark extraction
- **Match Score** — Uses Euclidean distance on 128-D face descriptors (dlib model)

## Troubleshooting

### Face Recognition Model Not Found

```
Error: "Face recognition model not available"
```

**Solution:** Download dlib models as shown in Installation section.

### Tesseract Not Found

```
Error: tesseract is not installed or cannot be found
```

**Solution:** Install tesseract-ocr and set TESSDATA_PREFIX environment variable.

### No Face Detected

- Ensure lighting is adequate
- Face should be clearly visible
- Try with higher resolution images

### Low Match Scores

- Ensure faces are in similar lighting conditions
- Keep face relatively neutral (not tilted)
- Use high-quality camera for liveness
