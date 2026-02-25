# ArcFace Recognition (RTBD)

Simple face-recognition project using **InsightFace (ArcFace)**, with:
- local webcam recognition
- face database generation from images in `known_face/`
- Flask APIs for embedding extraction

## Project Structure

- `webcam.py` — realtime webcam recognition against known faces
- `database.py` — builds and saves `face_db.pkl` from `known_face/`
- `attendance.py` — Flask API to return embeddings for all detected faces
- `ml_service.py` — Flask API to return embedding for exactly one face
- `test_arcface.py` — quick local test script
- `known_face/` — store known people images (`.jpg` / `.png`)

## Requirements

- Python 3.x
- `insightface`
- `opencv-python`
- `numpy`
- `flask`

If you use the included virtual environment (`env`), activate it first.

## Setup (Windows PowerShell)

```powershell
cd C:\Users\24sai\Desktop\RTBD\ArcFace
.\env\Scripts\Activate.ps1
```

If needed, install dependencies:

```powershell
pip install insightface opencv-python numpy flask
```

## Prepare Known Faces

1. Add images to `known_face/` (file name becomes person name).
2. Run database builder:

```powershell
python database.py
```

This creates/updates `face_db.pkl`.

## Run Webcam Recognition

```powershell
python webcam.py
```

- Press `q` to quit.
- Matching threshold is currently `0.5` in `webcam.py`.

## Run API Services

### Option 1: Multi-face embedding API

```powershell
python attendance.py
```

Endpoint:
- `POST /query-embedding`
- Form-data field: `image`

Response includes:
- `faces_detected`
- list of `bbox` + `embedding`

### Option 2: Single-face embedding API

```powershell
python ml_service.py
```

Endpoint:
- `POST /generate-embedding`
- Form-data field: `image`

Returns error unless image contains exactly one face.

> Note: `attendance.py` and `ml_service.py` both use port `5001`. Run one at a time, or change one port.

## Quick Test

```powershell
python test_arcface.py
```

It reads `test.jpeg` and prints detected faces + embedding shape.

## GPU / CPU Note

Current scripts use:

```python
model.prepare(ctx_id=0)
```

If GPU/CUDA is unavailable, switch to CPU by changing `ctx_id=0` to `ctx_id=-1` in scripts.

## GitHub Push (already initialized locally)

If your remote is set:

```powershell
git branch -M main
git push -u origin main
```
