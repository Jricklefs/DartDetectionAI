# DartDetectionAI

Standalone dartboard detection and scoring API service.

## Overview

A modular dart detection system that:
1. **Calibrates** cameras by analyzing dartboard images and storing the transformation matrix
2. **Detects** dart positions in subsequent images using the stored calibration
3. **Scores** darts by mapping detected positions to dartboard segments

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      API Service                             │
├─────────────────────────────────────────────────────────────┤
│  POST /calibrate     - Calibrate camera(s) from images      │
│  POST /detect        - Detect dart and return score         │
│  GET  /calibrations  - List stored calibrations             │
│  DELETE /calibration/{camera_id} - Remove calibration       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Core Detection Engine                     │
├─────────────────────────────────────────────────────────────┤
│  calibration.py   - Dartboard detection & homography        │
│  scoring.py       - Segment/zone calculation                │
│  geometry.py      - Dartboard constants & math              │
│  storage.py       - In-memory calibration storage           │
└─────────────────────────────────────────────────────────────┘
```

## API Endpoints

### POST /calibrate
Calibrate one or more cameras.

**Request:**
```json
{
  "cameras": [
    {
      "camera_id": "cam1",
      "image": "<base64 encoded image>"
    }
  ]
}
```

**Response:**
```json
{
  "results": [
    {
      "camera_id": "cam1",
      "success": true,
      "quality": 0.95,
      "overlay_image": "<base64 encoded image with grid overlay>",
      "segment_at_top": 20
    }
  ]
}
```

### POST /detect
Detect dart position and calculate score.

**Request:**
```json
{
  "camera_id": "cam1",
  "image": "<base64 encoded image>"
}
```

**Response:**
```json
{
  "success": true,
  "dart": {
    "position": {"x": 123, "y": 456},
    "score": 60,
    "multiplier": 3,
    "segment": 20,
    "zone": "triple"
  }
}
```

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run API server
python -m uvicorn app.main:app --reload --port 8000

# Run tests
pytest
```

## Project Structure

```
DartDetectionAI/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI app entry point
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py     # API endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   ├── calibration.py    # Dartboard calibration
│   │   ├── detection.py      # Dart tip detection
│   │   ├── scoring.py        # Score calculation
│   │   ├── geometry.py       # Dartboard geometry constants
│   │   └── storage.py        # Calibration storage
│   └── models/
│       ├── __init__.py
│       └── schemas.py    # Pydantic models
├── tests/
│   └── ...
├── ui/                   # Minimal test UI
│   └── index.html
├── requirements.txt
└── README.md
```
