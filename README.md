# DartDetect API

**Stateless dart detection as a service.**

Send dartboard images → Get dart tip positions and scores.

## Features

- 🎯 **Multi-camera support** — Combine views for better accuracy
- 📐 **Automatic calibration** — Detect dartboard from any angle
- ⚡ **Fast inference** — GPU-accelerated YOLO models
- 🔑 **API key authentication** — Per-customer access control
- 📊 **Usage tracking** — Monitor API calls per key

## API Endpoints

### Calibration

```http
POST /v1/calibrate
Authorization: Bearer <api_key>

{
  "cameras": [
    {"camera_id": "cam1", "image": "base64..."}
  ]
}
```

Stores calibration data associated with your API key. Returns overlay image showing detected zones.

### Detection

```http
POST /v1/detect
Authorization: Bearer <api_key>

{
  "cameras": [
    {"camera_id": "cam1", "image": "base64..."},
    {"camera_id": "cam2", "image": "base64..."}
  ]
}
```

Returns all detected dart tips with positions and scores:

```json
{
  "request_id": "req_abc123",
  "processing_ms": 42,
  "tips": [
    {
      "x_mm": 0.0,
      "y_mm": -99.5,
      "segment": 20,
      "multiplier": 3,
      "zone": "triple",
      "score": 60,
      "confidence": 0.94,
      "cameras_seen": ["cam1", "cam2"]
    }
  ]
}
```

### Calibration Management

```http
GET /v1/calibrations              # List your calibrations
DELETE /v1/calibrations/{cam_id}  # Delete a calibration
```

## Quick Start

### Self-Hosted

```bash
# Clone
git clone https://github.com/Jricklefs/DartDetectAPI.git
cd DartDetectAPI

# Setup
python -m venv .venv
.venv/Scripts/activate  # Windows
pip install -r requirements.txt

# Run
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker run -p 8000:8000 -e API_KEYS=your_key_here dartdetect/api
```

## Models

Place YOLO models in the `models/` directory:
- `dartboard_calibration/` — Detects calibration points (wire intersections)
- `dart_tips/` — Detects dart tip positions

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEYS` | (none) | Comma-separated valid API keys |
| `REQUIRE_AUTH` | `true` | Set to `false` for local dev |
| `MAX_IMAGE_SIZE` | `10485760` | Max image size in bytes (10MB) |
| `MODEL_DEVICE` | `auto` | `cpu`, `cuda`, or `auto` |

## License

Commercial license required for production use. Contact for pricing.
