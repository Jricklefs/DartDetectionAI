"""
DartDetect API - Stateless Dart Detection Service

Accepts images + calibration data, returns dart positions/scores.
NO camera access, NO motion detection, NO state.

This can run on a powerful GPU box while sensors run on lightweight devices.
"""
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path

from app.api.routes import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
API_TITLE = "DartDetect API"
API_VERSION = "3.0.0"
API_DESCRIPTION = """
Stateless dart detection API - accepts images, returns dart positions.

## Architecture (v3.0 - Separated Services)

This API is **stateless** - it has no camera access, no motion detection.

### Services:
1. **DartDetect API** (this service, port 8000) - GPU-intensive dart detection
2. **DartSensor** (separate, runs locally) - Lightweight camera monitoring
3. **DartGame API** (port 5000) - Game logic

### Flow:
1. DartSensor monitors cameras locally
2. On motion detected, DartSensor sends image to DartDetect `/v1/detect`
3. DartDetect returns dart positions/scores
4. DartSensor forwards result to DartGame API

## Endpoints

### Detection
- `POST /v1/detect` - Detect darts in images (requires calibration data)

### Calibration
- `POST /v1/calibrate` - Calibrate from dartboard images
- `GET /v1/calibrations` - List stored calibrations
- `POST /v1/calibrations/{camera_id}/mark20` - Mark segment 20 position

### Health
- `GET /health` - Service health check
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("DartDetect API starting (stateless mode)")
    yield
    logger.info("DartDetect API shutting down")


app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS - allow all for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)

# Static files directory
static_path = Path(__file__).parent / "static"

# Serve test UI (only for development)
ui_path = Path(__file__).parent.parent / "ui"
if ui_path.exists():
    app.mount("/ui", StaticFiles(directory=str(ui_path), html=True), name="ui")

# Serve static files
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path), html=True), name="static")


@app.get("/focus")
async def focus_helper():
    """Camera focus helper tool."""
    focus_file = static_path / "focus.html"
    if focus_file.exists():
        return FileResponse(focus_file)
    return {"error": "Focus helper not found"}


@app.get("/")
async def root():
    """API info and links."""
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "mode": "stateless",
        "docs": "/docs",
        "health": "/health",
        "focus_helper": "/focus",
        "note": "This is a stateless detection API. Use DartSensor for camera monitoring."
    }
