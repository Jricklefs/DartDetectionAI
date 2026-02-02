"""
DartDetect API - Stateless Dart Detection Service

Send dartboard images → Get dart tip positions and scores.
"""
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path

from app.api.routes import router

# Configuration
API_TITLE = "DartDetect API"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
Stateless dart detection as a service.

## Features
- Multi-camera support for better accuracy
- Automatic dartboard calibration
- Fast GPU-accelerated inference
- API key authentication

## Quick Start
1. Calibrate your cameras: `POST /v1/calibrate`
2. Detect darts: `POST /v1/detect`
"""

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS - allow all for now (configure in production)
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
        "docs": "/docs",
        "health": "/health",
        "focus_helper": "/focus"
    }
