"""
DartDetectionAI - FastAPI Application Entry Point
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from app.api.routes import router

app = FastAPI(
    title="DartDetectionAI",
    description="Dartboard detection and scoring API",
    version="0.1.0"
)

# Include API routes
app.include_router(router, prefix="/api")

# Serve test UI (only for development/testing)
ui_path = Path(__file__).parent.parent / "ui"
if ui_path.exists():
    app.mount("/ui", StaticFiles(directory=str(ui_path), html=True), name="ui")

@app.get("/")
async def root():
    """Redirect to API docs or UI"""
    return {
        "message": "DartDetectionAI API",
        "docs": "/docs",
        "ui": "/ui/index.html"
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}
