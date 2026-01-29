"""
Pydantic schemas for API request/response models
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


# === Calibration ===

class CameraImage(BaseModel):
    """Single camera image for calibration"""
    camera_id: str = Field(..., description="Unique identifier for this camera")
    image: str = Field(..., description="Base64 encoded image data")


class CalibrateRequest(BaseModel):
    """Request to calibrate one or more cameras"""
    cameras: List[CameraImage] = Field(..., description="List of camera images to calibrate")


class CameraCalibrationResult(BaseModel):
    """Result of calibrating a single camera"""
    camera_id: str
    success: bool
    quality: Optional[float] = Field(None, description="Calibration quality 0-1")
    overlay_image: Optional[str] = Field(None, description="Base64 image with scoring zones overlay")
    segment_at_top: Optional[int] = Field(None, description="Segment number at 12 o'clock position")
    error: Optional[str] = None
    calibration_data: Optional[Dict[str, Any]] = Field(None, exclude=True)  # Internal, not returned to client


class CalibrateResponse(BaseModel):
    """Response from calibration endpoint"""
    results: List[CameraCalibrationResult]


# === Detection ===

class DetectRequest(BaseModel):
    """Request to detect dart in an image"""
    camera_id: str = Field(..., description="Camera ID (must be calibrated)")
    image: str = Field(..., description="Base64 encoded image data")


class DartPosition(BaseModel):
    """Detected dart position"""
    x: float
    y: float
    confidence: Optional[float] = None


class DartScore(BaseModel):
    """Calculated dart score"""
    score: int = Field(..., description="Total points (e.g., 60 for T20)")
    multiplier: int = Field(..., description="1=single, 2=double, 3=triple")
    segment: int = Field(..., description="Segment number (1-20, 0 for bull)")
    zone: str = Field(..., description="Zone name: inner_bull, outer_bull, triple, double, single_inner, single_outer, miss")
    is_bullseye: Optional[bool] = False
    is_outer_bull: Optional[bool] = False


class DetectedDart(BaseModel):
    """A single detected dart with position and score"""
    position: Dict[str, Any] = Field(..., description="Dart tip position (x, y, confidence)")
    score: Dict[str, Any] = Field(..., description="Score info (score, multiplier, segment, zone)")


class DetectResponse(BaseModel):
    """Response from dart detection"""
    success: bool
    darts: Optional[List[DetectedDart]] = Field(None, description="List of detected darts")
    overlay_image: Optional[str] = Field(None, description="Base64 image with detection overlay")
    message: Optional[str] = None
    error: Optional[str] = None
    
    # Legacy single-dart fields for backwards compatibility
    dart: Optional[DartScore] = None
    position: Optional[DartPosition] = None
    confidence: Optional[float] = None


# === Calibration Storage ===

class CalibrationInfo(BaseModel):
    """Info about a stored calibration"""
    camera_id: str
    created_at: datetime
    quality: float
    segment_at_top: Optional[int] = None
