"""
Pydantic schemas for DartDetect API

Stateless API with motion detection service support.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


# === Calibration ===

class CameraImage(BaseModel):
    """Single camera image"""
    camera_id: str = Field(..., description="Unique identifier for this camera")
    image: str = Field(..., description="Base64 encoded image data")


class CalibrateRequest(BaseModel):
    """Request to calibrate cameras"""
    cameras: List[CameraImage] = Field(..., description="Camera images to calibrate")


class CameraCalibrationResult(BaseModel):
    """Result of calibrating a single camera"""
    camera_id: str
    success: bool
    quality: Optional[float] = Field(None, description="Calibration quality 0-1")
    overlay_image: Optional[str] = Field(None, description="Base64 image with overlay")
    segment_at_top: Optional[int] = Field(None, description="Segment at 12 o'clock")
    error: Optional[str] = None
    calibration_data: Optional[dict] = Field(None, description="Calibration data for storage")


class CalibrateResponse(BaseModel):
    """Response from calibration endpoint"""
    results: List[CameraCalibrationResult]


class CalibrationInfo(BaseModel):
    """Info about a stored calibration"""
    camera_id: str
    created_at: datetime
    quality: float
    segment_at_top: Optional[int] = None


# === Detection (Stateless) ===

class DetectRequest(BaseModel):
    """Request to detect dart tips in images"""
    cameras: List[CameraImage] = Field(..., description="Images from calibrated cameras")
    rotation_offset_degrees: Optional[float] = Field(None, description="Board rotation offset in degrees (20 segment angle)")


class DetectedTip(BaseModel):
    """A detected dart tip with position and score"""
    x_mm: float = Field(..., description="X position in mm from center")
    y_mm: float = Field(..., description="Y position in mm from center")
    segment: int = Field(..., description="Segment number 1-20, or 0 for bull")
    multiplier: int = Field(..., description="1=single, 2=double, 3=triple")
    zone: str = Field(..., description="Zone: single, double, triple, inner_bull, outer_bull, miss")
    score: int = Field(..., description="Points (segment * multiplier, or 25/50 for bulls)")
    confidence: float = Field(..., description="Detection confidence 0-1")
    cameras_seen: List[str] = Field(default_factory=list, description="Which cameras detected this tip")


class CameraDetectionResult(BaseModel):
    """Per-camera detection details"""
    camera_id: str
    tips_detected: int
    error: Optional[str] = None
    calibration_data: Optional[dict] = Field(None, description="Calibration data for storage")


class DetectResponse(BaseModel):
    """Response from detection endpoint - all tips currently visible"""
    request_id: str = Field(..., description="Unique request identifier")
    processing_ms: int = Field(..., description="Processing time in milliseconds")
    tips: List[DetectedTip] = Field(default_factory=list, description="All detected dart tips")
    camera_results: List[CameraDetectionResult] = Field(default_factory=list)


# === Health/Status ===

class HealthResponse(BaseModel):
    """API health status"""
    status: str
    version: str
    models_loaded: bool
    calibrations_count: int
    motion_detection_running: bool = Field(False, description="Whether motion detection service is active")


class UsageInfo(BaseModel):
    """API usage statistics for an API key"""
    api_key_prefix: str
    requests_today: int
    requests_total: int
    last_request: Optional[datetime] = None


# === Motion Detection Service ===

class DetectionStartRequest(BaseModel):
    """Request to start motion detection"""
    board_id: str = Field("default", description="Board ID to report detections for")
    game_api_url: Optional[str] = Field(None, description="DartGame API URL")
    camera_indices: Optional[List[int]] = Field(None, description="Specific camera indices to use")


class DetectionConfigRequest(BaseModel):
    """Request to configure detection parameters"""
    motion_threshold: Optional[float] = Field(None, description="Pixel intensity difference threshold")
    min_contour_area: Optional[int] = Field(None, description="Minimum contour area in pixels")
    detection_interval_ms: Optional[int] = Field(None, description="Detection loop interval in ms")


class CameraStatus(BaseModel):
    """Status of a single camera in motion detection"""
    index: int
    has_baseline: bool
    baseline_captured_at: Optional[str] = None
    last_tips_count: int


class DetectionStatus(BaseModel):
    """Full motion detection service status"""
    running: bool
    board_id: str
    game_api_url: str
    cameras: Dict[str, CameraStatus]
    known_darts_count: int
    motion_threshold: float
    detection_interval_ms: int


# === Dart Event (sent to DartGame API) ===

class DartDetectedEvent(BaseModel):
    """Dart detection event sent to DartGame API"""
    camera_id: str = Field(..., alias="cameraId")
    segment: int
    multiplier: int
    score: int
    x_mm: float = Field(..., alias="xMm")
    y_mm: float = Field(..., alias="yMm")
    confidence: float
    zone: str
    image_base64: Optional[str] = Field(None, alias="imageBase64")

    class Config:
        populate_by_name = True


# === Legacy schemas ===
  
class DartPosition(BaseModel):  
    x_mm: float  
    y_mm: float  
  
class DartScore(BaseModel):  
    segment: int  
    multiplier: int  
    zone: str  
    score: int
