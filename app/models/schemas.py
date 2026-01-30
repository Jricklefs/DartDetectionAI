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


# === Detection (Multi-camera, stateful, board-scoped) ===

class CameraDetection(BaseModel):
    """Detection from a single camera"""
    camera_id: str
    image: str = Field(..., description="Base64 encoded image")


class MultiDetectRequest(BaseModel):
    """Request to detect darts from multiple cameras"""
    board_id: str = Field(..., description="Unique identifier for the physical dartboard")
    cameras: List[CameraDetection] = Field(..., description="Images from each camera")


class DartInfo(BaseModel):
    """Information about a detected dart"""
    dart_id: str = Field(..., description="Unique ID for this dart")
    dart_index: int = Field(..., description="0-based index (0, 1, 2)")
    segment: int = Field(..., description="Segment number 1-20, or 0 for bull")
    multiplier: int = Field(..., description="1=single, 2=double, 3=triple")
    score: int = Field(..., description="Points (segment * multiplier, or 25/50)")
    zone: str = Field(..., description="Zone name")
    confidence: float = Field(..., description="Detection confidence 0-1")
    x_mm: float = Field(..., description="X position in mm from center")
    y_mm: float = Field(..., description="Y position in mm from center")
    is_new: bool = Field(default=False, description="True if this dart was just detected")


class CameraResult(BaseModel):
    """Per-camera detection result"""
    camera_id: str
    tips_detected: int
    tips: List[Dict[str, Any]] = Field(default_factory=list)


class ConsensusResult(BaseModel):
    """Consensus score from multiple cameras"""
    segment: int
    multiplier: int
    score: int
    zone: str
    confidence: float


class MultiDetectResponse(BaseModel):
    """Response from multi-camera detection"""
    detection_id: str
    board_id: str
    timestamp: float
    
    # The new dart if one was detected
    new_dart: Optional[DartInfo] = None
    
    # All darts currently on board
    all_darts: List[DartInfo] = Field(default_factory=list)
    
    # Consensus for the new dart
    consensus: Optional[ConsensusResult] = None
    
    # Per-camera results
    camera_results: List[CameraResult] = Field(default_factory=list)
    
    # State
    dart_count: int
    board_cleared: bool = False


class TrackerState(BaseModel):
    """Current state of the dart tracker for a board"""
    board_id: str
    dart_count: int
    darts: List[DartInfo]
    last_detection_id: Optional[str] = None


class BoardInfo(BaseModel):
    """Info about an active board"""
    board_id: str
    dart_count: int
    created_at: float
    last_activity: float


class BoardListResponse(BaseModel):
    """Response listing all active boards"""
    boards: List[BoardInfo]


# === Legacy single-camera detection ===

class DetectRequest(BaseModel):
    """Request to detect dart in an image (single camera)"""
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
    zone: str = Field(..., description="Zone name")
    is_bullseye: Optional[bool] = False
    is_outer_bull: Optional[bool] = False


class DetectedDart(BaseModel):
    """A single detected dart with position and score"""
    position: Dict[str, Any]
    score: Dict[str, Any]


class DetectResponse(BaseModel):
    """Response from dart detection (legacy single-camera)"""
    success: bool
    darts: Optional[List[DetectedDart]] = None
    overlay_image: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None
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
