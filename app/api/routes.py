"""
DartDetect API Routes - Mask-Based Differential Detection

Like Machine Darts: maintains a 3-value mask (0=bg, 76=new, 152=old) per camera.
New dart pixels are marked 76, confirmed darts become 152.
YOLO tips are filtered to only those in "new" (76) regions.
"""
import os
import time
import uuid
import math
import json
import logging
import threading
from fastapi import APIRouter, HTTPException, Depends, Header, Request
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel
import numpy as np
import cv2
import base64

from app.core.calibration import DartboardCalibrator
from app.core.scoring import scoring_system
from app.core.geometry import (
    DARTBOARD_SEGMENTS,
    BULL_RADIUS_MM,
    OUTER_BULL_RADIUS_MM,
    TRIPLE_INNER_RADIUS_MM,
    TRIPLE_OUTER_RADIUS_MM,
    DOUBLE_INNER_RADIUS_MM,
    DOUBLE_OUTER_RADIUS_MM
)

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("dartdetect.routes")

router = APIRouter()

# === Mask Cache for Differential Detection ===
# Like Machine Darts: 0=background, 76=new dart, 152=old dart
MASK_NEW = 76
MASK_OLD = 152
MASK_BG = 0

# Structure: { board_id: { "baseline": {cam_id: np.ndarray}, "masks": {cam_id: np.ndarray}, "timestamp": float, "dart_count": int } }
_mask_cache: Dict[str, Dict[str, Any]] = {}
_cache_lock = threading.Lock()
CACHE_TTL_SECONDS = 300  # 5 minutes

def _cleanup_old_cache():
    """Remove cache entries older than TTL."""
    now = time.time()
    with _cache_lock:
        expired = [bid for bid, data in _mask_cache.items() 
                   if now - data.get("timestamp", 0) > CACHE_TTL_SECONDS]
        for bid in expired:
            del _mask_cache[bid]
            logger.info(f"Cache expired for board {bid}")

def init_board_cache(board_id: str, baseline_images: Dict[str, np.ndarray]):
    """Initialize cache with baseline (empty board) images."""
    with _cache_lock:
        masks = {}
        for cam_id, img in baseline_images.items():
            h, w = img.shape[:2]
            masks[cam_id] = np.zeros((h, w), dtype=np.uint8)
        
        _mask_cache[board_id] = {
            "baseline": {k: v.copy() for k, v in baseline_images.items()},
            "masks": masks,
            "timestamp": time.time(),
            "dart_count": 0
        }
    logger.info(f"Initialized cache for board {board_id} with {len(baseline_images)} cameras")

def update_masks_with_diff(board_id: str, current_images: Dict[str, np.ndarray], threshold: int = 40) -> Dict[str, np.ndarray]:
    """
    Compute diff vs baseline, mark NEW pixels as 76 in mask.
    Returns the updated masks.
    """
    with _cache_lock:
        cache = _mask_cache.get(board_id)
        if not cache:
            return {}
        
        baseline = cache.get("baseline", {})
        masks = cache.get("masks", {})
        
        for cam_id, current in current_images.items():
            if cam_id not in baseline:
                continue
            
            base_img = baseline[cam_id]
            mask = masks.get(cam_id)
            if mask is None:
                h, w = current.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
            
            # Compute diff
            if len(current.shape) == 3:
                current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
            else:
                current_gray = current
            if len(base_img.shape) == 3:
                base_gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
            else:
                base_gray = base_img
            
            diff = cv2.absdiff(current_gray, base_gray)
            _, diff_thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
            
            # Dilate to fill gaps
            kernel = np.ones((5, 5), np.uint8)
            diff_thresh = cv2.dilate(diff_thresh, kernel, iterations=2)
            
            # Mark NEW pixels (not already old) as 76
            new_pixels = (diff_thresh > 0) & (mask != MASK_OLD)
            mask[new_pixels] = MASK_NEW
            
            masks[cam_id] = mask
            
            new_count = np.count_nonzero(mask == MASK_NEW)
            old_count = np.count_nonzero(mask == MASK_OLD)
            logger.info(f"[MASK] {cam_id}: {new_count} new pixels, {old_count} old pixels")
        
        cache["masks"] = masks
        cache["timestamp"] = time.time()
        return {k: v.copy() for k, v in masks.items()}

def promote_new_to_old(board_id: str):
    """After dart confirmed, promote all NEW (76) pixels to OLD (152)."""
    with _cache_lock:
        cache = _mask_cache.get(board_id)
        if not cache:
            return
        
        masks = cache.get("masks", {})
        for cam_id, mask in masks.items():
            mask[mask == MASK_NEW] = MASK_OLD
            masks[cam_id] = mask
        
        cache["masks"] = masks
        cache["dart_count"] = cache.get("dart_count", 0) + 1
        logger.info(f"[MASK] Promoted new→old for board {board_id}, dart_count={cache['dart_count']}")

def get_new_region_bbox(mask: np.ndarray, margin: int = 20) -> Optional[tuple]:
    """Get bounding box of NEW (76) region with margin."""
    new_pixels = (mask == MASK_NEW)
    if not np.any(new_pixels):
        return None
    
    coords = np.argwhere(new_pixels)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    h, w = mask.shape
    x1 = max(0, x_min - margin)
    y1 = max(0, y_min - margin)
    x2 = min(w, x_max + margin)
    y2 = min(h, y_max + margin)
    
    return (x1, y1, x2, y2)

def get_new_region_centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    """Get centroid (center of mass) of NEW (76) region."""
    new_pixels = (mask == MASK_NEW)
    if not np.any(new_pixels):
        return None
    
    coords = np.argwhere(new_pixels)  # Returns (y, x) pairs
    centroid_y = coords[:, 0].mean()
    centroid_x = coords[:, 1].mean()
    return (centroid_x, centroid_y)

def point_in_new_region(x: float, y: float, mask: np.ndarray, margin: int = 10) -> bool:
    """Check if point is within or near NEW (76) region."""
    h, w = mask.shape
    
    # Check a small area around the point
    x_min = max(0, int(x) - margin)
    x_max = min(w, int(x) + margin)
    y_min = max(0, int(y) - margin)
    y_max = min(h, int(y) + margin)
    
    region = mask[y_min:y_max, x_min:x_max]
    return np.any(region == MASK_NEW)

def clear_cache(board_id: str):
    """Clear cache for a board (called on board clear)."""
    with _cache_lock:
        if board_id in _mask_cache:
            del _mask_cache[board_id]
            logger.info(f"Cache cleared for board {board_id}")

def get_cached_dart_count(board_id: str) -> int:
    """Get the dart count for cached board."""
    with _cache_lock:
        cache = _mask_cache.get(board_id)
        if cache:
            return cache.get("dart_count", 0)
    return 0

def has_cache(board_id: str) -> bool:
    """Check if board has cache initialized."""
    with _cache_lock:
        return board_id in _mask_cache

def decode_image(base64_str: str) -> np.ndarray:
    """Decode base64 image to numpy array."""
    img_bytes = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def compute_diff_mask(current: np.ndarray, previous: np.ndarray, threshold: int = 50) -> np.ndarray:
    """
    Compute difference mask between current and previous frame.
    Returns binary mask where the NEW dart appears.
    """
    # Convert to grayscale
    curr_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    
    # Compute absolute difference
    diff = cv2.absdiff(curr_gray, prev_gray)
    
    # Threshold to get binary mask
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Dilate to fill gaps and expand region around new dart
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    
    return mask

def get_diff_bounding_box(mask: np.ndarray, margin: int = 50):
    """
    Get bounding box of the diff region (where new dart is).
    Returns (x1, y1, x2, y2) or None if no significant diff.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # Get bounding box of all contours combined
    all_points = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(all_points)
    
    # Add margin
    h_img, w_img = mask.shape[:2]
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(w_img, x + w + margin)
    y2 = min(h_img, y + h + margin)
    
    return (x1, y1, x2, y2)

def point_in_bbox(x: float, y: float, bbox: tuple) -> bool:
    """Check if point is inside bounding box."""
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2
    return mask

def apply_mask_to_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply mask to image - keep only the new dart region."""
    # Create 3-channel mask
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # Apply mask
    return cv2.bitwise_and(image, mask_3ch)


# Standard segment angle offset: 20 is at top (-90 degrees = -π/2 radians)
SEGMENT_ANGLE_OFFSET = -math.pi / 2


def point_in_ellipse(point, ellipse):
    """
    Check if a point is inside an ellipse.
    
    ellipse format: ((cx, cy), (width, height), angle_degrees)
    - or stored as: [[cx, cy], [width, height], angle_degrees]
    point format: (x, y)
    
    Returns True if point is inside the ellipse.
    """
    if ellipse is None:
        return False
    
    # Handle both tuple and list formats
    try:
        cx, cy = ellipse[0][0], ellipse[0][1]
        w, h = ellipse[1][0], ellipse[1][1]
        angle_deg = ellipse[2]
    except (IndexError, TypeError):
        return False
    
    # Semi-axes
    a = w / 2.0
    b = h / 2.0
    
    if a <= 0 or b <= 0:
        return False
    
    # Translate point to ellipse center
    px = point[0] - cx
    py = point[1] - cy
    
    # Rotate point by negative angle (to align with ellipse axes)
    angle_rad = math.radians(-angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    px_rot = px * cos_a - py * sin_a
    py_rot = px * sin_a + py * cos_a
    
    # Check ellipse equation: (x/a)^2 + (y/b)^2 <= 1
    return (px_rot / a) ** 2 + (py_rot / b) ** 2 <= 1.0


def get_segment_from_boundaries(point_angle_deg: float, segment_angles: List[float], segment_20_index: int) -> int:
    """
    Determine which segment a point is in using actual wire boundary angles.
    
    This properly accounts for perspective distortion where segments appear
    different sizes (ranging from ~10° to ~30° instead of uniform 18°).
    
    Args:
        point_angle_deg: The angle from center to the dart tip (0-360 degrees)
        segment_angles: List of wire boundary angles in radians (20 values)
        segment_20_index: Which boundary index starts segment 20
    
    Returns:
        The segment number (1-20)
    """
    # Convert point angle to radians for comparison
    point_rad = math.radians(point_angle_deg)
    if point_rad > math.pi:
        point_rad -= 2 * math.pi  # Normalize to -pi to pi
    
    num_boundaries = len(segment_angles)
    
    # Debug: show segment_20_index and first few boundaries
    logger.info(f"[SCORE] Boundaries: segment_20_index={segment_20_index}, point_angle={point_angle_deg:.1f}°, point_rad={point_rad:.3f}")
    logger.info(f"[SCORE] First 5 boundaries (rad): {[f'{b:.3f}' for b in segment_angles[:5]]}")
    
    # Find which boundary pair contains the point
    for i in range(num_boundaries):
        start = segment_angles[i]
        end = segment_angles[(i + 1) % num_boundaries]
        
        # Check if point is in this segment (handle wrap-around)
        if end < start:
            # Segment wraps around from +pi to -pi
            if point_rad >= start or point_rad < end:
                # Found it! Now map boundary index to segment number
                # boundary_index i corresponds to segment at position (i - segment_20_index) % 20
                segment_pos = (i - segment_20_index) % 20
                logger.info(f"[SCORE] Found in boundary {i} (wrap), segment_pos={segment_pos}, segment={DARTBOARD_SEGMENTS[segment_pos]}")
                return DARTBOARD_SEGMENTS[segment_pos]
        else:
            if start <= point_rad < end:
                segment_pos = (i - segment_20_index) % 20
                logger.info(f"[SCORE] Found in boundary {i}, segment_pos={segment_pos}, segment={DARTBOARD_SEGMENTS[segment_pos]}")
                return DARTBOARD_SEGMENTS[segment_pos]
    
    # Fallback: shouldn't reach here, but use segment 20 as default
    logger.warning(f"[SCORE] Could not find segment for angle {point_angle_deg}°, defaulting to 20")
    return 20


def score_with_calibration(tip_data: Dict[str, Any], calibration_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate score using ellipse-based zone detection (like original Machine Darts code).
    
    Segment: Simple 18° division with rotation_offset_deg (proven approach from Machine Darts)
    Zone: Ellipse containment checks for triple/double/single
    """
    x_px = tip_data.get('x_px', 0)
    y_px = tip_data.get('y_px', 0)
    point = (x_px, y_px)
    
    center = calibration_data.get('center', [0, 0])
    
    # Get calibrated ellipses
    bullseye_ellipse = calibration_data.get('bullseye_ellipse')
    bull_ellipse = calibration_data.get('bull_ellipse')
    board_edge = calibration_data.get('outer_double_ellipse')      # Board outer edge
    double_inner = calibration_data.get('inner_triple_ellipse')    # Inner edge of double ring
    triple_outer = calibration_data.get('outer_triple_ellipse')    # Outer edge of triple ring  
    triple_inner = calibration_data.get('inner_double_ellipse')    # Inner edge of triple ring
    
    # Calculate angle from center in PIXEL space (in degrees, 0-360)
    dx = x_px - center[0]
    dy = y_px - center[1]
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360
    
    # DEBUG: Print to stdout
    print(f"[SCORE] x_px={x_px:.1f}, y_px={y_px:.1f}, center={center}, pixel_angle={angle_deg:.1f}")
    
    # Check bullseye first
    if point_in_ellipse(point, bullseye_ellipse):
        print(f"[SCORE] Zone: inner_bull (bullseye)")
        return {"score": 50, "multiplier": 1, "segment": 0, "zone": "inner_bull"}
    
    # Check outer bull
    if point_in_ellipse(point, bull_ellipse):
        print(f"[SCORE] Zone: outer_bull")
        return {"score": 25, "multiplier": 1, "segment": 0, "zone": "outer_bull"}
    
    # === SEGMENT DETECTION: Use wire boundaries from calibration ===
    # This uses the SAME boundaries that the overlay shows, ensuring consistency
    segment_angles = calibration_data.get("segment_angles", [])
    segment_20_index = calibration_data.get("segment_20_index", 0)
    
    if segment_angles and len(segment_angles) >= 20:
        # Use the actual wire boundaries - same as overlay
        segment = get_segment_from_boundaries(angle_deg, segment_angles, segment_20_index)
        print(f"[SCORE] Using wire boundaries: segment={segment}")
    else:
        # Fallback to simple 18° division if no boundaries available
        rotation_offset_deg = calibration_data.get("rotation_offset_deg", 0)
        adjusted_angle = (rotation_offset_deg - angle_deg) % 360
        segment_angle = (adjusted_angle + 9) % 360
        segment_index = int(segment_angle / 18) % 20
        segment = DARTBOARD_SEGMENTS[segment_index]
        print(f"[SCORE] Fallback 18° division: rotation_offset={rotation_offset_deg:.1f}, segment={segment}")
    
    # Check if outside the board
    in_board = point_in_ellipse(point, board_edge)
    if not in_board:
        print(f"[SCORE] Zone: miss (outside board)")
        return {"score": 0, "multiplier": 0, "segment": segment, "zone": "miss"}
    
    # Check zones using ellipse containment (from outside in)
    in_double_inner = point_in_ellipse(point, double_inner)
    in_triple_outer = point_in_ellipse(point, triple_outer)
    in_triple_inner = point_in_ellipse(point, triple_inner)
    
    print(f"[SCORE] Ellipse checks: board={in_board}, double_inner={in_double_inner}, triple_outer={in_triple_outer}, triple_inner={in_triple_inner}")
    
    # Determine zone (from outside in)
    if not in_double_inner:
        # Between board edge and double_inner = DOUBLE ring
        zone = "double"
        multiplier = 2
    elif in_triple_outer and not in_triple_inner:
        # Between triple_outer and triple_inner = TRIPLE ring
        zone = "triple"
        multiplier = 3
    elif not in_triple_outer:
        # Between double_inner and triple_outer = single outer
        zone = "single_outer"
        multiplier = 1
    else:
        # Inside triple_inner = single inner
        zone = "single_inner"
        multiplier = 1
    
    print(f"[SCORE] Zone: {zone}, Segment: {segment}, Multiplier: {multiplier}")
    
    score = segment * multiplier
    return {"score": score, "multiplier": multiplier, "segment": segment, "zone": zone}


# Configuration
REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "false").lower() == "true"
API_KEYS = set(os.getenv("API_KEYS", "").split(",")) if os.getenv("API_KEYS") else set()

# Shared calibrator instance
calibrator = DartboardCalibrator()


# === Request/Response Models ===

class CameraInput(BaseModel):
    """Camera image with inline calibration data."""
    camera_id: str
    image: str  # base64 encoded
    calibration: Optional[Dict[str, Any]] = None  # Made optional for backwards compat


class DetectRequest(BaseModel):
    """Detection request with calibrations inline."""
    cameras: List[CameraInput]
    rotation_offset_degrees: Optional[float] = 0.0
    board_id: Optional[str] = "default"  # For caching/differential detection
    dart_number: Optional[int] = 1  # 1, 2, or 3 - which dart we're detecting


class DetectedTip(BaseModel):
    """A detected dart tip with consensus scoring."""
    x_mm: float
    y_mm: float
    segment: int  # 1-20 (0 for bull)
    multiplier: int  # 1=single, 2=double, 3=triple
    zone: str  # inner_bull, outer_bull, triple, double, single_inner, single_outer, miss
    score: int  # segment * multiplier (or 25/50 for bulls)
    confidence: float  # Consensus confidence after voting
    cameras_seen: List[str]  # Which cameras detected this dart


class CameraResult(BaseModel):
    """Per-camera detection result."""
    camera_id: str
    tips_detected: int
    error: Optional[str] = None


class DetectResponse(BaseModel):
    """Detection response."""
    request_id: str
    processing_ms: int
    tips: List[DetectedTip]
    camera_results: List[CameraResult]


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: bool


# === Authentication ===

async def verify_api_key(authorization: Optional[str] = Header(None)) -> str:
    """Verify API key from Authorization header."""
    if not REQUIRE_AUTH:
        return "local"
    
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid Authorization format")
    
    if parts[1] not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return parts[1]


# === Health ===

@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="3.1.0",
        models_loaded=calibrator.detector.is_initialized if calibrator.detector else False
    )


# === Raw Request Logger for debugging ===

@router.post("/v1/detect-debug")
async def detect_debug(request: Request):
    """Debug endpoint - logs raw request body."""
    body = await request.body()
    body_str = body.decode('utf-8')[:2000]  # Truncate for logging
    logger.info(f"[DEBUG] Raw request body: {body_str}")
    
    try:
        data = json.loads(body)
        logger.info(f"[DEBUG] Parsed JSON keys: {list(data.keys())}")
        if 'cameras' in data:
            for i, cam in enumerate(data['cameras']):
                cam_keys = list(cam.keys()) if isinstance(cam, dict) else "NOT A DICT"
                logger.info(f"[DEBUG] Camera {i} keys: {cam_keys}")
                if isinstance(cam, dict):
                    has_calibration = 'calibration' in cam and cam['calibration'] is not None
                    image_len = len(cam.get('image', ''))
                    logger.info(f"[DEBUG] Camera {i}: has_calibration={has_calibration}, image_len={image_len}")
    except Exception as e:
        logger.error(f"[DEBUG] Failed to parse JSON: {e}")
    
    return {"received": True, "body_preview": body_str[:500]}


# === Detection Endpoint (Fully Stateless) ===

@router.post("/v1/detect", response_model=DetectResponse)
async def detect_tips(
    request: DetectRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Detect dart tips in images with mask-based differential detection.
    
    Like Machine Darts: maintains 3-value mask (0=bg, 76=new, 152=old).
    Dart 1: Init baseline, detect tips, mark as NEW, then promote to OLD.
    Dart 2+: Update mask with diff, filter tips to NEW regions only.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())[:12]
    board_id = request.board_id or "default"
    dart_number = request.dart_number or 1
    
    logger.info(f"")
    logger.info(f"{'='*60}")
    logger.info(f"[DETECT] DART {dart_number} - Board: {board_id} - Request: {request_id}")
    logger.info(f"{'='*60}")
    
    # Cleanup old cache entries periodically
    _cleanup_old_cache()
    
    # Decode all images first
    current_images: Dict[str, np.ndarray] = {}
    for cam in request.cameras:
        try:
            current_images[cam.camera_id] = decode_image(cam.image)
        except Exception as e:
            logger.error(f"[DETECT] Failed to decode image for {cam.camera_id}: {e}")
    
    # For dart 1: initialize baseline (these images become the reference)
    # For dart 2+: update masks with diff from baseline
    masks: Dict[str, np.ndarray] = {}
    
    if dart_number == 1:
        # First dart - initialize cache with current images as baseline
        init_board_cache(board_id, current_images)
        logger.info(f"[DETECT] Initialized baseline for board {board_id}")
    else:
        # Subsequent darts - compute diff and update masks
        if has_cache(board_id):
            masks = update_masks_with_diff(board_id, current_images, threshold=40)
            logger.info(f"[DETECT] Updated masks for {len(masks)} cameras")
        else:
            # No baseline - treat as dart 1
            logger.warning(f"[DETECT] No baseline for board {board_id}, treating as dart 1")
            init_board_cache(board_id, current_images)
            dart_number = 1
    
    all_tips = []
    camera_results = []
    
    for cam in request.cameras:
        logger.info(f"[DETECT] Camera {cam.camera_id}: image_len={len(cam.image)}, has_calibration={cam.calibration is not None}")
        
        if not cam.calibration:
            logger.warning(f"[DETECT] Camera {cam.camera_id}: No calibration data provided!")
            camera_results.append(CameraResult(
                camera_id=cam.camera_id,
                tips_detected=0,
                error="No calibration data provided. Include 'calibration' object with each camera."
            ))
            continue
        
        try:
            current_img = current_images.get(cam.camera_id)
            if current_img is None:
                continue
            
            # Unwrap calibration data - handle multiple formats
            calibration_data = cam.calibration
            logger.debug(f"[DETECT] Camera {cam.camera_id}: Raw calibration keys={list(calibration_data.keys()) if calibration_data else 'None'}")
            
            # Check if this is already the real calibration data (has 'center' key)
            if calibration_data and 'center' in calibration_data:
                pass
            elif calibration_data and 'calibration_data' in calibration_data:
                inner = calibration_data['calibration_data']
                if isinstance(inner, str) and inner:
                    calibration_data = json.loads(inner)
                elif isinstance(inner, dict):
                    calibration_data = inner
                else:
                    camera_results.append(CameraResult(
                        camera_id=cam.camera_id,
                        tips_detected=0,
                        error="Calibration data missing. Re-run camera calibration."
                    ))
                    continue
            else:
                camera_results.append(CameraResult(
                    camera_id=cam.camera_id,
                    tips_detected=0,
                    error="Invalid calibration format"
                ))
                continue
            
            if not calibration_data or 'center' not in calibration_data:
                camera_results.append(CameraResult(
                    camera_id=cam.camera_id,
                    tips_detected=0,
                    error="Invalid calibration - missing center. Re-run calibration."
                ))
                continue
            
            # Detect tips using YOLO - always on full image
            tips = calibrator.detect_tips(
                camera_id=cam.camera_id,
                image_base64=cam.image,
                calibration_data=calibration_data
            )
            
            logger.info(f"[DETECT] Camera {cam.camera_id}: YOLO found {len(tips)} tips")
            
            # For dart 2+, filter tips to only those in NEW (76) regions of the mask
            mask = masks.get(cam.camera_id)
            if dart_number > 1 and tips and mask is not None:
                original_count = len(tips)
                filtered_tips = []
                for t in tips:
                    tx, ty = t.get('x_px', 0), t.get('y_px', 0)
                    in_new = point_in_new_region(tx, ty, mask, margin=15)
                    logger.info(f"[DETECT] Camera {cam.camera_id}: Tip ({tx:.1f}, {ty:.1f}) in NEW region? {in_new}")
                    if in_new:
                        filtered_tips.append(t)
                tips = filtered_tips
                logger.info(f"[DETECT] Camera {cam.camera_id}: Filtered {original_count} → {len(tips)} tips (NEW region only)")
            
            # Only keep ONE tip per camera - the one closest to NEW region centroid
            # This ensures we pick the actual new dart, not an old dart with high confidence
            if len(tips) > 1:
                centroid = get_new_region_centroid(mask) if mask is not None else None
                if centroid:
                    cx, cy = centroid
                    # Sort by distance to centroid (ascending)
                    tips = sorted(tips, key=lambda t: math.sqrt((t.get('x_px', 0) - cx)**2 + (t.get('y_px', 0) - cy)**2))
                    best_tip = tips[0]
                    dist = math.sqrt((best_tip.get('x_px', 0) - cx)**2 + (best_tip.get('y_px', 0) - cy)**2)
                    logger.info(f"[DETECT] Camera {cam.camera_id}: Picked tip closest to NEW centroid ({cx:.1f}, {cy:.1f}), dist={dist:.1f}")
                    tips = [best_tip]
                else:
                    # Fallback: highest confidence
                    tips = sorted(tips, key=lambda t: t.get('confidence', 0), reverse=True)[:1]
                    logger.info(f"[DETECT] Camera {cam.camera_id}: Fallback - taking best conf tip (conf={tips[0].get('confidence', 0):.3f})")
            
            # Calculate score for each tip
            for tip in tips:
                score_info = score_with_calibration(tip, calibration_data)
                tip['camera_id'] = cam.camera_id
                tip['segment'] = score_info.get('segment', 0)
                tip['multiplier'] = score_info.get('multiplier', 1)
                tip['zone'] = score_info.get('zone', 'miss')
                tip['score'] = score_info.get('score', 0)
                all_tips.append(tip)
                
                logger.info(f"[DETECT] Tip: segment={tip['segment']}, multiplier={tip['multiplier']}, score={tip['score']}, conf={tip.get('confidence', 0):.3f}")
            
            camera_results.append(CameraResult(
                camera_id=cam.camera_id,
                tips_detected=len(tips)
            ))
            
        except Exception as e:
            logger.error(f"[DETECT] Camera {cam.camera_id}: Error - {e}", exc_info=True)
            camera_results.append(CameraResult(
                camera_id=cam.camera_id,
                tips_detected=0,
                error=str(e)
            ))
    
    # Cluster tips from multiple cameras and vote on score
    clustered_tips = cluster_tips_by_segment(all_tips)
    logger.info(f"[DETECT] Clustered {len(all_tips)} tips into {len(clustered_tips)} clusters")
    
    # If all cameras see 1 dart each but clustering failed (mm coords don't match),
    # do segment-based voting across all tips
    if len(clustered_tips) > 1 and all(len(c) == 1 for c in clustered_tips):
        logger.info(f"[DETECT] All cameras see 1 dart each - combining for segment voting")
        all_single_tips = [c[0] for c in clustered_tips]
        clustered_tips = [all_single_tips]
    
    detected_tips = vote_on_scores(clustered_tips)
    
    # DartSensor triggers once per dart - we should only return 1 tip per request
    # Take the most confident one if multiple were detected
    if len(detected_tips) > 1:
        logger.info(f"[DETECT] Multiple tips after voting ({len(detected_tips)}), taking best one")
        detected_tips = sorted(detected_tips, key=lambda t: t.confidence, reverse=True)[:1]
    
    # After successful detection, promote NEW pixels to OLD in the mask
    if detected_tips:
        promote_new_to_old(board_id)
        logger.info(f"[DETECT] Promoted new→old for board {board_id}")
    
    processing_ms = int((time.time() - start_time) * 1000)
    
    # Clear result summary
    if detected_tips:
        result_summary = ", ".join([f"{t.get('segment', '?')}x{t.get('multiplier', 1)}={t.get('score', 0)}" for t in detected_tips])
        logger.info(f"")
        logger.info(f">>> RESULT: DART {dart_number} = {result_summary} ({processing_ms}ms)")
        logger.info(f"{'='*60}")
    else:
        logger.info(f"")
        logger.info(f">>> RESULT: DART {dart_number} = NO DETECTION ({processing_ms}ms)")
        logger.info(f"{'='*60}")
    
    return DetectResponse(
        request_id=request_id,
        processing_ms=processing_ms,
        tips=detected_tips,
        camera_results=camera_results
    )


def cluster_tips_by_segment(tips: List[dict]) -> List[List[dict]]:
    """
    Cluster tips by segment+multiplier (same dart seen by multiple cameras).
    
    Since mm coordinates are unreliable due to camera angles, we cluster by
    what the dart scored as. Tips with the same (segment, multiplier) from
    different cameras are considered the same dart.
    """
    if not tips:
        return []
    
    # Group by (segment, multiplier)
    clusters_dict = {}
    for tip in tips:
        key = (tip.get('segment', 0), tip.get('multiplier', 1))
        if key not in clusters_dict:
            clusters_dict[key] = []
        clusters_dict[key].append(tip)
    
    return list(clusters_dict.values())


def cluster_tips_by_position(
    tips: List[dict],
    cluster_threshold_mm: float = 20.0
) -> List[List[dict]]:
    """
    Cluster nearby tips (same dart seen by multiple cameras).
    Tips within cluster_threshold_mm are considered the same dart.
    
    NOTE: This is unreliable due to camera angle distortion.
    Use cluster_tips_by_segment() instead for multi-camera setups.
    """
    if not tips:
        return []
    
    clusters = []
    used = set()
    
    for i, tip in enumerate(tips):
        if i in used:
            continue
        
        cluster = [tip]
        used.add(i)
        
        for j, other in enumerate(tips):
            if j in used:
                continue
            
            dist = math.sqrt(
                (tip['x_mm'] - other['x_mm'])**2 +
                (tip['y_mm'] - other['y_mm'])**2
            )
            
            if dist < cluster_threshold_mm:
                cluster.append(other)
                used.add(j)
        
        clusters.append(cluster)
    
    return clusters


def vote_on_scores(clusters: List[List[dict]]) -> List[DetectedTip]:
    """
    For each cluster (same dart seen by multiple cameras), vote on the score.
    
    Uses majority voting:
    - Each camera's detection votes for its calculated segment/multiplier
    - Majority wins (if 2+ cameras agree)
    - If no majority, use highest confidence detection
    """
    detected_tips = []
    
    for cluster in clusters:
        if not cluster:
            continue
        
        cameras_seen = list(set(t['camera_id'] for t in cluster))
        
        # Single camera - use directly
        if len(cluster) == 1:
            tip = cluster[0]
            detected_tips.append(DetectedTip(
                x_mm=round(tip['x_mm'], 2),
                y_mm=round(tip['y_mm'], 2),
                segment=tip['segment'],
                multiplier=tip['multiplier'],
                zone=tip['zone'],
                score=tip['score'],
                confidence=round(tip['confidence'], 3),
                cameras_seen=cameras_seen
            ))
            continue
        
        # Multiple cameras - vote on segment and multiplier
        votes = {}
        total_confidence = 0.0
        
        for tip in cluster:
            key = (tip['segment'], tip['multiplier'])
            weight = tip['confidence']
            votes[key] = votes.get(key, 0.0) + weight
            total_confidence += weight
        
        # Find winning vote
        winning_key = max(votes.keys(), key=lambda k: votes[k])
        winning_segment, winning_multiplier = winning_key
        
        # Consensus confidence
        agreeing_confidence = votes[winning_key]
        consensus_confidence = agreeing_confidence / total_confidence if total_confidence > 0 else 0.0
        
        # Determine zone and score
        if winning_segment == 0:
            # Bull - check inner vs outer
            inner_votes = sum(t['confidence'] for t in cluster if t['zone'] == 'inner_bull')
            outer_votes = sum(t['confidence'] for t in cluster if t['zone'] == 'outer_bull')
            if inner_votes >= outer_votes:
                winning_zone = 'inner_bull'
                winning_score = 50
            else:
                winning_zone = 'outer_bull'
                winning_score = 25
        elif winning_multiplier == 0:
            winning_zone = 'miss'
            winning_score = 0
        elif winning_multiplier == 3:
            winning_zone = 'triple'
            winning_score = winning_segment * 3
        elif winning_multiplier == 2:
            winning_zone = 'double'
            winning_score = winning_segment * 2
        else:
            winning_zone = 'single'
            winning_score = winning_segment
        
        # Average position for logging
        avg_x = sum(t['x_mm'] for t in cluster) / len(cluster)
        avg_y = sum(t['y_mm'] for t in cluster) / len(cluster)
        
        logger.info(f"[VOTE] Cluster: {len(cluster)} cameras, winner={winning_segment}x{winning_multiplier}={winning_score}, conf={consensus_confidence:.3f}")
        
        detected_tips.append(DetectedTip(
            x_mm=round(avg_x, 2),
            y_mm=round(avg_y, 2),
            segment=winning_segment,
            multiplier=winning_multiplier,
            zone=winning_zone,
            score=winning_score,
            confidence=round(consensus_confidence, 3),
            cameras_seen=cameras_seen
        ))
    
    return detected_tips


# === Calibration Endpoint (still useful for generating calibration data) ===

class CameraImage(BaseModel):
    camera_id: str
    image: str


class CalibrateRequest(BaseModel):
    cameras: List[CameraImage]


class CalibrationResult(BaseModel):
    camera_id: str
    success: bool
    quality: Optional[float] = None
    overlay_image: Optional[str] = None
    segment_at_top: Optional[int] = None
    calibration_data: Optional[Dict[str, Any]] = None  # Return full data to store in DB
    error: Optional[str] = None


class CalibrateResponse(BaseModel):
    results: List[CalibrationResult]


@router.post("/v1/calibrate", response_model=CalibrateResponse)
async def calibrate_cameras(
    request: CalibrateRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Calibrate cameras from dartboard images.
    
    Returns calibration data that should be stored in DartGame DB.
    DartDetect does not store this - it's returned to the caller.
    """
    results = []
    
    for camera in request.cameras:
        logger.info(f"[CALIBRATE] Camera {camera.camera_id}: image_len={len(camera.image)}")
        
        try:
            result = calibrator.calibrate(
                camera_id=camera.camera_id,
                image_base64=camera.image
            )
            
            logger.info(f"[CALIBRATE] Camera {camera.camera_id}: success={result.success}, quality={result.quality}")
            
            results.append(CalibrationResult(
                camera_id=camera.camera_id,
                success=result.success,
                quality=result.quality,
                overlay_image=result.overlay_image,
                segment_at_top=result.segment_at_top,
                calibration_data=result.calibration_data if result.success else None,
                error=result.error
            ))
            
        except Exception as e:
            logger.error(f"[CALIBRATE] Camera {camera.camera_id}: Error - {e}", exc_info=True)
            results.append(CalibrationResult(
                camera_id=camera.camera_id,
                success=False,
                error=str(e)
            ))
    
    return CalibrateResponse(results=results)


# === Rebase endpoint (for compatibility) ===

@router.post("/rebase")
async def rebase_noop():
    """Rebase endpoint - clears image cache for differential detection."""
    logger.info("[REBASE] Called - clearing all cached images")
    _cleanup_old_cache()
    return {
        "message": "Rebase acknowledged - cache cleaned",
        "note": "DartDetect image cache cleared for differential detection."
    }

@router.post("/v1/clear")
async def clear_board_cache(request: dict):
    """Clear cached images for a specific board."""
    board_id = request.get("board_id", "default")
    clear_cache(board_id)
    logger.info(f"[CLEAR] Cleared cache for board {board_id}")
    return {"message": f"Cache cleared for board {board_id}"}
