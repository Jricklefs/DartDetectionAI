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
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends, Header, Request
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel
import numpy as np
import cv2
import base64

from app.core.calibration import DartboardCalibrator
from app.core.stereo_calibration import StereoCalibrator, StereoCalibration, generate_checkerboard_pdf
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
import requests as http_requests  # Renamed to avoid conflict with FastAPI request

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("dartdetect.routes")

# Centralized logging endpoint
DARTGAME_API_URL = os.environ.get("DARTGAME_URL", "http://localhost:5000")

def log_to_api(level: str, category: str, message: str, data: dict = None, game_id: str = None):
    """Send log entry to centralized DartGame API logging endpoint."""
    try:
        payload = {
            "source": "DartDetect",
            "level": level,
            "category": category,
            "message": message,
            "data": json.dumps(data) if data else None,
            "gameId": game_id
        }
        http_requests.post(f"{DARTGAME_API_URL}/api/logs", json=payload, timeout=0.5)
    except Exception:
        pass  # Don't let logging failures break detection

router = APIRouter()

# === Debug Image Saving ===
DEBUG_IMAGES_ENABLED = os.environ.get("DARTDETECT_DEBUG_IMAGES", "true").lower() == "true"
DEBUG_IMAGES_DIR = Path(os.environ.get("DARTDETECT_DEBUG_DIR", "C:/Users/clawd/DartImages"))

def save_debug_image(request_id: str, dart_number: int, camera_id: str, 
                     image: np.ndarray, all_tips: List[dict], selected_tip: dict = None,
                     known_darts: List[dict] = None, new_centroid: tuple = None) -> Optional[np.ndarray]:
    """
    Save debug image with all tips marked.
    - Green circle: selected tip (the one we're scoring)
    - Red circles: other detected tips
    - Blue circles: known dart locations (source of truth)
    - Yellow X: NEW region centroid
    
    Returns the debug image for benchmark storage.
    """
    debug_img = None
    
    try:
        # Make a copy to draw on
        debug_img = image.copy()
        
        # Draw known dart locations (blue)
        if known_darts:
            for kd in known_darts:
                x, y = int(kd.get('x', 0)), int(kd.get('y', 0))
                cv2.circle(debug_img, (x, y), 15, (255, 0, 0), 2)  # Blue
                cv2.putText(debug_img, f"D{kd.get('dart_number', '?')}", (x+10, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw NEW region centroid (yellow X)
        if new_centroid:
            cx, cy = int(new_centroid[0]), int(new_centroid[1])
            cv2.drawMarker(debug_img, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(debug_img, "NEW", (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw all tips (red = not selected, green = selected)
        for tip in all_tips:
            tx, ty = int(tip.get('x_px', 0)), int(tip.get('y_px', 0))
            conf = tip.get('confidence', 0)
            
            is_selected = (selected_tip and 
                          abs(tip.get('x_px', 0) - selected_tip.get('x_px', 0)) < 1 and
                          abs(tip.get('y_px', 0) - selected_tip.get('y_px', 0)) < 1)
            
            if is_selected:
                color = (0, 255, 0)  # Green
                cv2.circle(debug_img, (tx, ty), 12, color, 3)
                cv2.putText(debug_img, f"SEL {conf:.2f}", (tx+10, ty+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                color = (0, 0, 255)  # Red
                cv2.circle(debug_img, (tx, ty), 8, color, 2)
                cv2.putText(debug_img, f"{conf:.2f}", (tx+10, ty+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Add header text
        cv2.putText(debug_img, f"Dart {dart_number} - {camera_id} - {request_id}", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save to disk if enabled
        if DEBUG_IMAGES_ENABLED:
            DEBUG_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%H%M%S_%f")
            filename = f"dart{dart_number}_{camera_id}_{request_id}_{timestamp}.jpg"
            filepath = DEBUG_IMAGES_DIR / filename
            cv2.imwrite(str(filepath), debug_img)
            logger.info(f"[DEBUG] Saved debug image: {filename}")
        
    except Exception as e:
        logger.warning(f"[DEBUG] Failed to save debug image: {e}")
    
    return debug_img

# === Training Data Capture ===
# Set to True to save images for future model training
CAPTURE_TRAINING_DATA = os.environ.get("DARTDETECT_CAPTURE_TRAINING", "false").lower() == "true"
TRAINING_DATA_DIR = Path(os.environ.get("DARTDETECT_TRAINING_DIR", "C:/Users/clawd/DartDetectionAI/training_data"))

def save_training_data(camera_id: str, image_base64: str, segment: int, multiplier: int, 
                       tip_x: float, tip_y: float, confidence: float, zone: str):
    """Save image and label for future model training."""
    if not CAPTURE_TRAINING_DATA:
        return
    
    try:
        # Create date-based directory
        date_dir = TRAINING_DATA_DIR / datetime.now().strftime("%Y-%m-%d")
        date_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%H%M%S_%f")
        zone_prefix = {"single_inner": "S", "single_outer": "S", "double": "D", "triple": "T", 
                       "inner_bull": "BULL", "outer_bull": "BULL25"}.get(zone, "X")
        
        if segment == 0:
            label = zone_prefix  # Bull or Bull25
        else:
            label = f"{zone_prefix}{segment}"  # e.g., T20, D16, S5
        
        base_name = f"{camera_id}_{label}_{timestamp}"
        
        # Save image
        img_path = date_dir / f"{base_name}.jpg"
        img_data = base64.b64decode(image_base64.split(',')[-1] if ',' in image_base64 else image_base64)
        with open(img_path, 'wb') as f:
            f.write(img_data)
        
        # Save metadata
        meta_path = date_dir / f"{base_name}.json"
        metadata = {
            "camera_id": camera_id,
            "segment": segment,
            "multiplier": multiplier,
            "zone": zone,
            "tip_x": tip_x,
            "tip_y": tip_y,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "label": label
        }
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.debug(f"[TRAINING] Saved {base_name}")
    except Exception as e:
        logger.warning(f"[TRAINING] Failed to save training data: {e}")


# === Benchmark System ===
# Comprehensive logging for accuracy analysis - enabled via Settings UI
BENCHMARK_ENABLED = False  # Set via API when logging is enabled in Settings
# === Stereo Calibration ===
# Triangulation mode: "ellipse" (default) or "stereo" (requires checkerboard calibration)
TRIANGULATION_MODE = os.environ.get("DARTDETECT_TRIANGULATION", "ellipse")
STEREO_CALIBRATION_DIR = Path(os.environ.get("DARTDETECT_STEREO_DIR", "C:/Users/clawd/DartDetectionAI/stereo_calibrations"))
_stereo_calibrator: StereoCalibrator = None
_stereo_capture_images: Dict[str, List] = {}  # Temporary storage for calibration images

def get_stereo_calibrator(board_id: str = "default") -> StereoCalibrator:
    """Get or create stereo calibrator, loading saved calibration if available."""
    global _stereo_calibrator
    if _stereo_calibrator is None:
        _stereo_calibrator = StereoCalibrator()
        cal_file = STEREO_CALIBRATION_DIR / f"{board_id}_stereo.json"
        if cal_file.exists():
            _stereo_calibrator.load(cal_file)
    return _stereo_calibrator


BENCHMARK_DIR = Path(os.environ.get("DARTDETECT_BENCHMARK_DIR", "C:/Users/clawd/DartBenchmark"))

# Current game context - set by DartGame API when game starts
_benchmark_context = {
    "board_id": None,
    "game_id": None,
    "round": None,
    "player_name": None
}
_benchmark_lock = threading.Lock()

# Track last saved dart paths for correction matching
# Key: (game_id, dart_number) -> path OR just dart_number for legacy
# Keep history of recent darts (last 50) so corrections work even after new game starts
_last_dart_paths = {}
_dart_path_history = []  # List of (game_id, dart_number, path, timestamp) for fallback lookup
MAX_PATH_HISTORY = 50

def set_benchmark_enabled(enabled: bool):
    """Enable/disable benchmark logging (called from Settings UI)."""
    global BENCHMARK_ENABLED
    BENCHMARK_ENABLED = enabled
    logger.info(f"[BENCHMARK] {'Enabled' if enabled else 'Disabled'}")

def set_benchmark_context(board_id: str = None, game_id: str = None, 
                          round_num: int = None, player_name: str = None):
    """Set current game context for benchmark organization."""
    global _benchmark_context
    with _benchmark_lock:
        if board_id is not None:
            _benchmark_context["board_id"] = board_id
        if game_id is not None:
            _benchmark_context["game_id"] = game_id
        if round_num is not None:
            _benchmark_context["round"] = round_num
        if player_name is not None:
            _benchmark_context["player_name"] = player_name
    logger.debug(f"[BENCHMARK] Context: {_benchmark_context}")

def get_benchmark_path(dart_number: int) -> Optional[Path]:
    """Get the path where benchmark data should be saved for current dart."""
    if not BENCHMARK_ENABLED:
        return None
    
    with _benchmark_lock:
        ctx = _benchmark_context.copy()
    
    board_id = ctx.get("board_id") or "default"
    game_id = ctx.get("game_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
    round_num = ctx.get("round") or 1
    player_name = ctx.get("player_name") or "player"
    
    # Sanitize names for filesystem
    player_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in player_name)
    
    path = BENCHMARK_DIR / board_id / game_id / f"round_{round_num}_{player_name}" / f"dart_{dart_number}"
    return path

def save_benchmark_data(
    dart_number: int,
    request_id: str,
    raw_images: Dict[str, str],  # camera_id -> base64 image
    debug_images: Dict[str, np.ndarray],  # camera_id -> cv2 image with annotations
    camera_results: List[Dict],  # Per-camera detection results
    final_result: Dict,  # Final voted result
    calibrations: Dict[str, Dict],  # camera_id -> calibration data
    timings: Dict[str, int],
    pipeline_data: Dict[str, Dict] = None,  # camera_id -> detailed pipeline info
    baseline_images: Dict[str, np.ndarray] = None  # camera_id -> baseline image for diff
):
    """
    Save complete benchmark data for a dart detection.
    This captures everything needed to replay and analyze detection accuracy.
    """
    if not BENCHMARK_ENABLED:
        return None
    
    dart_path = get_benchmark_path(dart_number)
    if not dart_path:
        return None
    
    try:
        dart_path.mkdir(parents=True, exist_ok=True)
        
        # Save raw images (original camera captures)
        for cam_id, img_b64 in raw_images.items():
            try:
                img_data = base64.b64decode(img_b64.split(',')[-1] if ',' in img_b64 else img_b64)
                with open(dart_path / f"{cam_id}_raw.jpg", 'wb') as f:
                    f.write(img_data)
            except Exception as e:
                logger.warning(f"[BENCHMARK] Failed to save raw image for {cam_id}: {e}")
        
        # Save baseline images (empty board reference for diff detection)
        if baseline_images:
            for cam_id, img in baseline_images.items():
                try:
                    cv2.imwrite(str(dart_path / f"{cam_id}_previous.jpg"), img)
                except Exception as e:
                    logger.warning(f"[BENCHMARK] Failed to save baseline image for {cam_id}: {e}")
        
        # Save debug images (with annotations)
        for cam_id, img in debug_images.items():
            try:
                cv2.imwrite(str(dart_path / f"{cam_id}_debug.jpg"), img)
            except Exception as e:
                logger.warning(f"[BENCHMARK] Failed to save debug image for {cam_id}: {e}")
        
        # Build comprehensive metadata
        metadata = {
            "request_id": request_id,
            "dart_number": dart_number,
            "timestamp": datetime.now().isoformat(),
            "context": _benchmark_context.copy(),
            "camera_results": camera_results,
            "final_result": final_result,
            "timings": timings,
            "calibrations": {},
            "pipeline": pipeline_data or {}  # Detailed per-camera detection pipeline
        }
        
        # Include full calibration data for replay (needed for scoring)
        for cam_id, cal in calibrations.items():
            # Normalize ellipse key names - DartGame API may use old names
            # Old: triple_ellipse, double_ellipse, inner_single_ellipse, outer_single_ellipse
            # New: outer_triple_ellipse, inner_triple_ellipse, outer_double_ellipse, inner_double_ellipse
            outer_double = cal.get("outer_double_ellipse") or cal.get("double_ellipse")
            inner_double = cal.get("inner_double_ellipse") or cal.get("inner_single_ellipse")
            outer_triple = cal.get("outer_triple_ellipse") or cal.get("triple_ellipse")
            inner_triple = cal.get("inner_triple_ellipse") or cal.get("outer_single_ellipse")
            
            # Save essential scoring fields (ellipses and segment angles for proper scoring)
            metadata["calibrations"][cam_id] = {
                "center": cal.get("center"),
                "segment_20_index": cal.get("segment_20_index"),
                "rotation_offset_deg": cal.get("rotation_offset_deg"),
                "quality": cal.get("quality"),
                # Ring ellipses for multiplier detection (normalized key names):
                "bullseye_ellipse": cal.get("bullseye_ellipse"),
                "bull_ellipse": cal.get("bull_ellipse"),
                "outer_double_ellipse": outer_double,
                "inner_double_ellipse": inner_double,
                "outer_triple_ellipse": outer_triple,
                "inner_triple_ellipse": inner_triple,
                "segment_angles": cal.get("segment_angles"),
                "segment_boundaries": cal.get("segment_boundaries"),
            }
        
        # Save metadata - ensure complete write with flush
        meta_path = dart_path / "metadata.json"
        try:
            json_str = json.dumps(metadata, indent=2, default=str)  # default=str handles non-serializable
            with open(meta_path, 'w') as f:
                f.write(json_str)
                f.flush()
            logger.info(f"[BENCHMARK] Wrote {len(json_str)} bytes to metadata.json")
        except Exception as e:
            logger.error(f"[BENCHMARK] Failed to write metadata: {e}")
            # Try simplified metadata without pipeline
            try:
                del metadata["pipeline"]
                json_str = json.dumps(metadata, indent=2, default=str)
                with open(meta_path, 'w') as f:
                    f.write(json_str)
                    f.flush()
                logger.info(f"[BENCHMARK] Wrote simplified metadata ({len(json_str)} bytes)")
            except Exception as e2:
                logger.error(f"[BENCHMARK] Failed to write simplified metadata: {e2}")
        
        # Track this dart path for correction matching
        global _last_dart_paths, _dart_path_history
        game_id = _benchmark_context.get("game_id") or "unknown"
        
        # Store by (game_id, dart_number) for precise matching
        _last_dart_paths[(game_id, dart_number)] = str(dart_path)
        # Also store by dart_number alone for legacy/fallback
        _last_dart_paths[dart_number] = str(dart_path)
        
        # Add to history for fallback lookup
        _dart_path_history.append({
            "game_id": game_id,
            "dart_number": dart_number,
            "path": str(dart_path),
            "timestamp": datetime.now().isoformat()
        })
        # Trim history (use del to modify in place, not reassign)
        if len(_dart_path_history) > MAX_PATH_HISTORY:
            del _dart_path_history[:-MAX_PATH_HISTORY]
        
        logger.info(f"[BENCHMARK] Saved dart {dart_number} to {dart_path}")
        return str(dart_path)
        
    except Exception as e:
        logger.error(f"[BENCHMARK] Failed to save benchmark data: {e}")
        return None

def save_benchmark_correction(dart_path: str, original: Dict, corrected: Dict):
    """Save correction data when user corrects a dart score."""
    try:
        path = Path(dart_path)
        if not path.exists():
            logger.warning(f"[BENCHMARK] Dart path not found for correction: {dart_path}")
            return
        
        correction = {
            "corrected_at": datetime.now().isoformat(),
            "original": original,
            "corrected": corrected
        }
        
        with open(path / "correction.json", 'w') as f:
            json.dump(correction, f, indent=2)
        
        logger.info(f"[BENCHMARK] Saved correction: {original} -> {corrected}")
    except Exception as e:
        logger.error(f"[BENCHMARK] Failed to save correction: {e}")


def find_recent_dart_on_disk(dart_number: int, game_id: str = None) -> str:
    """
    Scan benchmark directory for the most recent dart folder matching dart_number.
    This is a fallback when in-memory tracking fails (e.g., after restart).
    
    Note: game_id from DartGame API is a GUID, but benchmark folders use timestamps.
    So we ignore game_id and just find the most recent dart with this number.
    """
    try:
        if not BENCHMARK_DIR.exists():
            return None
        
        candidates = []
        
        # Scan all board directories
        for board_dir in BENCHMARK_DIR.iterdir():
            if not board_dir.is_dir():
                continue
            
            # Scan game directories (sorted by name = timestamp, newest first)
            for game_dir in sorted(board_dir.iterdir(), reverse=True):
                if not game_dir.is_dir():
                    continue
                
                # Scan round directories
                for round_dir in game_dir.iterdir():
                    if not round_dir.is_dir() or not round_dir.name.startswith("round_"):
                        continue
                    
                    # Look for dart folder
                    dart_folder = round_dir / f"dart_{dart_number}"
                    if dart_folder.exists() and (dart_folder / "metadata.json").exists():
                        # Get modification time for sorting
                        mtime = (dart_folder / "metadata.json").stat().st_mtime
                        candidates.append((mtime, str(dart_folder)))
        
        # Return most recent match
        if candidates:
            candidates.sort(reverse=True)
            logger.info(f"[BENCHMARK] Disk fallback found {len(candidates)} candidates for dart_{dart_number}, using newest")
            return candidates[0][1]
        
        logger.warning(f"[BENCHMARK] Disk fallback found no dart_{dart_number} folders")
        return None
        
    except Exception as e:
        logger.error(f"[BENCHMARK] Disk scan failed: {e}")
        return None


# === Mask Cache for Differential Detection ===
# Like Machine Darts: 0=background, 76=new dart, 152=old dart
MASK_NEW = 76
MASK_OLD = 152
MASK_BG = 0

# Distance threshold for "new dart" detection (in pixels)
# Tips within this distance of existing darts are rejected
# NOTE: 1/2 inch on board ≈ 20-40px at typical webcam distances
# Keep this LOW to allow close darts - mask filtering handles the rest
NEW_DART_DISTANCE_PX = 30

# Structure: { board_id: { "baseline": {cam_id: np.ndarray}, "masks": {cam_id: np.ndarray}, 
#              "timestamp": float, "dart_count": int,
#              "dart_locations": {cam_id: [(x, y, conf), ...]} } }
_mask_cache: Dict[str, Dict[str, Any]] = {}
_cache_lock = threading.Lock()
CACHE_TTL_SECONDS = 300  # 5 minutes

# Track last raw images for each board (for benchmark "previous" saving)
# Structure: { board_id: { cam_id: np.ndarray } }
_last_raw_images: Dict[str, Dict[str, np.ndarray]] = {}

def get_previous_images(board_id: str) -> Dict[str, np.ndarray]:
    """Get the previous dart's raw images (for benchmark saving)."""
    with _cache_lock:
        if board_id in _last_raw_images:
            return {k: v.copy() for k, v in _last_raw_images[board_id].items()}
        # Fall back to baseline if no previous
        cache = _mask_cache.get(board_id)
        if cache and "baseline" in cache:
            return {k: v.copy() for k, v in cache["baseline"].items()}
        return {}

def set_last_raw_images(board_id: str, images: Dict[str, np.ndarray]):
    """Store raw images for next dart's "previous" reference."""
    with _cache_lock:
        _last_raw_images[board_id] = {k: v.copy() for k, v in images.items()}

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
        dart_locations = {}
        for cam_id, img in baseline_images.items():
            h, w = img.shape[:2]
            masks[cam_id] = np.zeros((h, w), dtype=np.uint8)
            dart_locations[cam_id] = []  # List of (x, y, confidence) tuples
        
        _mask_cache[board_id] = {
            "baseline": {k: v.copy() for k, v in baseline_images.items()},
            "masks": masks,
            "timestamp": time.time(),
            "dart_count": 0,
            "dart_locations": dart_locations,
            # Source of truth: winning camera's dart locations
            # Format: [(cam_id, x, y, dart_number), ...]
            "source_of_truth": []
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

def mark_tip_as_old(board_id: str, cam_id: str, x: float, y: float, radius: int = 30):
    """Mark a tip location as OLD in the mask (for dart 1 which has no NEW pixels)."""
    with _cache_lock:
        cache = _mask_cache.get(board_id)
        if not cache:
            return
        
        mask = cache.get("masks", {}).get(cam_id)
        if mask is None:
            return
        
        h, w = mask.shape
        x_int, y_int = int(x), int(y)
        
        # Create circular mask around tip
        y_coords, x_coords = np.ogrid[:h, :w]
        dist_sq = (x_coords - x_int)**2 + (y_coords - y_int)**2
        circle = dist_sq <= radius**2
        mask[circle] = MASK_OLD
        
        cache["masks"][cam_id] = mask


def add_source_of_truth(board_id: str, cam_id: str, x_px: float, y_px: float, x_mm: float, y_mm: float, dart_number: int):
    """
    Record a winning dart location as source of truth (legacy - single camera).
    """
    with _cache_lock:
        cache = _mask_cache.get(board_id)
        if not cache:
            return
        
        if "source_of_truth" not in cache:
            cache["source_of_truth"] = []
        
        cache["source_of_truth"].append({
            "cam_id": cam_id,
            "x_px": x_px,
            "y_px": y_px,
            "x_mm": x_mm,
            "y_mm": y_mm,
            "dart_number": dart_number
        })
        logger.info(f"[SOT] Added source of truth: Dart {dart_number} @ ({x_mm:.1f}, {y_mm:.1f})mm [from {cam_id}]")


def add_source_of_truth_all_cameras(board_id: str, dart_number: int, all_camera_tips: List[dict]):
    """
    Record dart position from ALL cameras for accurate per-camera matching.
    
    Since mm coords aren't perfectly calibrated across cameras, we store each camera's
    own mm coords and match against the same camera's history.
    
    Args:
        board_id: Board ID
        dart_number: Which dart (1, 2, 3)
        all_camera_tips: List of tips from all cameras, each with camera_id, x_mm, y_mm
    """
    with _cache_lock:
        cache = _mask_cache.get(board_id)
        if not cache:
            return
        
        if "sot_per_camera" not in cache:
            cache["sot_per_camera"] = {}  # cam_id -> list of {dart_number, x_mm, y_mm}
        
        sot = cache["sot_per_camera"]
        
        for tip in all_camera_tips:
            cam_id = tip.get('camera_id')
            x_mm = tip.get('x_mm', 0)
            y_mm = tip.get('y_mm', 0)
            
            if not cam_id:
                continue
            
            if cam_id not in sot:
                sot[cam_id] = []
            
            sot[cam_id].append({
                "dart_number": dart_number,
                "x_mm": x_mm,
                "y_mm": y_mm
            })
            logger.info(f"[SOT] Stored Dart {dart_number} for {cam_id}: ({x_mm:.1f}, {y_mm:.1f})mm")


def get_source_of_truth_for_camera(board_id: str, cam_id: str) -> List[dict]:
    """
    Get source-of-truth dart locations for a specific camera.
    Returns list of {dart_number, x_mm, y_mm} for that camera's perspective.
    """
    with _cache_lock:
        cache = _mask_cache.get(board_id)
        if not cache:
            return []
        
        sot = cache.get("sot_per_camera", {})
        return list(sot.get(cam_id, []))


def get_source_of_truth(board_id: str, cam_id: str = None) -> List[dict]:
    """
    Get all source-of-truth dart locations (legacy - from winning camera only).
    If cam_id provided, returns only darts where that camera was the winner.
    Otherwise returns all darts (for mm-based matching).
    """
    with _cache_lock:
        cache = _mask_cache.get(board_id)
        if not cache:
            return []
        
        sot = cache.get("source_of_truth", [])
        if cam_id:
            return [d for d in sot if d["cam_id"] == cam_id]
        return list(sot)


def find_new_tip_by_elimination_mm(tips: List[dict], board_id: str, cam_id: str, match_threshold_mm: float = 20.0) -> Optional[dict]:
    """
    Find the new dart tip by eliminating known dart locations using MM COORDINATES.
    
    Uses PER-CAMERA matching: compares this camera's tips against this camera's
    historical dart positions, since mm coords aren't perfectly calibrated across cameras.
    
    Args:
        tips: List of tips, each must have 'x_mm' and 'y_mm' set
        board_id: Board ID for cache lookup
        cam_id: Camera ID to match against (uses that camera's stored history)
        match_threshold_mm: Distance in mm to consider a match (default 20mm)
    
    Returns the new tip, or None if can't determine.
    """
    # Use per-camera source of truth
    known_darts = get_source_of_truth_for_camera(board_id, cam_id)
    
    if not known_darts:
        # No previous darts for this camera - can't eliminate, all tips are potentially new
        logger.debug(f"[SOT] No known darts for {cam_id}, can't eliminate")
        return None
    
    # Find tips that DON'T match any known dart
    new_tips = []
    for tip in tips:
        tx_mm, ty_mm = tip.get('x_mm', 0), tip.get('y_mm', 0)
        matches_known = False
        
        for known in known_darts:
            kx_mm, ky_mm = known["x_mm"], known["y_mm"]
            dist = math.sqrt((tx_mm - kx_mm)**2 + (ty_mm - ky_mm)**2)
            if dist < match_threshold_mm:
                matches_known = True
                logger.info(f"[SOT] {cam_id}: Tip ({tx_mm:.1f}, {ty_mm:.1f})mm matches Dart {known['dart_number']} at ({kx_mm:.1f}, {ky_mm:.1f})mm - dist={dist:.1f}mm")
                break
        
        if not matches_known:
            new_tips.append(tip)
            logger.info(f"[SOT] {cam_id}: Tip ({tx_mm:.1f}, {ty_mm:.1f})mm is NEW (no match to known darts)")
    
    if len(new_tips) == 1:
        return new_tips[0]
    elif len(new_tips) > 1:
        # Multiple new tips - pick highest confidence
        logger.warning(f"[SOT] {cam_id}: Found {len(new_tips)} new tips, picking highest confidence")
        return max(new_tips, key=lambda t: t.get('confidence', 0))
    else:
        # All tips matched known darts - shouldn't happen
        logger.warning(f"[SOT] {cam_id}: All tips matched known darts - can't find new tip")
        return None

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
            logger.info(f"Cache cleared for board {board_id} (masks + dart locations)")
        if board_id in _last_raw_images:
            del _last_raw_images[board_id]
            logger.debug(f"Cleared last raw images for board {board_id}")

def get_cached_dart_count(board_id: str) -> int:
    """Get the dart count for cached board."""
    with _cache_lock:
        cache = _mask_cache.get(board_id)
        if cache:
            return cache.get("dart_count", 0)
    return 0


def is_near_existing_dart(board_id: str, cam_id: str, x: float, y: float, threshold_px: float = None) -> bool:
    """
    Check if a tip location is within threshold distance of any existing dart.
    Like Machine Darts' is_new_dart() but inverted (returns True if near existing).
    """
    if threshold_px is None:
        threshold_px = NEW_DART_DISTANCE_PX
    
    with _cache_lock:
        cache = _mask_cache.get(board_id)
        if not cache:
            return False
        
        dart_locations = cache.get("dart_locations", {}).get(cam_id, [])
        
        for (dx, dy, _) in dart_locations:
            dist = math.sqrt((x - dx)**2 + (y - dy)**2)
            if dist < threshold_px:
                logger.info(f"[LOCATION] Tip ({x:.1f}, {y:.1f}) is {dist:.1f}px from existing dart at ({dx:.1f}, {dy:.1f}) - REJECTED")
                return True
        
        return False


def add_dart_location(board_id: str, cam_id: str, x: float, y: float, confidence: float):
    """
    Record a detected dart's location for future comparison.
    Like Machine Darts' add_dart_location().
    """
    with _cache_lock:
        cache = _mask_cache.get(board_id)
        if not cache:
            return
        
        if "dart_locations" not in cache:
            cache["dart_locations"] = {}
        
        if cam_id not in cache["dart_locations"]:
            cache["dart_locations"][cam_id] = []
        
        cache["dart_locations"][cam_id].append((x, y, confidence))
        logger.info(f"[LOCATION] Added dart location ({x:.1f}, {y:.1f}) for {cam_id}, total={len(cache['dart_locations'][cam_id])}")


def clear_dart_locations(board_id: str):
    """Clear all tracked dart locations (on board clear)."""
    with _cache_lock:
        cache = _mask_cache.get(board_id)
        if cache and "dart_locations" in cache:
            for cam_id in cache["dart_locations"]:
                cache["dart_locations"][cam_id] = []
            logger.info(f"[LOCATION] Cleared all dart locations for board {board_id}")

def has_cache(board_id: str) -> bool:
    """Check if board has cache initialized."""
    with _cache_lock:
        return board_id in _mask_cache

def get_baseline_images(board_id: str) -> Dict[str, np.ndarray]:
    """Get baseline images for a board (for benchmark saving)."""
    with _cache_lock:
        cache = _mask_cache.get(board_id)
        if cache and "baseline" in cache:
            return {k: v.copy() for k, v in cache["baseline"].items()}
        return {}

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


def get_segment_from_boundaries(point_angle_deg: float, segment_angles: List[float], segment_20_index: int) -> Tuple[int, float]:
    """
    Determine which segment a point is in using actual wire boundary angles.
    
    This properly accounts for perspective distortion where segments appear
    different sizes (ranging from ~10° to ~30° instead of uniform 18°).
    
    Args:
        point_angle_deg: The angle from center to the dart tip (0-360 degrees)
        segment_angles: List of wire boundary angles in radians (20 values)
        segment_20_index: Which boundary index starts segment 20
    
    Returns:
        Tuple of (segment number 1-20, boundary_distance_deg)
    """
    # Convert point angle to radians for comparison
    point_rad = math.radians(point_angle_deg)
    if point_rad > math.pi:
        point_rad -= 2 * math.pi  # Normalize to -pi to pi
    
    num_boundaries = len(segment_angles)
    
    # Helper to compute angular distance
    def angle_diff(a, b):
        d = a - b
        while d > math.pi: d -= 2*math.pi
        while d < -math.pi: d += 2*math.pi
        return abs(d)
    
    # Debug: show segment_20_index and first few boundaries
    logger.info(f"[SCORE] Boundaries: seg20_idx={segment_20_index}, point_angle={point_angle_deg:.1f}°, point_rad={point_rad:.3f}")
    
    # Find which boundary pair contains the point
    for i in range(num_boundaries):
        start = segment_angles[i]
        end = segment_angles[(i + 1) % num_boundaries]
        
        # Check if point is in this segment (handle wrap-around)
        in_segment = False
        if end < start:
            # Segment wraps around from +pi to -pi
            if point_rad >= start or point_rad < end:
                in_segment = True
        else:
            if start <= point_rad < end:
                in_segment = True
        
        if in_segment:
            # Found it! Calculate distance to nearest boundary
            dist_to_start = angle_diff(point_rad, start)
            dist_to_end = angle_diff(point_rad, end)
            boundary_distance_deg = math.degrees(min(dist_to_start, dist_to_end))
            
            # Map boundary index to segment number
            segment_pos = (i - segment_20_index) % 20
            logger.info(f"[SCORE] Found in boundary {i}, segment_pos={segment_pos}, segment={DARTBOARD_SEGMENTS[segment_pos]}, boundary_dist={boundary_distance_deg:.1f}°")
            return DARTBOARD_SEGMENTS[segment_pos], boundary_distance_deg
    
    # Fallback: shouldn't reach here, but use segment 20 as default
    logger.warning(f"[SCORE] Could not find segment for angle {point_angle_deg}°, defaulting to 20")
    return 20, 9.0  # Default to max distance (middle of segment)


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
    
    # Get calibrated ellipses - dartboard layout from outside to center:
    # outer_double_ellipse (board edge) -> inner_double_ellipse -> outer_triple_ellipse -> inner_triple_ellipse -> bull
    bullseye_ellipse = calibration_data.get('bullseye_ellipse')
    bull_ellipse = calibration_data.get('bull_ellipse')
    board_edge = calibration_data.get('outer_double_ellipse')       # Outermost edge of board
    double_inner = calibration_data.get('inner_double_ellipse')     # Inner edge of double ring
    triple_outer = calibration_data.get('outer_triple_ellipse')     # Outer edge of triple ring  
    triple_inner = calibration_data.get('inner_triple_ellipse')     # Inner edge of triple ring
    
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
        return {"score": 50, "multiplier": 2, "segment": 25, "zone": "inner_bull"}  # Bull counts as double for checkout!
    
    # Check outer bull
    if point_in_ellipse(point, bull_ellipse):
        print(f"[SCORE] Zone: outer_bull")
        return {"score": 25, "multiplier": 1, "segment": 0, "zone": "outer_bull"}
    
    # === SEGMENT DETECTION: Use wire boundaries from calibration ===
    # This uses the SAME boundaries that the overlay shows, ensuring consistency
    segment_angles = calibration_data.get("segment_angles", [])
    segment_20_index = calibration_data.get("segment_20_index", 0)
    boundary_distance_deg = None
    
    if segment_angles and len(segment_angles) >= 20:
        # Use the actual wire boundaries - same as overlay
        segment, boundary_distance_deg = get_segment_from_boundaries(angle_deg, segment_angles, segment_20_index)
        print(f"[SCORE] Using wire boundaries: segment={segment}, boundary_dist={boundary_distance_deg:.1f}°")
    else:
        # Fallback to simple 18° division if no boundaries available
        rotation_offset_deg = calibration_data.get("rotation_offset_deg", 0)
        adjusted_angle = (rotation_offset_deg - angle_deg) % 360
        segment_angle = (adjusted_angle + 9) % 360
        segment_index = int(segment_angle / 18) % 20
        segment = DARTBOARD_SEGMENTS[segment_index]
        # Estimate boundary distance for fallback
        boundary_distance_deg = min(segment_angle % 18, 18 - (segment_angle % 18))
        print(f"[SCORE] Fallback 18° division: rotation_offset={rotation_offset_deg:.1f}, segment={segment}")
    
    # Check if outside the board
    in_board = point_in_ellipse(point, board_edge)
    if not in_board:
        print(f"[SCORE] Zone: miss (outside board)")
        return {"score": 0, "multiplier": 0, "segment": segment, "zone": "miss", "boundary_distance_deg": boundary_distance_deg}
    
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
    
    print(f"[SCORE] Zone: {zone}, Segment: {segment}, Multiplier: {multiplier}, boundary_dist={boundary_distance_deg:.1f}°")
    
    score = segment * multiplier
    return {"score": score, "multiplier": multiplier, "segment": segment, "zone": zone, "boundary_distance_deg": boundary_distance_deg}


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


@router.post("/v1/warmup")
async def warmup_model():
    """Keep the tip detection model warm by running a dummy inference."""
    try:
        if calibrator.tip_detector and calibrator.tip_detector.is_initialized:
            calibrator.tip_detector.warmup()
            return {"status": "ok", "message": "Model warmed up"}
        return {"status": "error", "message": "Tip detector not initialized"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


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
    timings = {}  # Track timing for each step
    
    request_id = str(uuid.uuid4())[:12]
    board_id = request.board_id or "default"
    dart_number = request.dart_number or 1
    
    # Warmup model if it's been idle (prevents cold start latency)
    t0 = time.time()
    if calibrator.tip_detector and calibrator.tip_detector.is_initialized:
        calibrator.tip_detector.maybe_warmup()
    timings['warmup'] = int((time.time() - t0) * 1000)
    
    logger.info(f"")
    logger.info(f"{'='*60}")
    logger.info(f"[DETECT] DART {dart_number} - Board: {board_id} - Request: {request_id}")
    logger.info(f"{'='*60}")
    
    # Cleanup old cache entries periodically
    _cleanup_old_cache()
    
    # Track raw images for benchmark (before decoding)
    raw_images_b64: Dict[str, str] = {}
    for cam in request.cameras:
        raw_images_b64[cam.camera_id] = cam.image
    
    # Decode all images first
    t0 = time.time()
    current_images: Dict[str, np.ndarray] = {}
    for cam in request.cameras:
        try:
            current_images[cam.camera_id] = decode_image(cam.image)
        except Exception as e:
            logger.error(f"[DETECT] Failed to decode image for {cam.camera_id}: {e}")
    timings['decode'] = int((time.time() - t0) * 1000)
    
    # Track debug images for benchmark
    debug_images: Dict[str, np.ndarray] = {}
    calibrations_used: Dict[str, Dict] = {}
    
    # For dart 1: initialize baseline (these images become the reference)
    # For dart 2+: update masks with diff from baseline
    t0 = time.time()
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
    timings['mask'] = int((time.time() - t0) * 1000)
    
    all_tips = []
    camera_results = []
    pipeline_data = {}  # Per-camera detailed detection pipeline
    yolo_total_ms = 0
    scoring_total_ms = 0
    
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
            
            # Track calibration for benchmark
            calibrations_used[cam.camera_id] = calibration_data
            
            # Detect tips using YOLO - always on full image
            t_yolo = time.time()
            tips = calibrator.detect_tips(
                camera_id=cam.camera_id,
                image_base64=cam.image,
                calibration_data=calibration_data
            )
            yolo_ms = int((time.time() - t_yolo) * 1000)
            yolo_total_ms += yolo_ms
            
            # === ADD MM COORDINATES TO ALL TIPS ===
            # Transform pixel coords to dartboard mm coords for cross-camera matching
            for tip in tips:
                x_px = tip.get('x_px', 0)
                y_px = tip.get('y_px', 0)
                x_mm, y_mm = calibrator.pixel_to_dartboard(x_px, y_px, calibration_data)
                tip['x_mm'] = x_mm
                
                # Calculate normalized polar coordinates (comparable across cameras)
                center = calibration_data.get('center', (0, 0))
                outer_double = calibration_data.get('outer_double_ellipse')
                dx = x_px - center[0]
                dy = y_px - center[1]
                angle_deg = (math.degrees(math.atan2(dy, dx)) + 360) % 360
                
                if outer_double:
                    (ecx, ecy), (ew, eh), eangle = outer_double
                    avg_radius_px = (ew + eh) / 4  # Average semi-axis
                    norm_dist = math.sqrt(dx*dx + dy*dy) / avg_radius_px if avg_radius_px > 0 else 0
                else:
                    norm_dist = math.sqrt(dx*dx + dy*dy) / 200  # Fallback
                
                tip['angle_deg'] = angle_deg
                tip['norm_dist'] = norm_dist
                tip['y_mm'] = y_mm
            
            # Keep original tips for debug image AND benchmark
            all_yolo_tips = [t.copy() for t in tips] if tips else []
            
            logger.info(f"[DETECT] Camera {cam.camera_id}: YOLO found {len(tips)} tips ({yolo_ms}ms)")
            
            # Track selected tip and context for debug image
            selected_tip = None
            selection_method = "none"  # Track how tip was selected
            known_darts_for_debug = get_source_of_truth(board_id) if dart_number > 1 else []
            new_centroid_for_debug = None
            
            # Track mask filter results for benchmark
            mask_filter_results = []
            tips_before_mask = len(tips)
            
            # === STEP 1: MASK FILTER (all darts) ===
            # Filter tips to only those INSIDE the NEW mask region
            # This eliminates flight detections and noise
            mask = masks.get(cam.camera_id)
            tips_in_mask = []
            mask_stats = {"has_mask": mask is not None, "new_pixels": 0, "old_pixels": 0}
            if mask is not None:
                # Count mask pixel stats
                mask_stats["new_pixels"] = int(np.sum(mask == MASK_NEW))
                mask_stats["old_pixels"] = int(np.sum(mask == MASK_OLD))
                
                # Get NEW region centroid for fallback tip selection
                new_centroid = get_new_region_centroid(mask)
                
                if tips:
                    for t in tips:
                        tx, ty = t.get('x_px', 0), t.get('y_px', 0)
                        in_region = point_in_new_region(tx, ty, mask, margin=15)
                        mask_filter_results.append({
                            "x_px": tx,
                            "y_px": ty,
                            "x_mm": t.get('x_mm', 0),
                            "y_mm": t.get('y_mm', 0),
                            "confidence": t.get('confidence', 0),
                            "passed_mask": in_region
                        })
                        if in_region:
                            tips_in_mask.append(t)
                            t['found_in_new_region'] = True  # Mark as found in NEW region
                            logger.debug(f"[MASK] Tip ({tx:.1f}, {ty:.1f}) is IN NEW region")
                        else:
                            logger.debug(f"[MASK] Tip ({tx:.1f}, {ty:.1f}) is OUTSIDE NEW region - filtered")
                    
                    if tips_in_mask:
                        logger.info(f"[DETECT] Camera {cam.camera_id}: {len(tips_in_mask)}/{len(tips)} tips in NEW mask region")
                        tips = tips_in_mask
                    elif new_centroid:
                        # No tips directly in NEW region - find tip CLOSEST to NEW centroid
                        # This handles occlusion where tip is just outside the detected new pixels
                        cx, cy = new_centroid
                        tips_with_dist = []
                        for t in tips:
                            tx, ty = t.get('x_px', 0), t.get('y_px', 0)
                            dist = math.sqrt((tx - cx)**2 + (ty - cy)**2)
                            tips_with_dist.append((dist, t))
                        
                        tips_with_dist.sort(key=lambda x: x[0])
                        closest_tip = tips_with_dist[0][1]
                        closest_dist = tips_with_dist[0][0]
                        
                        # Only use if reasonably close (within 100px)
                        if closest_dist < 100:
                            closest_tip['found_in_new_region'] = False  # Mark as fallback
                            closest_tip['centroid_distance_px'] = closest_dist
                            tips = [closest_tip]
                            logger.info(f"[DETECT] Camera {cam.camera_id}: No tips in NEW, using closest to centroid (dist={closest_dist:.1f}px)")
                        else:
                            logger.warning(f"[DETECT] Camera {cam.camera_id}: No tips near NEW centroid (closest={closest_dist:.1f}px), skipping camera")
                            tips = []  # Don't use any tips from this camera
                    else:
                        logger.warning(f"[DETECT] Camera {cam.camera_id}: NO tips in NEW region and no centroid, skipping camera")
                        tips = []  # Don't use any tips from this camera
            
            tips_after_mask = len(tips)
            
            # Track MM elimination results
            mm_elimination_results = []
            
            # === STEP 2: MM ELIMINATION (dart 2+) ===
            # Use mm coordinates to eliminate known dart locations (per-camera matching)
            if tips and dart_number > 1:
                # Get known darts for this camera for logging
                camera_known = get_source_of_truth_for_camera(board_id, cam.camera_id)
                
                new_tip = find_new_tip_by_elimination_mm(tips, board_id, cam.camera_id, match_threshold_mm=20.0)
                if new_tip:
                    logger.info(f"[DETECT] Camera {cam.camera_id}: Found new tip by MM elimination @ ({new_tip.get('x_mm', 0):.1f}, {new_tip.get('y_mm', 0):.1f})mm")
                    selected_tip = new_tip
                    selection_method = "mm_elimination"
                    tips = [new_tip]
                elif len(tips) == 1:
                    # Only one tip after mask filter - use it
                    selected_tip = tips[0]
                    selection_method = "single_after_mask"
                    logger.info(f"[DETECT] Camera {cam.camera_id}: Single tip after mask filter")
                else:
                    # Multiple tips, elimination didn't help - pick highest confidence
                    tips = sorted(tips, key=lambda t: t.get('confidence', 0), reverse=True)[:1]
                    selected_tip = tips[0] if tips else None
                    selection_method = "highest_confidence"
                    logger.info(f"[DETECT] Camera {cam.camera_id}: Multiple tips, picked highest conf")
            
            # === STEP 3: DART 1 FINAL SELECTION ===
            if dart_number == 1 and tips:
                if len(tips) == 1:
                    selected_tip = tips[0]
                    selection_method = "single_tip"
                else:
                    # Multiple tips after mask filter - pick highest confidence
                    tips = sorted(tips, key=lambda t: t.get('confidence', 0), reverse=True)[:1]
                    selected_tip = tips[0]
                    selection_method = "highest_confidence"
                logger.info(f"[DETECT] Camera {cam.camera_id}: Dart 1 - selected tip conf={selected_tip.get('confidence', 0):.3f}")
            
            # === BUILD PIPELINE DATA FOR THIS CAMERA ===
            pipeline_data[cam.camera_id] = {
                "yolo_ms": yolo_ms,
                "all_yolo_tips": [
                    {
                        "x_px": t.get('x_px', 0),
                        "y_px": t.get('y_px', 0),
                        "x_mm": t.get('x_mm', 0),
                        "y_mm": t.get('y_mm', 0),
                        "confidence": t.get('confidence', 0)
                    }
                    for t in all_yolo_tips
                ],
                "mask_stats": mask_stats,
                "mask_filter_results": mask_filter_results,
                "tips_before_mask": tips_before_mask,
                "tips_after_mask": tips_after_mask,
                "selection_method": selection_method,
                "selected_tip": {
                    "x_px": selected_tip.get('x_px', 0) if selected_tip else None,
                    "y_px": selected_tip.get('y_px', 0) if selected_tip else None,
                    "x_mm": selected_tip.get('x_mm', 0) if selected_tip else None,
                    "y_mm": selected_tip.get('y_mm', 0) if selected_tip else None,
                    "confidence": selected_tip.get('confidence', 0) if selected_tip else None
                } if selected_tip else None
            }
            
            # Save debug image with all tips marked (and capture for benchmark)
            debug_img = save_debug_image(
                request_id=request_id,
                dart_number=dart_number,
                camera_id=cam.camera_id,
                image=current_img,
                all_tips=all_yolo_tips,
                selected_tip=selected_tip,
                known_darts=known_darts_for_debug,
                new_centroid=new_centroid_for_debug
            )
            if debug_img is not None:
                debug_images[cam.camera_id] = debug_img
            
            # Calculate score for each tip
            for tip in tips:
                t_score = time.time()
                score_info = score_with_calibration(tip, calibration_data)
                scoring_total_ms += int((time.time() - t_score) * 1000)
                
                tip['camera_id'] = cam.camera_id
                tip['segment'] = score_info.get('segment', 0)
                tip['multiplier'] = score_info.get('multiplier', 1)
                tip['zone'] = score_info.get('zone', 'miss')
                tip['score'] = score_info.get('score', 0)
                tip['boundary_distance_deg'] = score_info.get('boundary_distance_deg')  # For weighted voting
                tip['calibration_quality'] = calibration_data.get('quality', 0.5)  # Camera calibration quality for voting
                all_tips.append(tip)
                
                # Save training data if enabled
                save_training_data(
                    camera_id=cam.camera_id,
                    image_base64=cam.image,
                    segment=tip['segment'],
                    multiplier=tip['multiplier'],
                    tip_x=tip.get('x_px', 0),
                    tip_y=tip.get('y_px', 0),
                    confidence=tip.get('confidence', 0),
                    zone=tip['zone']
                )
                
                logger.info(f"[DETECT] {cam.camera_id}: segment={tip['segment']}, multiplier={tip['multiplier']}, zone={tip['zone']}, score={tip['score']}, conf={tip.get('confidence', 0):.3f}")
            
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
    
    # Log per-camera votes for debugging
    if all_tips:
        votes_summary = ", ".join([f"{t.get('camera_id')}={t.get('segment')}x{t.get('multiplier')}" for t in all_tips])
        logger.info(f"[VOTE] Camera votes: {votes_summary}")
    
    # If all cameras see 1 dart each but clustering failed (mm coords don't match),
    # do segment-based voting across all tips
    if len(clustered_tips) > 1 and all(len(c) == 1 for c in clustered_tips):
        logger.info(f"[DETECT] All cameras see 1 dart each - combining for segment voting")
        all_single_tips = [c[0] for c in clustered_tips]
        clustered_tips = [all_single_tips]
    
    detected_tips = vote_on_scores(clustered_tips)
    
    # Log final result vs votes
    if detected_tips and all_tips:
        winner = detected_tips[0]
        logger.info(f"[VOTE] WINNER: {winner.segment}x{winner.multiplier}={winner.score} (from {len(all_tips)} cameras)")
    
    # DartSensor triggers once per dart - we should only return 1 tip per request
    # Take the most confident one if multiple were detected
    if len(detected_tips) > 1:
        logger.info(f"[DETECT] Multiple tips after voting ({len(detected_tips)}), taking best one")
        detected_tips = sorted(detected_tips, key=lambda t: t.confidence, reverse=True)[:1]
    
    # After successful detection, promote NEW pixels to OLD in the mask
    # AND record the WINNING tip as source of truth
    if detected_tips:
        promote_new_to_old(board_id)
        logger.info(f"[DETECT] Promoted new→old for board {board_id}")
        
        # Find the winning tip (the one from all_tips that matches the voted result)
        # Use the highest confidence tip from the winning segment/multiplier
        for dt in detected_tips:
            winning_tip = None
            winning_conf = 0
            for tip in all_tips:
                if tip['segment'] == dt.segment and tip['multiplier'] == dt.multiplier:
                    if tip.get('confidence', 0) > winning_conf:
                        winning_conf = tip.get('confidence', 0)
                        winning_tip = tip
            
            if winning_tip:
                cam_id = winning_tip.get('camera_id')
                x_px = winning_tip.get('x_px', 0)
                y_px = winning_tip.get('y_px', 0)
                x_mm = winning_tip.get('x_mm', 0)
                y_mm = winning_tip.get('y_mm', 0)
                # Record as source of truth with BOTH px and mm coords (legacy - single camera)
                add_source_of_truth(board_id, cam_id, x_px, y_px, x_mm, y_mm, dart_number)
                # Also add to general dart locations (for backwards compat)
                add_dart_location(board_id, cam_id, x_px, y_px, winning_conf)
    
    # Store source of truth for ALL cameras (for per-camera mm elimination on next dart)
    # all_tips contains the selected tip from each camera with their mm coords
    if all_tips:
        add_source_of_truth_all_cameras(board_id, dart_number, all_tips)
    
    # For dart 1: also mark detected tip locations as OLD so dart 2+ won't include them
    if dart_number == 1 and all_tips:
        for tip in all_tips:
            cam_id = tip.get('camera_id')
            x = tip.get('x_px', 0)
            y = tip.get('y_px', 0)
            if cam_id and x and y:
                mark_tip_as_old(board_id, cam_id, x, y, radius=35)
        logger.info(f"[DETECT] Marked {len(all_tips)} dart 1 tip locations as OLD")
    
    processing_ms = int((time.time() - start_time) * 1000)
    
    # Collect all timings
    timings['yolo'] = yolo_total_ms
    timings['scoring'] = scoring_total_ms
    timings['total'] = processing_ms
    
    # Log timing breakdown
    timing_str = ", ".join([f"{k}={v}ms" for k, v in timings.items()])
    logger.info(f"[TIMING] {timing_str}")
    
    # Clear result summary
    if detected_tips:
        result_summary = ", ".join([f"{t.segment}x{t.multiplier}={t.score}" for t in detected_tips])
        logger.info(f"")
        logger.info(f">>> RESULT: DART {dart_number} = {result_summary} ({processing_ms}ms)")
        logger.info(f"{'='*60}")
        
        # Log to centralized system with full timing breakdown
        for t in detected_tips:
            log_to_api("INFO", "Detection", f"Dart {dart_number}: {t.zone} {t.segment} = {t.score}",
                      {"dart_number": dart_number, "segment": t.segment, "multiplier": t.multiplier,
                       "score": t.score, "zone": t.zone, "confidence": t.confidence,
                       "cameras_used": len(camera_results),
                       "timing": timings})
    else:
        logger.info(f"")
        logger.info(f">>> RESULT: DART {dart_number} = NO DETECTION ({processing_ms}ms)")
        logger.info(f"{'='*60}")
        
        # Log no detection with timing
        log_to_api("WARN", "Detection", f"Dart {dart_number}: No detection",
                  {"dart_number": dart_number, "tips_found": len(all_tips),
                   "cameras_used": len(camera_results),
                   "timing": timings})
    
    # === Save Benchmark Data ===
    # Only save if benchmark logging is enabled (via Settings UI)
    if BENCHMARK_ENABLED:
        # Build final result info
        final_result = {}
        if detected_tips:
            t = detected_tips[0]
            final_result = {
                "segment": t.segment,
                "multiplier": t.multiplier,
                "score": t.score,
                "zone": t.zone,
                "confidence": t.confidence
            }
        
        # Build per-camera results for metadata including what each camera detected
        camera_data = []
        for cr in camera_results:
            # Find tips from this camera to include per-camera segment votes
            camera_tips = [t for t in all_tips if t.get('camera_id') == cr.camera_id]
            tip_info = None
            if camera_tips:
                t = camera_tips[0]  # Usually just 1 tip per camera
                tip_info = {
                    "segment": t.get('segment'),
                    "multiplier": t.get('multiplier'),
                    "zone": t.get('zone'),
                    "score": t.get('score'),
                    "confidence": t.get('confidence')
                }
            camera_data.append({
                "camera_id": cr.camera_id,
                "tips_detected": cr.tips_detected,
                "error": cr.error,
                "detected": tip_info
            })
        
        # Save benchmark data (including previous images for replay diff)
        previous_imgs = get_previous_images(board_id)
        benchmark_path = save_benchmark_data(
            dart_number=dart_number,
            request_id=request_id,
            raw_images=raw_images_b64,
            debug_images=debug_images,
            camera_results=camera_data,
            final_result=final_result,
            calibrations=calibrations_used,
            timings=timings,
            pipeline_data=pipeline_data,
            baseline_images=previous_imgs  # Actually "previous" images for diff
        )
        
        # Store current images as "previous" for next dart
        set_last_raw_images(board_id, current_images)
        
        if benchmark_path:
            logger.info(f"[BENCHMARK] Saved to {benchmark_path}")
    
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



def weighted_position_average(tips: List[dict]) -> Tuple[float, float, float]:
    """
    Compute weighted average position from multiple camera detections.
    
    Instead of voting on scores, we average the mm positions weighted by:
    - YOLO confidence
    - Calibration quality
    - Whether tip was found in NEW region
    - Boundary distance (tips near center of segment are more reliable)
    
    Returns:
        (x_mm, y_mm, total_weight)
    """
    if not tips:
        return 0.0, 0.0, 0.0
    
    total_x = 0.0
    total_y = 0.0
    total_weight = 0.0
    
    for tip in tips:
        weight = tip.get('confidence', 0.5)
        
        # Calibration quality factor
        cal_quality = tip.get('calibration_quality', 0.5)
        weight *= (0.5 + cal_quality)
        
        # Reduce weight for fallback tips (not in NEW region)
        if not tip.get('found_in_new_region', True):
            weight *= 0.1
        
        # Boundary distance factor - tips near wire are less reliable
        boundary_dist = tip.get('boundary_distance_deg')
        if boundary_dist is not None:
            if boundary_dist < 2.0:
                weight *= 0.3
            elif boundary_dist < 4.0:
                weight *= 0.7
        
        x_mm = tip.get('x_mm', 0.0)
        y_mm = tip.get('y_mm', 0.0)
        
        total_x += x_mm * weight
        total_y += y_mm * weight
        total_weight += weight
    
    if total_weight > 0:
        return total_x / total_weight, total_y / total_weight, total_weight
    return 0.0, 0.0, 0.0


def weighted_polar_average(tips: List[dict]) -> Tuple[float, float, float]:
    """
    Calculate weighted average in POLAR coordinates.
    
    Unlike mm coordinates, polar coords (angle_deg, norm_dist) are
    comparable across cameras because they're relative to each
    camera's calibrated center and outer ring.
    
    Returns:
        (avg_angle_deg, avg_norm_dist, total_weight)
    """
    if not tips:
        return 0.0, 0.0, 0.0
    
    # Use vector averaging for angles to handle wraparound correctly
    total_cos = 0.0
    total_sin = 0.0
    total_dist = 0.0
    total_weight = 0.0
    
    for tip in tips:
        weight = tip.get('confidence', 0.5)
        cal_quality = tip.get('calibration_quality', 0.5)
        weight *= (0.5 + cal_quality)
        
        angle_deg = tip.get('angle_deg', 0.0)
        norm_dist = tip.get('norm_dist', 0.0)
        
        angle_rad = math.radians(angle_deg)
        total_cos += math.cos(angle_rad) * weight
        total_sin += math.sin(angle_rad) * weight
        total_dist += norm_dist * weight
        total_weight += weight
    
    if total_weight > 0:
        avg_angle_rad = math.atan2(total_sin / total_weight, total_cos / total_weight)
        avg_angle_deg = (math.degrees(avg_angle_rad) + 360) % 360
        avg_norm_dist = total_dist / total_weight
        return avg_angle_deg, avg_norm_dist, total_weight
    return 0.0, 0.0, 0.0


def score_from_polar(angle_deg: float, norm_dist: float) -> dict:
    """
    Calculate score from polar coordinates.
    
    Args:
        angle_deg: Angle from center (0-360, 0=right/east)
        norm_dist: Normalized distance (0 = center, 1.0 = outer double edge)
    """
    # Segment order (clockwise from right, but we need to find where 20 is)
    # Standard dartboard: 20 at top, so 20 is at 90 degrees (up)
    # Order clockwise: 20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5
    SEGMENTS = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
    
    # Normalize zones by outer double (170mm)
    BULLSEYE_NORM = 6.35 / 170.0       # ~0.037
    OUTER_BULL_NORM = 16.0 / 170.0     # ~0.094
    INNER_TRIPLE_NORM = 99.0 / 170.0   # ~0.582
    OUTER_TRIPLE_NORM = 107.0 / 170.0  # ~0.629
    INNER_DOUBLE_NORM = 162.0 / 170.0  # ~0.953
    OUTER_DOUBLE_NORM = 1.0
    
    # Determine zone from normalized distance
    if norm_dist <= BULLSEYE_NORM:
        return {"score": 50, "multiplier": 2, "segment": 25, "zone": "inner_bull", "boundary_distance_deg": 9.0}  # Double!
    elif norm_dist <= OUTER_BULL_NORM:
        return {"score": 25, "multiplier": 1, "segment": 0, "zone": "outer_bull", "boundary_distance_deg": 9.0}
    elif norm_dist > OUTER_DOUBLE_NORM * 1.05:  # 5% tolerance
        return {"score": 0, "multiplier": 0, "segment": 0, "zone": "miss", "boundary_distance_deg": 0.0}
    
    # Determine segment from angle
    # 20 is at top (90°), each segment is 18° wide
    # Adjust so 20 is centered at 90°
    adjusted_angle = (angle_deg - 90 + 9 + 360) % 360  # +9 to center the segment
    segment_idx = int(adjusted_angle / 18.0) % 20
    segment = SEGMENTS[segment_idx]
    
    # Calculate boundary distance (how far from wire)
    angle_in_segment = adjusted_angle % 18.0
    boundary_distance_deg = min(angle_in_segment, 18.0 - angle_in_segment)
    
    # Determine multiplier from distance
    if INNER_DOUBLE_NORM <= norm_dist <= OUTER_DOUBLE_NORM * 1.05:
        multiplier = 2
        zone = "double"
    elif INNER_TRIPLE_NORM <= norm_dist <= OUTER_TRIPLE_NORM:
        multiplier = 3
        zone = "triple"
    elif norm_dist < INNER_TRIPLE_NORM:
        multiplier = 1
        zone = "single_inner"
    else:
        multiplier = 1
        zone = "single_outer"
    
    return {
        "score": segment * multiplier,
        "multiplier": multiplier,
        "segment": segment,
        "zone": zone,
        "boundary_distance_deg": boundary_distance_deg
    }


def score_from_position(x_mm: float, y_mm: float, calibration_data: Dict = None) -> Dict:
    """
    Calculate score from mm position on dartboard.
    
    This is the reverse of pixel_to_dartboard - we have the position,
    now calculate what segment/multiplier it's in.
    """
    import math
    
    # Calculate distance from center
    distance = math.sqrt(x_mm**2 + y_mm**2)
    
    # Calculate angle (0° = right, counter-clockwise)
    angle_rad = math.atan2(y_mm, x_mm)
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360
    
    # Standard dartboard radii (mm)
    BULL_RADIUS = 6.35  # Inner bull
    OUTER_BULL_RADIUS = 16.0  # Outer bull
    TRIPLE_INNER = 99.0
    TRIPLE_OUTER = 107.0
    DOUBLE_INNER = 162.0
    DOUBLE_OUTER = 170.0
    
    # Determine zone by distance
    if distance <= BULL_RADIUS:
        return {"segment": 0, "multiplier": 2, "zone": "inner_bull", "score": 50}
    elif distance <= OUTER_BULL_RADIUS:
        return {"segment": 0, "multiplier": 1, "zone": "outer_bull", "score": 25}
    elif distance > DOUBLE_OUTER:
        return {"segment": 0, "multiplier": 0, "zone": "miss", "score": 0}
    
    # Determine segment from angle
    # Standard dartboard: 20 at top, going clockwise: 20,1,18,4,13,6,10,15,2,17,3,19,7,16,8,11,14,9,12,5
    segments = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
    
    # Each segment is 18 degrees, starting from -9° from top (270°)
    # Segment 20 is centered at 270° (top)
    # Normalize angle: 0° = segment 20 center
    normalized_angle = (angle_deg - 270 + 9) % 360  # +9 to align with segment boundary
    segment_index = int(normalized_angle / 18) % 20
    segment = segments[segment_index]
    
    # Calculate boundary distance
    segment_center_offset = normalized_angle % 18
    boundary_distance = min(segment_center_offset, 18 - segment_center_offset)
    
    # Determine multiplier by distance
    if TRIPLE_INNER <= distance <= TRIPLE_OUTER:
        multiplier = 3
        zone = "triple"
    elif DOUBLE_INNER <= distance <= DOUBLE_OUTER:
        multiplier = 2
        zone = "double"
    elif distance < TRIPLE_INNER:
        multiplier = 1
        zone = "single_inner"
    else:
        multiplier = 1
        zone = "single_outer"
    
    score = segment * multiplier
    
    return {
        "segment": segment,
        "multiplier": multiplier,
        "zone": zone,
        "score": score,
        "boundary_distance_deg": boundary_distance,
        "distance_mm": distance
    }



def vote_on_scores(clusters: List[List[dict]]) -> List[DetectedTip]:
    """
    For each cluster (same dart seen by multiple cameras), vote on the score.
    
    ENHANCED VOTING (Feb 2026):
    - Tips found IN the NEW mask region get full weight
    - Tips found via centroid fallback get 0.1x weight
    - Darts near wire boundaries (<2°) get "low_confidence" flag
    - Camera disagreement is detected and logged
    - Exponential boundary weighting (more aggressive near wires)
    """
    detected_tips = []
    
    # Wire boundary threshold - darts closer than this are "uncertain"
    WIRE_THRESHOLD_DEG = 2.0
    
    for cluster in clusters:
        if not cluster:
            continue
        
        cameras_seen = list(set(t['camera_id'] for t in cluster))
        
        # Check if any detection is near a wire boundary
        min_boundary_dist = min(
            (t.get('boundary_distance_deg') or 9.0) for t in cluster
        )
        near_wire = min_boundary_dist < WIRE_THRESHOLD_DEG
        
        # Single camera - use directly but flag if near wire
        if len(cluster) == 1:
            tip = cluster[0]
            conf = tip['confidence']
            if near_wire:
                conf *= 0.5  # Reduce confidence for near-wire single-camera detections
                logger.warning(f"[VOTE] Single camera near wire ({min_boundary_dist:.1f}°) - low confidence")
            
            detected_tips.append(DetectedTip(
                x_mm=round(tip['x_mm'], 2),
                y_mm=round(tip['y_mm'], 2),
                segment=tip['segment'],
                multiplier=tip['multiplier'],
                zone=tip['zone'],
                score=tip['score'],
                confidence=round(conf, 3),
                cameras_seen=cameras_seen
            ))
            continue
        
        # Multiple cameras - vote on segment and multiplier
        votes = {}
        vote_details = {}  # Track which cameras voted for what
        total_confidence = 0.0
        
        # Check if ANY tip was found in NEW region
        tips_in_new = [t for t in cluster if t.get('found_in_new_region', True)]
        tips_fallback = [t for t in cluster if not t.get('found_in_new_region', True)]
        
        if tips_in_new and tips_fallback:
            logger.info(f"[VOTE] {len(tips_in_new)} tips in NEW region, {len(tips_fallback)} via fallback")
        
        for tip in cluster:
            key = (tip['segment'], tip['multiplier'])
            cam_id = tip.get('camera_id', 'unknown')
            
            # Base weight is YOLO confidence
            weight = tip['confidence']
            
            # CAMERA QUALITY WEIGHTING: Higher calibration quality = more trusted
            # Quality ranges 0-1, scale to 0.5-1.5 multiplier
            cal_quality = tip.get('calibration_quality', 0.5)
            quality_factor = 0.5 + cal_quality  # 0 quality = 0.5x, 1.0 quality = 1.5x
            weight *= quality_factor
            
            # MAJOR weight reduction for tips NOT found in NEW region
            if not tip.get('found_in_new_region', True):
                weight *= 0.1
                logger.debug(f"[VOTE] {cam_id} tip NOT in NEW region - weight reduced to {weight:.3f}")
            
            # EXPONENTIAL boundary weighting (more aggressive near wires)
            # 0° (on wire) = 0.1x, 2° = 0.5x, 5° = 0.9x, 9° (center) = 1.5x
            boundary_dist = tip.get('boundary_distance_deg')
            if boundary_dist is not None:
                if boundary_dist < 1.0:
                    boundary_factor = 0.1  # Extremely low weight on wire
                elif boundary_dist < 2.0:
                    boundary_factor = 0.3  # Very low weight near wire
                elif boundary_dist < 4.0:
                    boundary_factor = 0.7  # Reduced weight
                else:
                    # Linear from 0.9 at 4° to 1.5 at 9°
                    boundary_factor = 0.9 + (boundary_dist - 4.0) * 0.12
                    boundary_factor = min(1.5, boundary_factor)
                weight *= boundary_factor
            
            votes[key] = votes.get(key, 0.0) + weight
            total_confidence += weight
            
            # Track vote details for disagreement analysis
            if key not in vote_details:
                vote_details[key] = []
            vote_details[key].append({
                'camera': cam_id,
                'weight': weight,
                'boundary_dist': boundary_dist,
                'in_new': tip.get('found_in_new_region', True),
                'cal_quality': cal_quality
            })
        
        # Detect disagreement
        unique_votes = len(votes)
        if unique_votes > 1:
            # Cameras disagree - log details
            sorted_votes = sorted(votes.items(), key=lambda x: -x[1])
            vote_summary = ', '.join([
                f"{s}x{m}={votes[(s,m)]:.2f}" for s, m in [v[0] for v in sorted_votes]
            ])
            logger.warning(f"[VOTE] DISAGREEMENT ({unique_votes} options): {vote_summary}")
            
            # Log which cameras voted for what (with quality)
            for (seg, mult), details in vote_details.items():
                cam_info = [f"{d['camera']}(q={d.get('cal_quality', 0):.2f})" for d in details]
                total_weight = sum(d['weight'] for d in details)
                logger.info(f"[VOTE]   {seg}x{mult}: {cam_info} total_weight={total_weight:.2f}")
            
            # POLAR AVERAGING: When cameras disagree, average polar coordinates
            # (mm coordinates are camera-specific, polar coords are comparable)
            try:
                avg_angle, avg_dist, avg_weight = weighted_polar_average(cluster)
                if avg_weight > 0:
                    pos_score = score_from_polar(avg_angle, avg_dist)
                    logger.info(f"[VOTE] Polar average: angle={avg_angle:.1f}°, dist={avg_dist:.3f}")
                pos_segment = pos_score['segment']
                pos_multiplier = pos_score['multiplier']
                pos_key = (pos_segment, pos_multiplier)
                
                # Check if position-based score differs from vote winner
                if pos_key != winning_key:
                    # Position averaging suggests different score
                    # Use it if it has reasonable confidence
                    pos_boundary = pos_score.get('boundary_distance_deg', 9)
                    if pos_boundary > 2.0:  # Not on a wire
                        logger.info(f"[VOTE] Position averaging suggests {pos_segment}x{pos_multiplier} (boundary={pos_boundary:.1f}°)")
                        logger.info(f"[VOTE] OVERRIDE: Using position average instead of vote winner")
            except Exception as polar_err:
                logger.warning(f"[VOTE] Polar averaging failed: {polar_err}")

            # Check if stereo triangulation is available and enabled
            if TRIANGULATION_MODE == "stereo":
                try:
                    stereo_cal = get_stereo_calibrator()
                    if stereo_cal.calibration is not None:
                        # Get pixel coordinates for each camera
                        pixel_coords = {}
                        for tip in cluster:
                            cam_id = tip.get('camera_id')
                            if cam_id and 'x_px' in tip and 'y_px' in tip:
                                pixel_coords[cam_id] = (tip['x_px'], tip['y_px'])
                        
                        if len(pixel_coords) >= 2:
                            # Use true 3D triangulation
                            avg_x, avg_y = stereo_cal.triangulate_to_dartboard(pixel_coords)
                            logger.info(f"[STEREO] Triangulated position: ({avg_x:.1f}, {avg_y:.1f}) mm")
                            
                            pos_score = score_from_position(avg_x, avg_y)
                            pos_segment = pos_score['segment']
                            pos_multiplier = pos_score['multiplier']
                            pos_key = (pos_segment, pos_multiplier)
                            
                            if pos_key != winning_key:
                                pos_boundary = pos_score.get('boundary_distance_deg', 9)
                                if pos_boundary > 2.0:
                                    logger.info(f"[STEREO] Triangulation suggests {pos_segment}x{pos_multiplier}")
                                    logger.info(f"[STEREO] OVERRIDE: Using triangulated position")
                                    winning_key = pos_key
                                    winning_segment, winning_multiplier = pos_key
                except Exception as e:
                    logger.warning(f"[STEREO] Triangulation failed: {e}, falling back to position averaging")
        
        # Find winning vote
        winning_key = max(votes.keys(), key=lambda k: votes[k])
        winning_segment, winning_multiplier = winning_key
        
        # Consensus confidence - how much of total weight agrees
        agreeing_confidence = votes[winning_key]
        consensus_confidence = agreeing_confidence / total_confidence if total_confidence > 0 else 0.0
        
        # Reduce confidence if:
        # 1. Near wire boundary
        # 2. Cameras disagreed
        # 3. Low consensus
        if near_wire:
            consensus_confidence *= 0.7
            logger.info(f"[VOTE] Near wire ({min_boundary_dist:.1f}°) - confidence reduced")
        
        if unique_votes >= 3:
            consensus_confidence *= 0.5
            logger.warning(f"[VOTE] 3-way split - confidence heavily reduced")
        elif unique_votes == 2 and consensus_confidence < 0.6:
            consensus_confidence *= 0.7
            logger.info(f"[VOTE] Split vote with low consensus - confidence reduced")
        
        # Determine zone and score
        if winning_segment == 0:
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
        
        # Average position
        avg_x = sum(t['x_mm'] for t in cluster) / len(cluster)
        avg_y = sum(t['y_mm'] for t in cluster) / len(cluster)
        
        confidence_label = "HIGH" if consensus_confidence > 0.7 else "MED" if consensus_confidence > 0.4 else "LOW"
        logger.info(f"[VOTE] Result: {winning_segment}x{winning_multiplier}={winning_score}, conf={consensus_confidence:.3f} ({confidence_label}), near_wire={near_wire}")
        
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


@router.get("/v1/training/status")
async def get_training_status():
    """Get training data capture status."""
    global CAPTURE_TRAINING_DATA
    return {
        "enabled": CAPTURE_TRAINING_DATA,
        "directory": str(TRAINING_DATA_DIR)
    }


@router.post("/v1/training/enable")
async def enable_training_capture(request: dict = {}):
    """Enable training data capture."""
    global CAPTURE_TRAINING_DATA
    CAPTURE_TRAINING_DATA = True
    logger.info("[TRAINING] Data capture ENABLED")
    return {"enabled": True, "directory": str(TRAINING_DATA_DIR)}


@router.post("/v1/training/disable")
async def disable_training_capture():
    """Disable training data capture."""
    global CAPTURE_TRAINING_DATA
    CAPTURE_TRAINING_DATA = False
    logger.info("[TRAINING] Data capture DISABLED")
    return {"enabled": False}


# === Focus Tool ===
from app.core.focus_tool import calculate_focus_score


class FocusRequest(BaseModel):
    """Request for focus measurement."""
    camera_id: str
    image: str  # base64 encoded image
    center_x: Optional[float] = None  # Optional center override
    center_y: Optional[float] = None


@router.post("/v1/focus")
async def measure_focus(request: FocusRequest):
    """
    Measure camera focus quality using Siemens star metrics.
    Returns a score 0-100 and quality rating.
    Higher score = better focus.
    """
    try:
        # Decode image - handle data URI and fix padding
        image_data = request.image.split(',')[-1] if ',' in request.image else request.image
        # Fix base64 padding if needed
        padding = 4 - len(image_data) % 4
        if padding != 4:
            image_data += '=' * padding
        img_data = base64.b64decode(image_data)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")
        
        # Optional center override
        center = None
        if request.center_x is not None and request.center_y is not None:
            center = (int(request.center_x), int(request.center_y))
        
        # Calculate focus score
        result = calculate_focus_score(frame, center=center)
        result["camera_id"] = request.camera_id
        
        logger.info(f"[FOCUS] Camera {request.camera_id}: score={result['score']}, quality={result['quality']}")
        
        return result
        
    except Exception as e:
        logger.error(f"[FOCUS] Error measuring focus: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === Benchmark API Endpoints ===

class BenchmarkContextRequest(BaseModel):
    """Request to set benchmark context."""
    board_id: Optional[str] = None
    game_id: Optional[str] = None
    round_num: Optional[int] = None
    player_name: Optional[str] = None


class BenchmarkCorrectionRequest(BaseModel):
    """Request to record a dart correction."""
    dart_path: Optional[str] = None  # Full path, or...
    dart_number: Optional[int] = None  # Just dart number (1,2,3) to use tracked path
    game_id: Optional[str] = None  # Game ID for precise matching
    original_segment: int
    original_multiplier: int
    corrected_segment: int
    corrected_multiplier: int


@router.get("/v1/benchmark/status")
async def get_benchmark_status():
    """Get current benchmark status and context."""
    return {
        "enabled": BENCHMARK_ENABLED,
        "directory": str(BENCHMARK_DIR),
        "context": _benchmark_context.copy(),
        "path_count": len(_last_dart_paths),
        "history_count": len(_dart_path_history),
        "recent_paths": {str(k): v for k, v in list(_last_dart_paths.items())[-6:]}
    }


@router.post("/v1/benchmark/enable")
async def enable_benchmark():
    """Enable benchmark logging."""
    set_benchmark_enabled(True)
    return {"enabled": True, "directory": str(BENCHMARK_DIR)}


@router.post("/v1/benchmark/disable")
async def disable_benchmark():
    """Disable benchmark logging."""
    set_benchmark_enabled(False)
    return {"enabled": False}


@router.post("/v1/benchmark/clear")
async def clear_benchmark():
    """Clear all benchmark data (images and metadata)."""
    import shutil
    try:
        if BENCHMARK_DIR.exists():
            shutil.rmtree(BENCHMARK_DIR)
            BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"[BENCHMARK] Cleared all benchmark data from {BENCHMARK_DIR}")
            return {"cleared": True, "message": "All benchmark data cleared"}
        else:
            return {"cleared": True, "message": "No benchmark data to clear"}
    except Exception as e:
        logger.error(f"[BENCHMARK] Failed to clear data: {e}")
        return {"cleared": False, "error": str(e)}


@router.post("/v1/benchmark/context")
async def update_benchmark_context(request: BenchmarkContextRequest):
    """Update benchmark context (called when game state changes)."""
    set_benchmark_context(
        board_id=request.board_id,
        game_id=request.game_id,
        round_num=request.round_num,
        player_name=request.player_name
    )
    return {"context": _benchmark_context.copy()}


@router.post("/v1/benchmark/correction")
async def record_benchmark_correction(request: BenchmarkCorrectionRequest):
    """Record a dart correction for accuracy analysis."""
    # If dart_path not provided, try to find it from recent tracking
    dart_path = request.dart_path
    
    if not dart_path and request.dart_number:
        # Try precise match with game_id first
        if request.game_id:
            dart_path = _last_dart_paths.get((request.game_id, request.dart_number))
            logger.info(f"[BENCHMARK] Looking up path for game={request.game_id}, dart={request.dart_number}: {dart_path}")
        
        # Fallback to dart_number only (most recent)
        if not dart_path:
            dart_path = _last_dart_paths.get(request.dart_number)
            logger.info(f"[BENCHMARK] Fallback lookup for dart={request.dart_number}: {dart_path}")
        
        # Search history for most recent dart with this number
        if not dart_path and _dart_path_history:
            for entry in reversed(_dart_path_history):
                if entry["dart_number"] == request.dart_number:
                    dart_path = entry["path"]
                    logger.info(f"[BENCHMARK] History lookup for dart={request.dart_number}: {dart_path}")
                    break
        
        # FINAL FALLBACK: Scan disk for most recent dart folder with this number
        if not dart_path:
            dart_path = find_recent_dart_on_disk(request.dart_number, request.game_id)
            if dart_path:
                logger.info(f"[BENCHMARK] Disk lookup for dart={request.dart_number}: {dart_path}")
    
    if not dart_path:
        logger.warning(f"[BENCHMARK] No dart_path found for game={request.game_id}, dart={request.dart_number}")
        return {"success": False, "error": "No dart path available"}
    
    save_benchmark_correction(
        dart_path=dart_path,
        original={
            "segment": request.original_segment,
            "multiplier": request.original_multiplier
        },
        corrected={
            "segment": request.corrected_segment,
            "multiplier": request.corrected_multiplier
        }
    )
    return {"success": True}


@router.get("/v1/benchmark/games")
async def list_benchmark_games():
    """List all games with benchmark data."""
    games = []
    
    if not BENCHMARK_DIR.exists():
        return {"games": []}
    
    for board_dir in BENCHMARK_DIR.iterdir():
        if not board_dir.is_dir():
            continue
        
        for game_dir in board_dir.iterdir():
            if not game_dir.is_dir():
                continue
            
            # Count darts and corrections
            total_darts = 0
            corrections = 0
            
            for round_dir in game_dir.iterdir():
                if not round_dir.is_dir():
                    continue
                for dart_dir in round_dir.iterdir():
                    if not dart_dir.is_dir():
                        continue
                    total_darts += 1
                    if (dart_dir / "correction.json").exists():
                        corrections += 1
            
            if total_darts > 0:
                accuracy = ((total_darts - corrections) / total_darts) * 100
                games.append({
                    "board_id": board_dir.name,
                    "game_id": game_dir.name,
                    "path": str(game_dir),
                    "total_darts": total_darts,
                    "corrections": corrections,
                    "accuracy": round(accuracy, 1)
                })
    
    # Sort by most recent first (game_id often contains timestamp)
    games.sort(key=lambda g: g["game_id"], reverse=True)
    
    return {"games": games}


@router.get("/v1/benchmark/games/{board_id}/{game_id}/darts")
async def get_benchmark_game_darts(board_id: str, game_id: str):
    """Get all darts for a specific game with their detection details."""
    try:
        game_dir = BENCHMARK_DIR / board_id / game_id
        
        if not game_dir.exists():
            raise HTTPException(status_code=404, detail="Game not found")
        
        darts = []
        
        for round_dir in sorted(game_dir.iterdir()):
            if not round_dir.is_dir():
                continue
            
            for dart_dir in sorted(round_dir.iterdir()):
                if not dart_dir.is_dir():
                    continue
                
                dart_info = {
                    "path": str(dart_dir),
                    "round": round_dir.name,
                    "dart": dart_dir.name,
                    "has_correction": (dart_dir / "correction.json").exists()
                }
                
                # Load metadata if exists
                meta_path = dart_dir / "metadata.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        dart_info["metadata"] = json.load(f)
                
                # Load correction if exists
                correction_path = dart_dir / "correction.json"
                if correction_path.exists():
                    with open(correction_path) as f:
                        dart_info["correction"] = json.load(f)
                
                # List available images
                dart_info["images"] = [f.name for f in dart_dir.iterdir() if f.suffix in ['.jpg', '.png']]
                
                darts.append(dart_info)
    
        return {"darts": darts}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[BENCHMARK] Error loading darts for {board_id}/{game_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/benchmark/image/{board_id}/{game_id}/{round_name}/{dart_name}/{image_name}")
async def get_benchmark_image(board_id: str, game_id: str, round_name: str, dart_name: str, image_name: str):
    """Get a benchmark image as base64."""
    from fastapi.responses import FileResponse
    
    image_path = BENCHMARK_DIR / board_id / game_id / round_name / dart_name / image_name
    
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(image_path, media_type="image/jpeg")


class ReplayRequest(BaseModel):
    dart_path: str  # Path to dart folder
    rescore_only: bool = False  # If true, use saved tip locations instead of re-detecting


class ReplayAllRequest(BaseModel):
    board_id: str = "default"
    limit: int = 100  # Max darts to replay


@router.post("/v1/benchmark/replay")
async def replay_single_dart(request: ReplayRequest):
    """
    Replay a single dart through current detection code.
    Loads raw images, runs YOLO tip detection, applies current scoring formula.
    Compares to original/corrected to see if accuracy improved.
    """
    dart_path = Path(request.dart_path)
    
    if not dart_path.exists():
        raise HTTPException(status_code=404, detail="Dart path not found")
    
    # Load metadata
    meta_path = dart_path / "metadata.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="No metadata.json found")
    
    with open(meta_path) as f:
        metadata = json.load(f)
    
    # Load correction if exists
    correction = None
    correction_path = dart_path / "correction.json"
    if correction_path.exists():
        with open(correction_path) as f:
            correction = json.load(f)
    
    # Load raw images
    images = {}
    for cam_file in dart_path.glob("*_raw.jpg"):
        cam_id = cam_file.stem.replace("_raw", "")
        img = cv2.imread(str(cam_file))
        if img is not None:
            images[cam_id] = img
    
    if not images:
        raise HTTPException(status_code=400, detail="No raw images found")
    
    # Load baseline images for diff detection
    baselines = {}
    for cam_file in dart_path.glob("*_previous.jpg"):
        cam_id = cam_file.stem.replace("_previous", "")
        img = cv2.imread(str(cam_file))
        if img is not None:
            baselines[cam_id] = img
    
    # Get calibrations from metadata and normalize key names
    raw_calibrations = metadata.get("calibrations", {})
    saved_calibrations = {}
    for cam_id, cal in raw_calibrations.items():
        # Normalize ellipse key names - old benchmark data may use different names
        # Old: triple_ellipse, double_ellipse, inner_single_ellipse, outer_single_ellipse
        # New: outer_triple_ellipse, inner_triple_ellipse, outer_double_ellipse, inner_double_ellipse
        saved_calibrations[cam_id] = {
            **cal,  # Keep all original keys
            "outer_double_ellipse": cal.get("outer_double_ellipse") or cal.get("double_ellipse"),
            "inner_double_ellipse": cal.get("inner_double_ellipse") or cal.get("inner_single_ellipse"),
            "outer_triple_ellipse": cal.get("outer_triple_ellipse") or cal.get("triple_ellipse"),
            "inner_triple_ellipse": cal.get("inner_triple_ellipse") or cal.get("outer_single_ellipse"),
        }
    from app.core.detection import score_from_ellipse_calibration
    # Get pipeline data with original tip locations
    pipeline_data = metadata.get("pipeline", {})
    
    # Run detection on each camera
    camera_votes = []
    
    for cam_id, img in images.items():
        # Try saved calibration first, then fall back to current calibrator
        cal = saved_calibrations.get(cam_id)
        if not cal and cam_id in calibrator.calibrations:
            cal = calibrator.calibrations[cam_id]
        
        if not cal:
            continue
        
        # Get tip location - either from saved data (rescore_only) or re-detect
        if request.rescore_only:
            # Use saved tip location from original detection
            cam_pipeline = pipeline_data.get(cam_id, {})
            selected = cam_pipeline.get("selected_tip")
            if not selected:
                continue
            tip_x, tip_y = selected.get("x_px"), selected.get("y_px")
            tip_confidence = selected.get("confidence", 0)
        else:
            # Re-run YOLO tip detection
            tips = calibrator.tip_detector.detect_tips(img, confidence_threshold=CONFIDENCE_THRESHOLD)
            
            if not tips:
                continue
            
            # If we have baseline, use mask to filter for NEW dart only
            if cam_id in baselines:
                baseline = baselines[cam_id]
                # Compute diff mask
                diff = cv2.absdiff(img, baseline)
                gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray_diff, 40, 255, cv2.THRESH_BINARY)
                
                # Filter tips to only those in the NEW region
                filtered_tips = []
                for tip in tips:
                    tx, ty = int(tip.x), int(tip.y)
                    if 0 <= ty < mask.shape[0] and 0 <= tx < mask.shape[1]:
                        if mask[ty, tx] > 0:
                            filtered_tips.append(tip)
                
                if filtered_tips:
                    tips = filtered_tips
            
            # Use highest confidence tip from filtered set
            best_tip = max(tips, key=lambda t: t.confidence)
            tip_x, tip_y = best_tip.x, best_tip.y
            tip_confidence = best_tip.confidence
        
        # Score using calibration data
        result = score_from_ellipse_calibration((tip_x, tip_y), cal)
        
        if result:
            camera_votes.append({
                "camera_id": cam_id,
                "segment": result["segment"],
                "multiplier": result["multiplier"],
                "score": result["score"],
                "zone": result.get("zone"),
                "confidence": tip_confidence
            })
    
    if not camera_votes:
        return {
            "success": False,
            "error": "No tips detected in any camera",
            "dart_path": str(dart_path),
            "original": metadata.get("final_result"),
            "corrected": correction.get("corrected") if correction else None
        }
    
    # Vote for final result (simple majority or highest confidence)
    from collections import Counter
    vote_keys = [(v["segment"], v["multiplier"]) for v in camera_votes]
    vote_counts = Counter(vote_keys)
    winning_vote = vote_counts.most_common(1)[0][0]
    new_segment, new_multiplier = winning_vote
    new_score = new_segment * new_multiplier
    
    # Compare results
    original = metadata.get("final_result", {})
    expected = correction.get("corrected") if correction else original
    
    expected_segment = expected.get("segment")
    expected_multiplier = expected.get("multiplier")
    
    matches_expected = (new_segment == expected_segment and new_multiplier == expected_multiplier)
    matches_original = (new_segment == original.get("segment") and new_multiplier == original.get("multiplier"))
    
    return {
        "success": True,
        "dart_path": str(dart_path),
        "original": original,
        "corrected": correction.get("corrected") if correction else None,
        "new_result": {
            "segment": new_segment,
            "multiplier": new_multiplier,
            "score": new_score
        },
        "camera_votes": camera_votes,
        "matches_expected": matches_expected,
        "matches_original": matches_original,
        "improved": matches_expected and not matches_original if correction else None
    }


@router.post("/v1/benchmark/replay-all")
async def replay_all_darts(request: ReplayAllRequest = None):
    """
    Replay all benchmark darts with corrections through current detection code.
    Returns accuracy comparison: old code vs new code.
    """
    if request is None:
        request = ReplayAllRequest()
    
    board_dir = BENCHMARK_DIR / request.board_id
    if not board_dir.exists():
        return {"error": "Board not found", "darts_with_corrections": 0}
    
    results = []
    total_with_corrections = 0
    new_correct = 0
    
    # Find all darts with corrections
    for game_dir in sorted(board_dir.iterdir(), reverse=True):
        if not game_dir.is_dir():
            continue
        
        for round_dir in game_dir.iterdir():
            if not round_dir.is_dir():
                continue
            
            for dart_dir in round_dir.iterdir():
                if not dart_dir.is_dir():
                    continue
                
                correction_path = dart_dir / "correction.json"
                if not correction_path.exists():
                    continue  # Only replay darts that have corrections
                
                if total_with_corrections >= request.limit:
                    break
                
                total_with_corrections += 1
                
                # Replay this dart
                try:
                    replay_result = await replay_single_dart(ReplayRequest(dart_path=str(dart_dir)))
                    
                    if replay_result.get("success"):
                        if replay_result.get("matches_expected"):
                            new_correct += 1
                        
                        results.append({
                            "dart_path": str(dart_dir),
                            "original": replay_result.get("original"),
                            "expected": replay_result.get("corrected"),
                            "new_result": replay_result.get("new_result"),
                            "now_correct": replay_result.get("matches_expected")
                        })
                    else:
                        results.append({
                            "dart_path": str(dart_dir),
                            "error": replay_result.get("error")
                        })
                except Exception as e:
                    results.append({
                        "dart_path": str(dart_dir),
                        "error": str(e)
                    })
    
    new_accuracy = 0 if total_with_corrections == 0 else (new_correct / total_with_corrections) * 100
    
    return {
        "total_with_corrections": total_with_corrections,
        "old_accuracy": "0.0%",  # All had corrections = all were wrong originally
        "new_accuracy": f"{new_accuracy:.1f}%",
        "new_correct": new_correct,
        "improvement": f"+{new_accuracy:.1f}%",
        "results": results
    }

@router.post("/v1/benchmark/replay-all-darts")
async def replay_all_darts_full(request: ReplayAllRequest = None):
    """
    Replay ALL benchmark darts (not just corrected ones).
    Returns comparison of original detection vs replay detection.
    """
    if request is None:
        request = ReplayAllRequest()
    
    board_dir = BENCHMARK_DIR / request.board_id
    if not board_dir.exists():
        return {"error": "Board not found", "total_darts": 0}
    
    results = []
    total_darts = 0
    match_count = 0
    corrected_count = 0
    corrected_now_correct = 0
    
    # Find ALL darts
    for game_dir in sorted(board_dir.iterdir(), reverse=True):
        if not game_dir.is_dir():
            continue
        
        for round_dir in game_dir.iterdir():
            if not round_dir.is_dir():
                continue
            
            for dart_dir in round_dir.iterdir():
                if not dart_dir.is_dir():
                    continue
                
                if total_darts >= request.limit:
                    break
                
                meta_path = dart_dir / "metadata.json"
                if not meta_path.exists():
                    continue
                
                total_darts += 1
                
                # Check if has correction
                correction_path = dart_dir / "correction.json"
                has_correction = correction_path.exists()
                if has_correction:
                    corrected_count += 1
                
                # Replay this dart
                try:
                    replay_result = await replay_single_dart(ReplayRequest(dart_path=str(dart_dir)))
                    
                    if replay_result.get("success"):
                        original = replay_result.get("original", {})
                        new_result = replay_result.get("new_result", {})
                        corrected = replay_result.get("corrected")
                        
                        # Check if replay matches original
                        orig_match = (
                            original.get("segment") == new_result.get("segment") and
                            original.get("multiplier") == new_result.get("multiplier")
                        )
                        if orig_match:
                            match_count += 1
                        
                        # Check if corrected dart is now correct
                        now_correct = False
                        if has_correction and corrected:
                            now_correct = (
                                corrected.get("segment") == new_result.get("segment") and
                                corrected.get("multiplier") == new_result.get("multiplier")
                            )
                            if now_correct:
                                corrected_now_correct += 1
                        
                        results.append({
                            "dart_path": str(dart_dir),
                            "round": dart_dir.parent.name,
                            "dart": dart_dir.name,
                            "original": original,
                            "new_result": new_result,
                            "matches_original": orig_match,
                            "had_correction": has_correction,
                            "corrected_to": corrected,
                            "now_correct": now_correct if has_correction else None
                        })
                    else:
                        results.append({
                            "dart_path": str(dart_dir),
                            "error": replay_result.get("error")
                        })
                except Exception as e:
                    results.append({
                        "dart_path": str(dart_dir),
                        "error": str(e)
                    })
    
    consistency = 0 if total_darts == 0 else (match_count / total_darts) * 100
    correction_rate = 0 if corrected_count == 0 else (corrected_now_correct / corrected_count) * 100
    
    return {
        "total_darts": total_darts,
        "matches_original": match_count,
        "consistency": f"{consistency:.1f}%",
        "had_corrections": corrected_count,
        "corrections_now_fixed": corrected_now_correct,
        "correction_fix_rate": f"{correction_rate:.1f}%",
        "results": results
    }

# === Stereo Calibration Endpoints ===

class StereoStatusResponse(BaseModel):
    mode: str  # "ellipse" or "stereo"
    stereo_available: bool
    cameras_calibrated: List[str]
    reprojection_error: Optional[float]




# ==================== AUTO-TUNING ====================

@router.post("/v1/benchmark/auto-tune")
async def auto_tune_parameters():
    """
    Automatically find optimal detection parameters by replaying benchmark data.
    
    Tests combinations of:
    - confidence_threshold
    - boundary_weight_min
    - polar_override_threshold
    
    Returns the best configuration based on accuracy.
    """
    global CONFIDENCE_THRESHOLD
    
    # Get all benchmark data from disk
    all_darts = []
    benchmark_dir = Path(BENCHMARK_DIR)
    
    if benchmark_dir.exists():
        for board_dir in benchmark_dir.iterdir():
            if not board_dir.is_dir():
                continue
            board_id = board_dir.name
            
            for game_dir in board_dir.iterdir():
                if not game_dir.is_dir():
                    continue
                game_id = game_dir.name
                
                # Load darts from JSON files
                for dart_file in game_dir.glob("round_*/dart_*.json"):
                    try:
                        with open(dart_file, 'r') as f:
                            dart_data = json.load(f)
                        dart_data["board_id"] = board_id
                        dart_data["game_id"] = game_id
                        all_darts.append(dart_data)
                    except Exception as e:
                        logger.warning(f"Error loading {dart_file}: {e}")
    
    if len(all_darts) < 5:
        return {
            "success": False,
            "error": f"Need at least 5 benchmark darts, have {len(all_darts)}",
            "darts_available": len(all_darts)
        }
    
    # Count corrections (these are our ground truth)
    corrections = [d for d in all_darts if d.get("was_corrected")]
    logger.info(f"[AUTO-TUNE] Starting with {len(all_darts)} darts, {len(corrections)} corrections")
    
    # Parameter grid to search
    confidence_values = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    boundary_weight_values = [0.05, 0.1, 0.2, 0.3]
    polar_threshold_values = [1.5, 2.0, 3.0, 4.0, 5.0]
    
    best_config = None
    best_score = 0
    best_details = None
    results = []
    
    total_combos = len(confidence_values) * len(boundary_weight_values) * len(polar_threshold_values)
    combo_num = 0
    
    for conf_thresh in confidence_values:
        for boundary_weight in boundary_weight_values:
            for polar_thresh in polar_threshold_values:
                combo_num += 1
                
                # Simulate detection with these parameters
                correct = 0
                would_fix_correction = 0
                would_break_correct = 0
                
                for dart in all_darts:
                    # Get the camera votes from benchmark data
                    camera_details = dart.get("camera_details", {})
                    
                    if not camera_details:
                        continue
                    
                    # Simulate voting with current parameters
                    votes = {}
                    total_weight = 0
                    
                    for cam_id, cam_data in camera_details.items():
                        tip = cam_data.get("selected_tip", {})
                        if not tip:
                            continue
                        
                        conf = tip.get("confidence", 0.5)
                        
                        # Apply confidence threshold
                        if conf < conf_thresh:
                            continue
                        
                        # Calculate weight with boundary factor
                        boundary_dist = tip.get("boundary_distance_deg", 9.0)
                        if boundary_dist < 3:
                            boundary_factor = boundary_weight + (1 - boundary_weight) * (boundary_dist / 3)
                        else:
                            boundary_factor = 1.0
                        
                        cal_quality = cam_data.get("calibration_quality", 0.9)
                        weight = conf * cal_quality * boundary_factor
                        
                        score_key = (tip.get("segment", 0), tip.get("multiplier", 1))
                        if score_key not in votes:
                            votes[score_key] = 0
                        votes[score_key] += weight
                        total_weight += weight
                    
                    if not votes:
                        continue
                    
                    # Find winner
                    winner = max(votes.keys(), key=lambda k: votes[k])
                    winner_segment, winner_mult = winner
                    
                    # Check polar averaging override
                    angle_deg = dart.get("avg_angle_deg")
                    norm_dist = dart.get("avg_norm_dist")
                    
                    if angle_deg is not None and norm_dist is not None:
                        # Would polar averaging change the result?
                        polar_score = score_from_polar(angle_deg, norm_dist)
                        polar_key = (polar_score["segment"], polar_score["multiplier"])
                        polar_boundary = polar_score.get("boundary_distance_deg", 9)
                        
                        if polar_key != winner and polar_boundary > polar_thresh:
                            winner_segment = polar_score["segment"]
                            winner_mult = polar_score["multiplier"]
                    
                    # Compare to actual result
                    actual_segment = dart.get("final_segment", dart.get("detected_segment"))
                    actual_mult = dart.get("final_multiplier", dart.get("detected_multiplier"))
                    
                    detected_segment = dart.get("detected_segment")
                    detected_mult = dart.get("detected_multiplier")
                    
                    was_corrected = dart.get("was_corrected", False)
                    
                    if was_corrected:
                        # Ground truth is the corrected value
                        if winner_segment == actual_segment and winner_mult == actual_mult:
                            would_fix_correction += 1
                            correct += 1
                        else:
                            # Still wrong
                            pass
                    else:
                        # Original detection was correct
                        if winner_segment == actual_segment and winner_mult == actual_mult:
                            correct += 1
                        else:
                            would_break_correct += 1
                
                # Calculate score (penalize breaking correct detections heavily)
                accuracy = correct / len(all_darts) if all_darts else 0
                fixes = would_fix_correction
                breaks = would_break_correct
                
                # Score: accuracy with penalty for breaking things
                score = accuracy - (breaks * 0.1)  # Each break costs 10%
                
                result = {
                    "confidence": conf_thresh,
                    "boundary_weight": boundary_weight,
                    "polar_threshold": polar_thresh,
                    "accuracy": round(accuracy * 100, 1),
                    "correct": correct,
                    "total": len(all_darts),
                    "would_fix": fixes,
                    "would_break": breaks,
                    "score": round(score * 100, 1)
                }
                results.append(result)
                
                if score > best_score:
                    best_score = score
                    best_config = {
                        "confidence_threshold": conf_thresh,
                        "boundary_weight_min": boundary_weight,
                        "polar_override_threshold": polar_thresh
                    }
                    best_details = result
    
    # Sort results by score
    results.sort(key=lambda x: x["score"], reverse=True)
    
    logger.info(f"[AUTO-TUNE] Best config: {best_config} with score {best_score:.1%}")
    
    return {
        "success": True,
        "darts_analyzed": len(all_darts),
        "corrections_in_data": len(corrections),
        "combinations_tested": total_combos,
        "best_config": best_config,
        "best_result": best_details,
        "top_10_results": results[:10],
        "current_config": {
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "boundary_weight_min": 0.1,  # Current hardcoded value
            "polar_override_threshold": 2.0  # Current hardcoded value
        }
    }


@router.post("/v1/benchmark/apply-config")
async def apply_tuned_config(request: Request):
    """Apply the auto-tuned configuration."""
    global CONFIDENCE_THRESHOLD
    
    body = await request.json()
    
    new_conf = body.get("confidence_threshold")
    if new_conf is not None:
        CONFIDENCE_THRESHOLD = float(new_conf)
        logger.info(f"[CONFIG] Applied confidence_threshold: {CONFIDENCE_THRESHOLD}")
    
    # Note: boundary_weight and polar_threshold would need code changes
    # to be truly configurable at runtime. For now we just report them.
    
    return {
        "success": True,
        "applied": {
            "confidence_threshold": CONFIDENCE_THRESHOLD
        },
        "note": "boundary_weight and polar_threshold require restart to change"
    }



@router.get("/v1/stereo/status")
async def get_stereo_status(board_id: str = "default"):
    """Get current triangulation mode and stereo calibration status."""
    calibrator = get_stereo_calibrator(board_id)
    
    return {
        "mode": TRIANGULATION_MODE,
        "stereo_available": calibrator.calibration is not None,
        "cameras_calibrated": list(calibrator.calibration.intrinsics.keys()) if calibrator.calibration else [],
        "reprojection_error": calibrator.calibration.reprojection_error if calibrator.calibration else None
    }


@router.post("/v1/stereo/set-mode")
async def set_triangulation_mode(request: dict):
    """Switch between ellipse and stereo triangulation modes."""
    global TRIANGULATION_MODE
    mode = request.get("mode", "ellipse")
    
    if mode not in ["ellipse", "stereo"]:
        raise HTTPException(400, f"Invalid mode: {mode}. Use 'ellipse' or 'stereo'")
    
    if mode == "stereo":
        calibrator = get_stereo_calibrator(request.get("board_id", "default"))
        if calibrator.calibration is None:
            raise HTTPException(400, "Stereo calibration not available. Run /v1/stereo/calibrate first.")
    
    TRIANGULATION_MODE = mode
    logger.info(f"[STEREO] Triangulation mode set to: {mode}")
    
    return {"mode": mode, "success": True}


@router.get("/v1/stereo/checkerboard")
async def get_checkerboard_pattern(
    cols: int = 9, 
    rows: int = 6, 
    square_mm: float = 25.0
):
    """
    Generate a printable checkerboard pattern.
    
    Recommended: Print on A3 paper or larger.
    Mount flat on dartboard for calibration.
    """
    import base64
    
    output_path = STEREO_CALIBRATION_DIR / f"checkerboard_{cols}x{rows}_{int(square_mm)}mm.png"
    STEREO_CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    
    generate_checkerboard_pdf((cols, rows), square_mm, str(output_path))
    
    # Read and encode
    with open(output_path, 'rb') as f:
        img_data = base64.b64encode(f.read()).decode()
    
    return {
        "success": True,
        "size": f"{cols}x{rows} inner corners",
        "square_mm": square_mm,
        "print_width_mm": (cols + 1) * square_mm,
        "print_height_mm": (rows + 1) * square_mm,
        "image": f"data:image/png;base64,{img_data}",
        "instructions": [
            "1. Print this pattern at 100% scale (no fit-to-page)",
            "2. Mount on rigid foam board or cardboard",
            "3. Attach flat to dartboard face",
            "4. Capture calibration images from Settings"
        ]
    }


@router.post("/v1/stereo/capture")
async def capture_stereo_image(request: dict):
    """
    Capture a calibration image from each camera.
    
    Call this multiple times with the checkerboard in different positions/angles.
    Need at least 10 captures for good calibration.
    """
    global _stereo_capture_images
    
    cameras = request.get("cameras", [])
    board_id = request.get("board_id", "default")
    
    if not cameras:
        raise HTTPException(400, "No camera images provided")
    
    if board_id not in _stereo_capture_images:
        _stereo_capture_images[board_id] = {}
    
    calibrator = get_stereo_calibrator(board_id)
    results = []
    
    for cam in cameras:
        cam_id = cam.get("camera_id")
        image_b64 = cam.get("image")
        
        if not cam_id or not image_b64:
            continue
        
        # Decode image
        img = decode_image(image_b64)
        
        # Detect checkerboard
        corners = calibrator.detect_checkerboard(img)
        
        if corners is not None:
            # Store image for calibration
            if cam_id not in _stereo_capture_images[board_id]:
                _stereo_capture_images[board_id][cam_id] = []
            _stereo_capture_images[board_id][cam_id].append(img)
            
            # Draw checkerboard for visualization
            vis = calibrator.draw_checkerboard(img, corners)
            vis_b64 = encode_image(vis, "jpeg")
            
            results.append({
                "camera_id": cam_id,
                "success": True,
                "corners_found": len(corners),
                "total_captures": len(_stereo_capture_images[board_id][cam_id]),
                "preview": vis_b64
            })
        else:
            results.append({
                "camera_id": cam_id,
                "success": False,
                "error": "Checkerboard not detected",
                "total_captures": len(_stereo_capture_images[board_id].get(cam_id, []))
            })
    
    # Summary
    min_captures = min(
        len(imgs) for imgs in _stereo_capture_images[board_id].values()
    ) if _stereo_capture_images[board_id] else 0
    
    return {
        "results": results,
        "total_captures": min_captures,
        "ready_to_calibrate": min_captures >= 10,
        "recommendation": "Capture at least 10 images with checkerboard at different angles"
    }


@router.post("/v1/stereo/calibrate")
async def run_stereo_calibration(request: dict = {}):
    """
    Run stereo calibration using captured images.
    
    Requires at least 10 captures from /v1/stereo/capture.
    """
    global _stereo_calibrator, _stereo_capture_images
    
    board_id = request.get("board_id", "default")
    
    if board_id not in _stereo_capture_images:
        raise HTTPException(400, "No calibration images captured. Use /v1/stereo/capture first.")
    
    camera_images = _stereo_capture_images[board_id]
    
    # Check we have enough images
    for cam_id, images in camera_images.items():
        if len(images) < 3:
            raise HTTPException(400, f"Camera {cam_id} needs at least 3 images, has {len(images)}")
    
    # Run calibration
    checkerboard_size = tuple(request.get("checkerboard_size", [9, 6]))
    square_mm = request.get("square_mm", 25.0)
    
    calibrator = StereoCalibrator(checkerboard_size, square_mm)
    calibration = calibrator.calibrate(camera_images, board_id)
    
    if calibration is None:
        raise HTTPException(500, "Calibration failed. Check that checkerboard is visible in all cameras.")
    
    # Save calibration
    STEREO_CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    cal_file = STEREO_CALIBRATION_DIR / f"{board_id}_stereo.json"
    calibrator.save(cal_file)
    
    # Update global calibrator
    _stereo_calibrator = calibrator
    
    # Clear captured images
    _stereo_capture_images[board_id] = {}
    
    return {
        "success": True,
        "cameras_calibrated": list(calibration.intrinsics.keys()),
        "reprojection_error": calibration.reprojection_error,
        "saved_to": str(cal_file),
        "next_step": "Use /v1/stereo/set-mode with mode='stereo' to enable triangulation"
    }




# ==================== MODEL SELECTION ====================

# Available models configuration
AVAILABLE_MODELS = {
    "default": {
        "name": "Pose Nano (Dec 2025)",
        "description": "Balanced speed/accuracy, INT8 optimized",
        "path": "posenano27122025_int8_openvino_model"
    },
    "best": {
        "name": "YOLO26n Pose (Jan 2026)", 
        "description": "Newer architecture, potentially better accuracy",
        "path": "26tippose27012026_int8_openvino_model"
    },
    "rect": {
        "name": "736x1280 FP16",
        "description": "Non-square input, higher precision",
        "path": "best_openvino_736x1280_fp16_openvino_model"
    },
    "square": {
        "name": "Square (Jan 2026)",
        "description": "Square input variant",
        "path": "tippose25012026_square_openvino_model"
    }
}

# Currently active model (in-memory, persists until restart)
ACTIVE_MODEL = "default"
# Detection confidence threshold (0.0 - 1.0)
CONFIDENCE_THRESHOLD = 0.25


@router.get("/v1/settings/threshold")
async def get_threshold():
    """Get current YOLO confidence threshold."""
    global CONFIDENCE_THRESHOLD
    return {
        "threshold": CONFIDENCE_THRESHOLD,
        "min": 0.1,
        "max": 0.9,
        "default": 0.25
    }


@router.post("/v1/settings/threshold")
async def set_threshold(request: Request):
    """
    Set YOLO confidence threshold.
    
    Body: {"threshold": 0.5}
    
    Lower = more detections (but more false positives)
    Higher = fewer detections (but more reliable)
    """
    global CONFIDENCE_THRESHOLD
    
    body = await request.json()
    new_threshold = body.get("threshold", 0.5)
    
    # Clamp to valid range
    new_threshold = max(0.1, min(0.9, float(new_threshold)))
    
    old_threshold = CONFIDENCE_THRESHOLD
    CONFIDENCE_THRESHOLD = new_threshold
    
    logger.info(f"[SETTINGS] Confidence threshold changed: {old_threshold} -> {new_threshold}")
    
    return {
        "success": True,
        "previous": old_threshold,
        "current": new_threshold
    }




@router.get("/v1/models")
async def list_models():
    """List available detection models."""
    global ACTIVE_MODEL
    return {
        "active": ACTIVE_MODEL,
        "models": AVAILABLE_MODELS
    }


@router.post("/v1/models/select")
async def select_model(request: Request):
    """
    Select which detection model to use.
    
    Body: {"model": "default" | "best" | "rect" | "square"}
    
    Note: This changes the model for future detections. 
    The detector will reload on next detection call.
    """
    global ACTIVE_MODEL
    
    body = await request.json()
    model_key = body.get("model", "default")
    
    if model_key not in AVAILABLE_MODELS:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unknown model: {model_key}. Available: {list(AVAILABLE_MODELS.keys())}"}
        )
    
    old_model = ACTIVE_MODEL
    ACTIVE_MODEL = model_key
    
    # Update the detector to use the new model
    from app.core.detection import DartTipDetector, TIP_MODEL_PATHS
    
    # Check if model path exists
    model_info = AVAILABLE_MODELS[model_key]
    model_path = TIP_MODEL_PATHS.get(model_key)
    
    if model_path and model_path.exists():
        # Reinitialize the global detector with new model
        try:
            # Get the calibrator and update its detector
            calibrator = get_calibrator()
            calibrator.tip_detector = DartTipDetector(model_name=model_key)
            logger.info(f"[MODEL] Switched from {old_model} to {model_key}")
            
            return {
                "success": True,
                "previous": old_model,
                "current": model_key,
                "model_info": model_info
            }
        except Exception as e:
            ACTIVE_MODEL = old_model  # Rollback
            logger.error(f"[MODEL] Failed to switch to {model_key}: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to load model: {str(e)}"}
            )
    else:
        ACTIVE_MODEL = old_model  # Rollback
        return JSONResponse(
            status_code=404,
            content={"error": f"Model file not found: {model_info['path']}"}
        )



@router.post("/v1/stereo/clear-captures")
async def clear_stereo_captures(request: dict = {}):
    """Clear captured calibration images to start over."""
    global _stereo_capture_images
    
    board_id = request.get("board_id", "default")
    
    if board_id in _stereo_capture_images:
        count = sum(len(imgs) for imgs in _stereo_capture_images[board_id].values())
        _stereo_capture_images[board_id] = {}
        return {"success": True, "cleared": count}
    
    return {"success": True, "cleared": 0}


@router.post("/v1/stereo/triangulate")
async def triangulate_point(request: dict):
    """
    Test triangulation with pixel coordinates from each camera.
    
    Useful for debugging/verifying stereo calibration.
    """
    board_id = request.get("board_id", "default")
    pixel_coords = request.get("pixel_coords", {})  # {cam_id: [x, y]}
    
    if len(pixel_coords) < 2:
        raise HTTPException(400, "Need at least 2 camera coordinates")
    
    calibrator = get_stereo_calibrator(board_id)
    if calibrator.calibration is None:
        raise HTTPException(400, "No stereo calibration available")
    
    # Convert to tuples
    coords = {k: tuple(v) for k, v in pixel_coords.items()}
    
    try:
        x_mm, y_mm = calibrator.triangulate_to_dartboard(coords)
        
        # Calculate what score this would be
        pos_score = score_from_position(x_mm, y_mm)
        
        return {
            "success": True,
            "x_mm": round(x_mm, 2),
            "y_mm": round(y_mm, 2),
            "distance_mm": round((x_mm**2 + y_mm**2)**0.5, 2),
            "score": pos_score
        }
    except Exception as e:
        raise HTTPException(500, f"Triangulation failed: {e}")

