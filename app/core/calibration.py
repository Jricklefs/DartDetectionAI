"""
Dartboard Calibration Module

Uses YOLO to detect calibration points (wire intersections), 
then fits ellipses to determine the dartboard geometry.

Key insight: The detected calibration points ARE the segment boundary intersections.
Cluster them by angle to find segment boundaries, use line-ellipse intersection
for perspective-correct drawing.
"""
import cv2
import numpy as np
import math
import base64
import os
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
from pathlib import Path

from app.core.geometry import (
    DARTBOARD_SEGMENTS,
    BULL_RADIUS_MM,
    OUTER_BULL_RADIUS_MM,
    TRIPLE_INNER_RADIUS_MM,
    TRIPLE_OUTER_RADIUS_MM,
    DOUBLE_INNER_RADIUS_MM,
    DOUBLE_OUTER_RADIUS_MM,
    DARTBOARD_DIAMETER_MM,
    SEGMENT_ANGLE_OFFSET,
)
from app.core.scoring import scoring_system
from app.core.polygon_calibration import (
    generate_polygon_calibration_from_yolo,
    set_polygon_calibration,
    PolygonCalibration,
)
from app.models.schemas import CameraCalibrationResult, DetectResponse, DartScore, DartPosition

# Get the models directory
MODELS_DIR = Path(__file__).parent.parent.parent / "models"

# Available calibration models
CALIBRATION_MODELS = {
    "default": {
        "name": "Dartboard 1280 INT8",
        "description": "Original calibration model, proven accuracy",
        "path": MODELS_DIR / "dartboard1280imgz_int8_openvino_model",
        "imgsz": (1280, 1280),
        # Class mapping: ID -> key
        "class_map": {0: 'twenty', 2: 'bull', 3: 'cal', 4: 'cal1', 5: 'cal2', 6: 'cal3'}
    },
    "11m": {
        "name": "11M Dartboard Model",
        "description": "Larger model with segment detection (20, 3, 44)",
        "path": MODELS_DIR / "11mdartboard30-1-26_openvino_model",
        "imgsz": (736, 1280),
        # Class mapping: ID -> key (different from default!)
        "class_map": {0: 'twenty', 1: 'three', 2: 'fortyfour', 3: 'bull', 4: 'cal', 5: 'cal1', 6: 'cal2', 7: 'cal3'}
    }
}

# Get current class map
def get_calibration_class_map():
    """Get class mapping for current calibration model."""
    return CALIBRATION_MODELS[_active_calibration_model].get("class_map", {})

# Active calibration model (can be switched via API)
_active_calibration_model = "default"

def get_active_calibration_model():
    """Get the currently active calibration model name."""
    return _active_calibration_model

# Global reference to the calibration detector (set by DartboardCalibrator)
_calibration_detector_instance = None

def set_calibration_detector_instance(detector):
    """Set the global calibration detector instance for model reloading."""
    global _calibration_detector_instance
    _calibration_detector_instance = detector

def set_active_calibration_model(model_name: str) -> bool:
    """Set the active calibration model. Returns True if successful."""
    global _active_calibration_model, CALIBRATION_MODEL_PATH, CALIBRATION_IMAGE_SIZE
    if model_name not in CALIBRATION_MODELS:
        return False
    _active_calibration_model = model_name
    model_info = CALIBRATION_MODELS[model_name]
    CALIBRATION_MODEL_PATH = model_info["path"]
    CALIBRATION_IMAGE_SIZE = model_info["imgsz"]
    print(f"[CALIBRATION] Switched to model: {model_name} ({model_info['name']})")
    
    # Reload the detector if it exists
    if _calibration_detector_instance is not None:
        _calibration_detector_instance.reload_model()
    
    return True

def get_calibration_models():
    """Get list of available calibration models."""
    return {
        "active": _active_calibration_model,
        "models": {k: {"name": v["name"], "description": v["description"]} 
                   for k, v in CALIBRATION_MODELS.items()}
    }

# Default calibration model path (for backward compatibility)
CALIBRATION_MODEL_PATH = CALIBRATION_MODELS["default"]["path"]
CALIBRATION_IMAGE_SIZE = CALIBRATION_MODELS["default"]["imgsz"]


def decode_image(image_base64: str) -> np.ndarray:
    """Decode base64 image to OpenCV format."""
    if ',' in image_base64:
        image_base64 = image_base64.split(',')[1]
    
    image_data = base64.b64decode(image_base64)
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Failed to decode image")
    
    return image


def encode_image(image: np.ndarray, format: str = "png") -> str:
    """Encode OpenCV image to base64."""
    success, buffer = cv2.imencode(f'.{format}', image)
    if not success:
        raise ValueError("Failed to encode image")
    
    return base64.b64encode(buffer).decode('utf-8')


def line_ellipse_intersection(center: Tuple[float, float], direction: Tuple[float, float], ellipse: Tuple) -> Tuple[int, int]:
    """
    Find intersection of a line from center with an ellipse, accounting for ellipse rotation.
    
    Args:
        center: (cx, cy) - line origin
        direction: (dx, dy) - unit direction vector
        ellipse: ((cx, cy), (w, h), angle) - OpenCV ellipse format
        
    Returns:
        (x, y) - intersection point
    """
    (ecx, ecy), (w, h), angle = ellipse
    dx, dy = direction
    
    angle_rad = np.radians(-angle)
    cos_a = float(np.cos(angle_rad))
    sin_a = float(np.sin(angle_rad))
    
    x0 = center[0] - ecx
    y0 = center[1] - ecy
    
    x0_rot = x0 * cos_a - y0 * sin_a
    y0_rot = x0 * sin_a + y0 * cos_a
    dx_rot = dx * cos_a - dy * sin_a
    dy_rot = dx * sin_a + dy * cos_a
    
    a = w / 2
    b = h / 2
    
    if a <= 1e-6 or b <= 1e-6:
        return (int(center[0] + dx * 1000), int(center[1] + dy * 1000))
    
    A = (dx_rot / a) ** 2 + (dy_rot / b) ** 2
    B = 2 * (x0_rot * dx_rot / a ** 2 + y0_rot * dy_rot / b ** 2)
    C = (x0_rot / a) ** 2 + (y0_rot / b) ** 2 - 1
    
    disc = B ** 2 - 4 * A * C
    
    if disc < 0 or abs(A) < 1e-12:
        return (int(center[0] + dx * 1000), int(center[1] + dy * 1000))
    
    t1 = (-B + np.sqrt(disc)) / (2 * A)
    t2 = (-B - np.sqrt(disc)) / (2 * A)
    t = float(max(t1, t2))
    
    x_int_rot = x0_rot + t * dx_rot
    y_int_rot = y0_rot + t * dy_rot
    
    angle_rad = np.radians(angle)
    cos_a = float(np.cos(angle_rad))
    sin_a = float(np.sin(angle_rad))
    
    x_int = x_int_rot * cos_a - y_int_rot * sin_a
    y_int = x_int_rot * sin_a + y_int_rot * cos_a
    
    return (int(x_int + ecx), int(y_int + ecy))


@dataclass
class EllipseCalibration:
    """Stores the calibration ellipses and parameters."""
    center: Tuple[float, float]
    outer_double_ellipse: Optional[Tuple] = None  # ((cx,cy), (w,h), angle)
    inner_double_ellipse: Optional[Tuple] = None
    outer_triple_ellipse: Optional[Tuple] = None
    inner_triple_ellipse: Optional[Tuple] = None
    bull_ellipse: Optional[Tuple] = None
    bullseye_ellipse: Optional[Tuple] = None
    segment_angles: List[float] = field(default_factory=list)
    rotation_offset_deg: float = 0.0
    segment_20_index: int = 0


class YOLOCalibrationDetector:
    """
    Uses YOLO to detect dartboard calibration points.
    
    Detects:
    - Class 0: '20' - Segment 20 marker (for orientation)
    - Class 2: 'bull' - Bullseye center
    - Class 3: 'cal' - Outer double ring points
    - Class 4: 'cal1' - Outer triple ring points
    - Class 5: 'cal2' - Inner triple ring points
    - Class 6: 'cal3' - Inner double ring points
    """
    
    def __init__(self):
        self.model = None
        self.is_initialized = False
        self._load_model()
    
    def _load_model(self):
        """Load the YOLO calibration model."""
        try:
            from ultralytics import YOLO
            
            model_path = CALIBRATION_MODEL_PATH
            if not model_path.exists():
                print(f"Warning: Calibration model not found at {model_path}")
                return
            
            self.model = YOLO(str(model_path), task="detect")
            self.is_initialized = True
            print(f"Loaded calibration model from {model_path}")
            
        except ImportError:
            print("Warning: ultralytics not installed. YOLO detection disabled.")
        except Exception as e:
            print(f"Warning: Failed to load YOLO model: {e}")
    
    def reload_model(self):
        """Reload the model (call after switching calibration model)."""
        self.is_initialized = False
        self.model = None
        self._load_model()
        print(f"[CAL] Reloaded calibration model: {CALIBRATION_MODEL_PATH}")
    
    def detect_calibration_points(
        self, 
        image: np.ndarray,
        confidence_threshold: float = 0.5
    ) -> Dict[str, List[Tuple[float, float, float]]]:
        """
        Detect calibration points in the image.
        
        Returns dict with keys for each class containing lists of (x, y, confidence).
        Uses dynamic class mapping from current calibration model config.
        """
        if not self.is_initialized or self.model is None:
            return {'cal': [], 'cal1': [], 'cal2': [], 'cal3': [], 'bull': [], 'twenty': []}
        
        # Run inference
        results = self.model(image, imgsz=CALIBRATION_IMAGE_SIZE, conf=confidence_threshold, verbose=False)
        
        # Get class mapping for current model
        class_map = get_calibration_class_map()
        
        # Initialize points dict with all possible keys
        points = {
            'twenty': [],   # Segment 20 marker
            'three': [],    # Segment 3 marker (11M model)
            'fortyfour': [],# Segment 44 marker (11M model)
            'bull': [],     # Bullseye
            'cal': [],      # Outer double ring (board edge)
            'cal1': [],     # Inner double ring
            'cal2': [],     # Outer triple ring
            'cal3': [],     # Inner triple ring
        }
        
        for result in results:
            if result.boxes is None:
                continue
                
            boxes = result.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                
                # Get center of bounding box
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                cx = float((x1 + x2) / 2)
                cy = float((y1 + y2) / 2)
                
                # Use dynamic class mapping
                key = class_map.get(cls_id)
                if key and key in points:
                    points[key].append((cx, cy, conf))
        
        return points


def fit_ellipse_from_points(points: List[Tuple[float, float, float]], remove_outliers: bool = True) -> Optional[Tuple]:
    """
    Fit an ellipse to a set of points.
    
    Args:
        points: List of (x, y, confidence) tuples
        remove_outliers: Whether to remove outlier points
        
    Returns:
        OpenCV ellipse tuple ((cx, cy), (w, h), angle) or None
    """
    if len(points) < 5:
        return None
    
    # Extract x, y coordinates
    pts = np.array([(p[0], p[1]) for p in points], dtype=np.float32)
    
    if remove_outliers and len(pts) > 10:
        # Simple outlier removal based on distance from centroid
        centroid = np.mean(pts, axis=0)
        distances = np.linalg.norm(pts - centroid, axis=1)
        median_dist = np.median(distances)
        mask = distances < median_dist * 2
        pts = pts[mask]
        
        if len(pts) < 5:
            return None
    
    try:
        ellipse = cv2.fitEllipse(pts)
        return ellipse
    except cv2.error:
        return None


def cluster_angles_to_segment_boundaries(
    all_points: List[Tuple[float, float, float]], 
    center: Tuple[float, float],
    angle_tolerance_deg: float = 5.0
) -> List[float]:
    """
    Cluster calibration points by angle from center to find segment boundaries.
    
    Points on the same radial line (segment boundary) will cluster together.
    
    Args:
        all_points: All calibration points from all rings
        center: Dartboard center (cx, cy)
        angle_tolerance_deg: Tolerance for clustering (degrees)
        
    Returns:
        Sorted list of segment boundary angles in radians
    """
    # Calculate angle from center to each point
    point_angles = []
    for p in all_points:
        dx = p[0] - center[0]
        dy = p[1] - center[1]
        angle = math.atan2(dy, dx)
        point_angles.append(angle)
    
    point_angles.sort()
    
    angle_tolerance = math.radians(angle_tolerance_deg)
    segment_angles = []
    used = set()
    
    for i, ang in enumerate(point_angles):
        if i in used:
            continue
        
        cluster = [ang]
        used.add(i)
        
        for j, other_ang in enumerate(point_angles):
            if j in used:
                continue
            
            diff = abs(ang - other_ang)
            if diff > math.pi:
                diff = 2 * math.pi - diff
            
            if diff < angle_tolerance:
                cluster.append(other_ang)
                used.add(j)
        
        # Only use if multiple points confirm it
        if len(cluster) >= 2:
            sin_sum = sum(math.sin(a) for a in cluster)
            cos_sum = sum(math.cos(a) for a in cluster)
            avg_angle = math.atan2(sin_sum, cos_sum)
            segment_angles.append(avg_angle)
    
    segment_angles.sort()
    return segment_angles


def find_segment_20_index(
    twenty_points: List[Tuple[float, float, float]],
    center: Tuple[float, float],
    segment_angles: List[float]
) -> int:
    """
    Find which segment boundary index contains segment 20.
    
    Args:
        twenty_points: Detected "20" marker points
        center: Dartboard center
        segment_angles: List of segment boundary angles (in original order)
        
    Returns:
        Index into segment_angles where segment 20 starts
    """
    if len(segment_angles) < 20:
        return 0
    
    if twenty_points and len(twenty_points) > 0:
        # Use detected "20" marker position
        # If multiple points, use the one with highest confidence or average them
        if len(twenty_points) > 1:
            # Average all detected 20 marker positions
            avg_x = sum(p[0] for p in twenty_points) / len(twenty_points)
            avg_y = sum(p[1] for p in twenty_points) / len(twenty_points)
            dx = avg_x - center[0]
            dy = avg_y - center[1]
            print(f"[CAL] Averaging {len(twenty_points)} detected 20 markers")
        else:
            dx = twenty_points[0][0] - center[0]
            dy = twenty_points[0][1] - center[1]
        
        angle_to_20 = math.atan2(dy, dx)
        print(f"[CAL] 20 marker at angle {math.degrees(angle_to_20):.1f}A�")
    else:
        # FALLBACK: Assume 20 is at the top of the image (-90A� in image coords)
        angle_to_20 = -math.pi / 2  # -90 degrees
        print(f"[CAL] No 20 marker detected, assuming 20 at top (-90A�)")
    
    # The "20" marker is painted inside segment 20, so find which segment contains it
    # Find index i where segment_angles[i] <= angle_to_20 < segment_angles[i+1]
    # That means segment_angles[i] is the START boundary of the segment containing "20"
    
    # Normalize angle_to_20 to be in the same range as segment_angles
    # segment_angles are typically -pi to pi
    
    best_idx = 0
    
    for i in range(len(segment_angles)):
        a1 = segment_angles[i]
        a2 = segment_angles[(i + 1) % len(segment_angles)]
        
        # Handle wrap-around
        if a2 < a1:
            # Segment spans across -pi/pi boundary
            if angle_to_20 >= a1 or angle_to_20 < a2:
                best_idx = i
                break
        else:
            if a1 <= angle_to_20 < a2:
                best_idx = i
                break
    
    print(f"[CAL] Found 20 in segment starting at index {best_idx}: segment_angles[{best_idx}]={math.degrees(segment_angles[best_idx]):.1f}A�")
    return best_idx


class DartboardCalibrator:
    """
    Calibrates camera views of a dartboard using YOLO detection and ellipse fitting.
    """
    
    def __init__(self):
        self.detector = YOLOCalibrationDetector()
        # Register this detector instance for model reloading
        set_calibration_detector_instance(self.detector)
        # Pre-load tip detector so first dart detection is fast
        try:
            from app.core.detection import DartTipDetector
            print("Loading tip detector...")
            self.tip_detector = DartTipDetector()
            print(f"Tip detector pre-loaded: {self.tip_detector.is_initialized}")
        except Exception as e:
            print(f"Warning: Failed to pre-load tip detector: {e}")
            self.tip_detector = None
    
    def calibrate(
        self, 
        camera_id: str, 
        image_base64: str
    ) -> CameraCalibrationResult:
        """
        Calibrate from a dartboard image using YOLO detection.
        """
        try:
            # Decode image
            image = decode_image(image_base64)
            h, w = image.shape[:2]
            
            # Detect calibration points using YOLO
            points = self.detector.detect_calibration_points(image)
            
            cal_points = points.get('cal', [])
            cal1_points = points.get('cal1', [])
            cal2_points = points.get('cal2', [])
            cal3_points = points.get('cal3', [])
            bull_points = points.get('bull', [])
            twenty_points = points.get('twenty', [])
            
            total_points = len(cal_points) + len(cal1_points) + len(cal2_points) + len(cal3_points)
            print(f"Detected {total_points} calibration points: cal={len(cal_points)}, cal1={len(cal1_points)}, cal2={len(cal2_points)}, cal3={len(cal3_points)}")
            print(f"Bull: {len(bull_points)}, Twenty marker: {len(twenty_points)}")
            
            if len(cal_points) < 8:
                return CameraCalibrationResult(
                    camera_id=camera_id,
                    success=False,
                    error=f"Not enough calibration points detected. Found {len(cal_points)} outer ring points, need at least 8. Ensure full dartboard is visible and well-lit."
                )
            
            # Fit ellipses to the detected points
            # YOLO classes map to physical rings by size (verified by distance measurement):
            # cal (class 3) = 172.6px avg → outer_double (board edge, ~170mm)
            # cal1 (class 4) = 105.8px avg → outer_triple (~107mm)
            # cal2 (class 5) = 161.9px avg → inner_double (~162mm)
            # cal3 (class 6) = 96.0px avg  → inner_triple (~99mm)
            outer_double_ellipse = fit_ellipse_from_points(cal_points)
            outer_triple_ellipse = fit_ellipse_from_points(cal1_points) if len(cal1_points) >= 5 else None
            inner_double_ellipse = fit_ellipse_from_points(cal2_points) if len(cal2_points) >= 5 else None
            inner_triple_ellipse = fit_ellipse_from_points(cal3_points) if len(cal3_points) >= 5 else None
            
            if outer_double_ellipse is None:
                return CameraCalibrationResult(
                    camera_id=camera_id,
                    success=False,
                    error="Could not fit ellipse to detected points."
                )
            
            # Determine center from bull detection or ellipse center
            if bull_points:
                center = (
                    float(np.mean([p[0] for p in bull_points])),
                    float(np.mean([p[1] for p in bull_points]))
                )
            else:
                center = (outer_double_ellipse[0][0], outer_double_ellipse[0][1])
            
            # Cluster all calibration points to find segment boundaries
            all_ring_points = cal_points + cal1_points + cal2_points + cal3_points
            segment_angles = cluster_angles_to_segment_boundaries(all_ring_points, center)
            
            # Find segment 20 index and calculate rotation offset
            segment_20_index = find_segment_20_index(twenty_points, center, segment_angles)
            
            # Calculate rotation_offset_deg - should be the angle to the CENTER of segment 20
            # The simplest and most reliable approach: use the detected "20" marker directly
            # The "20" is painted at the center of segment 20, which is exactly what we need
            if twenty_points and len(twenty_points) > 0:
                # Use detected "20" marker position directly
                if len(twenty_points) > 1:
                    avg_x = sum(p[0] for p in twenty_points) / len(twenty_points)
                    avg_y = sum(p[1] for p in twenty_points) / len(twenty_points)
                    dx = avg_x - center[0]
                    dy = avg_y - center[1]
                else:
                    dx = twenty_points[0][0] - center[0]
                    dy = twenty_points[0][1] - center[1]
                
                angle_to_20_rad = math.atan2(dy, dx)
                rotation_offset_deg = math.degrees(angle_to_20_rad)
                if rotation_offset_deg < 0:
                    rotation_offset_deg += 360
                print(f"[CAL] 20 marker detected at {rotation_offset_deg:.1f}° - using as rotation_offset")
            elif len(segment_angles) >= 20 and 0 <= segment_20_index < 20:
                # Fallback: use wire boundary + 9° to estimate center
                boundary_rad = segment_angles[segment_20_index]
                boundary_deg = math.degrees(boundary_rad)
                if boundary_deg < 0:
                    boundary_deg += 360
                rotation_offset_deg = (boundary_deg + 9) % 360
                print(f"[CAL] No 20 marker, using boundary {segment_20_index} + 9° = {rotation_offset_deg:.1f}°")
            else:
                rotation_offset_deg = 0.0
                print("[CAL] No 20 marker or segment_angles, defaulting rotation_offset=0")
            
            # Build calibration data
            ellipse_cal = EllipseCalibration(
                center=center,
                outer_double_ellipse=outer_double_ellipse,
                outer_triple_ellipse=outer_triple_ellipse,
                inner_triple_ellipse=inner_triple_ellipse,
                inner_double_ellipse=inner_double_ellipse,
                segment_angles=segment_angles,
                segment_20_index=segment_20_index,
            )
            
            # Estimate bull ellipses from outer_triple
            ellipse_cal = self._estimate_bull_rings(ellipse_cal)
            
            # Calculate quality
            quality = self._calculate_quality(
                cal_points, cal1_points, cal2_points, cal3_points, 
                outer_double_ellipse, len(segment_angles), h, w
            )
            
            # Detect segment at top
            segment_at_top = self._get_segment_at_top(ellipse_cal)
            
            # Build calibration data dict for storage
            calibration_data = {
                "center": center,
                "outer_double_ellipse": outer_double_ellipse,
                "outer_triple_ellipse": outer_triple_ellipse,
                "inner_triple_ellipse": inner_triple_ellipse,
                "inner_double_ellipse": inner_double_ellipse,
                "bull_ellipse": ellipse_cal.bull_ellipse,
                "bullseye_ellipse": ellipse_cal.bullseye_ellipse,
                "segment_angles": segment_angles,
                "segment_20_index": segment_20_index,
                "rotation_offset_deg": rotation_offset_deg,
                "image_size": (w, h),
                "quality": quality,
                "segment_at_top": segment_at_top,
            }
            
            # === POLYGON CALIBRATION (Autodarts-style 20-point) ===
            # Generate polygon calibration from the same YOLO-detected points
            try:
                twenty_angle_rad = math.radians(rotation_offset_deg)
                polygon_data = generate_polygon_calibration_from_yolo(
                    cal_points=cal_points,      # outer double (board edge)
                    cal1_points=cal1_points,    # outer triple
                    cal2_points=cal2_points,    # inner double  
                    cal3_points=cal3_points,    # inner triple
                    center=center,
                    twenty_angle_rad=twenty_angle_rad,
                    image_width=w,
                    image_height=h
                )
                
                # Add polygon data to calibration_data
                calibration_data["polygon"] = polygon_data
                
                # Also store in global polygon calibrations for use during detection
                if polygon_data.get("valid", False):
                    poly_cal = PolygonCalibration(
                        camera_id=camera_id,
                        bull=center,
                        double_outers=polygon_data["double_outers"],
                        double_inners=polygon_data["double_inners"],
                        treble_outers=polygon_data["treble_outers"],
                        treble_inners=polygon_data["treble_inners"],
                        image_width=w,
                        image_height=h
                    )
                    set_polygon_calibration(camera_id, poly_cal)
                    print(f"[CAL] Generated polygon calibration for camera {camera_id}: {len(polygon_data['double_outers'])} points per ring")
                else:
                    print(f"[CAL] Polygon calibration incomplete for camera {camera_id} - not enough points detected")
                    
            except Exception as e:
                print(f"[CAL] Failed to generate polygon calibration: {e}")
                calibration_data["polygon"] = None
            
            # Generate overlay image
            overlay = self._draw_calibration_overlay(
                image, ellipse_cal, 
                cal_points, cal1_points, cal2_points, cal3_points,
                bull_points, twenty_points,
                rotation_offset_deg
            )
            overlay_base64 = encode_image(overlay)
            
            return CameraCalibrationResult(
                camera_id=camera_id,
                success=True,
                quality=quality,
                overlay_image=overlay_base64,
                segment_at_top=segment_at_top,
                calibration_data=calibration_data
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return CameraCalibrationResult(
                camera_id=camera_id,
                success=False,
                error=str(e)
            )
    
    def _estimate_bull_rings(self, cal: EllipseCalibration) -> EllipseCalibration:
        """Estimate bull ellipses from outer_triple ring using dartboard geometry."""
        # Use outer_triple for estimation (most reliable)
        if cal.outer_triple_ellipse:
            base = cal.outer_triple_ellipse
            # Outer bull / outer triple = 15.9mm / 107mm = 0.1486
            # Bullseye / outer triple = 6.35mm / 107mm = 0.0594
            bull_ratio = OUTER_BULL_RADIUS_MM / TRIPLE_OUTER_RADIUS_MM
            bullseye_ratio = BULL_RADIUS_MM / TRIPLE_OUTER_RADIUS_MM
        elif cal.outer_double_ellipse:
            base = cal.outer_double_ellipse
            bull_ratio = OUTER_BULL_RADIUS_MM / DOUBLE_OUTER_RADIUS_MM
            bullseye_ratio = BULL_RADIUS_MM / DOUBLE_OUTER_RADIUS_MM
        else:
            return cal
        
        (_, _), (bw, bh), bangle = base
        
        bull_w = bw * bull_ratio
        bull_h = bh * bull_ratio
        bullseye_w = bw * bullseye_ratio
        bullseye_h = bh * bullseye_ratio
        
        cal.bull_ellipse = ((cal.center[0], cal.center[1]), (bull_w, bull_h), bangle)
        cal.bullseye_ellipse = ((cal.center[0], cal.center[1]), (bullseye_w, bullseye_h), bangle)
        
        return cal
    
    def _get_segment_at_top(self, cal: EllipseCalibration) -> int:
        """Determine which segment is at the 12 o'clock position."""
        if not cal.segment_angles or len(cal.segment_angles) < 20:
            return 20
        
        # Angle pointing up (12 o'clock) is -I?/2
        top_angle = -math.pi / 2
        
        def normalize_angle(a):
            while a < 0:
                a += 2 * math.pi
            while a >= 2 * math.pi:
                a -= 2 * math.pi
            return a
        
        top_norm = normalize_angle(top_angle)
        sorted_angles = sorted([normalize_angle(a) for a in cal.segment_angles])
        
        # Find which segment top falls into
        for i in range(len(sorted_angles)):
            a1 = sorted_angles[i]
            a2 = sorted_angles[(i + 1) % len(sorted_angles)]
            
            if a2 < a1:
                if top_norm >= a1 or top_norm < a2:
                    segment_index = (i - cal.segment_20_index) % 20
                    return DARTBOARD_SEGMENTS[segment_index]
            else:
                if a1 <= top_norm < a2:
                    segment_index = (i - cal.segment_20_index) % 20
                    return DARTBOARD_SEGMENTS[segment_index]
        
        return 20
    
    def _calculate_quality(
        self,
        cal_points: List,
        cal1_points: List,
        cal2_points: List,
        cal3_points: List,
        ellipse: Tuple,
        num_segment_angles: int,
        img_h: int,
        img_w: int
    ) -> float:
        """Calculate calibration quality score."""
        total_points = len(cal_points) + len(cal1_points) + len(cal2_points) + len(cal3_points)
        point_score = min(1.0, total_points / 80)  # Expect ~80 points (20 per ring)
        
        (cx, cy), (w, h), angle = ellipse
        circularity = min(w, h) / max(w, h) if max(w, h) > 0 else 0
        
        ellipse_area = math.pi * (w/2) * (h/2)
        image_area = img_w * img_h
        coverage = ellipse_area / image_area
        coverage_score = min(1.0, coverage / 0.3)
        
        # Segment detection score
        segment_score = min(1.0, num_segment_angles / 20)
        
        quality = (point_score * 0.3 + circularity * 0.2 + coverage_score * 0.2 + segment_score * 0.3)
        return round(quality, 3)
    
    def _draw_calibration_overlay(
        self,
        image: np.ndarray,
        cal: EllipseCalibration,
        cal_points: List,
        cal1_points: List,
        cal2_points: List,
        cal3_points: List,
        bull_points: List,
        twenty_points: List,
        rotation_offset_deg: float = 0.0
    ) -> np.ndarray:
        """Draw dartboard overlay showing detected rings, points, and segment labels."""
        overlay = image.copy()
        
        # Draw fitted ellipses
        if cal.outer_double_ellipse:
            cv2.ellipse(overlay, 
                       (int(cal.outer_double_ellipse[0][0]), int(cal.outer_double_ellipse[0][1])),
                       (int(cal.outer_double_ellipse[1][0]/2), int(cal.outer_double_ellipse[1][1]/2)),
                       cal.outer_double_ellipse[2], 0, 360, (255, 255, 0), 2)
        
        if cal.inner_double_ellipse:
            cv2.ellipse(overlay,
                       (int(cal.inner_double_ellipse[0][0]), int(cal.inner_double_ellipse[0][1])),
                       (int(cal.inner_double_ellipse[1][0]/2), int(cal.inner_double_ellipse[1][1]/2)),
                       cal.inner_double_ellipse[2], 0, 360, (255, 255, 0), 1)
        
        if cal.outer_triple_ellipse:
            cv2.ellipse(overlay,
                       (int(cal.outer_triple_ellipse[0][0]), int(cal.outer_triple_ellipse[0][1])),
                       (int(cal.outer_triple_ellipse[1][0]/2), int(cal.outer_triple_ellipse[1][1]/2)),
                       cal.outer_triple_ellipse[2], 0, 360, (255, 255, 0), 2)
        
        if cal.inner_triple_ellipse:
            cv2.ellipse(overlay,
                       (int(cal.inner_triple_ellipse[0][0]), int(cal.inner_triple_ellipse[0][1])),
                       (int(cal.inner_triple_ellipse[1][0]/2), int(cal.inner_triple_ellipse[1][1]/2)),
                       cal.inner_triple_ellipse[2], 0, 360, (255, 255, 0), 1)
        
        if cal.bull_ellipse:
            cv2.ellipse(overlay,
                       (int(cal.bull_ellipse[0][0]), int(cal.bull_ellipse[0][1])),
                       (int(cal.bull_ellipse[1][0]/2), int(cal.bull_ellipse[1][1]/2)),
                       cal.bull_ellipse[2], 0, 360, (255, 255, 0), 1)
        
        if cal.bullseye_ellipse:
            cv2.ellipse(overlay,
                       (int(cal.bullseye_ellipse[0][0]), int(cal.bullseye_ellipse[0][1])),
                       (int(cal.bullseye_ellipse[1][0]/2), int(cal.bullseye_ellipse[1][1]/2)),
                       cal.bullseye_ellipse[2], 0, 360, (255, 255, 0), 1)
        
        # Calibration point dots removed for cleaner overlay
        # (Detection points are used for fitting but not displayed)
        
        # Center marker removed
        
        # Draw segment boundary lines from bull to outer double ring
        if cal.outer_double_ellipse and cal.segment_angles and len(cal.segment_angles) >= 10:
            for angle in cal.segment_angles:
                dx = math.cos(angle)
                dy = math.sin(angle)
                
                outer_pt = line_ellipse_intersection(cal.center, (dx, dy), cal.outer_double_ellipse)
                
                # Draw from center (or bull edge if available) to outer double
                if cal.bull_ellipse:
                    inner_pt = line_ellipse_intersection(cal.center, (dx, dy), cal.bull_ellipse)
                else:
                    inner_pt = (int(cal.center[0]), int(cal.center[1]))
                
                # White lines, thickness 2 for visibility
                cv2.line(overlay, inner_pt, outer_pt, (255, 255, 255), 2)
        
        # Draw segment numbers using the ACTUAL detected boundary angles
        # Place each number at the midpoint between its two boundary lines, projected outward
        if cal.outer_double_ellipse and cal.segment_angles and len(cal.segment_angles) >= 20:
            # Create text ellipse further out - at 1.18x the outer double ellipse
            text_w = cal.outer_double_ellipse[1][0] * 1.18
            text_h = cal.outer_double_ellipse[1][1] * 1.18
            text_ellipse = (cal.outer_double_ellipse[0], (text_w, text_h), cal.outer_double_ellipse[2])
            
            # Use segment_angles directly - they are already sorted and segment_20_index
            # is an index into this array
            segment_angles = cal.segment_angles
            idx_20 = cal.segment_20_index % len(segment_angles)
            
            for i, seg in enumerate(DARTBOARD_SEGMENTS):
                # segment_20_index is where segment 20 starts
                # DARTBOARD_SEGMENTS[0] = 20, [1] = 1, [2] = 18, etc.
                idx_start = (idx_20 + i) % len(segment_angles)
                idx_end = (idx_20 + i + 1) % len(segment_angles)
                
                a1 = segment_angles[idx_start]
                a2 = segment_angles[idx_end]
                
                # Handle wrap-around for midpoint calculation
                if a2 < a1:
                    # Wraps around from +I? to -I?
                    mid_angle = (a1 + (a2 + 2 * math.pi)) / 2
                    if mid_angle > math.pi:
                        mid_angle -= 2 * math.pi
                else:
                    mid_angle = (a1 + a2) / 2
                
                dx = math.cos(mid_angle)
                dy = math.sin(mid_angle)
                text_pt = line_ellipse_intersection(cal.center, (dx, dy), text_ellipse)
                
                text_str = str(seg)
                (tw, th), _ = cv2.getTextSize(text_str, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                
                # Draw text with black outline for readability
                text_x = text_pt[0] - tw//2
                text_y = text_pt[1] + th//2
                # Black outline (draw in 8 directions with offset 2)
                for ox, oy in [(-2,-2), (-2,0), (-2,2), (0,-2), (0,2), (2,-2), (2,0), (2,2)]:
                    cv2.putText(overlay, text_str, (text_x + ox, text_y + oy),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
                # Bright green for 20, white for all others
                text_color = (0, 255, 0) if seg == 20 else (255, 255, 255)
                cv2.putText(overlay, text_str, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)
        
        # Draw segment 20 as a filled cyan wedge (semi-transparent)
        # Use the actual detected segment boundaries for segment 20
        if cal.outer_double_ellipse and cal.bull_ellipse and cal.segment_angles and len(cal.segment_angles) >= 20:
            idx_20 = cal.segment_20_index % len(cal.segment_angles)
            idx_20_end = (idx_20 + 1) % len(cal.segment_angles)
            
            # Get the actual boundary angles for segment 20
            angle_start_rad = cal.segment_angles[idx_20]
            angle_end_rad = cal.segment_angles[idx_20_end]
            
            # Handle wrap-around
            if angle_end_rad < angle_start_rad:
                angle_end_rad += 2 * math.pi
            
            # Create a polygon for the segment 20 wedge
            num_arc_points = 20
            wedge_points = []
            
            # Outer arc (from angle_start to angle_end)
            for i in range(num_arc_points + 1):
                angle = angle_start_rad + (angle_end_rad - angle_start_rad) * i / num_arc_points
                dx, dy = math.cos(angle), math.sin(angle)
                pt = line_ellipse_intersection(cal.center, (dx, dy), cal.outer_double_ellipse)
                wedge_points.append(pt)
            
            # Inner arc (from angle_end back to angle_start) - use bull ellipse
            for i in range(num_arc_points, -1, -1):
                angle = angle_start_rad + (angle_end_rad - angle_start_rad) * i / num_arc_points
                dx, dy = math.cos(angle), math.sin(angle)
                pt = line_ellipse_intersection(cal.center, (dx, dy), cal.bull_ellipse)
                wedge_points.append(pt)
            
            # Draw filled polygon with transparency
            wedge_array = np.array(wedge_points, dtype=np.int32)
            overlay_layer = overlay.copy()
            cv2.fillPoly(overlay_layer, [wedge_array], (255, 255, 0))  # Cyan in BGR
            # Blend with 40% opacity
            cv2.addWeighted(overlay_layer, 0.4, overlay, 0.6, 0, overlay)
        
        # Add info text
        total_points = len(cal_points) + len(cal1_points) + len(cal2_points) + len(cal3_points)
        cv2.putText(overlay, f"Calibration points: {total_points} | Segments: {len(cal.segment_angles)}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return overlay
    
    def detect_tips(
        self,
        camera_id: str,
        image_base64: str,
        calibration_data: Dict[str, Any],
        confidence_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Detect dart tips in an image using YOLO.
        
        Args:
            camera_id: Camera identifier
            image_base64: Base64-encoded image
            calibration_data: Calibration data
            confidence_threshold: Minimum confidence
            
        Returns:
            List of detected tips with pixel and mm coordinates
        """
        try:
            from app.core.detection import DartTipDetector
            
            # Decode image
            image = decode_image(image_base64)
            
            # Initialize tip detector if needed
            if not hasattr(self, 'tip_detector'):
                self.tip_detector = DartTipDetector()
            
            if not self.tip_detector.is_initialized:
                return []
            
            # Detect dart tips
            tips = self.tip_detector.detect_tips(image, confidence_threshold=confidence_threshold)
            
            results = []
            for tip in tips:
                # Convert pixel to dartboard coordinates
                x_mm, y_mm = self.pixel_to_dartboard(tip.x, tip.y, calibration_data)
                
                results.append({
                    'x_px': tip.x,
                    'y_px': tip.y,
                    'x_mm': x_mm,
                    'y_mm': y_mm,
                    'confidence': tip.confidence
                })
            
            return results
            
        except Exception as e:
            print(f"Error detecting tips: {e}")
            return []
    
    def pixel_to_dartboard(
        self,
        x_px: float,
        y_px: float,
        calibration_data: Dict[str, Any]
    ) -> Tuple[float, float]:
        """
        Convert pixel coordinates to dartboard mm coordinates.
        
        Uses the calibration ellipses to estimate scale and perspective.
        
        Args:
            x_px: X pixel coordinate
            y_px: Y pixel coordinate
            calibration_data: Calibration data from calibrate()
            
        Returns:
            (x_mm, y_mm) in dartboard coordinates (0,0 = center)
        """
        center = calibration_data.get('center', (0, 0))
        outer_double = calibration_data.get('outer_double_ellipse')
        
        if not outer_double:
            # Fallback: assume 1 pixel = 1mm offset from center
            return (x_px - center[0], y_px - center[1])
        
        # Calculate relative position from center in pixels
        dx_px = x_px - center[0]
        dy_px = y_px - center[1]
        
        # Get ellipse parameters
        (ecx, ecy), (ew, eh), angle = outer_double
        
        # Convert angle to radians
        angle_rad = math.radians(-angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        # Rotate to align with ellipse axes
        dx_rot = dx_px * cos_a - dy_px * sin_a
        dy_rot = dx_px * sin_a + dy_px * cos_a
        
        # Scale by ellipse semi-axes to get normalized position
        # Then scale to dartboard mm (outer double = 170mm radius)
        a = ew / 2  # semi-major axis in pixels
        b = eh / 2  # semi-minor axis in pixels
        
        if a > 0 and b > 0:
            # Scale each axis by dartboard radius / ellipse radius
            x_mm = (dx_rot / a) * DOUBLE_OUTER_RADIUS_MM
            y_mm = (dy_rot / b) * DOUBLE_OUTER_RADIUS_MM
        else:
            x_mm = dx_px
            y_mm = dy_px
        
        # Rotate back
        angle_rad = math.radians(angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        x_final = x_mm * cos_a - y_mm * sin_a
        y_final = x_mm * sin_a + y_mm * cos_a
        
        return (x_final, y_final)
    
    def detect_dart(
        self,
        camera_id: str,
        image_base64: str,
        calibration_data: Dict[str, Any]
    ) -> DetectResponse:
        """
        Detect darts in the image and calculate their scores.
        
        Args:
            camera_id: Camera identifier
            image_base64: Base64-encoded image
            calibration_data: Calibration data from previous calibrate() call
            
        Returns:
            DetectResponse with dart positions and scores
        """
        try:
            from app.core.detection import DartTipDetector, score_from_ellipse_calibration
            
            # Decode image
            image = decode_image(image_base64)
            
            # Initialize tip detector if needed
            if not hasattr(self, 'tip_detector'):
                self.tip_detector = DartTipDetector()
            
            if not self.tip_detector.is_initialized:
                return DetectResponse(
                    success=False,
                    error="Tip detection model not loaded."
                )
            
            # Detect dart tips
            tips = self.tip_detector.detect_tips(image, confidence_threshold=0.5)
            
            if not tips:
                return DetectResponse(
                    success=True,
                    darts=[],
                    message="No darts detected in image."
                )
            
            # Calculate scores for each detected tip
            darts = []
            for tip in tips:
                # Calculate score using ellipse calibration
                score_result = score_from_ellipse_calibration(
                    (tip.x, tip.y),
                    calibration_data
                )
                
                darts.append({
                    "position": {
                        "x": tip.x,
                        "y": tip.y,
                        "confidence": tip.confidence
                    },
                    "score": score_result
                })
            
            # Create overlay image showing detections
            overlay = image.copy()
            for dart in darts:
                pos = dart["position"]
                score = dart["score"]
                
                # Draw tip position
                cv2.circle(overlay, (int(pos["x"]), int(pos["y"])), 10, (0, 255, 255), -1)
                cv2.circle(overlay, (int(pos["x"]), int(pos["y"])), 12, (0, 0, 0), 2)
                
                # Draw score text
                score_text = f"{score['score']}"
                if score['zone'] == 'triple':
                    score_text = f"T{score['segment']}"
                elif score['zone'] == 'double':
                    score_text = f"D{score['segment']}"
                elif score['zone'] == 'inner_bull':
                    score_text = "Bull"
                elif score['zone'] == 'outer_bull':
                    score_text = "25"
                
                cv2.putText(overlay, score_text, 
                           (int(pos["x"]) + 15, int(pos["y"]) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            overlay_base64 = encode_image(overlay)
            
            return DetectResponse(
                success=True,
                darts=darts,
                overlay_image=overlay_base64,
                message=f"Detected {len(darts)} dart(s)."
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return DetectResponse(
                success=False,
                error=str(e)
            )

