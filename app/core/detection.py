"""
Dart Tip Detection Module

Uses YOLO pose estimation to detect dart tips in images.
Based on the original DartDetector approach.
"""
import cv2
import numpy as np
import math
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from pathlib import Path

from app.core.geometry import (
    DARTBOARD_SEGMENTS,
    BULL_RADIUS_MM,
    OUTER_BULL_RADIUS_MM,
    TRIPLE_INNER_RADIUS_MM,
    TRIPLE_OUTER_RADIUS_MM,
    DOUBLE_INNER_RADIUS_MM,
    DOUBLE_OUTER_RADIUS_MM,
    SEGMENT_ANGLE_OFFSET,
    calculate_score,
)
from app.models.schemas import DetectResponse, DartScore, DartPosition

# Get the models directory
MODELS_DIR = Path(__file__).parent.parent.parent / "models"

# Available tip detection models
TIP_MODEL_PATHS = {
    "default": MODELS_DIR / "posenano27122025_int8_openvino_model",
    "rect": MODELS_DIR / "tippose25012026_rect_openvino_model",
    "square": MODELS_DIR / "tippose25012026_square_openvino_model",
}

# YOLO class IDs (from original code)
CLASS_TIP = 0  # Dart tip
CLASS_CAL = 1  # Outer double ring calibration points
CLASS_CAL1 = 2  # Triple ring calibration points


@dataclass
class DetectedTip:
    """A detected dart tip position."""
    x: float
    y: float
    confidence: float
    keypoints: Optional[np.ndarray] = None  # For pose models


class DartTipDetector:
    """
    Uses YOLO to detect dart tips in images.
    
    Supports both detection and pose estimation models.
    Pose models provide more accurate tip positioning via keypoints.
    """
    
    def __init__(self, model_name: str = "default"):
        """
        Initialize detector with specified model.
        
        Args:
            model_name: One of "default", "rect", "square"
        """
        self.model = None
        self.model_name = model_name
        self.is_initialized = False
        self.is_pose_model = False
        self.image_size = 1280  # Default size
        self._load_model()
    
    def _load_model(self):
        """Load the YOLO tip detection model."""
        try:
            from ultralytics import YOLO
            
            model_path = TIP_MODEL_PATHS.get(self.model_name, TIP_MODEL_PATHS["default"])
            if not model_path.exists():
                print(f"Warning: Tip model not found at {model_path}")
                return
            
            self.model = YOLO(str(model_path))
            self.is_initialized = True
            
            # Check if this is a pose model by looking at metadata
            self.is_pose_model = self._check_if_pose_model(model_path)
            
            # Adjust image size for rect models
            if "rect" in self.model_name:
                self.image_size = (1280, 736)
            
            print(f"Loaded tip model from {model_path} (pose={self.is_pose_model})")
            
        except ImportError:
            print("Warning: ultralytics not installed. Tip detection disabled.")
        except Exception as e:
            print(f"Warning: Failed to load YOLO tip model: {e}")
    
    def _check_if_pose_model(self, model_path: Path) -> bool:
        """Check if the model is a pose estimation model."""
        try:
            meta_file = model_path / "metadata.yaml"
            if meta_file.exists():
                content = meta_file.read_text()
                for line in content.splitlines():
                    if line.strip().startswith("task:"):
                        task = line.split(":", 1)[1].strip().lower()
                        return task == "pose"
        except Exception:
            pass
        return False
    
    def detect_tips(
        self, 
        image: np.ndarray,
        confidence_threshold: float = 0.5
    ) -> List[DetectedTip]:
        """
        Detect dart tips in the image.
        
        Args:
            image: Input image (BGR format)
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            List of DetectedTip objects
        """
        if not self.is_initialized or self.model is None:
            return []
        
        # Run inference
        results = self.model(
            image, 
            imgsz=self.image_size, 
            conf=confidence_threshold, 
            verbose=False
        )
        
        tips = []
        
        for result in results:
            if result.boxes is None:
                continue
            
            boxes = result.boxes
            keypoints = getattr(result, 'keypoints', None)
            
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                
                # Only process tip detections (class 0)
                if cls_id != CLASS_TIP:
                    continue
                
                conf = float(boxes.conf[i])
                
                # Get bounding box center
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                # If pose model, use keypoints for more accurate tip position
                kp = None
                if self.is_pose_model and keypoints is not None:
                    try:
                        kp_data = keypoints[i].data.cpu().numpy()
                        if len(kp_data) > 0:
                            # First keypoint is typically the tip
                            # Shape is (num_keypoints, 3) where 3 = (x, y, conf)
                            tip_kp = kp_data[0][0]  # First keypoint
                            if len(tip_kp) >= 2 and tip_kp[0] > 0 and tip_kp[1] > 0:
                                cx, cy = tip_kp[0], tip_kp[1]
                            kp = kp_data[0]
                    except Exception as e:
                        print(f"Error processing keypoints: {e}")
                
                tips.append(DetectedTip(
                    x=float(cx),
                    y=float(cy),
                    confidence=conf,
                    keypoints=kp
                ))
        
        return tips


def calculate_score_from_pixel_position(
    tip_x: float,
    tip_y: float,
    calibration_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate score from a dart tip's pixel position using calibration data.
    
    Uses ellipse-based calibration to handle perspective distortion.
    
    Args:
        tip_x, tip_y: Dart tip pixel coordinates
        calibration_data: Calibration data from DartboardCalibrator
        
    Returns:
        Score dictionary with score, multiplier, segment, zone
    """
    center = calibration_data.get("center")
    if not center:
        return {"score": 0, "multiplier": 1, "segment": 0, "zone": "unknown"}
    
    cx, cy = center
    
    # Calculate distance and angle from center
    dx = tip_x - cx
    dy = tip_y - cy
    pixel_distance = math.sqrt(dx * dx + dy * dy)
    angle = math.atan2(dy, dx)
    
    # Get the outer double ellipse to calculate scale
    outer_double = calibration_data.get("outer_double_ellipse")
    if not outer_double:
        return {"score": 0, "multiplier": 1, "segment": 0, "zone": "unknown"}
    
    # Outer double ellipse gives us the scale
    # outer_double format: ((cx, cy), (width, height), angle)
    ellipse_center, (ew, eh), ellipse_angle = outer_double
    
    # Calculate the effective radius at this angle (accounting for ellipse shape)
    # For a proper solution, we'd use the ellipse equation
    # For MVP, use average radius
    avg_outer_radius_px = (ew + eh) / 4  # /4 because ew, eh are diameters
    
    # Scale factor: pixels to mm
    scale = DOUBLE_OUTER_RADIUS_MM / avg_outer_radius_px if avg_outer_radius_px > 0 else 1.0
    
    # Convert pixel distance to mm
    distance_mm = pixel_distance * scale
    
    # Get rotation offset from calibration (if available)
    rotation_offset = calibration_data.get("rotation_offset_rad", 0.0)
    
    # Calculate score using geometry
    return calculate_score(distance_mm, angle, rotation_offset)


def point_in_ellipse(
    point: Tuple[float, float], 
    ellipse: Optional[Tuple]
) -> bool:
    """
    Check if a point is inside an ellipse.
    
    Args:
        point: (x, y) coordinates
        ellipse: ((cx, cy), (width, height), angle) - OpenCV ellipse format
        
    Returns:
        True if point is inside ellipse
    """
    if ellipse is None:
        return False
    
    (cx, cy), (w, h), angle = ellipse
    
    # Semi-axes
    a = w / 2
    b = h / 2
    
    if a <= 0 or b <= 0:
        return False
    
    # Translate point to ellipse center
    x = point[0] - cx
    y = point[1] - cy
    
    # Rotate point by negative ellipse angle
    angle_rad = np.radians(-angle)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    x_rot = x * cos_a - y * sin_a
    y_rot = x * sin_a + y * cos_a
    
    # Check ellipse equation: (x/a)² + (y/b)² <= 1
    return (x_rot / a) ** 2 + (y_rot / b) ** 2 <= 1.0


def score_from_ellipse_calibration(
    point: Tuple[float, float],
    calibration_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate score using fitted ellipses (from original DartDetector).
    
    This is more accurate than simple distance calculation because
    it accounts for perspective distortion.
    
    Args:
        point: (x, y) pixel coordinates of dart tip
        calibration_data: Calibration data containing ellipses
        
    Returns:
        Score dictionary
    """
    center = calibration_data.get("center")
    if not center:
        return {"score": 0, "multiplier": 1, "segment": 0, "zone": "unknown"}
    
    # Check bullseye first
    bullseye_ellipse = calibration_data.get("bullseye_ellipse")
    if bullseye_ellipse and point_in_ellipse(point, bullseye_ellipse):
        return {
            "score": 50,
            "multiplier": 1,
            "segment": 0,
            "zone": "inner_bull",
            "is_bullseye": True,
            "is_outer_bull": False
        }
    
    # Check outer bull
    bull_ellipse = calibration_data.get("bull_ellipse")
    if bull_ellipse and point_in_ellipse(point, bull_ellipse):
        return {
            "score": 25,
            "multiplier": 1,
            "segment": 0,
            "zone": "outer_bull",
            "is_bullseye": False,
            "is_outer_bull": True
        }
    
    # Get segment from angle
    cx, cy = center
    dx = point[0] - cx
    dy = point[1] - cy
    angle = math.atan2(dy, dx)
    
    # Get rotation offset
    rotation_offset = calibration_data.get("rotation_offset_rad", 0.0)
    
    # Calculate segment
    adjusted = angle - SEGMENT_ANGLE_OFFSET + rotation_offset
    while adjusted < 0:
        adjusted += 2 * math.pi
    while adjusted >= 2 * math.pi:
        adjusted -= 2 * math.pi
    
    segment_index = int((adjusted / (2 * math.pi)) * 20) % 20
    segment = DARTBOARD_SEGMENTS[segment_index]
    
    # Check ellipses from outside to inside
    outer_double = calibration_data.get("outer_double_ellipse")
    inner_double = calibration_data.get("inner_double_ellipse")
    outer_triple = calibration_data.get("outer_triple_ellipse")
    inner_triple = calibration_data.get("inner_triple_ellipse")
    
    # Check if outside board
    if outer_double and not point_in_ellipse(point, outer_double):
        return {
            "score": 0,
            "multiplier": 0,
            "segment": 0,
            "zone": "miss",
            "is_bullseye": False,
            "is_outer_bull": False
        }
    
    # Double ring (between inner and outer double)
    if outer_double and inner_double:
        in_outer = point_in_ellipse(point, outer_double)
        in_inner = point_in_ellipse(point, inner_double)
        if in_outer and not in_inner:
            return {
                "score": segment * 2,
                "multiplier": 2,
                "segment": segment,
                "zone": "double",
                "is_bullseye": False,
                "is_outer_bull": False
            }
    
    # Triple ring (between inner and outer triple)
    if outer_triple and inner_triple:
        in_outer = point_in_ellipse(point, outer_triple)
        in_inner = point_in_ellipse(point, inner_triple)
        if in_outer and not in_inner:
            return {
                "score": segment * 3,
                "multiplier": 3,
                "segment": segment,
                "zone": "triple",
                "is_bullseye": False,
                "is_outer_bull": False
            }
    
    # Single outer (between triple and double)
    if inner_double and outer_triple:
        in_double = point_in_ellipse(point, inner_double)
        in_triple = point_in_ellipse(point, outer_triple)
        if in_double and not in_triple:
            return {
                "score": segment,
                "multiplier": 1,
                "segment": segment,
                "zone": "single_outer",
                "is_bullseye": False,
                "is_outer_bull": False
            }
    
    # Single inner (between bull and triple)
    if inner_triple and bull_ellipse:
        in_triple = point_in_ellipse(point, inner_triple)
        in_bull = point_in_ellipse(point, bull_ellipse)
        if in_triple and not in_bull:
            return {
                "score": segment,
                "multiplier": 1,
                "segment": segment,
                "zone": "single_inner",
                "is_bullseye": False,
                "is_outer_bull": False
            }
    
    # Fallback to single
    return {
        "score": segment,
        "multiplier": 1,
        "segment": segment,
        "zone": "single",
        "is_bullseye": False,
        "is_outer_bull": False
    }
