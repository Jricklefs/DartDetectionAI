"""
Dart Tip Detection Module

Uses YOLO pose estimation to detect dart tips in images.
Based on the original DartDetector approach.
"""
import os
# Set OpenVINO to optimize for latency (keeps model warm)
os.environ.setdefault('OPENVINO_TPUT_MODE', 'LATENCY')

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
    "best": MODELS_DIR / "26tippose27012026_int8_openvino_model",  # Newer YOLO26n-pose model (Jan 2026)
    "rect": MODELS_DIR / "best_openvino_736x1280_fp16_openvino_model",  # Non-square 736x1280
    "square": MODELS_DIR / "tippose25012026_square_openvino_model",
}

# Use the best model (requires ultralytics 8.4+)
DEFAULT_TIP_MODEL = "default"

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
    
    def __init__(self, model_name: str = DEFAULT_TIP_MODEL):
        """
        Initialize detector with specified model.
        
        Args:
            model_name: One of "default", "best", "rect", "square"
        """
        self.model = None
        self.model_name = model_name
        self.is_initialized = False
        self.is_pose_model = False
        self.image_size = 1280  # Default size
        self._last_inference_time = 0
        self._warmup_interval = 3  # Run inference every 3 seconds to keep model hot
        self._warmup_thread = None
        self._stop_warmup = False
        self._warmup_image = None  # Store a calibration image for warmups
        self._warmup_counter = 0
        self._load_model()
        self._start_background_warmup()
    
    def set_warmup_image(self, image: np.ndarray):
        """Set a real camera image to use for warmups instead of dummy."""
        self._warmup_image = image.copy()
        print(f"Warmup image set: {image.shape}")
    
    def _fetch_camera_snapshot(self):
        """Try to fetch a live camera snapshot for warmup."""
        try:
            import requests
            import base64
            resp = requests.get("http://localhost:8001/cameras/0/snapshot", timeout=2)
            if resp.ok:
                data = resp.json()
                img_b64 = data.get('image', '')
                if img_b64:
                    img_bytes = base64.b64decode(img_b64)
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is not None:
                        return img
        except Exception:
            pass
        return None
    
    def _start_background_warmup(self):
        """Start background thread to keep model warm using real camera images."""
        import threading
        
        def warmup_loop():
            import time
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            
            while not self._stop_warmup:
                try:
                    if self.is_initialized and self.model is not None:
                        # Only warmup if no recent inference
                        if time.time() - self._last_inference_time > self._warmup_interval:
                            # Try to get a real image
                            warmup_img = None
                            
                            # Option 1: Use stored warmup image
                            if self._warmup_image is not None:
                                warmup_img = self._warmup_image
                            # Option 2: Fetch live camera snapshot (every 10th warmup)
                            elif self._warmup_counter % 10 == 0:
                                warmup_img = self._fetch_camera_snapshot()
                                if warmup_img is not None:
                                    self._warmup_image = warmup_img  # Cache it
                            
                            # Fallback to dummy
                            if warmup_img is None:
                                warmup_img = dummy_img
                            
                            # Run inference
                            _ = self.model(warmup_img, imgsz=self.image_size, verbose=False)
                            self._last_inference_time = time.time()
                            self._warmup_counter += 1
                except Exception as e:
                    pass
                time.sleep(2)  # Check every 2 seconds
        
        self._warmup_thread = threading.Thread(target=warmup_loop, daemon=True)
        self._warmup_thread.start()
        print("Background warmup thread started (uses real camera images)")
    
    def _load_model(self):
        """Load the YOLO tip detection model."""
        try:
            from ultralytics import YOLO
            import numpy as np
            
            model_path = TIP_MODEL_PATHS.get(self.model_name, TIP_MODEL_PATHS["default"])
            if not model_path.exists():
                print(f"Warning: Tip model not found at {model_path}")
                return
            
            # Check if this is a pose model BEFORE loading
            self.is_pose_model = self._check_if_pose_model(model_path)
            
            # Load with explicit task to avoid auto-detection issues
            task = "pose" if self.is_pose_model else "detect"
            self.model = YOLO(str(model_path), task=task)
            self.is_initialized = True
            
            # Adjust image size for rect models
            if "rect" in self.model_name:
                self.image_size = (1280, 736)
            
            print(f"Loaded tip model from {model_path} (pose={self.is_pose_model})")
            
            # Warmup inference - run a dummy image through to initialize OpenVINO
            print("Warming up model...")
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model(dummy_img, imgsz=self.image_size, verbose=False)
            print("Model warmup complete")
            
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
    
    def warmup(self):
        """Run a warmup inference to keep the model hot."""
        if not self.is_initialized or self.model is None:
            return
        
        import time
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = self.model(dummy_img, imgsz=self.image_size, verbose=False)
        self._last_inference_time = time.time()
    
    def maybe_warmup(self):
        """Warmup if model has been idle too long."""
        import time
        if time.time() - self._last_inference_time > self._warmup_interval:
            self.warmup()
    
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
        
        import time
        self._last_inference_time = time.time()
        
        # Cache this image for warmups (real camera image is better than dummy)
        if self._warmup_image is None and image is not None:
            self._warmup_image = image.copy()
        
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
                
                # Get bounding box
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                
                # Default to box center
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                # Prefer keypoint if available with good confidence
                kp = None
                if self.is_pose_model and keypoints is not None:
                    try:
                        kp_data = keypoints[i].data.cpu().numpy()
                        if len(kp_data) > 0:
                            tip_kp = kp_data[0][0]  # First keypoint
                            kp_conf = tip_kp[2] if len(tip_kp) > 2 else 0
                            # Use keypoint if confident and valid
                            if kp_conf > 0.5 and tip_kp[0] > 0 and tip_kp[1] > 0:
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
    angle = math.atan2(dy, dx)  # Angle in radians
    
    # Get the outer double ellipse to calculate scale
    outer_double = calibration_data.get("outer_double_ellipse")
    if not outer_double:
        return {"score": 0, "multiplier": 1, "segment": 0, "zone": "unknown"}
    
    # Outer double ellipse gives us the scale
    # outer_double format: ((cx, cy), (width, height), angle)
    ellipse_center, (ew, eh), ellipse_angle = outer_double
    
    # Calculate the radius of the ellipse at this specific angle
    # This accounts for perspective distortion
    a = ew / 2  # semi-major axis
    b = eh / 2  # semi-minor axis
    
    # Adjust angle for ellipse rotation
    theta = angle - math.radians(ellipse_angle)
    
    # Ellipse radius at angle theta: r = ab / sqrt((b*cos(θ))² + (a*sin(θ))²)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    
    if a > 0 and b > 0:
        ellipse_radius_at_angle = (a * b) / math.sqrt((b * cos_t)**2 + (a * sin_t)**2)
    else:
        ellipse_radius_at_angle = (a + b) / 2  # fallback to average
    
    # Scale factor: pixels to mm (at this specific angle)
    scale = DOUBLE_OUTER_RADIUS_MM / ellipse_radius_at_angle if ellipse_radius_at_angle > 0 else 1.0
    
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
    
    # Get segment from angle using boundary lookup
    cx, cy = center
    dx = point[0] - cx
    dy = point[1] - cy
    angle = math.atan2(dy, dx)
    
    segment_angles = calibration_data.get("segment_angles", [])
    segment_20_index = calibration_data.get("segment_20_index", 0)
    
    if len(segment_angles) == 20 and 0 <= segment_20_index < 20:
        # Find which boundary interval contains this angle
        found_idx = 0
        for i in range(20):
            a1 = segment_angles[i]
            a2 = segment_angles[(i + 1) % 20]
            
            if a2 > a1:  # normal case
                if a1 <= angle < a2:
                    found_idx = i
                    break
            else:  # wraps around (a2 < a1, crosses ±π boundary)
                if angle >= a1 or angle < a2:
                    found_idx = i
                    break
        
        # Convert boundary index to segment
        # segment_20_index points to where segment 20 starts
        # DARTBOARD_SEGMENTS[0] = 20, [1] = 1, [2] = 18, etc (clockwise)
        relative_idx = (found_idx - segment_20_index) % 20
        segment = DARTBOARD_SEGMENTS[relative_idx]
        
        print(f"[SCORE] angle={math.degrees(angle):.1f}°, found_idx={found_idx}, seg20_idx={segment_20_index}, relative={relative_idx}, segment={segment}")
    else:
        # Fallback if no segment_angles
        rotation_offset = calibration_data.get("rotation_offset_rad", 0.0)
        adjusted_angle = (angle - SEGMENT_ANGLE_OFFSET) + rotation_offset
        while adjusted_angle < 0:
            adjusted_angle += 2 * math.pi
        while adjusted_angle >= 2 * math.pi:
            adjusted_angle -= 2 * math.pi
        segment_index = int((adjusted_angle / (2 * math.pi)) * 20)
        if segment_index >= 20:
            segment_index = 0
        segment = DARTBOARD_SEGMENTS[segment_index]
        print(f"[SCORE] FALLBACK: angle={math.degrees(angle):.1f}°, segment={segment}")
    
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
