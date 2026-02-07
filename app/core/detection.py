"""
Dart Tip Detection Module

Uses YOLO pose estimation to detect dart tips in images.
Based on the original DartDetector approach.
"""
import os
# OpenVINO optimizations
os.environ.setdefault('OPENVINO_CACHE_DIR', r'C:\Users\clawd\openvino_cache')  # Compiled model cache
os.environ.setdefault('OV_GPU_CACHE_MODEL', '1')  # GPU kernel caching

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

def letterbox_image(image: np.ndarray, target_size: Tuple[int, int], 
                    color: Tuple[int, int, int] = (114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize image with letterboxing to maintain aspect ratio.
    Like Machine Darts / Ultralytics preprocessing.
    
    Args:
        image: Input image (H, W, C)
        target_size: (height, width) the model expects
        color: Padding color (default gray)
    
    Returns:
        (letterboxed_image, scale, (pad_x, pad_y))
        - scale: factor used to resize
        - pad_x, pad_y: padding added on each side
    """
    img_h, img_w = image.shape[:2]
    target_h, target_w = target_size
    
    # Calculate scale to fit image in target while maintaining aspect ratio
    scale = min(target_w / img_w, target_h / img_h)
    
    # New dimensions after scaling
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Calculate padding
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    
    # Pad evenly on both sides
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    
    # Apply padding
    letterboxed = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=color
    )
    
    return letterboxed, scale, (pad_left, pad_top)


def unletterbox_coords(x: float, y: float, scale: float, 
                        pad: Tuple[int, int]) -> Tuple[float, float]:
    """
    Convert coordinates from letterboxed image back to original image space.
    
    Args:
        x, y: Coordinates in letterboxed image
        scale: Scale factor used in letterboxing
        pad: (pad_x, pad_y) padding added
    
    Returns:
        (orig_x, orig_y) in original image coordinates
    """
    pad_x, pad_y = pad
    orig_x = (x - pad_x) / scale
    orig_y = (y - pad_y) / scale
    return orig_x, orig_y



# Get the models directory
MODELS_DIR = Path(__file__).parent.parent.parent / "models"

# Available tip detection models
TIP_MODEL_PATHS = {
    "default": MODELS_DIR / "posenano27122025_int8_openvino_model",
    "best": MODELS_DIR / "26tippose27012026_int8_openvino_model",  # Newer YOLO26n-pose model (Jan 2026)
    "rect": MODELS_DIR / "best_openvino_736x1280_fp16_openvino_model",  # Non-square 736x1280
    "square": MODELS_DIR / "tippose25012026_square_openvino_model",
    # New models from Machine Darts v1.0.25.1
    "384x640": MODELS_DIR / "y26-p-n-384x640_int8_openvino_model",  # 384x640 INT8
    "552x960": MODELS_DIR / "y26-p-n-552x960_int8_openvino_model",  # 552x960 INT8
    "736x1280": MODELS_DIR / "y26-p-n-736x1280_int8_openvino_model",  # 736x1280 INT8
    "11m": MODELS_DIR / "11m_dartboard_openvino_model",  # 11M param model
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




def refine_tip_position(image: np.ndarray, cx: float, cy: float, 
                        box: Tuple[float, float, float, float],
                        search_radius: int = 30) -> Tuple[float, float, float]:
    """
    Refine dart tip position using edge detection.
    
    Strategy:
    1. Crop region around initial detection
    2. Apply Canny edge detection
    3. Find the point closest to board center that's on an edge
    
    Args:
        image: Full camera image
        cx, cy: Initial tip estimate from YOLO
        box: Bounding box (x1, y1, x2, y2)
        search_radius: Pixels to search around initial point
    
    Returns:
        (refined_x, refined_y, refinement_dist) - refined position and how much it moved
    """
    import cv2
    import numpy as np
    
    try:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = box
        
        # Expand box slightly for search
        margin = search_radius
        crop_x1 = max(0, int(x1 - margin))
        crop_y1 = max(0, int(y1 - margin))
        crop_x2 = min(w, int(x2 + margin))
        crop_y2 = min(h, int(y2 + margin))
        
        # Crop and convert to grayscale
        crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0:
            return cx, cy, 0.0
        
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find edge points
        edge_points = np.column_stack(np.where(edges > 0))
        if len(edge_points) == 0:
            return cx, cy, 0.0
        
        # Convert initial point to crop coordinates
        local_cx = cx - crop_x1
        local_cy = cy - crop_y1
        
        # Find edge point closest to initial position
        # Prefer points that are "forward" (toward smaller y in most cases for dart tips)
        distances = np.sqrt((edge_points[:, 1] - local_cx)**2 + (edge_points[:, 0] - local_cy)**2)
        
        # Filter to points within search radius
        valid_mask = distances < search_radius
        if not np.any(valid_mask):
            return cx, cy, 0.0
        
        valid_points = edge_points[valid_mask]
        valid_distances = distances[valid_mask]
        
        # Pick the closest edge point
        min_idx = np.argmin(valid_distances)
        refined_local_y, refined_local_x = valid_points[min_idx]
        
        # Convert back to image coordinates
        refined_x = refined_local_x + crop_x1
        refined_y = refined_local_y + crop_y1
        
        refinement_dist = np.sqrt((refined_x - cx)**2 + (refined_y - cy)**2)
        
        # Only accept refinement if it's small (< 15 pixels)
        if refinement_dist > 15:
            return cx, cy, 0.0
        
        return float(refined_x), float(refined_y), float(refinement_dist)
        
    except Exception as e:
        print(f"[REFINE] Error: {e}")
        return cx, cy, 0.0


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
        self._warmup_interval = 0.5  # Warmup if no inference in last 0.5 second (very aggressive)
        self._warmup_thread = None
        self._stop_warmup = False
        self._warmup_image = None  # Store a calibration image for warmups
        self._warmup_counter = 0
        self._load_model()
        # self._start_background_warmup()  # Disabled - not helping with latency
    
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
        """Start background thread to keep model warm.
        
        OpenVINO goes cold after ~4-5 seconds idle, causing 300-500ms latency spike.
        This thread runs inference every 2 seconds to prevent that.
        """
        import threading
        
        def warmup_loop():
            import time
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            warmup_count = 0
            
            while not self._stop_warmup:
                try:
                    if self.is_initialized and self.model is not None:
                        # Only warmup if no real inference recently
                        time_since_last = time.time() - self._last_inference_time
                        if time_since_last > 0.5:
                            # Use cached camera image if available, else dummy
                            warmup_img = self._warmup_image if self._warmup_image is not None else dummy_img
                            
                            # Run inference silently
                            start = time.time()
                            _ = self.model(warmup_img, imgsz=self.image_size, verbose=False)
                            elapsed = (time.time() - start) * 1000
                            self._last_inference_time = time.time()
                            warmup_count += 1
                            if warmup_count % 30 == 1:  # Log every 30th warmup
                                print(f"[WARMUP] #{warmup_count} - {elapsed:.0f}ms (idle was {time_since_last:.1f}s)")
                except Exception as e:
                    print(f"[WARMUP] Error: {e}")
                time.sleep(0.3)  # Check every 0.3 seconds (very responsive)
        
        self._warmup_thread = threading.Thread(target=warmup_loop, daemon=True)
        self._warmup_thread.start()
        print("Background warmup thread started (keeps model hot)")
    
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
            
            # Read image size from model metadata
            self.image_size = self._get_model_imgsz(model_path)
            self.model_imgsz = self.image_size  # Store for letterboxing
            
            print(f"Loaded tip model from {model_path} (pose={self.is_pose_model})")
            
            # Warmup inference - run MULTIPLE inferences to fully compile OpenVINO kernels
            # First inference compiles, subsequent ones optimize - need 3-5 for full speed
            print("[WARMUP] Compiling OpenVINO kernels (this takes a few seconds on first start)...")
            import time
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            for i in range(5):
                start = time.time()
                _ = self.model(dummy_img, imgsz=self.image_size, verbose=False)
                elapsed = (time.time() - start) * 1000
                print(f"[WARMUP] Inference {i+1}/5: {elapsed:.0f}ms")
            self._last_inference_time = time.time()
            print("[WARMUP] Model fully warmed up and ready!")
            
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
    
    def _get_model_imgsz(self, model_path: Path) -> tuple:
        """Read imgsz from model metadata.yaml.
        
        Returns:
            (height, width) tuple or default (640, 640)
        """
        try:
            import yaml
            meta_file = model_path / "metadata.yaml"
            if meta_file.exists():
                with open(meta_file, 'r') as f:
                    meta = yaml.safe_load(f)
                imgsz = meta.get('imgsz', [640, 640])
                if isinstance(imgsz, list) and len(imgsz) == 2:
                    h, w = imgsz[0], imgsz[1]
                    print(f"[MODEL] Read imgsz from metadata: {h}x{w}")
                    return (h, w)
                elif isinstance(imgsz, int):
                    print(f"[MODEL] Read imgsz from metadata: {imgsz}x{imgsz}")
                    return (imgsz, imgsz)
        except Exception as e:
            print(f"[MODEL] Could not read imgsz from metadata: {e}")
        return (640, 640)  # Default
    
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
        time_since_last = time.time() - self._last_inference_time
        if time_since_last > self._warmup_interval:
            print(f"[WARMUP] Model was idle {time_since_last:.1f}s, warming up...")
            start = time.time()
            self.warmup()
            elapsed = (time.time() - start) * 1000
            print(f"[WARMUP] Done in {elapsed:.0f}ms")
    
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
        
        # Apply letterboxing for non-square models
        # This maintains aspect ratio and pads to model's expected size
        orig_h, orig_w = image.shape[:2]
        target_h, target_w = self.image_size if isinstance(self.image_size, tuple) else (self.image_size, self.image_size)
        
        # Check if letterboxing is needed (model expects different size)
        needs_letterbox = (target_h != target_w) or (orig_h != target_h or orig_w != target_w)
        
        if needs_letterbox:
            letterboxed, lb_scale, lb_pad = letterbox_image(image, (target_h, target_w))
            inference_image = letterboxed
        else:
            inference_image = image
            lb_scale = 1.0
            lb_pad = (0, 0)
        
        # Run inference
        results = self.model(
            inference_image, 
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
                
                # Default to box center (in letterboxed coords)
                box_cx = (x1 + x2) / 2
                box_cy = (y1 + y2) / 2
                cx, cy = box_cx, box_cy
                
                # Convert letterboxed coordinates back to original image space
                if needs_letterbox:
                    x1, y1 = unletterbox_coords(x1, y1, lb_scale, lb_pad)
                    x2, y2 = unletterbox_coords(x2, y2, lb_scale, lb_pad)
                    box_cx, box_cy = unletterbox_coords(box_cx, box_cy, lb_scale, lb_pad)
                    cx, cy = box_cx, box_cy
                used_keypoint = False
                
                # Prefer keypoint if available with good confidence
                kp = None
                if self.is_pose_model and keypoints is not None:
                    try:
                        kp_data = keypoints[i].data.cpu().numpy()
                        if len(kp_data) > 0:
                            tip_kp = kp_data[0][0]  # First keypoint
                            kp_conf = tip_kp[2] if len(tip_kp) > 2 else 0
                            kp_x, kp_y = tip_kp[0], tip_kp[1]
                            # Unletterbox keypoint coordinates
                            if needs_letterbox:
                                kp_x, kp_y = unletterbox_coords(kp_x, kp_y, lb_scale, lb_pad)
                            print(f"[YOLO] Box=({box_cx:.1f},{box_cy:.1f}) KP=({kp_x:.1f},{kp_y:.1f}) kp_conf={kp_conf:.2f} box_conf={conf:.2f}")
                            # Use keypoint if confident and valid
                            if kp_conf > 0.3 and kp_x > 0 and kp_y > 0:
                                cx, cy = kp_x, kp_y
                                used_keypoint = True
                            kp = kp_data[0]
                    except Exception as e:
                        print(f"Error processing keypoints: {e}")
                else:
                    print(f"[YOLO] Box=({box_cx:.1f},{box_cy:.1f}) NO KEYPOINTS (pose={self.is_pose_model})")
                
                # Try to refine tip position using edge detection
                refined_x, refined_y, refine_dist = refine_tip_position(
                    image, cx, cy, (x1, y1, x2, y2), search_radius=20
                )
                if refine_dist > 0:
                    print(f"[REFINE] Tip moved {refine_dist:.1f}px: ({cx:.1f},{cy:.1f}) -> ({refined_x:.1f},{refined_y:.1f})")
                    cx, cy = refined_x, refined_y
                
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
    
    boundary_distance_deg = None  # How far from nearest wire (in degrees)
    
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
        
        # Calculate distance to nearest boundary (in degrees)
        a1 = segment_angles[found_idx]
        a2 = segment_angles[(found_idx + 1) % 20]
        
        # Normalize angle differences
        def angle_diff(a, b):
            d = a - b
            while d > math.pi: d -= 2*math.pi
            while d < -math.pi: d += 2*math.pi
            return abs(d)
        
        dist_to_a1 = angle_diff(angle, a1)
        dist_to_a2 = angle_diff(angle, a2)
        boundary_distance_deg = math.degrees(min(dist_to_a1, dist_to_a2))
        
        # Convert boundary index to segment
        # segment_20_index points to where segment 20 starts
        # DARTBOARD_SEGMENTS[0] = 20, [1] = 1, [2] = 18, etc (clockwise)
        relative_idx = (found_idx - segment_20_index) % 20
        segment = DARTBOARD_SEGMENTS[relative_idx]
        
        # Log with boundary distance
        print(f"[SCORE] angle={math.degrees(angle):.1f}°, segment={segment}, boundary_dist={boundary_distance_deg:.1f}°")
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
    
    # Build result dict - we'll add boundary_distance_deg at the end
    result = None
    
    # Check if outside board
    if outer_double and not point_in_ellipse(point, outer_double):
        result = {
            "score": 0,
            "multiplier": 0,
            "segment": 0,
            "zone": "miss",
            "is_bullseye": False,
            "is_outer_bull": False
        }
    
    # Double ring (between inner and outer double)
    if result is None and outer_double and inner_double:
        in_outer = point_in_ellipse(point, outer_double)
        in_inner = point_in_ellipse(point, inner_double)
        if in_outer and not in_inner:
            result = {
                "score": segment * 2,
                "multiplier": 2,
                "segment": segment,
                "zone": "double",
                "is_bullseye": False,
                "is_outer_bull": False
            }
    
    # Triple ring (between inner and outer triple)
    if result is None and outer_triple and inner_triple:
        in_outer = point_in_ellipse(point, outer_triple)
        in_inner = point_in_ellipse(point, inner_triple)
        if in_outer and not in_inner:
            result = {
                "score": segment * 3,
                "multiplier": 3,
                "segment": segment,
                "zone": "triple",
                "is_bullseye": False,
                "is_outer_bull": False
            }
    
    # Single outer (between triple and double)
    if result is None and inner_double and outer_triple:
        in_double = point_in_ellipse(point, inner_double)
        in_triple = point_in_ellipse(point, outer_triple)
        if in_double and not in_triple:
            result = {
                "score": segment,
                "multiplier": 1,
                "segment": segment,
                "zone": "single_outer",
                "is_bullseye": False,
                "is_outer_bull": False
            }
    
    # Single inner (between bull and triple)
    if result is None and inner_triple and bull_ellipse:
        in_triple = point_in_ellipse(point, inner_triple)
        in_bull = point_in_ellipse(point, bull_ellipse)
        if in_triple and not in_bull:
            result = {
                "score": segment,
                "multiplier": 1,
                "segment": segment,
                "zone": "single_inner",
                "is_bullseye": False,
                "is_outer_bull": False
            }
    
    # Fallback to single
    if result is None:
        result = {
            "score": segment,
            "multiplier": 1,
            "segment": segment,
            "zone": "single",
            "is_bullseye": False,
            "is_outer_bull": False
        }
    
    # Add boundary distance (how close to wire, in degrees)
    if boundary_distance_deg is not None:
        result["boundary_distance_deg"] = boundary_distance_deg
    
    return result


# Global detector instance
_global_detector = None

def get_detector(model_name: str = None) -> DartTipDetector:
    """Get the global detector instance, creating if needed."""
    global _global_detector
    if _global_detector is None or (model_name and _global_detector.model_name != model_name):
        _global_detector = DartTipDetector(model_name=model_name or DEFAULT_TIP_MODEL)
    return _global_detector

def set_detector(detector: DartTipDetector):
    """Set the global detector instance."""
    global _global_detector
    _global_detector = detector
