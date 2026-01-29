"""
Dartboard Calibration Module

Uses YOLO to detect calibration points (wire intersections), 
then fits ellipses to determine the dartboard geometry.

Based on the original DartDetector approach.
"""
import cv2
import numpy as np
import math
import base64
import os
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
    DARTBOARD_DIAMETER_MM,
    SEGMENT_ANGLE_OFFSET,
)
from app.core.scoring import scoring_system
from app.models.schemas import CameraCalibrationResult, DetectResponse, DartScore, DartPosition

# Get the models directory
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
CALIBRATION_MODEL_PATH = MODELS_DIR / "dartboard1280imgz_int8_openvino_model"


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
    segment_angles: Optional[List[float]] = None
    rotation_offset_deg: float = 0.0


class YOLOCalibrationDetector:
    """
    Uses YOLO to detect dartboard calibration points.
    
    Detects:
    - 'cal': Outer double ring intersection points (wire meets outer ring)
    - 'cal1': Triple ring intersection points
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
            
            self.model = YOLO(str(model_path))
            self.is_initialized = True
            print(f"Loaded calibration model from {model_path}")
            
        except ImportError:
            print("Warning: ultralytics not installed. YOLO detection disabled.")
        except Exception as e:
            print(f"Warning: Failed to load YOLO model: {e}")
    
    def detect_calibration_points(
        self, 
        image: np.ndarray,
        confidence_threshold: float = 0.5
    ) -> Dict[str, List[Tuple[float, float, float]]]:
        """
        Detect calibration points in the image.
        
        Returns dict with keys 'cal', 'cal1' containing lists of (x, y, confidence).
        """
        if not self.is_initialized or self.model is None:
            return {'cal': [], 'cal1': []}
        
        # Resize image to model input size (1280x1280)
        h, w = image.shape[:2]
        input_size = 1280
        
        # Run inference
        results = self.model(image, imgsz=input_size, conf=confidence_threshold, verbose=False)
        
        points = {'cal': [], 'cal1': [], 'cal2': [], 'cal3': []}
        
        for result in results:
            if result.boxes is None:
                continue
                
            boxes = result.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                
                # Get center of bounding box
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                # Map class ID to name
                # Based on original: tip=0, cal=1, cal1=2
                if cls_id == 1:
                    points['cal'].append((cx, cy, conf))
                elif cls_id == 2:
                    points['cal1'].append((cx, cy, conf))
                elif cls_id == 3:
                    points['cal2'].append((cx, cy, conf))
                elif cls_id == 4:
                    points['cal3'].append((cx, cy, conf))
        
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


def estimate_center_from_points(cal_points: List, cal1_points: List) -> Optional[Tuple[float, float]]:
    """Estimate dartboard center from calibration points."""
    all_points = [(p[0], p[1]) for p in cal_points + cal1_points]
    if len(all_points) < 3:
        return None
    
    pts = np.array(all_points, dtype=np.float32)
    centroid = np.mean(pts, axis=0)
    return (float(centroid[0]), float(centroid[1]))


class DartboardCalibrator:
    """
    Calibrates camera views of a dartboard using YOLO detection and ellipse fitting.
    """
    
    def __init__(self):
        self.detector = YOLOCalibrationDetector()
        
        # Model dimensions
        self.image_width = 1280
        self.image_height = 720
    
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
            
            print(f"Detected {len(cal_points)} cal points, {len(cal1_points)} cal1 points")
            
            if len(cal_points) < 8:
                return CameraCalibrationResult(
                    camera_id=camera_id,
                    success=False,
                    error=f"Not enough calibration points detected. Found {len(cal_points)} outer ring points, need at least 8. Ensure full dartboard is visible and well-lit."
                )
            
            # Fit ellipses to the detected points
            outer_double_ellipse = fit_ellipse_from_points(cal_points)
            outer_triple_ellipse = fit_ellipse_from_points(cal1_points) if len(cal1_points) >= 5 else None
            
            if outer_double_ellipse is None:
                return CameraCalibrationResult(
                    camera_id=camera_id,
                    success=False,
                    error="Could not fit ellipse to detected points."
                )
            
            # Estimate center
            center = (outer_double_ellipse[0][0], outer_double_ellipse[0][1])
            
            # Build calibration data
            ellipse_cal = EllipseCalibration(
                center=center,
                outer_double_ellipse=outer_double_ellipse,
                outer_triple_ellipse=outer_triple_ellipse,
            )
            
            # Estimate inner ellipses based on dartboard proportions
            ellipse_cal = self._estimate_inner_rings(ellipse_cal)
            
            # Calculate quality
            quality = self._calculate_quality(cal_points, cal1_points, outer_double_ellipse, h, w)
            
            # Detect segment at top (TODO: use "20" detection from YOLO)
            segment_at_top = 20
            
            # Build calibration data dict
            calibration_data = {
                "center": center,
                "outer_double_ellipse": outer_double_ellipse,
                "outer_triple_ellipse": outer_triple_ellipse,
                "image_size": (w, h),
                "quality": quality,
                "segment_at_top": segment_at_top,
                "cal_points": cal_points,
                "cal1_points": cal1_points,
            }
            
            # Generate overlay image
            overlay = self._draw_calibration_overlay(image, ellipse_cal, cal_points, cal1_points)
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
    
    def _estimate_inner_rings(self, cal: EllipseCalibration) -> EllipseCalibration:
        """Estimate inner ring ellipses based on dartboard proportions."""
        if cal.outer_double_ellipse is None:
            return cal
        
        (cx, cy), (w, h), angle = cal.outer_double_ellipse
        
        # Proportions relative to outer double ring (170mm)
        ratios = {
            'inner_double': DOUBLE_INNER_RADIUS_MM / DOUBLE_OUTER_RADIUS_MM,  # 162/170
            'outer_triple': TRIPLE_OUTER_RADIUS_MM / DOUBLE_OUTER_RADIUS_MM,  # 107/170
            'inner_triple': TRIPLE_INNER_RADIUS_MM / DOUBLE_OUTER_RADIUS_MM,  # 99/170
            'bull': OUTER_BULL_RADIUS_MM / DOUBLE_OUTER_RADIUS_MM,            # 15.9/170
            'bullseye': BULL_RADIUS_MM / DOUBLE_OUTER_RADIUS_MM,              # 6.35/170
        }
        
        cal.inner_double_ellipse = ((cx, cy), (w * ratios['inner_double'], h * ratios['inner_double']), angle)
        
        if cal.outer_triple_ellipse is None:
            cal.outer_triple_ellipse = ((cx, cy), (w * ratios['outer_triple'], h * ratios['outer_triple']), angle)
        
        cal.inner_triple_ellipse = ((cx, cy), (w * ratios['inner_triple'], h * ratios['inner_triple']), angle)
        cal.bull_ellipse = ((cx, cy), (w * ratios['bull'], h * ratios['bull']), angle)
        cal.bullseye_ellipse = ((cx, cy), (w * ratios['bullseye'], h * ratios['bullseye']), angle)
        
        return cal
    
    def _calculate_quality(
        self,
        cal_points: List,
        cal1_points: List,
        ellipse: Tuple,
        img_h: int,
        img_w: int
    ) -> float:
        """Calculate calibration quality score."""
        # Factors:
        # 1. Number of detected points (more = better)
        # 2. Ellipse circularity (more circular = camera more perpendicular)
        # 3. Coverage (ellipse size relative to image)
        
        point_score = min(1.0, (len(cal_points) + len(cal1_points)) / 40)  # Expect ~40 points
        
        (cx, cy), (w, h), angle = ellipse
        circularity = min(w, h) / max(w, h) if max(w, h) > 0 else 0
        
        ellipse_area = math.pi * (w/2) * (h/2)
        image_area = img_w * img_h
        coverage = ellipse_area / image_area
        coverage_score = min(1.0, coverage / 0.3)  # Expect ~30% coverage
        
        quality = (point_score * 0.4 + circularity * 0.3 + coverage_score * 0.3)
        return round(quality, 3)
    
    def _draw_calibration_overlay(
        self,
        image: np.ndarray,
        cal: EllipseCalibration,
        cal_points: List,
        cal1_points: List
    ) -> np.ndarray:
        """Draw dartboard overlay showing detected rings and points."""
        overlay = image.copy()
        
        # Draw detected calibration points
        for x, y, conf in cal_points:
            cv2.circle(overlay, (int(x), int(y)), 5, (0, 255, 0), -1)  # Green for outer
        
        for x, y, conf in cal1_points:
            cv2.circle(overlay, (int(x), int(y)), 5, (255, 165, 0), -1)  # Orange for triple
        
        # Draw fitted ellipses
        if cal.outer_double_ellipse:
            cv2.ellipse(overlay, 
                       (int(cal.outer_double_ellipse[0][0]), int(cal.outer_double_ellipse[0][1])),
                       (int(cal.outer_double_ellipse[1][0]/2), int(cal.outer_double_ellipse[1][1]/2)),
                       cal.outer_double_ellipse[2], 0, 360, (0, 0, 255), 2)
        
        if cal.inner_double_ellipse:
            cv2.ellipse(overlay,
                       (int(cal.inner_double_ellipse[0][0]), int(cal.inner_double_ellipse[0][1])),
                       (int(cal.inner_double_ellipse[1][0]/2), int(cal.inner_double_ellipse[1][1]/2)),
                       cal.inner_double_ellipse[2], 0, 360, (0, 0, 200), 1)
        
        if cal.outer_triple_ellipse:
            cv2.ellipse(overlay,
                       (int(cal.outer_triple_ellipse[0][0]), int(cal.outer_triple_ellipse[0][1])),
                       (int(cal.outer_triple_ellipse[1][0]/2), int(cal.outer_triple_ellipse[1][1]/2)),
                       cal.outer_triple_ellipse[2], 0, 360, (0, 255, 0), 2)
        
        if cal.inner_triple_ellipse:
            cv2.ellipse(overlay,
                       (int(cal.inner_triple_ellipse[0][0]), int(cal.inner_triple_ellipse[0][1])),
                       (int(cal.inner_triple_ellipse[1][0]/2), int(cal.inner_triple_ellipse[1][1]/2)),
                       cal.inner_triple_ellipse[2], 0, 360, (0, 200, 0), 1)
        
        if cal.bull_ellipse:
            cv2.ellipse(overlay,
                       (int(cal.bull_ellipse[0][0]), int(cal.bull_ellipse[0][1])),
                       (int(cal.bull_ellipse[1][0]/2), int(cal.bull_ellipse[1][1]/2)),
                       cal.bull_ellipse[2], 0, 360, (255, 0, 0), 2)
        
        if cal.bullseye_ellipse:
            cv2.ellipse(overlay,
                       (int(cal.bullseye_ellipse[0][0]), int(cal.bullseye_ellipse[0][1])),
                       (int(cal.bullseye_ellipse[1][0]/2), int(cal.bullseye_ellipse[1][1]/2)),
                       cal.bullseye_ellipse[2], 0, 360, (255, 0, 0), -1)
        
        # Draw segment lines
        if cal.outer_double_ellipse:
            cx, cy = cal.center
            (_, _), (w, h), angle = cal.outer_double_ellipse
            outer_r = max(w, h) / 2
            inner_r = outer_r * (OUTER_BULL_RADIUS_MM / DOUBLE_OUTER_RADIUS_MM)
            
            for i in range(20):
                seg_angle = SEGMENT_ANGLE_OFFSET + (i * math.pi / 10)
                
                x_outer = cx + math.cos(seg_angle) * outer_r
                y_outer = cy + math.sin(seg_angle) * outer_r
                x_inner = cx + math.cos(seg_angle) * inner_r
                y_inner = cy + math.sin(seg_angle) * inner_r
                
                cv2.line(overlay, (int(x_inner), int(y_inner)), (int(x_outer), int(y_outer)), (255, 255, 255), 1)
            
            # Draw segment numbers
            text_r = outer_r * ((TRIPLE_OUTER_RADIUS_MM + DOUBLE_INNER_RADIUS_MM) / 2 / DOUBLE_OUTER_RADIUS_MM)
            for i, segment in enumerate(DARTBOARD_SEGMENTS):
                seg_angle = SEGMENT_ANGLE_OFFSET + (i * math.pi / 10) + (math.pi / 20)
                x = cx + math.cos(seg_angle) * text_r
                y = cy + math.sin(seg_angle) * text_r
                
                cv2.putText(overlay, str(segment), (int(x)-10, int(y)+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Draw center point
        cv2.circle(overlay, (int(cal.center[0]), int(cal.center[1])), 8, (0, 0, 255), -1)
        
        # Add info text
        cv2.putText(overlay, f"Cal points: {len(cal_points)} outer, {len(cal1_points)} triple",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return overlay
    
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
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
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
