"""
Dartboard Calibration Module

Handles:
1. Detecting the dartboard in an image
2. Computing homography transformation
3. Drawing scoring zone overlays
4. Storing calibration for reuse
"""
import cv2
import numpy as np
import math
import base64
from io import BytesIO
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

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
    DEGREES_PER_SEGMENT
)
from app.core.scoring import scoring_system
from app.models.schemas import CameraCalibrationResult, DetectResponse, DartScore, DartPosition


def decode_image(image_base64: str) -> np.ndarray:
    """Decode base64 image to OpenCV format."""
    # Handle data URL prefix if present
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
class EllipseParams:
    """Parameters of a fitted ellipse."""
    center: Tuple[float, float]
    axes: Tuple[float, float]  # (major, minor) or (width, height)
    angle: float  # rotation in degrees


class DartboardCalibrator:
    """
    Calibrates camera views of a dartboard using ellipse fitting.
    
    The dartboard appears as an ellipse when viewed from an angle.
    We detect the outer ring, fit an ellipse, and use that to
    establish the coordinate system.
    """
    
    def __init__(self):
        # Model dimensions in pixels (for overlay generation)
        self.model_width = 800
        self.model_height = 800
        self.model_center = (self.model_width // 2, self.model_height // 2)
        
        # Scale: pixels per mm in the model
        self.model_scale = self.model_height / (DARTBOARD_DIAMETER_MM * 1.1)  # Leave margin
    
    def calibrate(
        self, 
        camera_id: str, 
        image_base64: str
    ) -> CameraCalibrationResult:
        """
        Calibrate from a dartboard image.
        
        Args:
            camera_id: Unique camera identifier
            image_base64: Base64 encoded image
            
        Returns:
            CameraCalibrationResult with overlay image
        """
        try:
            # Decode image
            image = decode_image(image_base64)
            h, w = image.shape[:2]
            
            # Detect dartboard (find the outer ring ellipse)
            ellipse = self._detect_dartboard_ellipse(image)
            
            if ellipse is None:
                return CameraCalibrationResult(
                    camera_id=camera_id,
                    success=False,
                    error="Could not detect dartboard. Ensure the full board is visible."
                )
            
            # Calculate calibration quality based on ellipse properties
            quality = self._calculate_calibration_quality(ellipse, image.shape)
            
            # Detect which segment is at top (for rotation)
            segment_at_top = self._detect_segment_at_top(image, ellipse)
            
            # Build calibration data
            calibration_data = {
                "center": ellipse.center,
                "axes": ellipse.axes,
                "angle": ellipse.angle,
                "image_size": (w, h),
                "quality": quality,
                "segment_at_top": segment_at_top,
                "pixels_per_mm": self._calculate_pixels_per_mm(ellipse),
                "rotation_offset": self._calculate_rotation_offset(segment_at_top)
            }
            
            # Generate overlay image
            overlay = self._draw_calibration_overlay(image, ellipse, segment_at_top)
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
            return CameraCalibrationResult(
                camera_id=camera_id,
                success=False,
                error=str(e)
            )
    
    def detect_dart(
        self,
        camera_id: str,
        image_base64: str,
        calibration_data: Dict[str, Any]
    ) -> DetectResponse:
        """
        Detect dart in an image using stored calibration.
        
        For now, this is a placeholder - actual dart tip detection
        would use motion detection or YOLO models.
        """
        # This is where dart detection would happen
        # For now, return a placeholder indicating the system works
        return DetectResponse(
            success=False,
            error="Dart detection not yet implemented. Use calibration overlay to verify setup."
        )
    
    def _detect_dartboard_ellipse(self, image: np.ndarray) -> Optional[EllipseParams]:
        """
        Detect the dartboard's outer ring as an ellipse.
        
        Uses edge detection and ellipse fitting.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate to connect nearby edges
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find the largest contour that could be the dartboard
        best_ellipse = None
        best_area = 0
        
        for contour in contours:
            # Need at least 5 points to fit an ellipse
            if len(contour) < 5:
                continue
            
            # Fit ellipse
            try:
                ellipse = cv2.fitEllipse(contour)
                (cx, cy), (w, h), angle = ellipse
                
                # Filter: must be reasonably circular (aspect ratio)
                aspect = min(w, h) / max(w, h) if max(w, h) > 0 else 0
                if aspect < 0.5:  # Too elongated
                    continue
                
                # Filter: must be reasonably sized (at least 20% of image)
                area = math.pi * (w/2) * (h/2)
                image_area = image.shape[0] * image.shape[1]
                if area < image_area * 0.05:  # Too small
                    continue
                
                if area > best_area:
                    best_area = area
                    best_ellipse = EllipseParams(
                        center=(cx, cy),
                        axes=(w, h),
                        angle=angle
                    )
            except cv2.error:
                continue
        
        return best_ellipse
    
    def _calculate_calibration_quality(
        self, 
        ellipse: EllipseParams, 
        image_shape: Tuple[int, ...]
    ) -> float:
        """
        Calculate calibration quality score (0-1).
        
        Based on:
        - Ellipse circularity (more circular = better angle)
        - Size relative to image (larger = more detail)
        - Center position (centered = better)
        """
        h, w = image_shape[:2]
        
        # Circularity score (1.0 = perfect circle)
        aspect = min(ellipse.axes) / max(ellipse.axes) if max(ellipse.axes) > 0 else 0
        circularity_score = aspect
        
        # Size score (prefer dartboard taking up 50-80% of image)
        ellipse_area = math.pi * (ellipse.axes[0]/2) * (ellipse.axes[1]/2)
        image_area = w * h
        coverage = ellipse_area / image_area
        if coverage < 0.2:
            size_score = coverage / 0.2  # Too small
        elif coverage > 0.9:
            size_score = 1.0 - (coverage - 0.9) / 0.1  # Too large
        else:
            size_score = 1.0
        
        # Center score (prefer center of image)
        cx_norm = abs(ellipse.center[0] - w/2) / (w/2)
        cy_norm = abs(ellipse.center[1] - h/2) / (h/2)
        center_score = 1.0 - (cx_norm + cy_norm) / 2
        
        # Weighted average
        quality = (circularity_score * 0.4 + size_score * 0.3 + center_score * 0.3)
        
        return round(quality, 3)
    
    def _detect_segment_at_top(
        self, 
        image: np.ndarray, 
        ellipse: EllipseParams
    ) -> int:
        """
        Detect which segment is at the top (12 o'clock position).
        
        For now, assumes 20 is at top (standard orientation).
        TODO: Use color/pattern detection to find actual orientation.
        """
        # Placeholder - would use color analysis to find the 20 segment
        return 20
    
    def _calculate_pixels_per_mm(self, ellipse: EllipseParams) -> float:
        """Calculate scale factor from ellipse size."""
        # Average of major/minor axes, divided by known diameter
        avg_diameter_px = (ellipse.axes[0] + ellipse.axes[1]) / 2
        return avg_diameter_px / DARTBOARD_DIAMETER_MM
    
    def _calculate_rotation_offset(self, segment_at_top: int) -> float:
        """
        Calculate rotation offset based on which segment is at top.
        
        Returns offset in radians.
        """
        if segment_at_top not in DARTBOARD_SEGMENTS:
            return 0.0
        
        # Find index of segment at top
        idx = DARTBOARD_SEGMENTS.index(segment_at_top)
        
        # Each segment is 18 degrees
        return math.radians(idx * DEGREES_PER_SEGMENT)
    
    def _draw_calibration_overlay(
        self, 
        image: np.ndarray, 
        ellipse: EllipseParams,
        segment_at_top: int = 20
    ) -> np.ndarray:
        """
        Draw dartboard scoring zones overlay on the image.
        """
        overlay = image.copy()
        cx, cy = ellipse.center
        
        # Calculate radii in pixels for each ring
        px_per_mm = self._calculate_pixels_per_mm(ellipse)
        
        # Scale factors for ellipse distortion
        scale_x = ellipse.axes[0] / (DARTBOARD_DIAMETER_MM * px_per_mm)
        scale_y = ellipse.axes[1] / (DARTBOARD_DIAMETER_MM * px_per_mm)
        
        def get_ellipse_axes(radius_mm: float) -> Tuple[int, int]:
            """Convert mm radius to ellipse axes in pixels."""
            base_px = radius_mm * px_per_mm * 2  # diameter
            return (int(base_px * scale_x), int(base_px * scale_y))
        
        # Draw rings (outer to inner)
        rings = [
            (DOUBLE_OUTER_RADIUS_MM, (0, 0, 255), 2),      # Red - double outer
            (DOUBLE_INNER_RADIUS_MM, (0, 0, 200), 1),      # Dark red - double inner
            (TRIPLE_OUTER_RADIUS_MM, (0, 255, 0), 2),      # Green - triple outer
            (TRIPLE_INNER_RADIUS_MM, (0, 200, 0), 1),      # Dark green - triple inner
            (OUTER_BULL_RADIUS_MM, (255, 0, 0), 2),        # Blue - outer bull
            (BULL_RADIUS_MM, (255, 0, 0), -1),             # Blue filled - inner bull
        ]
        
        for radius_mm, color, thickness in rings:
            axes = get_ellipse_axes(radius_mm)
            cv2.ellipse(
                overlay, 
                (int(cx), int(cy)), 
                (axes[0]//2, axes[1]//2),
                ellipse.angle,
                0, 360,
                color,
                thickness
            )
        
        # Draw segment lines
        rotation_offset = self._calculate_rotation_offset(segment_at_top)
        
        for i in range(20):
            # Angle for this segment boundary (between segments)
            angle = SEGMENT_ANGLE_OFFSET + (i * math.pi / 10) - rotation_offset
            
            # Convert ellipse angle to radians for rotation
            ellipse_angle_rad = math.radians(ellipse.angle)
            
            # Calculate line endpoint on outer ring
            # Account for ellipse rotation and aspect ratio
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            
            # Apply ellipse transformation
            outer_r = DOUBLE_OUTER_RADIUS_MM * px_per_mm
            inner_r = OUTER_BULL_RADIUS_MM * px_per_mm
            
            # Simple approximation (for perfect view)
            x_outer = cx + cos_a * outer_r * scale_x
            y_outer = cy + sin_a * outer_r * scale_y
            x_inner = cx + cos_a * inner_r * scale_x
            y_inner = cy + sin_a * inner_r * scale_y
            
            cv2.line(
                overlay,
                (int(x_inner), int(y_inner)),
                (int(x_outer), int(y_outer)),
                (255, 255, 255),
                1
            )
        
        # Draw segment numbers
        for i, segment in enumerate(DARTBOARD_SEGMENTS):
            # Angle to center of segment
            angle = SEGMENT_ANGLE_OFFSET + (i * math.pi / 10) + (math.pi / 20) - rotation_offset
            
            # Position text between triple and double rings
            text_r = (TRIPLE_OUTER_RADIUS_MM + DOUBLE_INNER_RADIUS_MM) / 2 * px_per_mm
            
            x = cx + math.cos(angle) * text_r * scale_x
            y = cy + math.sin(angle) * text_r * scale_y
            
            # Draw segment number
            text = str(segment)
            font_scale = 0.6
            thickness = 2
            
            # Get text size for centering
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            cv2.putText(
                overlay,
                text,
                (int(x - text_w/2), int(y + text_h/2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 255, 255),  # Yellow
                thickness
            )
        
        # Add quality indicator
        cv2.putText(
            overlay,
            f"Calibrated - {segment_at_top} at top",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        
        return overlay
