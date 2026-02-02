"""
Camera focus measurement tool using multiple sharpness metrics.
Helps users achieve optimal camera focus before calibration.

Based on Autodarts approach - uses Laplacian, Tenengrad, and Brenner metrics.
"""
import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


def laplacian_variance(gray: np.ndarray) -> float:
    """Calculate Laplacian variance (edge sharpness)."""
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    return float(lap.var())


def tenengrad(gray: np.ndarray) -> float:
    """Calculate Tenengrad (gradient magnitude)."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    t = gx * gx + gy * gy
    return float(t.mean())


def brenner(gray: np.ndarray) -> float:
    """Calculate Brenner focus measure."""
    diff = gray[:, 2:].astype(np.float64) - gray[:, :-2].astype(np.float64)
    return float((diff * diff).mean())


def normalize_score(x: float, lo: float, hi: float, clip: bool = True) -> float:
    """Normalize value to 0-1 range."""
    if hi == lo:
        return 0.0
    s = (x - lo) / (hi - lo)
    if clip:
        s = max(0.0, min(1.0, s))
    return s


@dataclass
class FocusMetrics:
    """Focus measurement results."""
    laplacian: float
    tenengrad: float
    brenner: float
    combined_score: float  # 0-100 scale
    quality: str  # "poor", "fair", "good", "excellent"
    
    def to_dict(self) -> Dict:
        return {
            "laplacian": round(self.laplacian, 2),
            "tenengrad": round(self.tenengrad, 2),
            "brenner": round(self.brenner, 2),
            "combined_score": round(self.combined_score, 1),
            "quality": self.quality
        }


def calculate_focus_score(
    frame: np.ndarray,
    center: Optional[Tuple[int, int]] = None,
    roi_size: int = 400
) -> FocusMetrics:
    """
    Calculate comprehensive focus score for a frame.
    
    Args:
        frame: BGR image
        center: Center point for ROI (defaults to image center)
        roi_size: Size of ROI to analyze
        
    Returns:
        FocusMetrics with individual metrics and combined score (0-100)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    
    # Default to image center
    if center is None:
        center = (w // 2, h // 2)
    
    # Extract ROI around center
    cx, cy = center
    half = roi_size // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)
    
    roi = gray[y1:y2, x1:x2]
    
    if roi.size == 0:
        return FocusMetrics(0, 0, 0, 0, "poor")
    
    # Calculate metrics
    lap = laplacian_variance(roi)
    ten = tenengrad(roi)
    bre = brenner(roi)
    
    # Normalize to 0-1 based on typical ranges
    # These ranges are calibrated for 1080p dartboard images
    lap_norm = normalize_score(lap, 50, 2000)      # Laplacian variance range
    ten_norm = normalize_score(ten, 500, 20000)    # Tenengrad range
    bre_norm = normalize_score(bre, 100, 5000)     # Brenner range
    
    # Combined score (weighted average)
    combined = (lap_norm * 0.4 + ten_norm * 0.4 + bre_norm * 0.2) * 100
    
    # Quality assessment
    if combined >= 75:
        quality = "excellent"
    elif combined >= 50:
        quality = "good"
    elif combined >= 25:
        quality = "fair"
    else:
        quality = "poor"
    
    return FocusMetrics(
        laplacian=lap,
        tenengrad=ten,
        brenner=bre,
        combined_score=combined,
        quality=quality
    )


def draw_focus_overlay(
    frame: np.ndarray,
    metrics: FocusMetrics,
    center: Optional[Tuple[int, int]] = None,
    roi_size: int = 400
) -> np.ndarray:
    """
    Draw focus helper overlay on frame.
    Shows ROI box and score indicator.
    """
    result = frame.copy()
    h, w = frame.shape[:2]
    
    if center is None:
        center = (w // 2, h // 2)
    
    cx, cy = center
    half = roi_size // 2
    
    # Draw ROI box
    color = (0, 255, 0) if metrics.quality in ["good", "excellent"] else \
            (0, 255, 255) if metrics.quality == "fair" else (0, 0, 255)
    
    cv2.rectangle(result, (cx - half, cy - half), (cx + half, cy + half), color, 2)
    
    # Draw score bar
    bar_width = 200
    bar_height = 20
    bar_x = 20
    bar_y = h - 50
    
    # Background
    cv2.rectangle(result, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
    
    # Fill based on score
    fill_width = int(bar_width * metrics.combined_score / 100)
    cv2.rectangle(result, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)
    
    # Score text
    text = f"Focus: {metrics.combined_score:.0f}% ({metrics.quality})"
    cv2.putText(result, text, (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return result
