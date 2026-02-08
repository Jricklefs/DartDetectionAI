"""
Skeleton-based Dart Tip Detection v5

Simpler approach: The tip is the point on the detected dart region
that is CLOSEST to the board center (darts point at the bullseye).
"""
import cv2
import numpy as np
import logging
from typing import Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger("dartdetect.skeleton")


@dataclass
class SkeletonTip:
    """Detected tip from skeleton analysis."""
    x: float
    y: float
    confidence: float
    method: str = "skeleton"


def find_dart_mask(current_frame: np.ndarray, previous_frame: np.ndarray,
                   threshold: float = 20.0) -> np.ndarray:
    """Create binary mask of new dart using frame differencing."""
    if len(current_frame.shape) == 3:
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    else:
        current_gray = current_frame
        
    if len(previous_frame.shape) == 3:
        previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    else:
        previous_gray = previous_frame
    
    diff = cv2.absdiff(current_gray, previous_gray)
    diff = cv2.GaussianBlur(diff, (3, 3), 0)
    _, binary = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Connect nearby regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.dilate(binary, kernel, iterations=2)
    binary = cv2.erode(binary, kernel, iterations=1)
    
    return binary.astype(np.uint8)


def find_tip_closest_to_center(mask: np.ndarray, board_center: Tuple[float, float],
                                min_area: int = 50) -> Optional[Tuple[float, float, float]]:
    """
    Find the point on the dart mask that's closest to board center.
    Returns (x, y, confidence) or None.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    cx, cy = board_center
    
    best_point = None
    best_dist = float('inf')
    best_area = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        
        # Check every point on the contour
        for pt in cnt:
            px, py = pt[0]
            dist = np.sqrt((px - cx)**2 + (py - cy)**2)
            if dist < best_dist:
                best_dist = dist
                best_point = (float(px), float(py))
                best_area = area
    
    if best_point is None:
        return None
    
    confidence = min(1.0, best_area / 1000.0)
    return (best_point[0], best_point[1], confidence)


def detect_tip_skeleton(current_frame: np.ndarray, previous_frame: np.ndarray,
                        board_center: Tuple[float, float] = None,
                        diff_threshold: float = 20.0,
                        min_contour_area: int = 50) -> Optional[SkeletonTip]:
    """
    Detect dart tip: find the point on the diff mask closest to board center.
    """
    if board_center is None:
        board_center = (current_frame.shape[1] // 2, current_frame.shape[0] // 2)
    
    # Get dart mask
    mask = find_dart_mask(current_frame, previous_frame, diff_threshold)
    
    # Find point closest to center
    result = find_tip_closest_to_center(mask, board_center, min_contour_area)
    
    if result is None:
        return None
    
    x, y, confidence = result
    
    return SkeletonTip(x=x, y=y, confidence=confidence)
