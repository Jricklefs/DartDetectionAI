"""
Skeleton-based dart tip detection (Autodarts-style classical CV).

Pipeline:
1. Frame differencing to isolate new dart
2. Skeletonization of the dart shape
3. Line fitting to find dart axis
4. Lowest point (closest to board center) is the tip
"""
import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
import math


def detect_dart_skeleton(
    current_frame: np.ndarray,
    previous_frame: np.ndarray,
    center: Tuple[float, float],
    mask: Optional[np.ndarray] = None,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Detect dart tip using skeleton-based approach.
    
    Args:
        current_frame: Current camera frame (BGR)
        previous_frame: Previous frame before dart landed (BGR)
        center: Board center (x, y) in pixels
        mask: Optional mask to restrict detection area
        debug: Return debug images
        
    Returns:
        Dict with tip position, confidence, and optional debug info
    """
    result = {
        "tip": None,
        "confidence": 0.0,
        "method": "skeleton",
        "debug": {}
    }
    
    # 1. Frame differencing
    diff = cv2.absdiff(current_frame, previous_frame)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray_diff, (5, 5), 0)
    
    # Threshold to get binary mask of changes
    _, thresh = cv2.threshold(blurred, 25, 255, cv2.THRESH_BINARY)
    
    # Apply board mask if provided
    if mask is not None:
        thresh = cv2.bitwise_and(thresh, mask)
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    if debug:
        result["debug"]["diff_mask"] = thresh.copy()
    
    # 2. Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return result
    
    # Filter contours by area (dart should be reasonable size)
    min_area = 100
    max_area = 50000
    valid_contours = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
    
    if not valid_contours:
        return result
    
    # Take largest contour as the dart
    dart_contour = max(valid_contours, key=cv2.contourArea)
    
    # 3. Create mask for just the dart
    dart_mask = np.zeros(thresh.shape, dtype=np.uint8)
    cv2.drawContours(dart_mask, [dart_contour], -1, 255, -1)
    
    # 4. Skeletonize
    skeleton = skeletonize(dart_mask)
    
    if debug:
        result["debug"]["skeleton"] = skeleton.copy()
    
    # 5. Find skeleton points
    skel_points = np.column_stack(np.where(skeleton > 0))  # (row, col) = (y, x)
    
    if len(skel_points) < 3:
        # Fallback to contour centroid
        M = cv2.moments(dart_contour)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            result["tip"] = (cx, cy)
            result["confidence"] = 0.3
        return result
    
    # 6. Fit line to skeleton
    # Convert to (x, y) format for cv2.fitLine
    points_xy = skel_points[:, ::-1].astype(np.float32)  # (x, y)
    
    try:
        vx, vy, x0, y0 = cv2.fitLine(points_xy, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    except:
        # Fallback
        M = cv2.moments(dart_contour)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            result["tip"] = (cx, cy)
            result["confidence"] = 0.3
        return result
    
    # 7. Find the point on skeleton closest to board center
    # (the tip is the end pointing toward the center)
    cx, cy = center
    
    # Calculate distance from each skeleton point to center
    distances = np.sqrt((points_xy[:, 0] - cx)**2 + (points_xy[:, 1] - cy)**2)
    
    # The tip is the skeleton point closest to center
    tip_idx = np.argmin(distances)
    tip_x, tip_y = points_xy[tip_idx]
    
    # Refine tip position using contour
    # Find contour point closest to the skeleton tip
    contour_points = dart_contour.reshape(-1, 2)
    contour_dists = np.sqrt((contour_points[:, 0] - tip_x)**2 + (contour_points[:, 1] - tip_y)**2)
    closest_contour_idx = np.argmin(contour_dists)
    refined_tip = contour_points[closest_contour_idx]
    
    result["tip"] = (float(refined_tip[0]), float(refined_tip[1]))
    result["confidence"] = 0.7
    
    if debug:
        # Draw debug visualization
        debug_img = current_frame.copy()
        cv2.drawContours(debug_img, [dart_contour], -1, (0, 255, 0), 2)
        cv2.circle(debug_img, (int(refined_tip[0]), int(refined_tip[1])), 5, (0, 0, 255), -1)
        cv2.circle(debug_img, (int(cx), int(cy)), 5, (255, 0, 0), -1)
        result["debug"]["visualization"] = debug_img
    
    return result


def skeletonize(binary_image: np.ndarray) -> np.ndarray:
    """
    Skeletonize a binary image using morphological operations.
    Zhang-Suen thinning algorithm approximation.
    """
    skeleton = np.zeros(binary_image.shape, dtype=np.uint8)
    img = binary_image.copy()
    
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    while True:
        eroded = cv2.erode(img, kernel)
        dilated = cv2.dilate(eroded, kernel)
        diff = cv2.subtract(img, dilated)
        skeleton = cv2.bitwise_or(skeleton, diff)
        img = eroded.copy()
        
        if cv2.countNonZero(img) == 0:
            break
    
    return skeleton


# Global detection method state
_detection_method = "yolo"  # "yolo" or "skeleton"


def set_detection_method(method: str) -> bool:
    """Set the active detection method."""
    global _detection_method
    if method not in ("yolo", "skeleton"):
        return False
    _detection_method = method
    print(f"[DETECTION] Method set to: {method}")
    return True


def get_detection_method() -> str:
    """Get the current detection method."""
    return _detection_method
