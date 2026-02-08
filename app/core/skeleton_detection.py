#!/usr/bin/env python3
"""
Dart tip detection using medial axis + local width.
The TIP is the narrowest part of the skeleton.
"""

import cv2
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

# Detection method switching
_DETECTION_METHOD = "skeleton"  # "yolo" or "skeleton"

def set_detection_method(method: str) -> bool:
    """Set the detection method (yolo or skeleton)."""
    global _DETECTION_METHOD
    if method.lower() in ("yolo", "skeleton"):
        _DETECTION_METHOD = method.lower()
        logger.info(f"Detection method set to: {_DETECTION_METHOD}")
        return True
    return False

def get_detection_method() -> str:
    """Get the current detection method."""
    return _DETECTION_METHOD



def find_skeleton_endpoints(skeleton):
    """Find endpoints (pixels with exactly 1 neighbor)."""
    skel = (skeleton > 0).astype(np.uint8)
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel, -1, kernel)
    endpoints = np.where((skel == 1) & (neighbor_count == 1))
    return list(zip(endpoints[1], endpoints[0]))  # (x, y)


def detect_dart_skeleton(
    current_frame: np.ndarray,
    previous_frame: np.ndarray,
    center: tuple = None,
    mask: np.ndarray = None,
    debug: bool = True
) -> dict:
    """
    Detect dart tip using skeleton + distance transform.
    The tip is at the skeleton endpoint with MINIMUM width.
    """
    result = {"tip": None, "confidence": 0.0, "method": "skeleton_width"}
    
    if current_frame is None or previous_frame is None:
        return result
    
    if center is None:
        center = (current_frame.shape[1] // 2, current_frame.shape[0] // 2)
    
    cx, cy = center
    
    # 1. Get differential mask
    if mask is not None:
        thresh = np.where(mask == 76, 255, 0).astype(np.uint8)
    else:
        diff = cv2.absdiff(current_frame, previous_frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_diff, 20, 255, cv2.THRESH_BINARY)
    
    # 2. Morphology cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # 3. Find contours - get most elongated
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return result
    
    best_contour = None
    best_aspect = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 200 or area > 50000:
            continue
        
        rect = cv2.minAreaRect(contour)
        w, h = rect[1]
        if w == 0 or h == 0:
            continue
        
        aspect = max(w, h) / min(w, h)
        if aspect > best_aspect:
            best_aspect = aspect
            best_contour = contour
    
    if best_contour is None or best_aspect < 2.0:
        return result
    
    # 4. Simple approach: Find contour point closest to board center
    # The tip is always the point closest to where it hits the board
    contour_points = best_contour.reshape(-1, 2)
    
    # Calculate distance from each contour point to board center
    distances = np.sqrt((contour_points[:, 0] - cx)**2 + (contour_points[:, 1] - cy)**2)
    closest_idx = np.argmin(distances)
    tip_x, tip_y = float(contour_points[closest_idx][0]), float(contour_points[closest_idx][1])
    
    # Confidence based on aspect ratio (more elongated = more confident)
    result["confidence"] = min(1.0, best_aspect / 5.0)  # Max confidence at aspect 5:1
    result["tip"] = (tip_x, tip_y)
    
    # Keep skeleton stuff for debug compatibility
    endpoint_widths = []
    endpoints = [(int(tip_x), int(tip_y))]
    
    # Debug - simplified for contour-based approach
    if debug:
        debug_dir = r"C:\Users\clawd\DartDetectionAI\skeleton_debug"
        os.makedirs(debug_dir, exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        cv2.imwrite(os.path.join(debug_dir, f"{timestamp}_1_thresh.jpg"), thresh)
        
        result_img = current_frame.copy()
        cv2.drawContours(result_img, [best_contour], -1, (0, 255, 0), 2)
        
        # Mark tip
        cv2.circle(result_img, (int(tip_x), int(tip_y)), 8, (0, 0, 255), -1)  # Red=tip
        cv2.circle(result_img, (int(cx), int(cy)), 5, (255, 0, 0), -1)  # Blue=center
        
        cv2.imwrite(os.path.join(debug_dir, f"{timestamp}_2_result.jpg"), result_img)
    
    return result
