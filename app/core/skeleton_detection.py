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
    
    # 1. Use provided mask if available (from pipeline's differential detection)
    # The mask already identifies NEW dart pixels (value=76 for NEW region)
    if mask is not None and np.any(mask > 0):
        # Pipeline mask uses: 0=background, 76=NEW, 152=OLD
        # Extract just the NEW region
        thresh = np.where(mask == 76, 255, 0).astype(np.uint8)
        
        # If no NEW pixels, fall back to frame diff
        if np.sum(thresh) == 0:
            diff = cv2.absdiff(current_frame, previous_frame)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray_diff, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 25, 255, cv2.THRESH_BINARY)
    else:
        # No mask - use frame differencing
        diff = cv2.absdiff(current_frame, previous_frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_diff, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 25, 255, cv2.THRESH_BINARY)
    
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
    
    # 6. Fit line to skeleton to get dart axis direction
    # Convert to (x, y) format for cv2.fitLine
    points_xy = skel_points[:, ::-1].astype(np.float32)  # (x, y)
    
    try:
        line_params = cv2.fitLine(points_xy, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        vx, vy, x0, y0 = line_params
    except:
        # Fallback to centroid
        M = cv2.moments(dart_contour)
        if M["m00"] > 0:
            cx_contour = M["m10"] / M["m00"]
            cy_contour = M["m01"] / M["m00"]
            result["tip"] = (cx_contour, cy_contour)
            result["confidence"] = 0.3
        return result
    
    # 7. Find the two endpoints of the skeleton along the fitted line
    # Project all skeleton points onto the line direction
    cx, cy = center
    
    # Line direction vector (normalized)
    line_vec = np.array([vx, vy])
    line_vec = line_vec / np.linalg.norm(line_vec)
    
    # Project each skeleton point onto the line
    projections = []
    for pt in points_xy:
        # Vector from line point to skeleton point
        diff = pt - np.array([x0, y0])
        proj_dist = np.dot(diff, line_vec)
        projections.append(proj_dist)
    
    projections = np.array(projections)
    
    # Find the two extreme points (endpoints of skeleton)
    min_idx = np.argmin(projections)
    max_idx = np.argmax(projections)
    
    endpoint1 = points_xy[min_idx]
    endpoint2 = points_xy[max_idx]
    
    # 8. The TIP is the endpoint CLOSER to the board center
    dist1 = np.sqrt((endpoint1[0] - cx)**2 + (endpoint1[1] - cy)**2)
    dist2 = np.sqrt((endpoint2[0] - cx)**2 + (endpoint2[1] - cy)**2)
    
    if dist1 < dist2:
        tip_endpoint = endpoint1
    else:
        tip_endpoint = endpoint2
    
    # 9. EXTEND the line beyond the detected contour toward board center
    # The actual metal tip is not visible in frame diff - extend ~40px along dart axis
    
    # Direction from flight toward tip (toward center)
    if dist1 < dist2:
        # endpoint1 is closer to center, endpoint2 is flight
        tip_direction = endpoint1 - endpoint2
    else:
        # endpoint2 is closer to center, endpoint1 is flight  
        tip_direction = endpoint2 - endpoint1
    
    # Normalize direction
    dir_len = np.linalg.norm(tip_direction)
    if dir_len > 0:
        tip_direction = tip_direction / dir_len
    
    # Extend 40 pixels beyond the detected endpoint toward center
    extension_px = 40
    extended_tip = tip_endpoint + tip_direction * extension_px
    
    # Make sure we don't go past the center
    dist_to_center = np.sqrt((extended_tip[0] - cx)**2 + (extended_tip[1] - cy)**2)
    if dist_to_center < 20:  # Too close to center, pull back
        extended_tip = tip_endpoint + tip_direction * (extension_px / 2)
    
    refined_tip = extended_tip
    
    result["tip"] = (float(refined_tip[0]), float(refined_tip[1]))
    result["confidence"] = 0.7
    
    # Always save debug images for now (to diagnose issues)
    import os
    from datetime import datetime
    debug_dir = r"C:\Users\clawd\DartDetectionAI\skeleton_debug"
    os.makedirs(debug_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    # Save diff mask
    cv2.imwrite(os.path.join(debug_dir, f"{timestamp}_1_diff.jpg"), thresh)
    
    # Save skeleton
    cv2.imwrite(os.path.join(debug_dir, f"{timestamp}_2_skeleton.jpg"), skeleton)
    
    # Save visualization with contour and detected tip
    debug_img = current_frame.copy()
    cv2.drawContours(debug_img, [dart_contour], -1, (0, 255, 0), 2)
    cv2.circle(debug_img, (int(refined_tip[0]), int(refined_tip[1])), 8, (0, 0, 255), -1)  # Red = detected tip
    cv2.circle(debug_img, (int(cx), int(cy)), 8, (255, 0, 0), -1)  # Blue = center
    # Draw skeleton points
    for pt in points_xy[:20]:  # First 20 skeleton points
        cv2.circle(debug_img, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)  # Yellow
    cv2.imwrite(os.path.join(debug_dir, f"{timestamp}_3_result.jpg"), debug_img)
    
    print(f"[SKELETON] Debug saved: {timestamp}, tip=({refined_tip[0]:.1f}, {refined_tip[1]:.1f})")
    
    if debug:
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
