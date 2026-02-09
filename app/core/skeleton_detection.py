"""
Skeleton-based dart detection with direction-aware tip finding.

Key insight from debug images:
- Skeleton covers entire dart (flight+shaft+tip)
- "Closest to center" picks wrong end because board center != dart tip location
- Dart tip is at the END of the skeleton that points TOWARD the board center

Approach:
1. Fit line to skeleton
2. The endpoint where the line vector points toward center is the TIP
3. The endpoint where the line vector points away from center is the FLIGHT
"""

import cv2
try:
    from skimage.morphology import skeletonize
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
import numpy as np
import os

_detection_method = "hough"

def set_detection_method(method: str) -> bool:
    global _detection_method
    if method in ("skeleton", "hough", "yolo"):
        _detection_method = method
        return True
    return False

def get_detection_method() -> str:
    return _detection_method


def find_skeleton_endpoints(skeleton):
    """Find endpoints of a skeletonized line (pixels with 1 neighbor)."""
    skel = (skeleton > 0).astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel, -1, kernel)
    endpoints = np.where((skel == 1) & (neighbor_count == 1))
    return list(zip(endpoints[1], endpoints[0]))  # (x, y) format


def fit_line_to_skeleton(skeleton):
    """Fit a line to skeleton points using cv2.fitLine."""
    points = np.column_stack(np.where(skeleton > 0))
    if len(points) < 10:
        return None
    points_xy = points[:, ::-1].reshape(-1, 1, 2).astype(np.float32)
    [vx, vy, x0, y0] = cv2.fitLine(points_xy, cv2.DIST_HUBER, 0, 0.01, 0.01)
    return float(vx[0]), float(vy[0]), float(x0[0]), float(y0[0])


def find_dart_contour(motion_mask, min_area=500, min_aspect_ratio=2.0):
    """
    Find the dart contour from motion mask.
    
    Darts are elongated - filter by aspect ratio to avoid noise blobs.
    """
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    best_contour = None
    best_score = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        
        # Get bounding rect to check aspect ratio
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect
        
        if w == 0 or h == 0:
            continue
        
        # Aspect ratio (length / width)
        aspect = max(w, h) / min(w, h)
        
        # Score by area * aspect ratio (prefer large elongated shapes)
        score = area * aspect
        
        if score > best_score and aspect >= min_aspect_ratio:
            best_score = score
            best_contour = cnt
    
    # If no elongated contour found, fall back to largest
    if best_contour is None and contours:
        best_contour = max(contours, key=cv2.contourArea)
    
    return best_contour


def find_tip_by_direction(endpoints, line_params, center):
    """
    Find the tip endpoint based on line direction.
    
    Key insight from camera geometry:
    - Cameras are above/around the board at 45° angle
    - The dart tip is always lower in the image (higher Y) than the flight
    - The line direction vector points along the dart
    - We want the endpoint in the direction that has positive vy (points downward)
    
    So: pick the endpoint that is in the +Y direction along the fitted line.
    """
    if len(endpoints) < 2 or line_params is None:
        return None
    
    vx, vy, x0, y0 = line_params
    cx, cy = center
    
    # Normalize line direction to point downward (+Y)
    # If vy is negative, flip the direction
    if vy < 0:
        vx, vy = -vx, -vy
    
    # Now (vx, vy) points in the "tip direction" (toward higher Y = lower in image)
    
    # For each endpoint, check which is in the tip direction from line midpoint
    best_endpoint = None
    best_score = -float('inf')
    
    for (ex, ey) in endpoints:
        # Vector from line midpoint to this endpoint
        to_endpoint = np.array([ex - x0, ey - y0])
        
        # Dot product with tip direction
        score = to_endpoint[0] * vx + to_endpoint[1] * vy
        
        if score > best_score:
            best_score = score
            best_endpoint = (ex, ey)
    
    return best_endpoint


def detect_dart_skeleton(
    current_frame: np.ndarray,
    previous_frame: np.ndarray,
    center: tuple = None,
    mask: np.ndarray = None,
    debug: bool = False
) -> dict:
    """
    Detect dart using skeleton-based approach with direction-aware tip finding.
    """
    result = {
        "tip": None, 
        "line": None,
        "confidence": 0.0, 
        "method": "skeleton"
    }
    
    if current_frame is None or previous_frame is None:
        return result
    
    if center is None:
        center = (current_frame.shape[1] // 2, current_frame.shape[0] // 2)
    
    cx, cy = center
    
    # 1. Frame differencing
    diff = cv2.absdiff(current_frame, previous_frame)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # 2. Threshold with higher value to reduce noise
    _, motion_mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
    
    # 3. Aggressive morphological cleanup
    # First erode to remove small noise
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    motion_mask = cv2.erode(motion_mask, erode_kernel, iterations=2)
    
    # Then dilate to restore dart size
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    motion_mask = cv2.dilate(motion_mask, dilate_kernel, iterations=2)
    
    # Close to fill holes
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, close_kernel)
    
    # 4. Find dart contour (elongated shape)
    dart_contour = find_dart_contour(motion_mask, min_area=300, min_aspect_ratio=1.5)
    
    if dart_contour is None:
        return result
    
    # Create clean mask from just the dart contour
    dart_mask = np.zeros_like(motion_mask)
    cv2.drawContours(dart_mask, [dart_contour], -1, 255, -1)
    
    # 5. Skeletonize
    if HAS_SKIMAGE:
        skeleton = skeletonize(dart_mask > 0).astype(np.uint8) * 255
    else:
        try:
            skeleton = cv2.ximgproc.thinning(dart_mask)
        except AttributeError:
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            skeleton = dart_mask.copy()
            for _ in range(20):
                skeleton = cv2.erode(skeleton, kernel, iterations=1)
                if np.sum(skeleton) < 100:
                    break
    
    if np.sum(skeleton) < 50:
        return result
    
    # 6. Fit line to skeleton
    line_params = fit_line_to_skeleton(skeleton)
    if line_params:
        result["line"] = line_params
    
    # 7. Find endpoints
    endpoints = find_skeleton_endpoints(skeleton)
    
    if len(endpoints) >= 2 and line_params:
        # Use direction-based tip finding
        tip = find_tip_by_direction(endpoints, line_params, center)
        if tip:
            result["tip"] = (float(tip[0]), float(tip[1]))
            result["confidence"] = 0.8
    elif len(endpoints) == 1:
        tip = endpoints[0]
        result["tip"] = (float(tip[0]), float(tip[1]))
        result["confidence"] = 0.6
    elif len(endpoints) >= 2:
        # Fallback: closest to center
        distances = [np.sqrt((x - cx)**2 + (y - cy)**2) for (x, y) in endpoints]
        closest_idx = np.argmin(distances)
        tip = endpoints[closest_idx]
        result["tip"] = (float(tip[0]), float(tip[1]))
        result["confidence"] = 0.5
    else:
        # Fallback: closest skeleton point to center
        skel_points = np.column_stack(np.where(skeleton > 0))
        if len(skel_points) > 0:
            distances = np.sqrt((skel_points[:, 1] - cx)**2 + (skel_points[:, 0] - cy)**2)
            closest_idx = np.argmin(distances)
            result["tip"] = (float(skel_points[closest_idx, 1]), float(skel_points[closest_idx, 0]))
            result["confidence"] = 0.4
    
    if debug and result["tip"]:
        debug_dir = r"C:\Users\clawd\DartDetectionAI\skeleton_debug"
        os.makedirs(debug_dir, exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Save threshold image
        cv2.imwrite(os.path.join(debug_dir, f"{timestamp}_1_thresh.jpg"), motion_mask)
        
        # Save skeleton
        cv2.imwrite(os.path.join(debug_dir, f"{timestamp}_3_skel.jpg"), skeleton)
        
        # Save result overlay
        debug_img = current_frame.copy()
        # Draw dart contour
        cv2.drawContours(debug_img, [dart_contour], -1, (0, 255, 0), 2)
        # Draw skeleton
        debug_img[skeleton > 0] = [255, 255, 0]
        # Draw center
        cv2.circle(debug_img, (int(cx), int(cy)), 10, (255, 0, 0), 2)
        # Draw endpoints with labels
        for i, (ex, ey) in enumerate(endpoints):
            # Calculate if this is the tip (toward center)
            to_center = np.array([cx - ex, cy - ey])
            label = f"d={np.linalg.norm(to_center):.0f}"
            cv2.circle(debug_img, (int(ex), int(ey)), 6, (255, 255, 0), -1)
            cv2.putText(debug_img, label, (int(ex)+10, int(ey)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        # Draw tip
        tip_x, tip_y = int(result["tip"][0]), int(result["tip"][1])
        cv2.circle(debug_img, (tip_x, tip_y), 10, (0, 0, 255), 3)
        cv2.imwrite(os.path.join(debug_dir, f"{timestamp}_4_result.jpg"), debug_img)
    
    return result


def detect_dart_hough(
    current_frame: np.ndarray,
    previous_frame: np.ndarray,
    center: tuple = None,
    mask: np.ndarray = None,
    existing_dart_locations: list = None,
    debug: bool = False
) -> dict:
    """Detect dart using Hough line detection."""
    result = {"tip": None, "line": None, "confidence": 0.0, "method": "hough"}
    
    if current_frame is None or previous_frame is None:
        return result
    
    if center is None:
        center = (current_frame.shape[1] // 2, current_frame.shape[0] // 2)
    
    if existing_dart_locations is None:
        existing_dart_locations = []
    
    cx, cy = center
    
    # Frame difference
    diff = cv2.absdiff(current_frame, previous_frame)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_diff = clahe.apply(gray_diff)
    
    # Threshold
    blurred = cv2.GaussianBlur(enhanced_diff, (3, 3), 0)
    _, motion_mask = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
    
    # Morphological cleanup
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    motion_mask = cv2.erode(motion_mask, erode_kernel, iterations=1)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    motion_mask = cv2.dilate(motion_mask, dilate_kernel, iterations=2)
    
    if existing_dart_locations:
        for (ex, ey) in existing_dart_locations:
            cv2.circle(motion_mask, (int(ex), int(ey)), 40, 0, -1)
    
    # Find dart contour
    dart_contour = find_dart_contour(motion_mask, min_area=300, min_aspect_ratio=1.5)
    
    if dart_contour is None:
        return result
    
    # Create mask from dart contour only
    dart_mask = np.zeros_like(motion_mask)
    cv2.drawContours(dart_mask, [dart_contour], -1, 255, -1)
    
    # Canny + Hough on dart region only
    edges = cv2.Canny(gray_diff, 15, 50)
    edges_masked = cv2.bitwise_and(edges, dart_mask)
    
    h, w = edges_masked.shape[:2]
    scale = max(w / 640.0, 1.0)
    
    lines = cv2.HoughLinesP(edges_masked, 
                             rho=1, 
                             theta=np.pi/180, 
                             threshold=int(20 * scale),
                             minLineLength=int(30 * scale),
                             maxLineGap=int(10 * scale))
    
    if lines is None or len(lines) == 0:
        return detect_dart_skeleton(current_frame, previous_frame, center, mask, debug)
    
    # Filter for dart-like lines
    dart_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        angle = abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
        if angle > 90:
            angle = 180 - angle
        if angle < 20:  # Skip near-horizontal
            continue
        
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        to_center = np.array([cx - mid_x, cy - mid_y])
        to_center_len = np.linalg.norm(to_center)
        if to_center_len < 1:
            continue
        to_center_norm = to_center / to_center_len
        
        line_dir = np.array([x2 - x1, y2 - y1])
        line_len = np.linalg.norm(line_dir)
        if line_len < 1:
            continue
        line_dir_norm = line_dir / line_len
        
        alignment = abs(np.dot(to_center_norm, line_dir_norm))
        if alignment < 0.4:
            continue
        
        dart_lines.append((x1, y1, x2, y2, length, alignment))
    
    if not dart_lines:
        return detect_dart_skeleton(current_frame, previous_frame, center, mask, debug)
    
    # Select best line
    best = max(dart_lines, key=lambda l: l[4] * l[5])
    x1, y1, x2, y2, length, alignment = best
    
    vx = (x2 - x1) / length
    vy = (y2 - y1) / length
    x0 = (x1 + x2) / 2
    y0 = (y1 + y2) / 2
    result["line"] = (vx, vy, x0, y0)
    
    # Find tip endpoint using direction toward center
    endpoints = [(x1, y1), (x2, y2)]
    tip = find_tip_by_direction(endpoints, (vx, vy, x0, y0), center)
    
    if tip:
        result["tip"] = (float(tip[0]), float(tip[1]))
        result["confidence"] = min(1.0, alignment * (length / 80.0))
    else:
        # Fallback: closer to center
        d1 = np.sqrt((x1 - cx)**2 + (y1 - cy)**2)
        d2 = np.sqrt((x2 - cx)**2 + (y2 - cy)**2)
        tip = (x1, y1) if d1 < d2 else (x2, y2)
        result["tip"] = (float(tip[0]), float(tip[1]))
        result["confidence"] = min(1.0, alignment * (length / 80.0))
    
    return result
