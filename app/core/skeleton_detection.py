"""
Skeleton-based dart detection following Autodarts approach.

Pipeline:
1. Frame differencing to isolate new dart
2. Skeletonization (thinning) to get 1-pixel-wide dart representation
3. Line fitting (cv2.fitLine) to get dart direction vector
4. Project along line toward board center to find tip
5. Return tip position and line direction
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


def fit_line_to_skeleton(skeleton):
    """Fit a line to skeleton points using cv2.fitLine."""
    points = np.column_stack(np.where(skeleton > 0))  # (y, x) format
    if len(points) < 10:
        return None
    
    points_xy = points[:, ::-1].reshape(-1, 1, 2).astype(np.float32)
    [vx, vy, x0, y0] = cv2.fitLine(points_xy, cv2.DIST_HUBER, 0, 0.01, 0.01)
    return float(vx[0]), float(vy[0]), float(x0[0]), float(y0[0])


def find_tip_by_line_projection(skeleton, line_params, center, dart_mask):
    """
    Find tip by projecting along the fitted line toward the board center.
    
    The line gives us the dart direction. We project from the line's midpoint
    toward the center, and find where the projection intersects the dart edge.
    """
    if line_params is None:
        return None
    
    vx, vy, x0, y0 = line_params
    cx, cy = center
    
    # Determine which direction along the line goes toward center
    # Vector from line point to center
    to_center = np.array([cx - x0, cy - y0])
    line_dir = np.array([vx, vy])
    
    # If dot product > 0, line_dir points toward center
    dot = np.dot(to_center, line_dir)
    if dot < 0:
        # Flip direction to point toward center
        vx, vy = -vx, -vy
        line_dir = np.array([vx, vy])
    
    # Project along line toward center, find where we exit the dart mask
    # Start from a point on the skeleton and move toward center
    max_steps = 500
    tip_x, tip_y = x0, y0
    
    # First, find where the dart mask ends in the direction toward center
    for step in range(1, max_steps):
        test_x = int(x0 + step * vx)
        test_y = int(y0 + step * vy)
        
        # Bounds check
        if test_x < 0 or test_x >= dart_mask.shape[1] or test_y < 0 or test_y >= dart_mask.shape[0]:
            break
        
        # Check if still on dart
        if dart_mask[test_y, test_x] > 0:
            tip_x, tip_y = test_x, test_y
        else:
            # Just exited the dart - previous point is the tip
            break
    
    return (float(tip_x), float(tip_y))


def detect_dart_skeleton(
    current_frame: np.ndarray,
    previous_frame: np.ndarray,
    center: tuple = None,
    mask: np.ndarray = None,
    debug: bool = False
) -> dict:
    """
    Detect dart using skeleton-based approach.
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
    
    # 1. Frame differencing
    diff = cv2.absdiff(current_frame, previous_frame)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # 2. Threshold
    _, motion_mask = cv2.threshold(gray_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
    
    # 4. Find largest contour
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return result
    
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 200:
        return result
    
    dart_mask = np.zeros_like(motion_mask)
    cv2.drawContours(dart_mask, [largest], -1, 255, -1)
    
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
        
        # 7. Find tip by projecting along line toward center
        tip = find_tip_by_line_projection(skeleton, line_params, center, dart_mask)
        
        if tip:
            result["tip"] = tip
            result["confidence"] = 0.8
    
    if debug and result["tip"]:
        debug_dir = r"C:\Users\clawd\DartDetectionAI\skeleton_debug"
        os.makedirs(debug_dir, exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        debug_img = current_frame.copy()
        debug_img[skeleton > 0] = [0, 255, 0]
        tip_x, tip_y = int(result["tip"][0]), int(result["tip"][1])
        cv2.circle(debug_img, (tip_x, tip_y), 8, (0, 0, 255), -1)
        cv2.circle(debug_img, (int(center[0]), int(center[1])), 10, (255, 255, 0), 2)
        if result["line"]:
            vx, vy, x0, y0 = result["line"]
            pt1 = (int(x0 - 100*vx), int(y0 - 100*vy))
            pt2 = (int(x0 + 100*vx), int(y0 + 100*vy))
            cv2.line(debug_img, pt1, pt2, (255, 0, 0), 2)
        cv2.imwrite(os.path.join(debug_dir, f"skeleton_{timestamp}.jpg"), debug_img)
    
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
    _, motion_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological cleanup
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, morph_kernel)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, morph_kernel)
    
    if existing_dart_locations:
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        motion_mask = cv2.erode(motion_mask, erode_kernel, iterations=1)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        motion_mask = cv2.dilate(motion_mask, dilate_kernel, iterations=1)
        for (ex, ey) in existing_dart_locations:
            cv2.circle(motion_mask, (int(ex), int(ey)), 40, 0, -1)
    else:
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        motion_mask = cv2.dilate(motion_mask, dilate_kernel, iterations=1)
    
    # Canny + Hough
    edges = cv2.Canny(gray_diff, 15, 50)
    edges_masked = cv2.bitwise_and(edges, motion_mask)
    
    h, w = edges_masked.shape[:2]
    scale = max(w / 640.0, 1.0)
    
    lines = cv2.HoughLinesP(edges_masked, 
                             rho=1, 
                             theta=np.pi/180, 
                             threshold=int(25 * scale),
                             minLineLength=int(40 * scale),
                             maxLineGap=int(8 * scale))
    
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
        if angle < 30:
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
        if alignment < 0.5:
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
    
    # Find tip endpoint (closer to center)
    d1 = np.sqrt((x1 - cx)**2 + (y1 - cy)**2)
    d2 = np.sqrt((x2 - cx)**2 + (y2 - cy)**2)
    
    if d1 < d2:
        tip_end = np.array([x1, y1], dtype=float)
        flight_end = np.array([x2, y2], dtype=float)
    else:
        tip_end = np.array([x2, y2], dtype=float)
        flight_end = np.array([x1, y1], dtype=float)
    
    dart_dir = tip_end - flight_end
    dart_dir = dart_dir / (np.linalg.norm(dart_dir) + 1e-6)
    
    # Extrapolate to find true tip
    extended_tip = tip_end.copy()
    line_length = np.linalg.norm(tip_end - flight_end)
    max_extrapolation = min(100, int(line_length * 0.5))
    
    signal_history = []
    
    for step in range(1, max_extrapolation + 50):
        test_point = tip_end + dart_dir * step
        tx, ty = int(test_point[0]), int(test_point[1])
        
        if tx < 0 or tx >= gray_diff.shape[1] or ty < 0 or ty >= gray_diff.shape[0]:
            break
        
        y_lo = max(0, ty - 3)
        y_hi = min(gray_diff.shape[0], ty + 4)
        x_lo = max(0, tx - 3)
        x_hi = min(gray_diff.shape[1], tx + 4)
        
        local_diff = gray_diff[y_lo:y_hi, x_lo:x_hi]
        max_diff = np.max(local_diff)
        
        signal_history.append((step, max_diff, test_point.copy()))
        
        if len(signal_history) >= 5:
            recent_peak = max(s[1] for s in signal_history[-10:])
            
            if recent_peak > 50 and max_diff < recent_peak * 0.4:
                for prev_step, prev_signal, prev_point in reversed(signal_history[:-3]):
                    if prev_signal > recent_peak * 0.6:
                        extended_tip = prev_point
                        break
                break
        
        if len(signal_history) >= 8:
            recent_signals = [s[1] for s in signal_history[-8:]]
            if max(recent_signals) < 30:
                extended_tip = signal_history[-8][2]
                break
    
    result["tip"] = (float(extended_tip[0]), float(extended_tip[1]))
    result["confidence"] = min(1.0, alignment * (length / 100.0))
    
    return result
