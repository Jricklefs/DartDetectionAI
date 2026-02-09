import cv2
import numpy as np
import os

# Current detection method: 'skeleton' (legacy) or 'hough' (new)
_detection_method = "hough"

def set_detection_method(method: str) -> bool:
    """Set the detection method to use."""
    global _detection_method
    if method in ("skeleton", "hough", "yolo"):
        _detection_method = method
        return True
    return False

def get_detection_method() -> str:
    """Get the current detection method."""
    return _detection_method

def find_skeleton_endpoints(skeleton):
    """Find endpoints of a skeletonized line (pixels with 1 neighbor)."""
    skel = (skeleton > 0).astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel, -1, kernel)
    endpoints = np.where((skel == 1) & (neighbor_count == 1))
    return list(zip(endpoints[1], endpoints[0]))



def _find_tip_by_contour(mask: np.ndarray, center: tuple, gray_diff: np.ndarray = None) -> tuple:
    """
    Contour-based fallback for tip detection.
    
    Finds the point on the largest contour closest to board center
    with strong diff signal. Used when Hough line detection fails.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Get the largest contour (likely the dart)
    largest = max(contours, key=cv2.contourArea)
    
    if cv2.contourArea(largest) < 100:  # Too small to be a dart
        return None
    
    cx, cy = center
    
    # Find point closest to center with strong diff signal
    min_dist = float('inf')
    best_point = None
    
    for pt in largest:
        px, py = pt[0]
        dist = np.sqrt((px - cx)**2 + (py - cy)**2)
        
        # Prefer points with strong diff signal (actual dart, not noise)
        if gray_diff is not None:
            y_lo = max(0, py - 2)
            y_hi = min(gray_diff.shape[0], py + 3)
            x_lo = max(0, px - 2)
            x_hi = min(gray_diff.shape[1], px + 3)
            local_diff = np.mean(gray_diff[y_lo:y_hi, x_lo:x_hi])
            
            # Weight distance by inverse of diff (prefer strong signal)
            if local_diff > 20:  # Minimum signal threshold
                weighted_dist = dist / (local_diff / 50.0)
                if weighted_dist < min_dist:
                    min_dist = weighted_dist
                    best_point = (float(px), float(py))
        else:
            if dist < min_dist:
                min_dist = dist
                best_point = (float(px), float(py))
    
    # Fallback: find sharpest corner
    if best_point is None:
        epsilon = 0.02 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        
        if len(approx) >= 3:
            min_angle = float('inf')
            sharpest = None
            
            for i in range(len(approx)):
                p1 = approx[i-1][0]
                p2 = approx[i][0]
                p3 = approx[(i+1) % len(approx)][0]
                
                v1 = p1 - p2
                v2 = p3 - p2
                
                len1 = np.linalg.norm(v1)
                len2 = np.linalg.norm(v2)
                if len1 < 1 or len2 < 1:
                    continue
                
                cos_angle = np.dot(v1, v2) / (len1 * len2)
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                
                if angle < min_angle:
                    min_angle = angle
                    sharpest = (float(p2[0]), float(p2[1]))
            
            best_point = sharpest
    
    return best_point


def detect_dart_hough(
    current_frame: np.ndarray,
    previous_frame: np.ndarray,
    center: tuple = None,
    mask: np.ndarray = None,
    existing_dart_locations: list = None,
    debug: bool = False
) -> dict:
    """
    Detect dart tip using Hough line detection on Canny edges.
    
    Improvements (Feb 9, 2026):
    1. Erode mask to separate overlapping darts
    2. Mask out existing dart locations
    3. Contour-based fallback when Hough fails
    """
    result = {"tip": None, "confidence": 0.0, "method": "hough"}
    
    if current_frame is None or previous_frame is None:
        return result
    
    if center is None:
        center = (current_frame.shape[1] // 2, current_frame.shape[0] // 2)
    
    if existing_dart_locations is None:
        existing_dart_locations = []
    
    cx, cy = center
    
    # 1. Compute frame difference
    diff = cv2.absdiff(current_frame, previous_frame)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # 2. Motion mask for region of interest
    _, motion_mask = cv2.threshold(gray_diff, 20, 255, cv2.THRESH_BINARY)
    
    # FIX 1a: Only erode/dilate when there are existing darts to separate from
    if existing_dart_locations:
        # Erode to break thin connections between overlapping darts
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        motion_mask = cv2.erode(motion_mask, erode_kernel, iterations=1)
        
        # Dilate to restore dart size
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        motion_mask = cv2.dilate(motion_mask, dilate_kernel, iterations=1)
        
        # FIX 1b: Mask out existing dart locations
        for (ex, ey) in existing_dart_locations:
            cv2.circle(motion_mask, (int(ex), int(ey)), 40, 0, -1)
    else:
        # For first dart, just dilate slightly to connect fragments
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        motion_mask = cv2.dilate(motion_mask, dilate_kernel, iterations=1)
    
    # 3. Canny edges (low thresholds to catch faint dart shaft)
    edges = cv2.Canny(gray_diff, 15, 50)
    edges_masked = cv2.bitwise_and(edges, motion_mask)
    
    # 4. Hough line detection
    lines = cv2.HoughLinesP(edges_masked, 
                             rho=1, 
                             theta=np.pi/180, 
                             threshold=30,
                             minLineLength=50,
                             maxLineGap=10)
    
    # FIX 2: Contour-based fallback if Hough finds nothing
    if lines is None or len(lines) == 0:
        contour_tip = _find_tip_by_contour(motion_mask, center, gray_diff)
        if contour_tip is not None:
            result["tip"] = contour_tip
            result["confidence"] = 0.5
            result["method"] = "hough_contour_fallback"
            return result
        return result
    
    # 5. Filter for dart-like lines
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
        contour_tip = _find_tip_by_contour(motion_mask, center, gray_diff)
        if contour_tip is not None:
            result["tip"] = contour_tip
            result["confidence"] = 0.5
            result["method"] = "hough_contour_fallback"
            return result
        return result
    
    # 6. Select best line (longest that points at center)
    best = max(dart_lines, key=lambda l: l[4] * l[5])
    x1, y1, x2, y2, length, alignment = best
    
    # 7. Determine which endpoint is toward the tip (closer to center)
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
    
    # 8. Extrapolate along line until diff signal drops
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
    result["line"] = (x1, y1, x2, y2)
    
    if debug:
        debug_dir = r"C:\Users\clawd\DartDetectionAI\hough_debug"
        os.makedirs(debug_dir, exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        result_img = current_frame.copy()
        cv2.line(result_img, (int(flight_end[0]), int(flight_end[1])),
                 (int(extended_tip[0]), int(extended_tip[1])), (0, 255, 0), 2)
        cv2.circle(result_img, (int(extended_tip[0]), int(extended_tip[1])), 8, (0, 0, 255), -1)
        cv2.circle(result_img, (int(cx), int(cy)), 5, (255, 0, 0), -1)
        cv2.imwrite(os.path.join(debug_dir, f"hough_{timestamp}.jpg"), result_img)
    
    return result


def detect_dart_skeleton(
    current_frame: np.ndarray,
    previous_frame: np.ndarray,
    center: tuple = None,
    mask: np.ndarray = None,
    debug: bool = False
) -> dict:
    """
    Main entry point - dispatches to appropriate detection method.
    """
    method = get_detection_method()
    
    if method == "hough":
        return detect_dart_hough(current_frame, previous_frame, center, mask, debug)
    else:
        # Legacy skeleton method (mostly for backwards compat)
        return detect_dart_hough(current_frame, previous_frame, center, mask, debug)
