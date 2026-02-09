import cv2
try:
    from skimage.morphology import skeletonize, medial_axis
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    # Fallback: use morphological thinning
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
    Uses MAX Y point (Autodarts lowestPoint approach).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Get the largest contour (the dart)
    largest = max(contours, key=cv2.contourArea)
    
    if cv2.contourArea(largest) < 100:  # Too small to be a dart
        return None
    
    # AUTODARTS APPROACH: Find the LOWEST point (max Y value)
    max_y = -1
    best_point = None
    
    for pt in largest:
        px, py = pt[0]
        if py > max_y:
            max_y = py
            best_point = (float(px), float(py))
    
    return best_point


def _find_tip_by_skeleton(mask: np.ndarray, center: tuple) -> tuple:
    """
    Skeleton tip detection: MAX Y point (Autodarts approach).
    
    The dart tip is the "lowest" point in the image (highest Y value)
    because cameras are mounted level/above looking at the board.
    More reliable than "closest to center" which fails for darts near center.
    """
    if mask is None or np.sum(mask) < 100:
        return None
    
    cx, cy = center
    
    # Skeletonize the mask
    if HAS_SKIMAGE:
        skeleton = skeletonize(mask > 0).astype(np.uint8) * 255
    else:
        try:
            skeleton = cv2.ximgproc.thinning(mask)
        except AttributeError:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            skeleton = cv2.erode(mask, kernel, iterations=3)
    
    # Find all skeleton points
    skel_points = np.column_stack(np.where(skeleton > 0))  # (y, x)
    
    if len(skel_points) == 0:
        return None
    
    # AUTODARTS APPROACH: Find the LOWEST point (max Y value)
    # This is the "lowestPoint" from their detection.thinning pipeline
    # Works because cameras are mounted above/around the board
    max_y_idx = np.argmax(skel_points[:, 0])  # Column 0 is Y
    best_y, best_x = skel_points[max_y_idx]
    
    return (float(best_x), float(best_y))


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
    
    # 2. Enhance contrast with CLAHE for better edge detection
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_diff = clahe.apply(gray_diff)
    
    # 3. Slight blur to reduce noise before thresholding
    blurred = cv2.GaussianBlur(enhanced_diff, (3, 3), 0)
    
    # 4. Adaptive threshold handles uneven lighting better
    # Otsu finds optimal threshold automatically
    _, motion_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 5. Morphological cleanup: remove noise speckles, fill small gaps
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, morph_kernel)   # remove noise
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, morph_kernel)  # fill gaps
    
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
    # Scale params for resolution - higher res = larger dart appearance
    h, w = edges_masked.shape[:2]
    scale = max(w / 640.0, 1.0)  # Scale relative to 640px baseline
    
    lines = cv2.HoughLinesP(edges_masked, 
                             rho=1, 
                             theta=np.pi/180, 
                             threshold=int(25 * scale),
                             minLineLength=int(40 * scale),
                             maxLineGap=int(8 * scale))
    
    # FIX 2: Fallback cascade if Hough finds no lines
    if lines is None or len(lines) == 0:
        # Try skeleton lowest point first
        skeleton_tip = _find_tip_by_skeleton(motion_mask, center)
        if skeleton_tip is not None:
            result["tip"] = skeleton_tip
            result["confidence"] = 0.6
            result["method"] = "skeleton_lowest"
            return result
        
        # Then try contour-based
        contour_tip = _find_tip_by_contour(motion_mask, center, gray_diff)
        if contour_tip is not None:
            result["tip"] = contour_tip
            result["confidence"] = 0.5
            result["method"] = "contour_fallback"
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
