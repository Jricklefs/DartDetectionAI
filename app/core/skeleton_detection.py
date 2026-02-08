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


def detect_dart_hough(
    current_frame: np.ndarray,
    previous_frame: np.ndarray,
    center: tuple = None,
    mask: np.ndarray = None,
    debug: bool = False
) -> dict:
    """
    Detect dart tip using Hough line detection on Canny edges.
    
    Algorithm:
    1. Canny edge detection on frame diff (captures dart shaft)
    2. Hough line transform to find line segments
    3. Filter for lines pointing toward board center
    4. Extrapolate along line direction until diff signal ends
    """
    result = {"tip": None, "confidence": 0.0, "method": "hough"}
    
    if current_frame is None or previous_frame is None:
        return result
    
    if center is None:
        center = (current_frame.shape[1] // 2, current_frame.shape[0] // 2)
    
    cx, cy = center
    
    # 1. Compute frame difference
    diff = cv2.absdiff(current_frame, previous_frame)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # 2. Motion mask for region of interest
    _, motion_mask = cv2.threshold(gray_diff, 20, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    motion_mask = cv2.dilate(motion_mask, kernel, iterations=1)
    
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
    
    if lines is None:
        return result
    
    # 5. Filter for dart-like lines
    dart_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        # Angle from horizontal (0-90 degrees)
        angle = abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
        if angle > 90:
            angle = 180 - angle
        
        # Skip nearly horizontal lines (not dart-like)
        if angle < 30:
            continue
        
        # Check if line points toward center
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
    
    # 8. Extrapolate along line until diff signal ends
    extended_tip = tip_end.copy()
    
    for step in range(1, 150):
        test_point = tip_end + dart_dir * step
        tx, ty = int(test_point[0]), int(test_point[1])
        
        if tx < 0 or tx >= gray_diff.shape[1] or ty < 0 or ty >= gray_diff.shape[0]:
            break
        
        # Sample diff intensity in small neighborhood
        y_lo = max(0, ty - 3)
        y_hi = min(gray_diff.shape[0], ty + 4)
        x_lo = max(0, tx - 3)
        x_hi = min(gray_diff.shape[1], tx + 4)
        
        local_diff = gray_diff[y_lo:y_hi, x_lo:x_hi]
        max_diff = np.max(local_diff)
        
        if max_diff < 10:
            extended_tip = tip_end + dart_dir * max(0, step - 1)
            break
        else:
            extended_tip = test_point
    
    result["tip"] = (float(extended_tip[0]), float(extended_tip[1]))
    result["confidence"] = min(1.0, alignment * (length / 100.0))
    result["line"] = (x1, y1, x2, y2)
    
    # Debug output
    if debug:
        debug_dir = r"C:\Users\clawd\DartDetectionAI\hough_debug"
        os.makedirs(debug_dir, exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        result_img = current_frame.copy()
        cv2.line(result_img, (int(flight_end[0]), int(flight_end[1])),
                 (int(extended_tip[0]), int(extended_tip[1])), (0, 255, 0), 2)
        cv2.circle(result_img, (int(extended_tip[0]), int(extended_tip[1])), 8, (0, 0, 255), -1)
        cv2.circle(result_img, (cx, cy), 5, (255, 0, 0), -1)
        cv2.imwrite(os.path.join(debug_dir, f"{timestamp}_result.jpg"), result_img)
    
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
