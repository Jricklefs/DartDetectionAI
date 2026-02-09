"""
Skeleton-based dart detection with line projection tip finding.

Key insight: erosion removes the thin dart tip from the mask.
Solution: After skeleton, project the fitted line in +Y direction
until exiting the ORIGINAL (pre-erosion) motion mask.
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

DEBUG_DIR = "C:/Users/clawd/DartDetectionAI/debug_images"

def save_debug_image(name: str, current_frame: np.ndarray, previous_frame: np.ndarray, 
                     motion_mask: np.ndarray, dart_contour, tip: tuple, center: tuple,
                     original_mask: np.ndarray = None, line_result: tuple = None):
    """Save a debug visualization showing the detection process."""
    try:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        
        # Create visualization - 2x2 grid
        h, w = current_frame.shape[:2]
        viz = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        
        # Top-left: Current frame with contour and tip
        frame_copy = current_frame.copy()
        if dart_contour is not None:
            cv2.drawContours(frame_copy, [dart_contour], -1, (0, 255, 0), 2)
        if tip is not None:
            cv2.circle(frame_copy, (int(tip[0]), int(tip[1])), 8, (0, 0, 255), -1)
            cv2.circle(frame_copy, (int(tip[0]), int(tip[1])), 10, (255, 255, 255), 2)
        if center is not None:
            cv2.circle(frame_copy, (int(center[0]), int(center[1])), 5, (255, 0, 0), -1)
        if line_result is not None:
            vx, vy, x0, y0 = line_result
            # Draw line through image
            pt1 = (int(x0 - vx * 200), int(y0 - vy * 200))
            pt2 = (int(x0 + vx * 200), int(y0 + vy * 200))
            cv2.line(frame_copy, pt1, pt2, (255, 255, 0), 2)
        viz[0:h, 0:w] = frame_copy
        
        # Top-right: Frame difference
        diff = cv2.absdiff(current_frame, previous_frame)
        viz[0:h, w:w*2] = diff
        
        # Bottom-left: Motion mask (cleaned up)
        mask_colored = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
        if dart_contour is not None:
            cv2.drawContours(mask_colored, [dart_contour], -1, (0, 255, 0), 2)
        viz[h:h*2, 0:w] = mask_colored
        
        # Bottom-right: Original mask (pre-erosion) if available
        if original_mask is not None:
            orig_colored = cv2.cvtColor(original_mask, cv2.COLOR_GRAY2BGR)
            if tip is not None:
                cv2.circle(orig_colored, (int(tip[0]), int(tip[1])), 8, (0, 0, 255), -1)
            viz[h:h*2, w:w*2] = orig_colored
        else:
            viz[h:h*2, w:w*2] = mask_colored
        
        # Add labels
        cv2.putText(viz, "Current + Contour + Tip", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(viz, "Frame Diff", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(viz, "Motion Mask", (10, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(viz, "Original Mask", (w + 10, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save
        filepath = os.path.join(DEBUG_DIR, f"{name}.jpg")
        cv2.imwrite(filepath, viz)
        return filepath
    except Exception as e:
        print(f"Debug save failed: {e}")
        return None

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
        
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect
        
        if w == 0 or h == 0:
            continue
        
        aspect = max(w, h) / min(w, h)
        score = area * aspect
        
        if score > best_score and aspect >= min_aspect_ratio:
            best_score = score
            best_contour = cnt
    
    if best_contour is None and contours:
        best_contour = max(contours, key=cv2.contourArea)
    
    return best_contour


def project_to_tip(skeleton_endpoint, line_params, original_mask, max_extend=50):
    """
    Project from skeleton endpoint along line direction to find true tip.
    
    The skeleton may be shortened due to erosion. Project along the
    fitted line in the +Y direction until we exit the original mask.
    
    Args:
        skeleton_endpoint: (x, y) endpoint from skeleton
        line_params: (vx, vy, x0, y0) from cv2.fitLine
        original_mask: Pre-erosion motion mask
        max_extend: Maximum pixels to extend beyond skeleton
    
    Returns:
        (tip_x, tip_y) - projected tip position
    """
    if line_params is None:
        return skeleton_endpoint
    
    vx, vy, x0, y0 = line_params
    ex, ey = skeleton_endpoint
    
    # Normalize direction to point +Y (downward in image)
    if vy < 0:
        vx, vy = -vx, -vy
    
    # Normalize to unit vector
    length = np.sqrt(vx*vx + vy*vy)
    if length < 0.001:
        return skeleton_endpoint
    vx, vy = vx/length, vy/length
    
    h, w = original_mask.shape[:2]
    
    # Start from endpoint and walk in tip direction
    best_x, best_y = ex, ey
    
    for step in range(1, max_extend + 1):
        nx = int(ex + vx * step)
        ny = int(ey + vy * step)
        
        # Check bounds
        if nx < 0 or nx >= w or ny < 0 or ny >= h:
            break
        
        # Check if still in mask
        if original_mask[ny, nx] > 0:
            best_x, best_y = nx, ny
        else:
            # Exited mask - we've found the tip
            break
    
    return (float(best_x), float(best_y))


def find_tip_endpoint(endpoints, line_params):
    """
    Find which skeleton endpoint is the tip end (in +Y direction).
    """
    if len(endpoints) < 2 or line_params is None:
        return endpoints[0] if endpoints else None
    
    vx, vy, x0, y0 = line_params
    
    # Normalize to point +Y
    if vy < 0:
        vx, vy = -vx, -vy
    
    # Pick endpoint in +Y direction from line midpoint
    best_endpoint = None
    best_score = -float('inf')
    
    for (ex, ey) in endpoints:
        to_endpoint = np.array([ex - x0, ey - y0])
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
    Detect dart using skeleton-based approach with line projection.
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
    
    # 2. Lower threshold to capture dart tip (tip is thin, low contrast)
    _, motion_mask_raw = cv2.threshold(gray_diff, 20, 255, cv2.THRESH_BINARY)
    
    # Keep original for tip projection (before erosion)
    original_mask = motion_mask_raw.copy()
    
    # 3. Morphological cleanup for skeleton
    # Erode to remove noise (but this may shrink tip)
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    motion_mask = cv2.erode(motion_mask_raw, erode_kernel, iterations=1)  # Less erosion
    
    # Dilate to restore
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    motion_mask = cv2.dilate(motion_mask, dilate_kernel, iterations=2)
    
    # Close to fill holes
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, close_kernel)
    
    # 4. Find dart contour
    dart_contour = find_dart_contour(motion_mask, min_area=300, min_aspect_ratio=1.5)
    
    if dart_contour is None:
        return result
    
    # Calculate view quality based on contour aspect ratio
    # Higher aspect ratio = better view (seeing dart from side)
    rect = cv2.minAreaRect(dart_contour)
    (_, _), (w, h), _ = rect
    if w > 0 and h > 0:
        aspect_ratio = max(w, h) / min(w, h)
        # Normalize: aspect ratio of 3+ is excellent, 1.5 is minimum
        view_quality = min(1.0, (aspect_ratio - 1.5) / 2.5)
        result["view_quality"] = max(0.3, view_quality)  # Floor at 0.3
    else:
        result["view_quality"] = 0.5
    
    # Create clean mask from dart contour
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
    
    # 7. Find tip endpoint from skeleton
    endpoints = find_skeleton_endpoints(skeleton)
    
    if len(endpoints) >= 2 and line_params:
        # Find which endpoint is the tip end
        skeleton_tip = find_tip_endpoint(endpoints, line_params)
        
        # Project along line to find true tip in original mask
        tip = project_to_tip(skeleton_tip, line_params, original_mask, max_extend=100)
        result["tip"] = tip
        result["confidence"] = 0.8
        
    elif len(endpoints) == 1:
        tip = project_to_tip(endpoints[0], line_params, original_mask, max_extend=100)
        result["tip"] = tip
        result["confidence"] = 0.6
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
        
        cv2.imwrite(os.path.join(debug_dir, f"{timestamp}_1_raw.jpg"), motion_mask_raw)
        cv2.imwrite(os.path.join(debug_dir, f"{timestamp}_2_clean.jpg"), motion_mask)
        cv2.imwrite(os.path.join(debug_dir, f"{timestamp}_3_skel.jpg"), skeleton)
        
        debug_img = current_frame.copy()
        cv2.drawContours(debug_img, [dart_contour], -1, (0, 255, 0), 2)
        debug_img[skeleton > 0] = [255, 255, 0]
        cv2.circle(debug_img, (int(cx), int(cy)), 10, (255, 0, 0), 2)
        for (ex, ey) in endpoints:
            cv2.circle(debug_img, (int(ex), int(ey)), 6, (255, 255, 0), -1)
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
    debug: bool = False,
    debug_name: str = "hough_debug"
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
    
    # Fixed low threshold for original mask (used for tip projection)
    _, motion_mask_raw = cv2.threshold(gray_diff, 20, 255, cv2.THRESH_BINARY)
    original_mask = motion_mask_raw.copy()
    
    # CLAHE enhancement for better edge detection
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_diff = clahe.apply(gray_diff)
    
    # Threshold the enhanced version for contour finding
    blurred = cv2.GaussianBlur(enhanced_diff, (3, 3), 0)
    _, motion_mask = cv2.threshold(blurred, 25, 255, cv2.THRESH_BINARY)
    
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
    
    # Calculate view quality based on contour aspect ratio
    rect = cv2.minAreaRect(dart_contour)
    (_, _), (w, h), _ = rect
    if w > 0 and h > 0:
        aspect_ratio = max(w, h) / min(w, h)
        view_quality = min(1.0, (aspect_ratio - 1.5) / 2.5)
        result["view_quality"] = max(0.3, view_quality)
    else:
        result["view_quality"] = 0.5
    
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
        if angle < 20:
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
    
    # IMPROVED: Find tip by projecting from processed contour through original mask
    # The processed contour gives us the dart body, then we extend into original mask
    
    # Normalize direction to point toward board center (usually +Y but verify)
    to_center = np.array([cx - x0, cy - y0])
    to_center_len = np.linalg.norm(to_center)
    if to_center_len > 0:
        to_center_norm = to_center / to_center_len
        # Check if line direction aligns with toward-center direction
        if vx * to_center_norm[0] + vy * to_center_norm[1] < 0:
            # Line points away from center, flip it
            vx, vy = -vx, -vy
    
    # First find the extreme point of the PROCESSED contour (dart body)
    contour_tip = None
    contour_best_score = -float('inf')
    
    for point in dart_contour.reshape(-1, 2):
        px, py = point
        score = (px - x0) * vx + (py - y0) * vy
        if score > contour_best_score:
            contour_best_score = score
            contour_tip = (float(px), float(py))
    
    # Now extend from that contour tip along the line direction through original_mask
    if contour_tip:
        # Walk along the line from contour_tip until we exit original_mask
        best_tip = project_to_tip(contour_tip, (vx, vy, x0, y0), original_mask, max_extend=150)
    else:
        best_tip = None
    
    if best_tip:
        result["tip"] = best_tip
        result["confidence"] = min(1.0, alignment * (length / 80.0))
    else:
        d1 = np.sqrt((x1 - cx)**2 + (y1 - cy)**2)
        d2 = np.sqrt((x2 - cx)**2 + (y2 - cy)**2)
        tip = (x1, y1) if d1 < d2 else (x2, y2)
        result["tip"] = (float(tip[0]), float(tip[1]))
        result["confidence"] = min(1.0, alignment * (length / 80.0))
    
    # Save debug visualization if requested
    if debug and result["tip"]:
        save_debug_image(debug_name, current_frame, previous_frame, 
                        motion_mask, dart_contour, result["tip"], center,
                        original_mask, result.get("line"))
    
    return result
