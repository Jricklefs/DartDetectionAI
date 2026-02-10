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

_detection_method = "skeleton"

def set_detection_method(method: str) -> bool:
    global _detection_method
    if method in ("skeleton", "hough", "yolo"):
        _detection_method = method
        return True
    return False

def get_detection_method() -> str:
    return _detection_method


def save_debug_image(debug_name, current_frame, diff, motion_mask, original_mask, 
                     dart_contour, tip, center, line_params=None):
    """Save debug visualization showing detection stages."""
    debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "debug_images")
    os.makedirs(debug_dir, exist_ok=True)
    
    h, w = current_frame.shape[:2]
    
    # Create 2x2 grid
    grid = np.zeros((h, w * 2, 3), dtype=np.uint8)
    
    # Top-left: current frame with overlays
    frame_vis = current_frame.copy()
    if dart_contour is not None:
        cv2.drawContours(frame_vis, [dart_contour], -1, (0, 255, 0), 2)
    if tip:
        cv2.circle(frame_vis, (int(tip[0]), int(tip[1])), 8, (0, 0, 255), -1)
    if center:
        cv2.circle(frame_vis, (int(center[0]), int(center[1])), 5, (255, 0, 0), -1)
    if line_params:
        vx, vy, x0, y0 = line_params
        pt1 = (int(x0 - vx * 200), int(y0 - vy * 200))
        pt2 = (int(x0 + vx * 200), int(y0 + vy * 200))
        cv2.line(frame_vis, pt1, pt2, (255, 255, 0), 2)
    cv2.putText(frame_vis, "Current + Contour + Tip", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    # Top-right: diff
    diff_vis = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR) if len(diff.shape) == 2 else diff
    cv2.putText(diff_vis, "Frame Diff", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    # Stack top row
    top_row = np.hstack([frame_vis, diff_vis])
    
    # Bottom-left: motion mask
    mask_vis = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
    if dart_contour is not None:
        cv2.drawContours(mask_vis, [dart_contour], -1, (0, 255, 0), 2)
    cv2.putText(mask_vis, "Motion Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    # Bottom-right: original mask with tip
    orig_vis = cv2.cvtColor(original_mask, cv2.COLOR_GRAY2BGR)
    if tip:
        cv2.circle(orig_vis, (int(tip[0]), int(tip[1])), 8, (0, 0, 255), -1)
    cv2.putText(orig_vis, "Original Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    # Stack bottom row
    bottom_row = np.hstack([mask_vis, orig_vis])
    
    # Full grid
    full_grid = np.vstack([top_row, bottom_row])
    
    # Save
    out_path = os.path.join(debug_dir, f"{debug_name}.jpg")
    cv2.imwrite(out_path, full_grid)
    return out_path


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


def find_dart_contour(motion_mask, min_area=500, min_aspect_ratio=2.0, existing_locations=None, exclude_radius=120):
    """
    Find the dart contour from motion mask.
    Darts are elongated - filter by aspect ratio to avoid noise blobs.
    
    Args:
        existing_locations: List of (x, y) tuples of previous dart tips to exclude
        exclude_radius: Pixels - contours with centroid within this radius of existing darts are excluded
    """
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    if existing_locations is None:
        existing_locations = []
    
    # DEBUG: Log existing locations
    if existing_locations:
        print(f"[CONTOUR] Excluding {len(existing_locations)} existing dart locations: {existing_locations}")
    
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
        
        # Check if this contour is near an existing dart - if so, skip it
        # Use the LOWEST point (max Y) of the contour as the tip estimate
        # This is more accurate than centroid for dart comparison
        lowest_point = max(cnt.reshape(-1, 2), key=lambda p: p[1])
        cnt_tip_x, cnt_tip_y = float(lowest_point[0]), float(lowest_point[1])
        
        is_existing = False
        for (ex, ey) in existing_locations:
            dist = np.sqrt((cnt_tip_x - ex)**2 + (cnt_tip_y - ey)**2)
            if dist < exclude_radius:
                is_existing = True
                print(f"[CONTOUR] Excluding contour with tip ({cnt_tip_x:.0f}, {cnt_tip_y:.0f}) - {dist:.0f}px from existing ({ex:.0f}, {ey:.0f})")
                break
        
        if is_existing:
            continue
        
        aspect = max(w, h) / min(w, h)
        score = area * aspect
        
        if score > best_score and aspect >= min_aspect_ratio:
            best_score = score
            best_contour = cnt
    
    # Fallback: if no good contour found (all were excluded or too small), 
    # try largest non-excluded contour
    if best_contour is None:
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area / 2:  # Lower threshold for fallback
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            
            is_existing = False
            for (ex, ey) in existing_locations:
                dist = np.sqrt((cx - ex)**2 + (cy - ey)**2)
                if dist < exclude_radius:
                    is_existing = True
                    break
            
            if not is_existing:
                valid_contours.append(cnt)
        
        if valid_contours:
            best_contour = max(valid_contours, key=cv2.contourArea)
    
    return best_contour


def project_to_tip(skeleton_endpoint, line_params, original_mask, max_extend=50, board_center=None, dart_contour=None):
    """
    Find the tip by walking along the fitted line in +Y direction.
    
    The tip is the lowest point (max Y) of the dart along the line,
    constrained to be near the dart contour if provided.
    
    Args:
        skeleton_endpoint: (x, y) starting point (fallback)
        line_params: (vx, vy, x0, y0) from cv2.fitLine
        original_mask: Full motion mask (pre-erosion)
        max_extend: Maximum pixels to search
        board_center: (unused)
        dart_contour: If provided, only consider points within this contour's bounding box
    
    Returns:
        (tip_x, tip_y) - the lowest point on line that's in the mask
    """
    if line_params is None:
        return skeleton_endpoint
    
    vx, vy, x0, y0 = line_params
    h, w = original_mask.shape[:2]
    
    # Get bounding box of dart contour to constrain search
    min_x, max_x = 0, w
    min_y, max_y = 0, h
    if dart_contour is not None:
        x, y, cw, ch = cv2.boundingRect(dart_contour)
        # Expand bounding box a bit to catch the tip
        min_x = max(0, x - 50)
        max_x = min(w, x + cw + 50)
        min_y = max(0, y - 20)
        max_y = min(h, y + ch + 100)  # More room below for tip
    
    # Normalize direction to point +Y (downward = toward tip)
    if vy < 0:
        vx, vy = -vx, -vy
    
    # Normalize to unit vector
    length = np.sqrt(vx*vx + vy*vy)
    if length < 0.001:
        return skeleton_endpoint
    vx, vy = vx/length, vy/length
    
    # Start from the line origin point and walk in BOTH directions to find full extent
    best_x, best_y = x0, y0
    
    # Walk along line in both directions
    for step in range(-max_extend, max_extend + 1):
        nx = int(x0 + vx * step)
        ny = int(y0 + vy * step)
        
        # Check bounds
        if nx < 0 or nx >= w or ny < 0 or ny >= h:
            continue
        
        # Check if within dart contour bounding box
        if nx < min_x or nx > max_x or ny < min_y or ny > max_y:
            continue
        
        # Check if in mask and has higher Y (lower on screen = closer to tip)
        if original_mask[ny, nx] > 0:
            if ny > best_y:
                best_x, best_y = nx, ny
    
    return (float(best_x), float(best_y))


def find_tip_endpoint(endpoints, line_params, board_center=None):
    """
    Find which skeleton endpoint is the tip end.
    
    The tip is simply the endpoint with the highest Y value (lowest point in image).
    Darts point downward into the board.
    """
    if len(endpoints) < 2:
        return endpoints[0] if endpoints else None
    
    # Tip = highest Y value (lowest point on screen)
    return max(endpoints, key=lambda p: p[1])


def detect_dart_skeleton(
    current_frame: np.ndarray,
    previous_frame: np.ndarray,
    center: tuple = None,
    mask: np.ndarray = None,
    existing_dart_locations: list = None,
    debug: bool = False,
    debug_name: str = "skeleton_debug"
) -> dict:
    """
    Detect dart using skeleton-based approach with line projection.
    
    Args:
        existing_dart_locations: List of (x, y) tuples of previous dart tips to exclude
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
    
    # 4. Find dart contour (excluding existing dart locations)
    if existing_dart_locations is None:
        existing_dart_locations = []
    dart_contour = find_dart_contour(motion_mask, min_area=300, min_aspect_ratio=1.5, 
                                     existing_locations=existing_dart_locations)
    
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
        # Find which endpoint is the tip end (highest Y = lowest on screen)
        skeleton_tip = find_tip_endpoint(endpoints, line_params, board_center=center)
        
        # Project along line to find true tip in original mask (need long range for full dart)
        tip = project_to_tip(skeleton_tip, line_params, original_mask, max_extend=400, board_center=center, dart_contour=dart_contour)
        result["tip"] = tip
        result["confidence"] = 0.8
        
    elif len(endpoints) == 1:
        tip = project_to_tip(endpoints[0], line_params, original_mask, max_extend=400, board_center=center, dart_contour=dart_contour)
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
        save_debug_image(debug_name, current_frame, gray_diff, motion_mask, original_mask,
                        dart_contour, result["tip"], center, result.get("line"))
    
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
    
    # Find tip endpoint and project
    endpoints = [(x1, y1), (x2, y2)]
    skeleton_tip = find_tip_endpoint(endpoints, (vx, vy, x0, y0), board_center=center)
    
    if skeleton_tip:
        tip = project_to_tip(skeleton_tip, (vx, vy, x0, y0), original_mask, max_extend=100, board_center=center)
        result["tip"] = tip
        result["confidence"] = min(1.0, alignment * (length / 80.0))
    else:
        d1 = np.sqrt((x1 - cx)**2 + (y1 - cy)**2)
        d2 = np.sqrt((x2 - cx)**2 + (y2 - cy)**2)
        tip = (x1, y1) if d1 < d2 else (x2, y2)
        result["tip"] = (float(tip[0]), float(tip[1]))
        result["confidence"] = min(1.0, alignment * (length / 80.0))
    
    # Save debug image if requested
    if debug and result["tip"]:
        save_debug_image(debug_name, current_frame, gray_diff, motion_mask, original_mask,
                        dart_contour, result["tip"], center, result.get("line"))
    
    return result
