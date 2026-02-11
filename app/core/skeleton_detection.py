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

# Optional background model for adaptive thresholding
try:
    from app.core.background_model import get_background_manager
    HAS_BG_MODEL = True
except ImportError:
    HAS_BG_MODEL = False

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
    
    # Top-right: diff with contour overlay
    diff_vis = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR) if len(diff.shape) == 2 else diff.copy()
    if dart_contour is not None:
        cv2.drawContours(diff_vis, [dart_contour], -1, (0, 255, 0), 2)
    if tip:
        cv2.circle(diff_vis, (int(tip[0]), int(tip[1])), 8, (0, 0, 255), -1)
    if line_params:
        vx, vy, x0, y0 = line_params
        pt1 = (int(x0 - vx * 200), int(y0 - vy * 200))
        pt2 = (int(x0 + vx * 200), int(y0 + vy * 200))
        cv2.line(diff_vis, pt1, pt2, (255, 255, 0), 2)
    cv2.putText(diff_vis, "Frame Diff + Contour", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
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


def find_dart_contour(motion_mask, min_area=500, min_aspect_ratio=2.0, existing_locations=None, exclude_radius=80, board_center=None):
    """
    Find the dart contour from motion mask.
    Darts are elongated - filter by aspect ratio to avoid noise blobs.
    BUT: Bullseye darts look circular from camera angle - give bonus to center proximity.
    
    Args:
        existing_locations: List of (x, y) tuples of previous dart tips to exclude
        exclude_radius: Pixels - contours with centroid within this radius of existing darts are excluded
        board_center: (x, y) center of the dartboard for bullseye detection
    """
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    if existing_locations is None:
        existing_locations = []
    
    # Get image center as fallback for board_center
    if board_center is None:
        h, w = motion_mask.shape[:2]
        board_center = (w // 2, h // 2)
    
    best_contour = None
    best_score = 0
    
    candidates = []  # Track all valid candidates with scores
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect
        
        if w == 0 or h == 0:
            continue
        
        # Check if this contour is near an existing dart - if so, skip it
        is_existing = False
        for (ex, ey) in existing_locations:
            dist = np.sqrt((cx - ex)**2 + (cy - ey)**2)
            if dist < exclude_radius:
                is_existing = True
                break
        
        if is_existing:
            continue
        
        aspect = max(w, h) / min(w, h)
        
        # Distance from board center
        dist_to_center = np.sqrt((cx - board_center[0])**2 + (cy - board_center[1])**2)
        
        # Base score
        base_score = area * aspect
        
        # CENTER PROXIMITY BONUS for bullseye detection:
        # If contour is near center AND somewhat circular (aspect < 2.5), boost score significantly
        # Bull/bullseye area is ~50mm radius, at typical resolution ~30-60px
        BULLSEYE_RADIUS_PX = 60
        
        if dist_to_center < BULLSEYE_RADIUS_PX:
            # Contour is in bullseye region - massive bonus
            # This overrides aspect ratio preference for circular bullseye blobs
            center_bonus = 5.0  # 5x multiplier for bullseye proximity
            score = base_score * center_bonus
            candidates.append((cnt, score, "bullseye_region", dist_to_center, aspect))
        elif dist_to_center < BULLSEYE_RADIUS_PX * 2:
            # Near-center bonus (outer bull area)
            center_bonus = 2.0
            score = base_score * center_bonus
            candidates.append((cnt, score, "near_center", dist_to_center, aspect))
        elif aspect >= min_aspect_ratio:
            # Regular dart-shaped contour
            score = base_score
            candidates.append((cnt, score, "dart_shape", dist_to_center, aspect))
    
    # Pick best candidate
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_contour = candidates[0][0]
        # Debug: uncomment to see selection
        # print(f"[CONTOUR] Selected: {candidates[0][2]}, dist={candidates[0][3]:.0f}, aspect={candidates[0][4]:.1f}, score={candidates[0][1]:.0f}")
        return best_contour
    
    # Fallback: if no good contour found, try largest non-excluded contour
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


def project_to_tip(skeleton_endpoint, line_params, original_mask, max_extend=50, board_center=None):
    """
    Find the dart tip: the furthest point of the detected dart mask
    toward the board center, projected onto the centerline.
    
    The tip is where the dart meets the board surface - NOT the board center,
    but the point on the dart closest to center (i.e. the pointy end).
    
    Strategy:
    1. Find ALL mask pixels
    2. For each, project onto centerline to get position along dart axis
    3. The tip is the mask pixel furthest toward board center along the centerline
    4. Project that point onto the centerline for a clean position
    
    Args:
        skeleton_endpoint: (x, y) endpoint from skeleton (fallback)
        line_params: (vx, vy, x0, y0) from cv2.fitLine
        original_mask: Motion mask including dart
        max_extend: Maximum pixels to extend beyond mask (for gap bridging)
        board_center: (cx, cy) center of dartboard
    
    Returns:
        (tip_x, tip_y) - tip position ON the centerline
    """
    if line_params is None:
        return skeleton_endpoint
    
    vx, vy, x0, y0 = line_params
    
    # Normalize direction to unit vector
    length = np.sqrt(vx*vx + vy*vy)
    if length < 0.001:
        return skeleton_endpoint
    vx, vy = vx/length, vy/length
    
    # Orient direction toward board center
    if board_center is not None:
        cx, cy = board_center
        to_center = np.array([cx - x0, cy - y0])
        dot = to_center[0] * vx + to_center[1] * vy
        if dot < 0:
            vx, vy = -vx, -vy
    else:
        if vy < 0:
            vx, vy = -vx, -vy
    
    h, w = original_mask.shape[:2]
    
    # Find the furthest mask pixel along the centerline toward the board
    # Sample along the centerline in both directions to find the extent
    best_t = 0
    best_x, best_y = float(skeleton_endpoint[0]), float(skeleton_endpoint[1])
    
    # Project skeleton endpoint onto line to get starting t
    ex, ey = skeleton_endpoint
    t_start = (ex - x0) * vx + (ey - y0) * vy
    
    # Search forward (toward board center) from way behind the dart
    # to find the furthest mask pixel along the line
    search_start = int(t_start - 200)  # Start well behind the flight
    search_end = int(t_start + max_extend + 200)  # Search well past skeleton endpoint
    
    furthest_t = search_start
    gap_count = 0
    max_gap = 15  # Allow gaps in the mask (thin shaft might have holes)
    in_dart = False
    
    for t in range(search_start, search_end):
        px = int(x0 + vx * t)
        py = int(y0 + vy * t)
        
        if px < 0 or px >= w or py < 0 or py >= h:
            continue
        
        # Check a small neighborhood perpendicular to line (not just centerline pixel)
        # This catches mask pixels slightly off the center
        nx, ny = -vy, vx  # perpendicular
        hit = False
        for offset in range(-5, 6):
            sx = int(px + nx * offset)
            sy = int(py + ny * offset)
            if 0 <= sx < w and 0 <= sy < h and original_mask[sy, sx] > 0:
                hit = True
                break
        
        if hit:
            in_dart = True
            furthest_t = t
            gap_count = 0
        elif in_dart:
            gap_count += 1
            if gap_count > max_gap:
                break  # Exited the dart
    
    if in_dart:
        best_x = x0 + vx * furthest_t
        best_y = y0 + vy * furthest_t
    
    return (float(best_x), float(best_y))



def project_line_to_board(line_params, board_center, board_radius, start_point=None):
    """
    Project the dart centerline until it intersects the board boundary.
    
    The dart tip is where the centerline meets the board surface.
    This works even when the actual tip is occluded or behind a wire.
    
    Args:
        line_params: (vx, vy, x0, y0) from cv2.fitLine
        board_center: (cx, cy) center of dartboard in pixels
        board_radius: radius of board (outer double) in pixels
        start_point: optional starting point to determine direction
    
    Returns:
        (tip_x, tip_y) where line intersects board, or None if no intersection
    """
    if line_params is None or board_center is None:
        return None
    
    vx, vy, x0, y0 = line_params
    cx, cy = board_center
    
    # Normalize direction
    length = np.sqrt(vx*vx + vy*vy)
    if length < 0.001:
        return None
    vx, vy = vx/length, vy/length
    
    # Determine direction: walk TOWARD board center
    if start_point is not None:
        sx, sy = start_point
    else:
        sx, sy = x0, y0
    
    to_center = np.array([cx - sx, cy - sy])
    dot = to_center[0] * vx + to_center[1] * vy
    if dot < 0:
        vx, vy = -vx, -vy  # Flip to point toward center
    
    # Line equation: P(t) = (x0, y0) + t * (vx, vy)
    # Circle equation: (x - cx)^2 + (y - cy)^2 = r^2
    # Substitute and solve quadratic for t
    
    # (x0 + t*vx - cx)^2 + (y0 + t*vy - cy)^2 = r^2
    # Let dx = x0 - cx, dy = y0 - cy
    dx = x0 - cx
    dy = y0 - cy
    
    # a*t^2 + b*t + c = 0
    a = vx*vx + vy*vy  # = 1 since normalized
    b = 2 * (dx*vx + dy*vy)
    c = dx*dx + dy*dy - board_radius*board_radius
    
    discriminant = b*b - 4*a*c
    
    if discriminant < 0:
        # Line doesn't intersect circle - shouldn't happen for valid darts
        return None
    
    sqrt_disc = np.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2*a)
    t2 = (-b + sqrt_disc) / (2*a)
    
    # We want the intersection in the direction we're walking (positive t if pointing right way)
    # Pick the one that's in front of us (positive t and closest)
    candidates = []
    for t in [t1, t2]:
        if t > -50:  # Allow slight negative for points just outside
            px = x0 + t * vx
            py = y0 + t * vy
            candidates.append((t, px, py))
    
    if not candidates:
        return None
    
    # Pick smallest positive t (closest intersection in walking direction)
    candidates.sort(key=lambda x: abs(x[0]))
    _, tip_x, tip_y = candidates[0]
    
    return (float(tip_x), float(tip_y))




def line_segment_intersection(p1, p2, p3, p4):
    """
    Find intersection point of line segment p1-p2 with line segment p3-p4.
    Returns (x, y, t) where t is parameter along p1-p2, or None if no intersection.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None  # Parallel lines
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    
    # Check if intersection is within both segments
    if 0 <= u <= 1:  # Must be on the polygon edge
        # t can be any value (we extend the dart line infinitely)
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return (x, y, t)
    
    return None


def project_line_to_polygon(line_params, polygon_points, board_center, start_point=None):
    """
    Project the dart centerline until it intersects the polygon boundary.
    
    Uses the 20-point calibration polygon (double_outers) for accurate
    board edge detection that accounts for perspective and lens distortion.
    
    Args:
        line_params: (vx, vy, x0, y0) from cv2.fitLine
        polygon_points: List of 20 (x, y) points defining board boundary
        board_center: (cx, cy) center of dartboard
        start_point: Starting point to determine direction toward board
    
    Returns:
        (tip_x, tip_y) where line intersects polygon, or None if no intersection
    """
    if line_params is None or not polygon_points or len(polygon_points) < 3:
        return None
    
    vx, vy, x0, y0 = line_params
    cx, cy = board_center
    
    # Normalize direction
    length = np.sqrt(vx*vx + vy*vy)
    if length < 0.001:
        return None
    vx, vy = vx/length, vy/length
    
    # Determine direction: walk TOWARD board center
    if start_point is not None:
        sx, sy = start_point
    else:
        sx, sy = x0, y0
    
    to_center = np.array([cx - sx, cy - sy])
    dot = to_center[0] * vx + to_center[1] * vy
    if dot < 0:
        vx, vy = -vx, -vy  # Flip to point toward center
    
    # Create line segment from current position extending far in both directions
    # We use a large extension to ensure we hit the polygon
    extend = 1000
    line_start = (x0 - extend * vx, y0 - extend * vy)
    line_end = (x0 + extend * vx, y0 + extend * vy)
    
    # Find all intersections with polygon edges
    intersections = []
    n = len(polygon_points)
    
    for i in range(n):
        p1 = polygon_points[i]
        p2 = polygon_points[(i + 1) % n]
        
        result = line_segment_intersection(line_start, line_end, p1, p2)
        if result:
            ix, iy, t = result
            # Calculate distance from start_point in the walking direction
            if start_point:
                dx = ix - sx
                dy = iy - sy
                # Project onto direction vector to get signed distance
                dist = dx * vx + dy * vy
            else:
                dist = t
            intersections.append((ix, iy, dist, i))
    
    if not intersections:
        return None
    
    # We want the intersection that's in FRONT of us (positive distance)
    # and closest to us (smallest positive distance)
    forward_intersections = [(x, y, d, i) for x, y, d, i in intersections if d > -50]
    
    if not forward_intersections:
        return None
    
    # Sort by distance, take closest
    forward_intersections.sort(key=lambda x: x[2])
    tip_x, tip_y, _, segment_idx = forward_intersections[0]
    
    # Debug log
    with open(r"C:\Users\clawd\polygon_intersection.txt", "a") as f:
        f.write(f"line=({x0:.0f},{y0:.0f})->({vx:.2f},{vy:.2f}) hit polygon at ({tip_x:.0f},{tip_y:.0f}) edge={segment_idx}\n")
    
    return (float(tip_x), float(tip_y))



def find_tip_endpoint(endpoints, line_params, board_center=None, board_radius=None, dart_mask=None):
    """
    Find which skeleton endpoint is the tip end.
    
    Strategy (in order of preference):
    1. If dart_mask provided: tip is the NARROWER end (less local white pixels)
    2. If board_center provided: tip is CLOSER to board center
    3. Fallback: use line direction
    
    The tip is thin and points toward the board. The flight is wide.
    """
    if len(endpoints) < 2 or line_params is None:
        return endpoints[0] if endpoints else None
    
    vx, vy, x0, y0 = line_params
    
    # Method 1: Width-based (most reliable)
    # Count white pixels near each endpoint - tip end is narrower
    if dart_mask is not None:
        endpoint_widths = []
        h, w = dart_mask.shape[:2]
        
        for (ex, ey) in endpoints:
            ex_int, ey_int = int(ex), int(ey)
            # Sample in 15x15 neighborhood
            y_min, y_max = max(0, ey_int-7), min(h, ey_int+8)
            x_min, x_max = max(0, ex_int-7), min(w, ex_int+8)
            local_region = dart_mask[y_min:y_max, x_min:x_max]
            width = np.sum(local_region > 0)
            endpoint_widths.append(((ex, ey), width))
        
        # Sort by width - tip is narrowest
        endpoint_widths.sort(key=lambda x: x[1])
        tip = endpoint_widths[0][0]
        
        # Debug log
        with open(r"C:\Users\clawd\skeleton_endpoints.txt", "a") as f:
            info = [f"({e[0]:.0f},{e[1]:.0f}) w={w}" for e, w in endpoint_widths]
            f.write(f"WIDTH-BASED: [{'; '.join(info)}] -> tip=({tip[0]:.0f},{tip[1]:.0f})\n")
        
        return tip
    
    # Method 2: Distance to board center (tip is closer to center)
    if board_center is not None:
        cx, cy = board_center
        
        best_endpoint = None
        best_dist = float('inf')
        
        endpoint_info = []
        for (ex, ey) in endpoints:
            dist_from_center = np.sqrt((ex - cx)**2 + (ey - cy)**2)
            endpoint_info.append(f"({ex:.0f},{ey:.0f}) d={dist_from_center:.0f}")
            
            if dist_from_center < best_dist:
                best_dist = dist_from_center
                best_endpoint = (ex, ey)
        
        # Debug log
        with open(r"C:\Users\clawd\skeleton_endpoints.txt", "a") as f:
            f.write(f"CENTER-BASED: center=({cx:.0f},{cy:.0f}) [{'; '.join(endpoint_info)}] -> tip=({best_endpoint[0]:.0f},{best_endpoint[1]:.0f})\n")
        
        return best_endpoint
    
    # Method 3: Fallback - use line direction
    if vy < 0:
        vx, vy = -vx, -vy
    
    best_endpoint = None
    best_score = -float('inf')
    
    for (ex, ey) in endpoints:
        to_endpoint = np.array([ex - x0, ey - y0])
        score = to_endpoint[0] * vx + to_endpoint[1] * vy
        
        if score > best_score:
            best_score = score
            best_endpoint = (ex, ey)
    
    return best_endpoint


def get_adaptive_motion_mask(current_frame, previous_frame, camera_id=None, 
                             fixed_threshold=8, use_background_model=False):
    """
    Autodarts-style simple diff: blur -> diff -> threshold.
    Minimal morphology to preserve dart shape, especially the tip.
    
    Args:
        current_frame: Current BGR frame (with dart)
        previous_frame: Previous BGR frame (before dart)
        camera_id: Unused (kept for API compatibility)
        fixed_threshold: Threshold value (default 20)
        use_background_model: Unused (kept for API compatibility)
    
    Returns:
        (gray_diff, motion_mask) tuple
    """
    # STEP 1: Gaussian blur to reduce sensor noise
    blur_current = cv2.GaussianBlur(current_frame, (5, 5), 0)
    blur_previous = cv2.GaussianBlur(previous_frame, (5, 5), 0)
    
    # STEP 2: Compute diff
    diff = cv2.absdiff(blur_current, blur_previous)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # STEP 3: Simple threshold
    # Autodarts uses minimal processing to preserve dart shape
    _, motion_mask = cv2.threshold(gray_diff, fixed_threshold, 255, cv2.THRESH_BINARY)
    
    # STEP 4: Select biggest elongated blob (dart-shaped)
    # Filters out motion artifacts, shadows, and noise
    blob_contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if blob_contours:
        best_blob = None
        best_score = 0
        for c in blob_contours:
            area = cv2.contourArea(c)
            if area < 500:
                continue
            x, y, w, h = cv2.boundingRect(c)
            aspect = max(w, h) / max(min(w, h), 1)
            # Score = area * aspect_bonus (darts are elongated)
            score = area * (1.0 + max(0, aspect - 1.5))
            if score > best_score:
                best_score = score
                best_blob = c
        
        if best_blob is not None:
            clean_mask = np.zeros_like(motion_mask)
            cv2.drawContours(clean_mask, [best_blob], -1, 255, -1)
            motion_mask = clean_mask
    
    return gray_diff, motion_mask



def extend_mask_along_centerline(gray_diff, dart_mask, line_params, board_center, low_threshold=8, corridor_width=15):
    """
    Extend the dart mask along the fitted centerline using a lower threshold.
    
    The flights/barrel are detected at the normal threshold, but the thin shaft
    is too faint. This function:
    1. Creates a narrow corridor along the centerline toward the board center
    2. Applies a much lower threshold within that corridor
    3. Merges detected shaft pixels into the dart mask
    
    Args:
        gray_diff: Grayscale absolute difference image
        dart_mask: Current binary mask (flights + barrel detected)
        line_params: (vx, vy, x0, y0) from cv2.fitLine
        board_center: (cx, cy) center of dartboard
        low_threshold: Lower threshold for shaft detection (default 8)
        corridor_width: Half-width of search corridor in pixels (default 15)
    
    Returns:
        Extended dart_mask with shaft pixels included
    """
    if line_params is None or board_center is None:
        return dart_mask
    
    vx, vy, x0, y0 = line_params
    cx, cy = board_center
    h, w = dart_mask.shape[:2]
    
    # Normalize direction
    length = np.sqrt(vx*vx + vy*vy)
    if length < 0.001:
        return dart_mask
    vx, vy = vx/length, vy/length
    
    # Direction toward board center from line point
    to_center = np.array([cx - x0, cy - y0])
    dot = to_center[0] * vx + to_center[1] * vy
    if dot < 0:
        vx, vy = -vx, -vy
    
    # Perpendicular direction for corridor width
    px, py = -vy, vx
    
    # Find the extent of current mask along the centerline
    # Walk from (x0,y0) toward center to find where mask currently ends
    mask_end_t = 0
    for t in range(0, 500):
        nx = int(x0 + vx * t)
        ny = int(y0 + vy * t)
        if nx < 0 or nx >= w or ny < 0 or ny >= h:
            break
        if dart_mask[ny, nx] > 0:
            mask_end_t = t
    
    # Now create corridor mask from mask_end onwards (where shaft should be)
    # Also include some overlap with existing mask for continuity
    corridor_mask = np.zeros_like(dart_mask)
    start_t = max(0, mask_end_t - 20)  # Start a bit before mask ends
    
    for t in range(start_t, start_t + 300):
        cx_line = x0 + vx * t
        cy_line = y0 + vy * t
        
        # Draw perpendicular line segment at this point
        for offset in range(-corridor_width, corridor_width + 1):
            nx = int(cx_line + px * offset)
            ny = int(cy_line + py * offset)
            if 0 <= nx < w and 0 <= ny < h:
                corridor_mask[ny, nx] = 255
    
    # Apply low threshold within corridor
    _, low_thresh_mask = cv2.threshold(gray_diff, low_threshold, 255, cv2.THRESH_BINARY)
    shaft_pixels = cv2.bitwise_and(low_thresh_mask, corridor_mask)
    
    # Light morphology to connect shaft fragments
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 7))
    shaft_pixels = cv2.morphologyEx(shaft_pixels, cv2.MORPH_CLOSE, close_kernel)
    
    # Merge shaft into dart mask
    extended_mask = cv2.bitwise_or(dart_mask, shaft_pixels)
    
    return extended_mask

def detect_dart_skeleton(
    current_frame: np.ndarray,
    previous_frame: np.ndarray,
    center: tuple = None,
    mask: np.ndarray = None,
    existing_dart_locations: list = None,
    debug: bool = False,
    debug_name: str = "skeleton_debug",
    camera_id: str = None,
    use_adaptive_threshold: bool = True,
    board_radius: float = None
) -> dict:
    """
    Detect dart using skeleton-based approach with line projection.
    
    Args:
        existing_dart_locations: List of (x, y) tuples of previous dart tips to exclude
        camera_id: Camera ID for adaptive background thresholding
        use_adaptive_threshold: Use background model for adaptive thresholding
        board_radius: Outer radius of board in pixels (from calibration) for tip selection
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
    
    # 1. Get motion mask (adaptive or fixed threshold)
    gray_diff, motion_mask_raw = get_adaptive_motion_mask(
        current_frame, previous_frame, 
        camera_id=camera_id,
        fixed_threshold=25,
        use_background_model=use_adaptive_threshold
    )
    
    # Keep original for tip projection (before erosion)
    original_mask = motion_mask_raw.copy()
    
    # 3. Autodarts-style: minimal morphology to preserve dart shape
    # Just use the raw threshold mask - the tip is preserved!
    motion_mask = motion_mask_raw
    
    # Optional: very light closing to connect broken segments (small kernel only)
    # close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, close_kernel)
    
    # 4. Find dart contour (excluding existing dart locations)
    if existing_dart_locations is None:
        existing_dart_locations = []
    dart_contour = find_dart_contour(motion_mask, min_area=300, min_aspect_ratio=1.5, 
                                     existing_locations=existing_dart_locations, board_center=center)
    
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
    
    # Extend dart mask along centerline to capture thin shaft
    # Step 1: Quick line fit on the contour to get direction
    contour_mask_for_line = np.zeros_like(motion_mask)
    cv2.drawContours(contour_mask_for_line, [dart_contour], -1, 255, -1)
    quick_skel_pts = np.column_stack(np.where(contour_mask_for_line > 0))
    if len(quick_skel_pts) > 10:
        # Fit line to contour points (y,x format from np.where, need x,y)
        pts_xy = quick_skel_pts[:, ::-1].astype(np.float32)
        quick_line = cv2.fitLine(pts_xy, cv2.DIST_L2, 0, 0.01, 0.01)
        quick_line = (float(quick_line[0]), float(quick_line[1]), float(quick_line[2]), float(quick_line[3]))
        
        # Step 2: Extend mask along centerline with low threshold
        dart_mask = extend_mask_along_centerline(gray_diff, dart_mask, quick_line, center, 
                                                  low_threshold=8, corridor_width=12)
    
    # Also update original_mask to include the expanded region for tip projection
    original_mask = cv2.bitwise_or(original_mask, dart_mask)
    
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
        # Find which endpoint is the tip (inside board boundary)
        skeleton_tip = find_tip_endpoint(endpoints, line_params, board_center=center, board_radius=board_radius, dart_mask=dart_mask)
        
        # Project along dart centerline to find tip
        # 1. First, walk along the mask to find where dart ends
        tip = project_to_tip(skeleton_tip, line_params, original_mask, max_extend=100, board_center=center)
        
        # 2. If tip is OUTSIDE board (beyond polygon), use polygon intersection instead
        #    This handles cases where tip is hidden behind wire or occluded
        if tip is not None and board_radius is not None:
            tip_dist = np.sqrt((tip[0] - center[0])**2 + (tip[1] - center[1])**2)
            if tip_dist > board_radius * 0.95:  # Tip seems to be at/beyond board edge
                # Try polygon intersection for more accuracy
                if hasattr(detect_dart_skeleton, '_polygon_cache') and camera_id in detect_dart_skeleton._polygon_cache:
                    polygon = detect_dart_skeleton._polygon_cache[camera_id]
                    poly_tip = project_line_to_polygon(line_params, polygon, center, start_point=skeleton_tip)
                    if poly_tip is not None:
                        tip = poly_tip
        result["tip"] = tip
        result["confidence"] = 0.8
        
    elif len(endpoints) == 1:
        # Walk along mask to find tip
        tip = project_to_tip(endpoints[0], line_params, original_mask, max_extend=100, board_center=center)
        
        # Use polygon only if tip seems to be at board edge
        if tip is not None and board_radius is not None:
            tip_dist = np.sqrt((tip[0] - center[0])**2 + (tip[1] - center[1])**2)
            if tip_dist > board_radius * 0.95:
                if hasattr(detect_dart_skeleton, '_polygon_cache') and camera_id in detect_dart_skeleton._polygon_cache:
                    polygon = detect_dart_skeleton._polygon_cache[camera_id]
                    poly_tip = project_line_to_polygon(line_params, polygon, center, start_point=endpoints[0])
                    if poly_tip is not None:
                        tip = poly_tip
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
    debug_name: str = "hough_debug",
    camera_id: str = None,
    use_adaptive_threshold: bool = True,
    board_radius: float = None
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
    
    # Get motion mask (adaptive or fixed)
    gray_diff, motion_mask_raw = get_adaptive_motion_mask(
        current_frame, previous_frame,
        camera_id=camera_id,
        fixed_threshold=25,
        use_background_model=use_adaptive_threshold
    )
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
        return detect_dart_skeleton(current_frame, previous_frame, center, mask, 
                                   existing_dart_locations, debug, debug_name,
                                   camera_id, use_adaptive_threshold)
    
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
        return detect_dart_skeleton(current_frame, previous_frame, center, mask,
                                   existing_dart_locations, debug, debug_name,
                                   camera_id, use_adaptive_threshold)
    
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
    skeleton_tip = find_tip_endpoint(endpoints, (vx, vy, x0, y0), board_center=center, board_radius=board_radius, dart_mask=original_mask)
    
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
