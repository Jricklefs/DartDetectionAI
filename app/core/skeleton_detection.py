"""
Dart Detection v10.2 — Shape-filtered
=======================================
Based on original v10 (which had clean masks on cam0/cam2).
Additions:
- Shape filtering: darts are elongated. Reject blobs that aren't dart-shaped.
- Previous dart pixel subtraction
- LAB + CLAHE only in the motion mask (not two-pass)
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


# =============================================================================
# PUBLIC API
# =============================================================================

def detect_dart(
    current_frame: np.ndarray,
    previous_frame: np.ndarray,
    board_center: Tuple[float, float] = (640, 360),
    board_radius: Optional[float] = None,
    existing_dart_mask: Optional[np.ndarray] = None,
    camera_id: str = "",
    debug: bool = False,
    debug_name: str = "",
) -> Dict[str, Any]:
    result = {
        "tip": None, "confidence": 0.0, "line": None,
        "dart_length": 0.0, "method": "none",
        "view_quality": 0.5, "debug_image": None,
    }
    
    # Step 1: Motion mask
    motion_mask = _compute_motion_mask(current_frame, previous_frame)
    
    # Step 2: Subtract existing darts
    if existing_dart_mask is not None:
        motion_mask = cv2.bitwise_and(motion_mask, cv2.bitwise_not(existing_dart_mask))
    
    # Step 3: Shape filter — keep only elongated (dart-shaped) blobs
    motion_mask = _shape_filter(motion_mask)
    
    # Step 4: Find the flight blob
    flight = _find_flight_blob(motion_mask, min_area=80)
    if flight is None:
        return result
    
    flight_centroid, flight_contour, flight_bbox = flight
    
    # Step 5: Find tip
    tip, tip_method, dart_length = _find_tip_from_flight(
        motion_mask, flight_centroid, flight_contour, flight_bbox
    )
    if tip is None:
        return result
    
    # Step 6: Sub-pixel refinement
    gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY) if len(current_frame.shape) == 3 else current_frame
    tip = _refine_tip_subpixel(tip, gray, motion_mask)
    
    # Step 7: Line + quality
    line = _compute_line(flight_centroid, tip)
    view_quality = min(1.0, dart_length / 150.0) if dart_length > 0 else 0.3
    
    result.update({
        "tip": tip, "confidence": 0.8, "line": line,
        "dart_length": dart_length, "method": tip_method,
        "view_quality": view_quality,
    })
    
    if debug:
        result["debug_image"] = _draw_debug(
            current_frame, motion_mask, flight_centroid, flight_contour,
            tip, line, tip_method, debug_name
        )
    
    return result


# =============================================================================
# Motion Mask (original v10 approach — works well)
# =============================================================================

def _compute_motion_mask(
    current: np.ndarray,
    previous: np.ndarray,
    blur_size: int = 5,
    threshold: int = 20,
) -> np.ndarray:
    if len(current.shape) == 3:
        gray_curr = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    else:
        gray_curr = current
    if len(previous.shape) == 3:
        gray_prev = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    else:
        gray_prev = previous
    
    blur_curr = cv2.GaussianBlur(gray_curr, (blur_size, blur_size), 0)
    blur_prev = cv2.GaussianBlur(gray_prev, (blur_size, blur_size), 0)
    diff = cv2.absdiff(blur_curr, blur_prev)
    
    # Multi-threshold hysteresis
    _, mask_high = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    _, mask_low = cv2.threshold(diff, max(5, threshold // 3), 255, cv2.THRESH_BINARY)
    
    # Aggressive close on low mask to bridge flight→shaft gaps
    # The shaft is thin and faint — need to connect it to the bright flight
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_low = cv2.morphologyEx(mask_low, cv2.MORPH_CLOSE, close_kernel)
    
    # Hysteresis: grow high-threshold seeds into connected low-threshold pixels
    seed = mask_high.copy()
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    for _ in range(50):  # more iterations to reach further along shaft
        expanded = cv2.dilate(seed, dilate_kernel, iterations=1)
        new_pixels = cv2.bitwise_and(expanded, mask_low)
        if np.array_equal(new_pixels, seed):
            break
        seed = new_pixels
    
    # Morphological OPENING to trim blobby protrusions ("dart herpes")
    # Erode shaves off thin noise fingers, dilate restores the main dart shape
    # Use 3x3 to be gentle — don't want to erase thin shaft pixels
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, open_kernel)
    
    return seed


# =============================================================================
# Shape Filter — reject non-dart blobs ("dart herpes")
# =============================================================================

def _shape_filter(mask: np.ndarray, min_aspect: float = 2.0, min_area: int = 100) -> np.ndarray:
    """
    Keep only blobs that are elongated (dart-shaped).
    Darts are long and thin. Shadows/noise are blobby/round.
    
    For each connected component:
    - Fit a minAreaRect
    - Check aspect ratio (length/width)
    - Keep if aspect >= min_aspect (elongated) OR area is large enough 
      that it could be a flight+shaft combo
    - Reject round/blobby shapes
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    
    filtered = np.zeros_like(mask)
    
    for label_id in range(1, num_labels):  # skip background (0)
        area = stats[label_id, cv2.CC_STAT_AREA]
        
        if area < min_area:
            continue
        
        # Get this component's pixels
        component_mask = (labels == label_id).astype(np.uint8) * 255
        ys, xs = np.nonzero(component_mask)
        
        if len(xs) < 5:
            continue
        
        # Fit oriented bounding box
        points = np.column_stack((xs, ys)).astype(np.float32)
        rect = cv2.minAreaRect(points)
        (_, (rw, rh), _) = rect
        
        # Aspect ratio
        long_side = max(rw, rh)
        short_side = min(rw, rh) + 1  # avoid div by zero
        aspect = long_side / short_side
        
        # Keep if:
        # 1. Elongated enough (aspect >= 2.0) — clearly dart-shaped
        # 2. OR very large area (>= 2000px) AND somewhat elongated (>= 1.3)
        #    — this catches the flight area which can be somewhat wide
        if aspect >= min_aspect:
            filtered = cv2.bitwise_or(filtered, component_mask)
        elif area >= 2000 and aspect >= 1.3:
            filtered = cv2.bitwise_or(filtered, component_mask)
        # else: rejected — too round/blobby
    
    return filtered


# =============================================================================
# Trim Perpendicular Noise — darts don't have perpendicular offshoots
# =============================================================================

def _trim_perpendicular_noise(mask: np.ndarray, max_perp_dist: float = 25.0) -> np.ndarray:
    """
    Find the main axis of each blob, then delete any pixels that are 
    too far perpendicular from that axis. Darts are linear — no perpendicular 
    offshoots. Shadows/reflections create blobby protrusions that stick out 
    sideways from the dart.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    cleaned = np.zeros_like(mask)
    
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < 50:
            continue
        
        component = (labels == label_id).astype(np.uint8) * 255
        ys, xs = np.nonzero(component)
        
        if len(xs) < 10:
            cleaned = cv2.bitwise_or(cleaned, component)
            continue
        
        points = np.column_stack((xs, ys)).astype(np.float32)
        
        # Fit the main axis via minAreaRect
        rect = cv2.minAreaRect(points)
        (cx, cy), (rw, rh), angle = rect
        
        # Ensure long side is the "length"
        if rw < rh:
            rw, rh = rh, rw
            angle += 90
        
        # If blob isn't elongated at all, keep as-is (might be a bull dart)
        aspect = rw / (rh + 1)
        if aspect < 1.5:
            cleaned = cv2.bitwise_or(cleaned, component)
            continue
        
        # Find the barrel line through progressive erosion for better axis
        barrel_vx, barrel_vy, barrel_x0, barrel_y0 = None, None, None, None
        for erode_iter in range(2, 10, 2):
            kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            eroded = cv2.erode(component, kern, iterations=erode_iter)
            ey, ex = np.nonzero(eroded)
            if len(ex) < 10:
                break
            erect = cv2.minAreaRect(np.column_stack((ex, ey)).astype(np.float32))
            (_, (erw, erh), _) = erect
            if max(erw, erh) / (min(erw, erh) + 1) > 2.5:
                line_params = cv2.fitLine(
                    np.column_stack((ex, ey)).astype(np.float32),
                    cv2.DIST_L2, 0, 0.01, 0.01
                )
                barrel_vx, barrel_vy, barrel_x0, barrel_y0 = line_params.flatten()
                break
        
        # Fallback: use minAreaRect angle for the line direction
        if barrel_vx is None:
            angle_rad = np.deg2rad(angle)
            barrel_vx = np.cos(angle_rad)
            barrel_vy = np.sin(angle_rad)
            barrel_x0 = cx
            barrel_y0 = cy
        
        # For each pixel, compute perpendicular distance from barrel line
        # Line: point (barrel_x0, barrel_y0) + t*(barrel_vx, barrel_vy)
        # Perp distance = |cross product| / |direction| (direction is unit so just cross)
        dx = xs.astype(np.float64) - barrel_x0
        dy = ys.astype(np.float64) - barrel_y0
        perp_dist = np.abs(dx * barrel_vy - dy * barrel_vx)
        
        # Allow wider near the flight (centroid area) — flight is naturally wider
        # Use distance along the line from centroid to scale the allowed width
        along_dist = np.abs(dx * barrel_vx + dy * barrel_vy)
        
        # Base max perpendicular distance, plus some extra near the center (flight area)
        # Flight can be ~30px wide, shaft is ~5-10px
        allowed_perp = max_perp_dist
        
        # Keep pixels within perpendicular tolerance
        keep = perp_dist <= allowed_perp
        
        # Create trimmed mask
        trimmed = np.zeros_like(component)
        trimmed[ys[keep], xs[keep]] = 255
        
        cleaned = cv2.bitwise_or(cleaned, trimmed)
    
    return cleaned


# =============================================================================
# Find Flight Blob
# =============================================================================

def _find_flight_blob(
    mask: np.ndarray, min_area: int = 80,
) -> Optional[Tuple[Tuple[float, float], np.ndarray, Tuple[int, int, int, int]]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None
    
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    x, y, w, h = cv2.boundingRect(largest)
    
    return (cx, cy), largest, (x, y, w, h)


# =============================================================================
# Find Tip (barrel erosion + taper profile)
# =============================================================================

def _find_tip_from_flight(
    mask: np.ndarray,
    flight_centroid: Tuple[float, float],
    flight_contour: np.ndarray,
    flight_bbox: Tuple[int, int, int, int],
) -> Tuple[Optional[Tuple[float, float]], str, float]:
    fcx, fcy = flight_centroid
    
    # Isolate dart's connected component
    num_labels, labels = cv2.connectedComponents(mask)
    flight_label = labels[int(fcy), int(fcx)]
    
    if flight_label == 0:
        for dy in range(-5, 6):
            for dx in range(-5, 6):
                py, px = int(fcy + dy), int(fcx + dx)
                if 0 <= py < mask.shape[0] and 0 <= px < mask.shape[1]:
                    if labels[py, px] > 0:
                        flight_label = labels[py, px]
                        break
            if flight_label > 0:
                break
    
    if flight_label == 0:
        return None, "none", 0
    
    dart_mask = (labels == flight_label).astype(np.uint8) * 255
    
    # Trim perpendicular noise from THIS dart's mask only
    dart_mask = _trim_perpendicular_noise(dart_mask)
    
    dart_ys, dart_xs = np.nonzero(dart_mask)
    if len(dart_xs) < 10:
        return None, "none", 0
    
    # Try barrel isolation via progressive erosion
    barrel_points = None
    for erode_iter in range(2, 12, 2):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eroded = cv2.erode(dart_mask, kernel, iterations=erode_iter)
        ey, ex = np.nonzero(eroded)
        if len(ex) < 10:
            break
        if len(ex) >= 5:
            rect = cv2.minAreaRect(np.column_stack((ex, ey)).astype(np.float32))
            (_, (rw, rh), _) = rect
            aspect = max(rw, rh) / (min(rw, rh) + 1)
            if aspect > 2.5:
                barrel_points = np.column_stack((ex, ey))
    
    if barrel_points is not None and len(barrel_points) >= 5:
        line_params = cv2.fitLine(barrel_points.astype(np.float32),
                                   cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = line_params.flatten()
        
        h_img, w_img = dart_mask.shape[:2]
        search_width = 15
        
        def walk_line(vx_dir, vy_dir):
            last_on = (float(x0), float(y0))
            gap = 0
            for t in range(1, 400):
                px = x0 + vx_dir * t
                py = y0 + vy_dir * t
                if px < 0 or px >= w_img or py < 0 or py >= h_img:
                    break
                found = False
                for offset in range(-search_width, search_width + 1, 2):
                    cx_c = int(px + (-vy_dir) * offset)
                    cy_c = int(py + vx_dir * offset)
                    if 0 <= cx_c < w_img and 0 <= cy_c < h_img:
                        if dart_mask[cy_c, cx_c] > 0:
                            found = True
                            last_on = (float(px), float(py))
                            gap = 0
                            break
                if not found:
                    gap += 1
                    if gap > 20:
                        break
            return last_on
        
        # Tip is ALWAYS the pixel with the highest Y in the dart mask
        # Cameras are above looking down — tip = bottom of image = highest Y
        # Don't rely on barrel line direction — just pick the lowest point
        tip_y_idx = np.argmax(dart_ys)
        tip_x = float(dart_xs[tip_y_idx])
        tip_y = float(dart_ys[tip_y_idx])
        
        dart_length = np.sqrt((tip_x - fcx)**2 + (tip_y - fcy)**2)
        method = "barrel_taper"
    else:
        # Fallback: taper profile on full dart mask
        tip_result = _taper_profile_fallback(dart_mask, dart_xs, dart_ys, flight_centroid)
        if tip_result is not None:
            tip_x, tip_y = tip_result
            dart_length = np.sqrt((tip_x - fcx)**2 + (tip_y - fcy)**2)
            method = "taper_fallback"
        else:
            dx = dart_xs.astype(np.float64) - fcx
            dy = dart_ys.astype(np.float64) - fcy
            dist_sq = dx*dx + dy*dy
            farthest_idx = np.argmax(dist_sq)
            tip_x = float(dart_xs[farthest_idx])
            tip_y = float(dart_ys[farthest_idx])
            dart_length = np.sqrt(dist_sq[farthest_idx])
            method = "farthest_fallback"
    
    if dart_length < 30:
        hough_tip = _find_tip_via_hough(dart_mask, flight_centroid)
        if hough_tip is not None:
            tip_x, tip_y = hough_tip
            dart_length = np.sqrt((tip_x - fcx)**2 + (tip_y - fcy)**2)
            method = "hough"
    
    return (tip_x, tip_y), method, dart_length


def _taper_profile_fallback(
    dart_mask: np.ndarray,
    dart_xs: np.ndarray,
    dart_ys: np.ndarray,
    flight_centroid: Tuple[float, float],
) -> Optional[Tuple[float, float]]:
    if len(dart_xs) < 20:
        return None
    
    points = np.column_stack((dart_xs, dart_ys)).astype(np.float32)
    rect = cv2.minAreaRect(points)
    (cx, cy), (rw, rh), angle = rect
    
    if rw < rh:
        rw, rh = rh, rw
        angle += 90
    
    if rw < 20:
        return None
    
    angle_rad = np.deg2rad(angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    dx = dart_xs.astype(np.float64) - cx
    dy = dart_ys.astype(np.float64) - cy
    rotated_x = dx * cos_a + dy * sin_a
    rotated_y = -dx * sin_a + dy * cos_a
    
    n_slices = 20
    x_min, x_max = rotated_x.min(), rotated_x.max()
    slice_width = (x_max - x_min) / n_slices
    if slice_width < 1:
        return None
    
    widths = []
    for i in range(n_slices):
        lo = x_min + i * slice_width
        hi = lo + slice_width
        in_slice = (rotated_x >= lo) & (rotated_x < hi)
        if np.any(in_slice):
            widths.append(np.ptp(rotated_y[in_slice]))
        else:
            widths.append(0)
    
    # Tip is always the lowest point (highest Y) — cameras above looking down
    # Find the mask pixel with the highest Y value in the bottom 20% of the oriented bbox
    bottom_mask = rotated_x > (x_max - (x_max - x_min) * 0.3)
    top_mask = rotated_x < (x_min + (x_max - x_min) * 0.3)
    
    # Check which end has higher Y (bottom of image = tip)
    if np.any(bottom_mask) and np.any(top_mask):
        bottom_avg_y = np.mean(dart_ys[bottom_mask])
        top_avg_y = np.mean(dart_ys[top_mask])
        if bottom_avg_y >= top_avg_y:
            tip_mask = bottom_mask
        else:
            tip_mask = top_mask
    else:
        # Fallback: just pick highest Y pixel
        tip_idx = np.argmax(dart_ys)
        return (float(dart_xs[tip_idx]), float(dart_ys[tip_idx]))
    
    if not np.any(tip_mask):
        return None
    
    # Among tip-end pixels, pick the one with highest Y
    tip_ys = dart_ys[tip_mask]
    tip_xs = dart_xs[tip_mask]
    highest_y_idx = np.argmax(tip_ys)
    
    return (float(tip_xs[highest_y_idx]), float(tip_ys[highest_y_idx]))


def _find_tip_via_hough(
    mask: np.ndarray,
    flight_centroid: Tuple[float, float],
) -> Optional[Tuple[float, float]]:
    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180,
                             threshold=20, minLineLength=25, maxLineGap=15)
    if lines is None:
        return None
    
    fcx, fcy = flight_centroid
    best_line, best_score = None, -1
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        if length < 15:
            continue
        mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
        dist = np.sqrt((mid_x-fcx)**2 + (mid_y-fcy)**2)
        score = length * max(0.1, 1.0 - dist/300)
        if score > best_score:
            best_score = score
            best_line = line[0]
    
    if best_line is None:
        return None
    
    x1, y1, x2, y2 = best_line
    d1 = (x1-fcx)**2 + (y1-fcy)**2
    d2 = (x2-fcx)**2 + (y2-fcy)**2
    return (float(x1), float(y1)) if d1 > d2 else (float(x2), float(y2))


# =============================================================================
# Sub-pixel Refinement
# =============================================================================

def _refine_tip_subpixel(tip, gray, mask, roi_size=10):
    tx, ty = int(tip[0]), int(tip[1])
    h, w = gray.shape[:2]
    x1, y1 = max(0, tx-roi_size), max(0, ty-roi_size)
    x2, y2 = min(w, tx+roi_size), min(h, ty+roi_size)
    if x2-x1 < 5 or y2-y1 < 5:
        return tip
    
    edges = cv2.Canny(gray[y1:y2, x1:x2], 30, 100)
    edges = cv2.bitwise_and(edges, mask[y1:y2, x1:x2])
    pts = np.column_stack(np.nonzero(edges))
    if len(pts) < 3:
        return tip
    
    coords = pts[:, ::-1] + np.array([x1, y1])
    dists = np.sqrt((coords[:,0]-tip[0])**2 + (coords[:,1]-tip[1])**2)
    idx = np.argmin(dists)
    if dists[idx] < roi_size:
        return (float(coords[idx, 0]), float(coords[idx, 1]))
    return tip


def _compute_line(flight_centroid, tip):
    fcx, fcy = flight_centroid
    tx, ty = tip
    dx, dy = tx-fcx, ty-fcy
    length = np.sqrt(dx*dx + dy*dy)
    if length < 5:
        return None
    return (dx/length, dy/length, fcx, fcy)


# =============================================================================
# Debug Visualization
# =============================================================================

def _draw_debug(frame, mask, flight_centroid, flight_contour, tip, line, method, name):
    vis = frame.copy()
    mask_color = np.zeros_like(vis)
    mask_color[mask > 0] = (0, 200, 0)
    vis = cv2.addWeighted(vis, 0.7, mask_color, 0.3, 0)
    
    cv2.drawContours(vis, [flight_contour], -1, (0, 255, 255), 2)
    
    fcx, fcy = int(flight_centroid[0]), int(flight_centroid[1])
    cv2.circle(vis, (fcx, fcy), 6, (0, 255, 255), -1)
    cv2.putText(vis, "FLIGHT", (fcx+10, fcy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
    
    if line:
        vx, vy, x0, y0 = line
        pt1 = (int(x0-vx*200), int(y0-vy*200))
        pt2 = (int(x0+vx*200), int(y0+vy*200))
        cv2.line(vis, pt1, pt2, (255,255,0), 1)
    
    tx, ty = int(tip[0]), int(tip[1])
    cv2.circle(vis, (tx, ty), 8, (0,0,255), 2)
    cv2.line(vis, (tx-15, ty), (tx+15, ty), (0,0,255), 2)
    cv2.line(vis, (tx, ty-15), (tx, ty+15), (0,0,255), 2)
    cv2.putText(vis, f"TIP ({method})", (tx+10, ty+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    
    cv2.arrowedLine(vis, (fcx, fcy), (tx, ty), (0,255,0), 2, tipLength=0.1)
    
    dart_len = np.sqrt((tx-fcx)**2 + (ty-fcy)**2)
    cv2.putText(vis, f"{name} | len={dart_len:.0f}px | {method}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return vis


# =============================================================================
# Backward Compatibility
# =============================================================================

def detect_dart_skeleton(
    current_frame, previous_frame, center=(640,360), mask=None,
    existing_dart_locations=None, camera_id="", use_adaptive_threshold=True,
    board_radius=None, debug=False, debug_name="", **kwargs,
):
    existing_mask = None
    if existing_dart_locations and len(existing_dart_locations) > 0:
        h, w = current_frame.shape[:2]
        existing_mask = np.zeros((h, w), dtype=np.uint8)
        for loc in existing_dart_locations:
            if isinstance(loc, (tuple, list)) and len(loc) >= 2:
                cv2.circle(existing_mask, (int(loc[0]), int(loc[1])), 40, 255, -1)
    
    result = detect_dart(
        current_frame=current_frame, previous_frame=previous_frame,
        board_center=center, board_radius=board_radius,
        existing_dart_mask=existing_mask, camera_id=camera_id,
        debug=debug, debug_name=debug_name,
    )
    
    if debug and result.get("debug_image") is not None and debug_name:
        try:
            cv2.imwrite(f"C:\\Users\\clawd\\DartDetectionAI\\debug_images\\{debug_name}.jpg", result["debug_image"])
        except Exception:
            pass
    return result

_current_method = "v10.2_shape_filtered"  # Default to skeleton (v10.2)

def set_detection_method(method: str):
    global _current_method
    if method == "skeleton":
        _current_method = "v10.2_shape_filtered"
    elif method == "yolo":
        _current_method = "yolo"
    elif method == "hough":
        _current_method = "hough"
    else:
        return False
    print(f"[DETECT] Method set to: {_current_method}")
    return True

def get_detection_method() -> str:
    return _current_method
