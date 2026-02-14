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
    prev_dart_masks: Optional[List[np.ndarray]] = None,
    camera_id: str = "",
    debug: bool = False,
    debug_name: str = "",
) -> Dict[str, Any]:
    barrel_info = []
    result = {
        "tip": None, "confidence": 0.0, "line": None,
        "dart_length": 0.0, "method": "none",
        "view_quality": 0.5, "debug_image": None,
        "mask_quality": 1.0,
    }
    
    # Step 1: Motion mask
    motion_mask, high_mask, positive_mask = _compute_motion_mask(current_frame, previous_frame)
    
    # Step 2: Subtract previous dart masks
    # When dart 2 or 3 lands, the motion mask includes ALL darts (current - baseline).
    # We subtract the actual detection masks from previous darts so only the NEW dart remains.
    # This handles the "dart shift" problem: when a new dart impacts, old darts physically move,
    # creating motion artifacts. By subtracting D1's mask from D2's detection (and D1+D2 from D3),
    # we isolate just the new dart. No dilation on prev masks - they're used as-is.
    if prev_dart_masks and len(prev_dart_masks) > 0:
        combined_prev = np.zeros_like(motion_mask)
        for pm in prev_dart_masks:
            if pm is not None and pm.shape == motion_mask.shape:
                combined_prev = cv2.bitwise_or(combined_prev, pm)
        motion_mask = cv2.bitwise_and(motion_mask, cv2.bitwise_not(combined_prev))
        # Also subtract from positive mask so flight detection ignores prev darts
        positive_mask = cv2.bitwise_and(positive_mask, cv2.bitwise_not(combined_prev))
    elif existing_dart_mask is not None:
        motion_mask = cv2.bitwise_and(motion_mask, cv2.bitwise_not(existing_dart_mask))
    
    # Step 3: Shape filter — keep only elongated (dart-shaped) blobs
    motion_mask = _shape_filter(motion_mask)
    
    # Step 3b: Mask quality — how clean is this mask?
    # A single dart mask is typically 2000-8000 px. Merged darts = much larger.
    mask_pixels = np.count_nonzero(motion_mask)
    num_components = 0
    if mask_pixels > 0:
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(motion_mask)
        # Count components > 200px (ignore tiny noise)
        num_components = sum(1 for i in range(1, n_labels) if stats[i, cv2.CC_STAT_AREA] > 200)
    
    # Quality heuristics:
    # - Normal single dart: 1000-8000px, 1-2 components → quality 1.0
    # - Merged darts: >12000px or >3 large components → quality drops
    mask_quality = 1.0
    if mask_pixels > 12000:
        # Oversized mask — likely merged darts
        mask_quality = min(1.0, 8000.0 / mask_pixels)
    if num_components > 3:
        # Many large blobs — messy mask
        mask_quality *= 0.5
    mask_quality = max(0.1, mask_quality)  # Floor at 0.1
    result["mask_quality"] = mask_quality
    
    # Step 4: Find the flight blob (largest bright region = flight feathers)
    # The flight is typically the most visible part of the dart from above.
    # We use it as an anchor point - the tip is always on the opposite end.
    flight = _find_flight_blob(motion_mask, min_area=80)
    if flight is not None:
        flight_centroid, flight_contour, flight_bbox = flight
        
        # Step 5: Find tip using flight as reference
        tip, tip_method, dart_length = _find_tip_from_flight(
            motion_mask, flight_centroid, flight_contour, flight_bbox,
            high_mask=positive_mask,
            board_center=board_center, board_radius=board_radius
        )
    else:
        tip = None
        tip_method = "none"
        dart_length = 0.0
    
    # Step 5b: Fallback when flight detection fails
    # Sometimes the flight blob doesn't pass shape filtering (e.g., partially occluded
    # by another dart, or the mask is fragmented). But the mask still has valid dart pixels.
    # In this case, skip flight detection entirely and just find the tip directly from the mask.
    # Rule: TIP = HIGHEST Y PIXEL (lowest point in image, closest to board surface).
    # Cameras are mounted above looking down, so highest Y = closest to board = dart tip.
    if tip is None and mask_pixels > 200:
        from scipy import ndimage as ndi
        clean = motion_mask.copy()
        lbl_all, n_all = ndi.label(clean)
        for lb in range(1, n_all + 1):
            if np.sum(lbl_all == lb) < 100:
                clean[lbl_all == lb] = 0
        ys, xs = np.nonzero(clean)
        if len(ys) == 0:
            ys, xs = np.nonzero(motion_mask)
        if len(ys) > 0:
            # Filter to board radius if available
            if board_center is not None and board_radius is not None and board_radius > 0:
                dists = np.sqrt((xs - board_center[0])**2 + (ys - board_center[1])**2)
                on_board = dists <= board_radius * 1.15
                if np.any(on_board):
                    xs, ys = xs[on_board], ys[on_board]
            idx = np.argmax(ys)
            tip = (float(xs[idx]), float(ys[idx]))
            tip_method = "highest_y_fallback"
            # Use mask centroid as fake flight for dart_length
            fcx, fcy = np.mean(xs), np.mean(ys)
            dart_length = np.sqrt((tip[0] - fcx)**2 + (tip[1] - fcy)**2)
            if flight is None:
                flight_centroid = (fcx, fcy)
    
    if tip is None:
        return result
    
    # Step 5c: Line projection for occluded tips
    # When a dart is partially hidden (e.g., D3 behind D1+D2), the visible mask
    # only shows the flight + some barrel. The highest-Y tip is just the bottom
    # of the visible portion, not the real tip. PCA line through the visible mask
    # gives the shaft direction - project it forward to find where the tip would be.
    # Only activates when dart_length is short (< 100px = likely occluded).
    if flight_centroid is not None and board_center is not None and board_radius is not None:
        projected_tip, proj_method = _project_tip_along_line(
            motion_mask, flight_centroid, tip, board_center, board_radius
        )
        if projected_tip is not None:
            tip = projected_tip
            tip_method = proj_method
            dart_length = np.sqrt((tip[0] - flight_centroid[0])**2 + (tip[1] - flight_centroid[1])**2)
    
    # Step 6: Sub-pixel refinement
    gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY) if len(current_frame.shape) == 3 else current_frame
    tip = _refine_tip_subpixel(tip, gray, motion_mask)
    
    # Step 7: Line + quality
    line = _compute_line(flight_centroid, tip)
    view_quality = min(1.0, dart_length / 150.0) if dart_length > 0 else 0.3
    
    # Step 7b: Line through dart shaft for line intersection triangulation.
    # Used by triangulate_with_line_intersection() to intersect lines from multiple cameras.
    # Strategy:
    #   1. Canny edge on GRAYSCALE IMAGE within mask -> find real barrel edges -> dual Hough -> average
    #   2. Fallback: cv2.fitLine on shaft contour (robust to outliers)
    #   3. Last resort: Full mask PCA
    pca_line = None
    if mask_pixels > 50:
        try:
            # Reference direction from flight->tip
            ref_angle = None
            if flight_centroid is not None and tip is not None:
                rdx = tip[0] - flight_centroid[0]
                rdy = tip[1] - flight_centroid[1]
                ref_len = np.sqrt(rdx*rdx + rdy*rdy)
                if ref_len > 10:
                    ref_angle = np.arctan2(rdy, rdx)

            # --- Method 1: Canny on grayscale image within dilated mask region ---
            # The actual barrel has sharp edges in the original image.
            # Dilate mask slightly to include edge pixels just outside mask boundary.
            gray_for_edges = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY) if len(current_frame.shape) == 3 else current_frame
            dilated_mask = cv2.dilate(motion_mask, np.ones((5,5), np.uint8), iterations=2)
            masked_gray = cv2.bitwise_and(gray_for_edges, dilated_mask)
            edges = cv2.Canny(masked_gray, 30, 100)
            # Only keep edges near the mask
            edges = cv2.bitwise_and(edges, dilated_mask)
            
            hough_lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=15,
                                          minLineLength=20, maxLineGap=10)
            
            if hough_lines is not None and len(hough_lines) > 0 and ref_angle is not None:
                # Filter lines parallel to dart axis (within 20 deg)
                parallel_lines = []
                for hl in hough_lines:
                    x1h, y1h, x2h, y2h = hl[0]
                    a = np.arctan2(y2h - y1h, x2h - x1h)
                    diff_a = abs(a - ref_angle)
                    diff_a = min(diff_a, np.pi - diff_a)
                    if diff_a < np.radians(20):
                        length = np.sqrt((x2h-x1h)**2 + (y2h-y1h)**2)
                        parallel_lines.append((hl[0], length, a))
                
                if len(parallel_lines) >= 2:
                    parallel_lines.sort(key=lambda x: -x[1])
                    cos_ref = np.cos(ref_angle)
                    sin_ref = np.sin(ref_angle)
                    
                    offsets = []
                    for seg, length, a in parallel_lines:
                        mx = (seg[0] + seg[2]) / 2.0
                        my = (seg[1] + seg[3]) / 2.0
                        if flight_centroid is not None:
                            dx_f = mx - flight_centroid[0]
                            dy_f = my - flight_centroid[1]
                            perp = -dx_f * sin_ref + dy_f * cos_ref
                        else:
                            perp = 0
                        offsets.append(perp)
                    
                    pos_lines = [(seg, l, a) for (seg, l, a), off in zip(parallel_lines, offsets) if off >= 0]
                    neg_lines = [(seg, l, a) for (seg, l, a), off in zip(parallel_lines, offsets) if off < 0]
                    
                    if pos_lines and neg_lines:
                        s1 = pos_lines[0][0]
                        s2 = neg_lines[0][0]
                        ax1 = (s1[0] + s2[0]) / 2.0
                        ay1 = (s1[1] + s2[1]) / 2.0
                        ax2 = (s1[2] + s2[2]) / 2.0
                        ay2 = (s1[3] + s2[3]) / 2.0
                        dx, dy = ax2 - ax1, ay2 - ay1
                        norm = np.sqrt(dx*dx + dy*dy)
                        if norm > 0:
                            vx, vy = dx/norm, dy/norm
                            if vy < 0:
                                vx, vy = -vx, -vy
                            pca_line = {
                                'vx': vx, 'vy': vy,
                                'x0': (ax1 + ax2) / 2.0, 'y0': (ay1 + ay2) / 2.0,
                                'elongation': (pos_lines[0][1] + neg_lines[0][1]) / 2,
                                'method': 'dual_edge_hough',
                            }
                
                if pca_line is None and parallel_lines:
                    seg = parallel_lines[0][0]
                    dx, dy = float(seg[2] - seg[0]), float(seg[3] - seg[1])
                    norm = np.sqrt(dx*dx + dy*dy)
                    if norm > 0:
                        vx, vy = dx/norm, dy/norm
                        if vy < 0:
                            vx, vy = -vx, -vy
                        pca_line = {
                            'vx': vx, 'vy': vy,
                            'x0': (seg[0] + seg[2]) / 2.0, 'y0': (seg[1] + seg[3]) / 2.0,
                            'elongation': parallel_lines[0][1],
                            'method': 'edge_hough',
                        }

            # --- Method 2: Robust fitLine on shaft contour ---
            if pca_line is None:
                contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    if len(largest_contour) > 10:
                        # If we have flight/tip info, filter to shaft region
                        pts = largest_contour.reshape(-1, 2)
                        if flight_centroid is not None and tip is not None:
                            fy, ty = flight_centroid[1], tip[1]
                            if abs(ty - fy) > 20:
                                y_min = min(fy, ty) + 0.15 * abs(ty - fy)
                                y_max = max(fy, ty) - 0.15 * abs(ty - fy)
                                shaft_mask_r = (pts[:, 1] >= y_min) & (pts[:, 1] <= y_max)
                                shaft_pts = pts[shaft_mask_r]
                                if len(shaft_pts) > 10:
                                    pts = shaft_pts
                        
                        line_params = cv2.fitLine(pts.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01)
                        vx, vy = float(line_params[0]), float(line_params[1])
                        x0, y0 = float(line_params[2]), float(line_params[3])
                        if vy < 0:
                            vx, vy = -vx, -vy
                        pca_line = {
                            'vx': vx, 'vy': vy,
                            'x0': x0, 'y0': y0,
                            'elongation': len(pts),
                            'method': 'fitline_huber',
                        }
            
            # --- Method 3: Full mask PCA (last resort) ---
            if pca_line is None:
                ys_pca, xs_pca = np.nonzero(motion_mask)
                if len(xs_pca) > 10:
                    pts = np.column_stack([xs_pca.astype(np.float64), ys_pca.astype(np.float64)])
                    mean = pts.mean(axis=0)
                    centered = pts - mean
                    cov = np.cov(centered.T)
                    eigenvalues, eigenvectors = np.linalg.eigh(cov)
                    pc = eigenvectors[:, np.argmax(eigenvalues)]
                    vx, vy = float(pc[0]), float(pc[1])
                    if vy < 0:
                        vx, vy = -vx, -vy
                    elongation = max(eigenvalues) / (min(eigenvalues) + 1e-6)
                    pca_line = {
                        'vx': vx, 'vy': vy,
                        'x0': float(mean[0]), 'y0': float(mean[1]),
                        'elongation': elongation,
                        'method': 'full_pca',
                    }
        except Exception:
            pass

    result.update({
        "tip": tip, "confidence": 0.8, "line": line,
        "dart_length": dart_length, "method": tip_method,
        "view_quality": view_quality,
        "pca_line": pca_line,
    })
    
    if debug:
        result["debug_image"] = _draw_debug(
            current_frame, motion_mask, flight_centroid, flight_contour,
            tip, line, tip_method, debug_name, barrel_info=barrel_info
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
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, open_kernel)
    
    # Signed diff: positive values = new pixels that APPEARED (new dart)
    # Negative values = pixels that DISAPPEARED (old dart shifted away)
    # Only keep strong positive signal as the "new object" mask
    signed_diff = blur_curr.astype(np.int16) - blur_prev.astype(np.int16)
    positive_mask = np.zeros_like(mask_high)
    positive_mask[signed_diff > threshold] = 255
    
    return seed, mask_high, positive_mask


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

def _trim_perpendicular_noise(mask: np.ndarray, margin: float = 2.0):
    """
    Barrel edge clipping: measure the barrel width from the clean side of
    the shaft (40-80% from flight to tip), then clip everything outside
    barrel walls + margin. Tapers wider near the flight.
    
    Measures both sides of the barrel independently and uses the NARROWER
    side's Q75 as the true barrel edge — whiskers/thorns on one side
    cannot influence the measurement.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    cleaned = np.zeros_like(mask)
    barrel_info_list = []
    
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
        rect = cv2.minAreaRect(points)
        (cx, cy), (rw, rh), angle = rect
        if rw < rh:
            rw, rh = rh, rw
            angle += 90
        
        # If blob isn't elongated, keep as-is (bull dart)
        if rw / (rh + 1) < 1.5:
            cleaned = cv2.bitwise_or(cleaned, component)
            continue
        
        # Fit barrel axis
        line_params = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        bvx, bvy, bx0, by0 = line_params.flatten()
        
        # Compute perpendicular and along-axis distances
        dx = xs.astype(np.float64) - bx0
        dy = ys.astype(np.float64) - by0
        perp_signed = dx * bvy - dy * bvx
        perp_dist = np.abs(perp_signed)
        along_dist = dx * bvx + dy * bvy
        
        # Find flight end (centroid) and tip end
        flight_along = (cx - bx0) * bvx + (cy - by0) * bvy
        
        # Shaft region: 40-80% from flight toward tip
        tip_along = np.max(along_dist) if bvy > 0 else np.min(along_dist)
        shaft_lo = flight_along + (tip_along - flight_along) * 0.4
        shaft_hi = flight_along + (tip_along - flight_along) * 0.8
        in_shaft = ((along_dist >= min(shaft_lo, shaft_hi)) & 
                    (along_dist <= max(shaft_lo, shaft_hi)))
        
        shaft_perps = perp_signed[in_shaft]
        if len(shaft_perps) < 10:
            # Not enough shaft pixels, fall back to simple trim
            keep = perp_dist <= 12.0
            trimmed = np.zeros_like(component)
            trimmed[ys[keep], xs[keep]] = 255
            cleaned = cv2.bitwise_or(cleaned, trimmed)
            continue
        
        # Measure each side independently
        left = np.abs(shaft_perps[shaft_perps < 0])
        right = np.abs(shaft_perps[shaft_perps >= 0])
        
        left_q75 = np.percentile(left, 75) if len(left) > 5 else 5.0
        right_q75 = np.percentile(right, 75) if len(right) > 5 else 5.0
        
        # Use the NARROWER side as the true barrel width
        barrel_half = min(left_q75, right_q75) + margin
        
        # Taper: wider near flight, barrel width in shaft
        flight_half_width = 35.0
        taper_length = 50.0
        dist_from_flight = np.abs(along_dist - flight_along)
        t = np.clip(dist_from_flight / taper_length, 0, 1)
        allowed = flight_half_width * (1 - t) + barrel_half * t
        
        # Store barrel info for debug visualization
        barrel_info_list.append({
            "axis_point": (float(bx0), float(by0)),
            "axis_dir": (float(bvx), float(bvy)),
            "barrel_half": float(barrel_half),
            "flight_along": float(flight_along),
            "tip_along": float(tip_along),
            "flight_half_width": float(flight_half_width),
            "taper_length": float(taper_length),
        })
        
        keep = perp_dist <= allowed
        trimmed = np.zeros_like(component)
        trimmed[ys[keep], xs[keep]] = 255
        cleaned = cv2.bitwise_or(cleaned, trimmed)
    
    return cleaned, barrel_info_list


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
    high_mask: np.ndarray = None,
    board_center: Tuple[float, float] = None,
    board_radius: float = None,
) -> Tuple[Optional[Tuple[float, float]], str, float]:
    """
    Find the dart tip: highest Y pixel in the flight's connected component.
    Cameras are above looking down, so highest Y = closest to board = tip.
    """
    fcx, fcy = flight_centroid
    h, w = mask.shape[:2]

    # If high_mask provided, re-find flight from strongest positive signal
    if high_mask is not None:
        contours_h, _ = cv2.findContours(high_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_h:
            flight_candidates = []
            for c in contours_h:
                area = cv2.contourArea(c)
                if area < 50:
                    continue
                M = cv2.moments(c)
                if M["m00"] == 0:
                    continue
                cy = M["m01"] / M["m00"]
                cx = M["m10"] / M["m00"]
                x, y, bw, bh = cv2.boundingRect(c)
                aspect = max(bw, bh) / max(1, min(bw, bh))
                if aspect > 6 and area < 200:
                    continue
                flight_candidates.append((cy, cx, c, area))

            if flight_candidates:
                flight_candidates.sort(key=lambda x: -x[3])
                best_cy, best_cx, best_contour, best_area = flight_candidates[0]
                fcx, fcy = best_cx, best_cy
                flight_contour = best_contour

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if num_labels <= 1:
        return None, "none", 0

    # Find which CC contains the flight
    flight_label = labels[int(fcy), int(fcx)]
    if flight_label == 0:
        for dy in range(-10, 11):
            for dx in range(-10, 11):
                py, px = int(fcy + dy), int(fcx + dx)
                if 0 <= py < h and 0 <= px < w and labels[py, px] > 0:
                    flight_label = labels[py, px]
                    break
            if flight_label > 0:
                break

    if flight_label == 0:
        return None, "none", 0

    # === TIP DETECTION: HIGHEST Y OF CLEANED MASK ===
    # The tip is ALWAYS the lowest point in the image (highest Y value).
    # Cameras are above looking down, so highest Y = closest to board = where tip is stuck.
    #
    # But we can't just blindly take max(Y) because:
    # 1. Tiny noise blobs (<100px) from lighting/shadow changes appear randomly
    # 2. Outlier blobs far from the dart body (e.g., old dart shift artifacts) can
    #    appear below the actual tip, stealing the "highest Y" position
    #
    # Cleaning strategy:
    # 1. Remove connected components < 100px (noise specks)
    # 2. Find the largest remaining CC (the main dart body)
    # 3. Remove other CCs that are BOTH far (>200px) AND small (<20% of main)
    #    - This keeps barrel/shaft fragments that are disconnected from flight
    #    - But removes distant noise blobs that would pull the tip off-target
    # 4. Filter to board radius (+ 15% margin) to exclude off-board pixels
    # 5. Take highest Y of what survives = tip
    from scipy import ndimage as ndi
    clean_mask = mask.copy()
    labeled_all, n_all = ndi.label(clean_mask)
    
    # Remove tiny blobs
    cc_sizes = []
    for lbl in range(1, n_all + 1):
        sz = np.sum(labeled_all == lbl)
        if sz < 100:
            clean_mask[labeled_all == lbl] = 0
        else:
            cc_sizes.append((lbl, sz))
    
    # Find largest CC and remove outlier blobs far from it
    if len(cc_sizes) > 1:
        cc_sizes.sort(key=lambda x: -x[1])
        main_label = cc_sizes[0][0]
        main_ys, main_xs = np.nonzero(labeled_all == main_label)
        main_cy, main_cx = np.mean(main_ys), np.mean(main_xs)
        
        # Remove blobs that are far from main AND small (noise outliers)
        # Keep large blobs even if distant (could be barrel/shaft separated from flight)
        main_size = cc_sizes[0][1]
        for lbl, sz in cc_sizes[1:]:
            blob_ys, blob_xs = np.nonzero(labeled_all == lbl)
            blob_cy, blob_cx = np.mean(blob_ys), np.mean(blob_xs)
            dist_to_main = np.sqrt((blob_cx - main_cx)**2 + (blob_cy - main_cy)**2)
            # Only remove if: far away AND small (< 20% of main blob)
            # Large blobs near the dart are likely barrel/shaft, keep them
            if dist_to_main > 200 and sz < main_size * 0.2:
                clean_mask[labeled_all == lbl] = 0
    
    all_ys, all_xs = np.nonzero(clean_mask)
    
    if len(all_ys) == 0:
        # Fallback to raw mask if filtering removed everything
        all_ys, all_xs = np.nonzero(mask)
        if len(all_ys) == 0:
            return None, "no_mask", 0.0
    
    # Filter to pixels within board radius (+ 10% margin) if we know the board
    # This prevents grabbing off-board noise as the "tip"
    if board_center is not None and board_radius is not None and board_radius > 0:
        dists = np.sqrt((all_xs - board_center[0])**2 + (all_ys - board_center[1])**2)
        on_board = dists <= board_radius * 1.15  # 15% margin for edge darts
        if np.any(on_board):
            all_xs = all_xs[on_board]
            all_ys = all_ys[on_board]
    
    # Highest Y across all surviving pixels = tip
    tip_idx = np.argmax(all_ys)
    tip_x = float(all_xs[tip_idx])
    tip_y = float(all_ys[tip_idx])
    
    dart_length = np.sqrt((tip_x - fcx)**2 + (tip_y - fcy)**2)
    
    return (tip_x, tip_y), "highest_y_clean_mask", dart_length




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



def _project_tip_along_line(mask, flight_centroid, tip, board_center, board_radius, 
                             min_dart_length=100):
    """
    Line projection for occluded tips.
    
    When a dart is partially hidden behind another dart, the visible mask may only
    show the flight + some barrel. The "highest Y" tip lands on the visible portion,
    not where the tip actually hits the board.
    
    Solution: fit a PCA line through ALL visible mask pixels to get the shaft direction,
    then project that line forward from the flight toward the board. The projected point
    is where the tip WOULD be if the dart continued in a straight line.
    
    Only activates when:
    - dart_length < min_dart_length (tip is suspiciously close to flight = likely occluded)
    - board_center and board_radius are available
    - PCA line has a clear direction (not a round blob)
    
    Returns:
        (projected_tip, method_name) if projection improved the tip
        (None, None) if projection not needed or failed
    """
    if board_center is None or board_radius is None or board_radius <= 0:
        return None, None
    
    # Check if dart is too short (likely occluded)
    fcx, fcy = flight_centroid
    tx, ty = tip
    dart_length = np.sqrt((tx - fcx)**2 + (ty - fcy)**2)
    
    if dart_length >= min_dart_length:
        return None, None  # Dart is long enough, no projection needed
    
    # Fit PCA line through all mask pixels
    ys, xs = np.nonzero(mask)
    if len(ys) < 50:
        return None, None  # Not enough pixels for reliable PCA
    
    # PCA: find principal axis of the mask
    mean_x, mean_y = np.mean(xs), np.mean(ys)
    coords = np.column_stack([xs - mean_x, ys - mean_y])
    cov = np.cov(coords.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Principal component = direction of most variance = shaft direction
    # eigh returns in ascending order, so [-1] is the largest eigenvalue
    pc = eigenvectors[:, -1]  # (dx, dy) direction
    
    # Check elongation: if eigenvalue ratio is low, mask is too round for reliable direction
    if eigenvalues[-1] < 1e-6:
        return None, None
    elongation = eigenvalues[-1] / max(eigenvalues[0], 1e-6)
    if elongation < 3.0:
        return None, None  # Not elongated enough - direction unreliable
    
    # Orient the line so it points AWAY from flight (toward tip = higher Y)
    # Flight is at the top (low Y), tip is at the bottom (high Y)
    vx, vy = pc[0], pc[1]
    
    # Direction should go from flight toward higher Y (toward board)
    # Check: does moving along (vx, vy) from flight centroid increase Y?
    if vy < 0:
        vx, vy = -vx, -vy  # Flip to point downward (toward board)
    
    # Project from the flight centroid along the PCA direction
    # Walk until we reach a reasonable tip distance or hit board edge
    # A typical dart is 150-300px long in the image
    # Project to where the tip should be: estimate based on board_radius
    # The tip should be somewhere on the board, so project until we reach
    # board_radius distance from center, or a max reasonable dart length
    
    best_tip = None
    best_y = -1
    
    # Walk along the line in small steps
    for step in range(10, 500, 5):
        px = fcx + vx * step
        py = fcy + vy * step
        
        # Check if still on/near board
        dist_from_center = np.sqrt((px - board_center[0])**2 + (py - board_center[1])**2)
        if dist_from_center > board_radius * 1.1:
            break  # Past the board edge, stop
        
        # Track the farthest point on the board (highest Y within board)
        if py > best_y:
            best_y = py
            best_tip = (float(px), float(py))
    
    if best_tip is None:
        return None, None
    
    # Only use projection if it's significantly better than original tip
    proj_length = np.sqrt((best_tip[0] - fcx)**2 + (best_tip[1] - fcy)**2)
    if proj_length < dart_length * 1.5:
        return None, None  # Projection didn't extend much, not worth it
    
    return best_tip, "line_projection"

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

def _draw_debug(frame, mask, flight_centroid, flight_contour, tip, line, method, name, barrel_info=None):
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
    # Draw barrel edge lines
    if barrel_info:
        for bi in barrel_info:
            bx0, by0 = bi["axis_point"]
            bvx, bvy = bi["axis_dir"]
            bh = bi["barrel_half"]
            # Perpendicular direction
            px, py = -bvy, bvx
            # Draw barrel walls as cyan lines along the full dart length
            extent = 200  # pixels along axis
            for side in [1, -1]:
                # Offset from axis by barrel_half in perpendicular direction
                ox = px * bh * side
                oy = py * bh * side
                p1 = (int(bx0 - bvx*100 + ox), int(by0 - bvy*100 + oy))
                p2 = (int(bx0 + bvx*extent + ox), int(by0 + bvy*extent + oy))
                cv2.line(vis, p1, p2, (255, 255, 0), 1)  # cyan barrel edges
    
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
        existing_dart_mask=existing_mask,
        prev_dart_masks=kwargs.get("prev_dart_masks"),
        camera_id=camera_id,
        debug=debug, debug_name=debug_name,
    )
    
    if debug and result.get("debug_image") is not None and debug_name:
        try:
            cv2.imwrite(f"C:\\Users\\clawd\\DartDetectionAI\\debug_images\\{debug_name}.jpg", result["debug_image"])
        except Exception:
            pass
    return result

_current_detection_method = "v10.2_shape_filtered"

def set_detection_method(method: str):
    global _current_detection_method
    method_map = {
        "skeleton": "v10.2_shape_filtered",
        "yolo": "yolo",
        "hough": "hough",
    }
    _current_detection_method = method_map.get(method, method)
    return True

def get_detection_method() -> str:
    return _current_detection_method
