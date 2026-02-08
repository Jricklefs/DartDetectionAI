#!/usr/bin/env python3
"""
Dart tip detection using medial axis + local width.
The TIP is the narrowest part of the skeleton.
"""

import cv2
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)


def find_skeleton_endpoints(skeleton):
    """Find endpoints (pixels with exactly 1 neighbor)."""
    skel = (skeleton > 0).astype(np.uint8)
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel, -1, kernel)
    endpoints = np.where((skel == 1) & (neighbor_count == 1))
    return list(zip(endpoints[1], endpoints[0]))  # (x, y)


def detect_dart_skeleton(
    current_frame: np.ndarray,
    previous_frame: np.ndarray,
    center: tuple = None,
    mask: np.ndarray = None,
    debug: bool = True
) -> dict:
    """
    Detect dart tip using skeleton + distance transform.
    The tip is at the skeleton endpoint with MINIMUM width.
    """
    result = {"tip": None, "confidence": 0.0, "method": "skeleton_width"}
    
    if current_frame is None or previous_frame is None:
        return result
    
    if center is None:
        center = (current_frame.shape[1] // 2, current_frame.shape[0] // 2)
    
    cx, cy = center
    
    # 1. Get differential mask
    if mask is not None:
        thresh = np.where(mask == 76, 255, 0).astype(np.uint8)
    else:
        diff = cv2.absdiff(current_frame, previous_frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_diff, 40, 255, cv2.THRESH_BINARY)
    
    # 2. Morphology cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # 3. Find contours - get most elongated
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return result
    
    best_contour = None
    best_aspect = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 300 or area > 50000:
            continue
        
        rect = cv2.minAreaRect(contour)
        w, h = rect[1]
        if w == 0 or h == 0:
            continue
        
        aspect = max(w, h) / min(w, h)
        if aspect > best_aspect:
            best_aspect = aspect
            best_contour = contour
    
    if best_contour is None or best_aspect < 2.0:
        return result
    
    # 4. Create mask for this contour
    contour_mask = np.zeros(thresh.shape, dtype=np.uint8)
    cv2.drawContours(contour_mask, [best_contour], -1, 255, -1)
    
    # 5. Compute distance transform (distance to nearest edge)
    # This gives us "width" at each point
    dist_transform = cv2.distanceTransform(contour_mask, cv2.DIST_L2, 5)
    
    # 6. Skeletonize
    try:
        skeleton = cv2.ximgproc.thinning(contour_mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    except AttributeError:
        skeleton = np.zeros(contour_mask.shape, dtype=np.uint8)
        temp = contour_mask.copy()
        k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while True:
            eroded = cv2.erode(temp, k)
            opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, k)
            subset = eroded - opened
            skeleton = cv2.bitwise_or(skeleton, subset)
            temp = eroded.copy()
            if cv2.countNonZero(temp) == 0:
                break
    
    # 7. Get distance values along skeleton
    dist_on_skel = dist_transform * (skeleton > 0)
    
    # 8. Find skeleton endpoints
    endpoints = find_skeleton_endpoints(skeleton)
    
    if len(endpoints) < 2:
        # Fallback: pick point on skeleton closest to center
        skel_points = np.column_stack(np.where(skeleton > 0))[:, ::-1]
        if len(skel_points) > 0:
            distances = np.sqrt((skel_points[:, 0] - cx)**2 + (skel_points[:, 1] - cy)**2)
            idx = np.argmin(distances)
            tip_x, tip_y = float(skel_points[idx][0]), float(skel_points[idx][1])
            result["tip"] = (tip_x, tip_y)
            result["confidence"] = 0.5
        return result
    
    # 9. For each endpoint, measure local width (average in small radius)
    endpoint_widths = []
    for (x, y) in endpoints:
        # Sample width in 7x7 neighborhood
        y_min, y_max = max(0, y-3), min(dist_transform.shape[0], y+4)
        x_min, x_max = max(0, x-3), min(dist_transform.shape[1], x+4)
        local_region = dist_transform[y_min:y_max, x_min:x_max]
        
        # Only consider skeleton pixels for width
        skel_region = skeleton[y_min:y_max, x_min:x_max]
        skel_pixels = local_region[skel_region > 0]
        
        if len(skel_pixels) > 0:
            avg_width = np.mean(skel_pixels)
        else:
            avg_width = dist_transform[y, x] if 0 <= y < dist_transform.shape[0] and 0 <= x < dist_transform.shape[1] else 0
        
        endpoint_widths.append(avg_width)
    
    # 10. TIP is the endpoint with MINIMUM width
    min_width_idx = np.argmin(endpoint_widths)
    tip_x, tip_y = float(endpoints[min_width_idx][0]), float(endpoints[min_width_idx][1])
    
    # Confidence based on width difference (bigger diff = more certain)
    width_ratio = min(endpoint_widths) / (max(endpoint_widths) + 1e-6)
    result["confidence"] = 1.0 - width_ratio  # 0 = same width, 1 = very different
    
    result["tip"] = (tip_x, tip_y)
    result["widths"] = endpoint_widths
    
    # Debug
    if debug:
        debug_dir = r"C:\Users\clawd\DartDetectionAI\skeleton_debug"
        os.makedirs(debug_dir, exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        cv2.imwrite(os.path.join(debug_dir, f"{timestamp}_1_thresh.jpg"), thresh)
        
        # Visualize distance transform
        dist_vis = (dist_transform / dist_transform.max() * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(debug_dir, f"{timestamp}_2_dist.jpg"), dist_vis)
        
        cv2.imwrite(os.path.join(debug_dir, f"{timestamp}_3_skel.jpg"), skeleton)
        
        result_img = current_frame.copy()
        cv2.drawContours(result_img, [best_contour], -1, (0, 255, 0), 2)
        
        # Mark all endpoints with their widths
        for i, (ex, ey) in enumerate(endpoints):
            color = (0, 0, 255) if i == min_width_idx else (0, 255, 255)  # Red=tip, Yellow=flight
            cv2.circle(result_img, (int(ex), int(ey)), 6, color, -1)
            cv2.putText(result_img, f"w={endpoint_widths[i]:.1f}", 
                        (int(ex)+5, int(ey)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        cv2.circle(result_img, (int(cx), int(cy)), 5, (255, 0, 0), -1)
        
        cv2.imwrite(os.path.join(debug_dir, f"{timestamp}_4_result.jpg"), result_img)
    
    return result
