#!/usr/bin/env python3
"""
Dart tip detection - find the skeleton endpoint closest to board center.
NOT just any skeleton point, but specifically an ENDPOINT of the skeleton.
"""

import cv2
import numpy as np
from datetime import datetime
import os
import logging

logger = logging.getLogger(__name__)


def find_skeleton_endpoints(skeleton: np.ndarray) -> list:
    """
    Find endpoints of skeleton (pixels with exactly 1 neighbor).
    These are the termination points of the skeleton line.
    """
    endpoints = []
    
    # Pad to avoid edge issues
    padded = np.pad(skeleton, 1, mode='constant', constant_values=0)
    
    # 8-connectivity kernel (exclude center)
    for y in range(1, padded.shape[0] - 1):
        for x in range(1, padded.shape[1] - 1):
            if padded[y, x] > 0:
                # Count neighbors in 8-connectivity
                neighbors = np.sum(padded[y-1:y+2, x-1:x+2] > 0) - 1  # -1 for self
                if neighbors == 1:
                    # This is an endpoint (only 1 connected neighbor)
                    endpoints.append((x - 1, y - 1))  # Adjust for padding
    
    return endpoints


def detect_dart_skeleton(
    current_frame: np.ndarray,
    previous_frame: np.ndarray,
    center: tuple = None,
    mask: np.ndarray = None,
    debug: bool = True
) -> dict:
    """
    Detect dart tip by finding skeleton endpoints and picking the one
    closest to board center.
    """
    result = {"tip": None, "confidence": 0.0, "method": "skeleton_endpoint"}
    
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
        if area < 200 or area > 50000:
            continue
        
        rect = cv2.minAreaRect(contour)
        w, h = rect[1]
        if w == 0 or h == 0:
            continue
        
        aspect = max(w, h) / min(w, h)
        if aspect > best_aspect:
            best_aspect = aspect
            best_contour = contour
    
    if best_contour is None:
        return result
    
    # 4. Isolate this contour
    contour_mask = np.zeros(thresh.shape, dtype=np.uint8)
    cv2.drawContours(contour_mask, [best_contour], -1, 255, -1)
    isolated = cv2.bitwise_and(thresh, contour_mask)
    
    # 5. Skeletonize
    try:
        skeleton = cv2.ximgproc.thinning(isolated, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    except AttributeError:
        skeleton = np.zeros(isolated.shape, dtype=np.uint8)
        temp = isolated.copy()
        k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while True:
            eroded = cv2.erode(temp, k)
            opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, k)
            subset = eroded - opened
            skeleton = cv2.bitwise_or(skeleton, subset)
            temp = eroded.copy()
            if cv2.countNonZero(temp) == 0:
                break
    
    # 6. Find skeleton endpoints
    endpoints = find_skeleton_endpoints(skeleton)
    
    if len(endpoints) < 2:
        # Fallback: use all skeleton points
        skel_points = np.column_stack(np.where(skeleton > 0))[:, ::-1]
        if len(skel_points) < 5:
            return result
        # Pick point closest to center
        distances = np.sqrt((skel_points[:, 0] - cx)**2 + (skel_points[:, 1] - cy)**2)
        idx = np.argmin(distances)
        tip_x, tip_y = float(skel_points[idx][0]), float(skel_points[idx][1])
    else:
        # 7. Pick the endpoint closest to center = the TIP
        # (Flight end is further from center)
        endpoints = np.array(endpoints)
        distances = np.sqrt((endpoints[:, 0] - cx)**2 + (endpoints[:, 1] - cy)**2)
        tip_idx = np.argmin(distances)
        tip_x, tip_y = float(endpoints[tip_idx][0]), float(endpoints[tip_idx][1])
    
    result["tip"] = (tip_x, tip_y)
    result["confidence"] = min(best_aspect / 5.0, 1.0)
    result["num_endpoints"] = len(endpoints) if 'endpoints' in dir() and isinstance(endpoints, np.ndarray) else 0
    
    # Debug
    if debug:
        debug_dir = r"C:\Users\clawd\DartDetectionAI\skeleton_debug"
        os.makedirs(debug_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        cv2.imwrite(os.path.join(debug_dir, f"{timestamp}_1_diff.jpg"), thresh)
        cv2.imwrite(os.path.join(debug_dir, f"{timestamp}_2_skeleton.jpg"), skeleton)
        
        result_img = current_frame.copy()
        cv2.drawContours(result_img, [best_contour], -1, (0, 255, 0), 2)
        
        # Draw all endpoints (yellow)
        if 'endpoints' in dir() and len(endpoints) > 0:
            for ep in endpoints:
                cv2.circle(result_img, (int(ep[0]), int(ep[1])), 5, (0, 255, 255), -1)
        
        cv2.circle(result_img, (int(cx), int(cy)), 5, (255, 0, 0), -1)
        cv2.circle(result_img, (int(tip_x), int(tip_y)), 8, (0, 0, 255), -1)
        
        cv2.imwrite(os.path.join(debug_dir, f"{timestamp}_3_result.jpg"), result_img)
        logger.info(f"[SKELETON] tip=({tip_x:.1f}, {tip_y:.1f})")
    
    return result
