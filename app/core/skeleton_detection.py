"""
Skeleton-based Dart Tip Detection v2

Improved approach based on Autodarts analysis:
1. Frame differencing to isolate new dart
2. Skeletonization (thinning) to find dart shaft
3. Line fitting on skeleton to find dart axis
4. Pick endpoint closest to board center as tip

Key insight: Darts point TOWARD the center, so the tip is the 
endpoint of the skeleton line that's closest to center.
"""
import cv2
import numpy as np
import logging
from typing import Optional, Tuple, List
from dataclasses import dataclass

logger = logging.getLogger("dartdetect.skeleton")


@dataclass
class SkeletonTip:
    """Detected tip from skeleton analysis."""
    x: float
    y: float
    confidence: float
    method: str = "skeleton"


def zhang_suen_thinning(binary_image: np.ndarray) -> np.ndarray:
    """
    Zhang-Suen thinning algorithm for skeletonization.
    OpenCV's ximgproc.thinning is preferred if available.
    """
    try:
        return cv2.ximgproc.thinning(binary_image, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    except AttributeError:
        # Fallback to morphological skeleton
        skeleton = np.zeros_like(binary_image)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        img = binary_image.copy()
        
        while True:
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            img = eroded.copy()
            
            if cv2.countNonZero(img) == 0:
                break
        
        return skeleton


def find_dart_mask(current_frame: np.ndarray, previous_frame: np.ndarray,
                   threshold: float = 30.0) -> np.ndarray:
    """
    Create binary mask of new dart using frame differencing.
    """
    # Convert to grayscale if needed
    if len(current_frame.shape) == 3:
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    else:
        current_gray = current_frame
        
    if len(previous_frame.shape) == 3:
        previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    else:
        previous_gray = previous_frame
    
    # Compute absolute difference
    diff = cv2.absdiff(current_gray, previous_gray)
    
    diff_max = np.max(diff)
    diff_mean = np.mean(diff)
    diff_nonzero = np.count_nonzero(diff > threshold)
    logger.info(f"[SKEL-DIFF] max={diff_max:.0f}, mean={diff_mean:.1f}, pixels>{threshold}={diff_nonzero}")
    
    # Apply Gaussian blur to reduce noise
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    
    # Threshold to binary
    _, binary = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)   # Remove noise
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # Fill gaps
    
    # Additional erosion to thin the dart shape before skeletonization
    binary = cv2.erode(binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    
    white_pixels = np.count_nonzero(binary)
    logger.info(f"[SKEL-MASK] After morph: {white_pixels} white pixels")
    
    return binary.astype(np.uint8)


def find_dart_contours(mask: np.ndarray, min_area: int = 100, max_area: int = 50000) -> List[np.ndarray]:
    """Find contours that could be darts (filter by size/shape)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    logger.info(f"[SKEL-CONTOUR] Found {len(contours)} raw contours")
    
    dart_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        
        # Check aspect ratio - darts are elongated
        rect = cv2.minAreaRect(cnt)
        w, h = rect[1]
        if w == 0 or h == 0:
            continue
        aspect = max(w, h) / min(w, h)
        
        # Darts should be elongated (aspect > 2)
        if aspect < 1.5:
            logger.debug(f"[SKEL-CONTOUR] Skipping contour: area={area:.0f}, aspect={aspect:.1f} (too round)")
            continue
            
        dart_contours.append(cnt)
        logger.info(f"[SKEL-CONTOUR] Candidate dart: area={area:.0f}, aspect={aspect:.1f}")
    
    return dart_contours


def find_skeleton_endpoints(skeleton: np.ndarray) -> List[Tuple[int, int]]:
    """
    Find endpoints of the skeleton (pixels with only 1 neighbor).
    These are the tip and flight-end of the dart.
    """
    # Kernel to count neighbors
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    # Count neighbors for each skeleton pixel
    neighbor_count = cv2.filter2D(skeleton // 255, -1, kernel)
    
    # Endpoints have exactly 1 neighbor and are on the skeleton
    endpoints_mask = (neighbor_count == 1) & (skeleton > 0)
    
    # Get endpoint coordinates
    endpoints = np.column_stack(np.where(endpoints_mask))
    
    # Convert from (row, col) to (x, y)
    endpoints_xy = [(int(p[1]), int(p[0])) for p in endpoints]
    
    logger.info(f"[SKEL-ENDPOINTS] Found {len(endpoints_xy)} endpoints")
    
    return endpoints_xy


def fit_line_to_skeleton(skeleton: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    """
    Fit a line to the skeleton points using PCA or cv2.fitLine.
    Returns (vx, vy, x0, y0) - direction vector and point on line.
    """
    points = np.column_stack(np.where(skeleton > 0))
    
    if len(points) < 10:
        return None
    
    # Convert to (x, y) format for fitLine
    points_xy = points[:, ::-1].astype(np.float32)
    
    # Fit line: returns (vx, vy, x0, y0)
    line = cv2.fitLine(points_xy, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = line.flatten()
    
    logger.info(f"[SKEL-LINE] Fitted line: dir=({vx:.2f},{vy:.2f}), point=({x0:.1f},{y0:.1f})")
    
    return (vx, vy, x0, y0)


def find_tip_from_endpoints(endpoints: List[Tuple[int, int]], 
                            board_center: Tuple[float, float],
                            skeleton: np.ndarray = None) -> Optional[Tuple[float, float]]:
    """
    Given skeleton endpoints, pick the one closest to board center (the tip).
    Darts point toward the center, so the tip is the near end.
    """
    if not endpoints:
        return None
    
    if len(endpoints) == 1:
        return (float(endpoints[0][0]), float(endpoints[0][1]))
    
    cx, cy = board_center
    
    # Calculate distance to center for each endpoint
    distances = []
    for (x, y) in endpoints:
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        distances.append((dist, x, y))
    
    # Sort by distance - closest first
    distances.sort()
    
    # The tip is the closest endpoint to center
    _, tip_x, tip_y = distances[0]
    
    logger.info(f"[SKEL-TIP] Endpoints: {endpoints}")
    logger.info(f"[SKEL-TIP] Distances to center: {[(f'{d:.0f}', x, y) for d, x, y in distances]}")
    logger.info(f"[SKEL-TIP] Selected tip (closest): ({tip_x}, {tip_y})")
    
    return (float(tip_x), float(tip_y))


def detect_tip_skeleton(current_frame: np.ndarray, previous_frame: np.ndarray,
                        board_center: Tuple[float, float] = None,
                        diff_threshold: float = 30.0,
                        min_contour_area: int = 100) -> Optional[SkeletonTip]:
    """
    Detect dart tip using skeleton-based method.
    
    Improved algorithm:
    1. Frame diff → binary mask
    2. Find dart contour (elongated shape)
    3. Skeletonize the contour
    4. Find skeleton endpoints
    5. Pick endpoint closest to board center (that's the tip)
    """
    logger.info(f"[SKEL] Starting detection: threshold={diff_threshold}, min_area={min_contour_area}")
    
    if board_center is None:
        board_center = (current_frame.shape[1] // 2, current_frame.shape[0] // 2)
    
    # 1. Get dart mask via frame differencing
    mask = find_dart_mask(current_frame, previous_frame, diff_threshold)
    
    # 2. Find dart contours
    contours = find_dart_contours(mask, min_area=min_contour_area)
    if not contours:
        logger.warning("[SKEL] No dart-shaped contours found")
        return None
    
    # Use the largest dart-like contour
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    
    # 3. Create clean mask from contour
    dart_mask = np.zeros_like(mask)
    cv2.drawContours(dart_mask, [contour], -1, 255, -1)
    
    # 4. Skeletonize
    skeleton = zhang_suen_thinning(dart_mask)
    skel_pixels = np.count_nonzero(skeleton)
    logger.info(f"[SKEL] Skeleton has {skel_pixels} pixels")
    
    if skel_pixels < 5:
        logger.warning("[SKEL] Skeleton too small")
        return None
    
    # 5. Find endpoints
    endpoints = find_skeleton_endpoints(skeleton)
    
    if len(endpoints) < 2:
        # If we don't have 2 clear endpoints, fall back to closest point to center
        logger.warning(f"[SKEL] Only {len(endpoints)} endpoints, using closest-to-center fallback")
        points = np.column_stack(np.where(skeleton > 0))
        if len(points) == 0:
            return None
        points_xy = points[:, ::-1]
        cx, cy = board_center
        distances = np.sqrt((points_xy[:, 0] - cx)**2 + (points_xy[:, 1] - cy)**2)
        min_idx = np.argmin(distances)
        tip = (float(points_xy[min_idx, 0]), float(points_xy[min_idx, 1]))
    else:
        # 6. Pick endpoint closest to center
        tip = find_tip_from_endpoints(endpoints, board_center, skeleton)
    
    if tip is None:
        return None
    
    # Confidence based on contour area and skeleton quality
    confidence = min(1.0, area / 2000.0) * min(1.0, skel_pixels / 50.0)
    logger.info(f"[SKEL] ✓ Detected tip at ({tip[0]:.1f}, {tip[1]:.1f}), confidence={confidence:.2f}")
    
    return SkeletonTip(
        x=tip[0],
        y=tip[1],
        confidence=confidence
    )


def save_debug_visualization(current_frame: np.ndarray, previous_frame: np.ndarray,
                             tip: Optional[SkeletonTip], save_path: str,
                             board_center: Tuple[float, float] = None):
    """Save debug visualization of skeleton detection."""
    mask = find_dart_mask(current_frame, previous_frame)
    contours = find_dart_contours(mask)
    
    viz = current_frame.copy()
    
    # Draw all contours in green
    cv2.drawContours(viz, contours, -1, (0, 255, 0), 2)
    
    if contours:
        # Skeletonize largest contour
        dart_mask = np.zeros_like(mask)
        cv2.drawContours(dart_mask, [max(contours, key=cv2.contourArea)], -1, 255, -1)
        skeleton = zhang_suen_thinning(dart_mask)
        
        # Draw skeleton in blue
        viz[skeleton > 0] = (255, 0, 0)
        
        # Draw endpoints in yellow
        endpoints = find_skeleton_endpoints(skeleton)
        for (x, y) in endpoints:
            cv2.circle(viz, (x, y), 5, (0, 255, 255), -1)
    
    # Draw detected tip in red
    if tip is not None:
        cv2.circle(viz, (int(tip.x), int(tip.y)), 8, (0, 0, 255), -1)
        cv2.circle(viz, (int(tip.x), int(tip.y)), 12, (0, 0, 255), 2)
    
    # Draw board center in magenta
    if board_center is not None:
        cv2.circle(viz, (int(board_center[0]), int(board_center[1])), 5, (255, 0, 255), -1)
    
    cv2.imwrite(save_path, viz)
