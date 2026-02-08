"""
Skeleton-based Dart Tip Detection

Alternative to YOLO - uses classical computer vision:
1. Frame differencing to isolate new dart
2. Skeletonization (thinning) to find dart shaft
3. Lowest point detection to find tip

Based on Autodarts approach.
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
        # Try OpenCV's built-in thinning (requires opencv-contrib)
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
                   threshold: float = 25.0) -> np.ndarray:
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
    
    # Log diff stats
    diff_max = np.max(diff)
    diff_mean = np.mean(diff)
    diff_nonzero = np.count_nonzero(diff > threshold)
    logger.info(f"[SKEL-DIFF] max={diff_max:.0f}, mean={diff_mean:.1f}, pixels>{threshold}={diff_nonzero}")
    
    # Apply Gaussian blur to reduce noise
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    
    # Threshold to binary
    _, binary = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)  # Remove noise
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # Fill gaps
    
    white_pixels = np.count_nonzero(binary)
    logger.info(f"[SKEL-MASK] After morph: {white_pixels} white pixels")
    
    return binary.astype(np.uint8)


def find_largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    """Find the largest contour in a binary mask (the dart)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    logger.info(f"[SKEL-CONTOUR] Found {len(contours)} contours")
    
    if not contours:
        return None
    
    largest = max(contours, key=cv2.contourArea)
    logger.info(f"[SKEL-CONTOUR] Largest contour area: {cv2.contourArea(largest):.0f} px")
    
    return largest


def find_lowest_point(skeleton: np.ndarray, contour: np.ndarray = None,
                      board_center: Tuple[float, float] = None) -> Optional[Tuple[float, float]]:
    """
    Find the dart tip as the lowest point on the skeleton.
    """
    # Get all skeleton points
    points = np.column_stack(np.where(skeleton > 0))
    
    logger.info(f"[SKEL-POINTS] Skeleton has {len(points)} points")
    
    if len(points) == 0:
        return None
    
    # points is (row, col) = (y, x), convert to (x, y)
    points_xy = points[:, ::-1]  # Now (x, y)
    
    if board_center is not None:
        cx, cy = board_center
        # Find point closest to center (tip points toward center)
        distances = np.sqrt((points_xy[:, 0] - cx)**2 + (points_xy[:, 1] - cy)**2)
        min_idx = np.argmin(distances)
        tip_x, tip_y = float(points_xy[min_idx, 0]), float(points_xy[min_idx, 1])
        logger.info(f"[SKEL-TIP] Closest to center ({cx:.0f},{cy:.0f}): ({tip_x:.1f},{tip_y:.1f}) dist={distances[min_idx]:.1f}")
        return tip_x, tip_y
    else:
        # Find point with highest y (lowest on screen)
        max_y_idx = np.argmax(points_xy[:, 1])
        return float(points_xy[max_y_idx, 0]), float(points_xy[max_y_idx, 1])


def detect_tip_skeleton(current_frame: np.ndarray, previous_frame: np.ndarray,
                        board_center: Tuple[float, float] = None,
                        diff_threshold: float = 25.0,
                        min_contour_area: int = 100) -> Optional[SkeletonTip]:
    """
    Detect dart tip using skeleton-based method.
    """
    logger.info(f"[SKEL] Starting detection: threshold={diff_threshold}, min_area={min_contour_area}")
    
    # 1. Get dart mask via frame differencing
    mask = find_dart_mask(current_frame, previous_frame, diff_threshold)
    
    # 2. Find largest contour (the dart)
    contour = find_largest_contour(mask)
    if contour is None:
        logger.warning("[SKEL] No contours found - no dart visible in diff")
        return None
    
    area = cv2.contourArea(contour)
    if area < min_contour_area:
        logger.warning(f"[SKEL] Contour too small: {area:.0f} < {min_contour_area}")
        return None
    
    # 3. Create clean mask from contour
    dart_mask = np.zeros_like(mask)
    cv2.drawContours(dart_mask, [contour], -1, 255, -1)
    
    # 4. Skeletonize
    skeleton = zhang_suen_thinning(dart_mask)
    
    # 5. Find lowest point (tip)
    tip = find_lowest_point(skeleton, contour, board_center)
    if tip is None:
        logger.warning("[SKEL] No tip found in skeleton")
        return None
    
    # Confidence based on contour area
    confidence = min(1.0, area / 2000.0)
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
    contour = find_largest_contour(mask)
    
    # Create side-by-side visualization
    viz = current_frame.copy()
    
    if contour is not None:
        cv2.drawContours(viz, [contour], -1, (0, 255, 0), 2)
    
    if tip is not None:
        cv2.circle(viz, (int(tip.x), int(tip.y)), 8, (0, 0, 255), -1)
        cv2.circle(viz, (int(tip.x), int(tip.y)), 12, (255, 255, 0), 2)
    
    if board_center is not None:
        cv2.circle(viz, (int(board_center[0]), int(board_center[1])), 5, (255, 0, 255), -1)
    
    cv2.imwrite(save_path, viz)
