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
from typing import Optional, Tuple, List
from dataclasses import dataclass


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
    
    Args:
        current_frame: Current frame with dart
        previous_frame: Previous frame without dart (or with fewer darts)
        threshold: Difference threshold for binary mask
    
    Returns:
        Binary mask where dart pixels are 255
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
    
    # Apply Gaussian blur to reduce noise
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    
    # Threshold to binary
    _, binary = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)  # Remove noise
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # Fill gaps
    
    return binary.astype(np.uint8)


def find_largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    """Find the largest contour in a binary mask (the dart)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Return largest by area
    return max(contours, key=cv2.contourArea)


def find_lowest_point(skeleton: np.ndarray, contour: np.ndarray = None,
                      board_center: Tuple[float, float] = None) -> Optional[Tuple[float, float]]:
    """
    Find the dart tip as the lowest point on the skeleton.
    
    "Lowest" means closest to the board center if provided,
    otherwise literally the lowest y-coordinate (assuming board at bottom).
    
    Args:
        skeleton: Skeletonized binary image
        contour: Optional contour to constrain search
        board_center: (cx, cy) of dartboard center for radial "lowest"
    
    Returns:
        (x, y) of the detected tip, or None
    """
    # Get all skeleton points
    points = np.column_stack(np.where(skeleton > 0))
    
    if len(points) == 0:
        return None
    
    # points is (row, col) = (y, x), convert to (x, y)
    points_xy = points[:, ::-1]  # Now (x, y)
    
    if board_center is not None:
        cx, cy = board_center
        # Find point closest to center (tip points toward center)
        distances = np.sqrt((points_xy[:, 0] - cx)**2 + (points_xy[:, 1] - cy)**2)
        min_idx = np.argmin(distances)
        return float(points_xy[min_idx, 0]), float(points_xy[min_idx, 1])
    else:
        # Find point with highest y (lowest on screen, assuming board at bottom)
        max_y_idx = np.argmax(points_xy[:, 1])
        return float(points_xy[max_y_idx, 0]), float(points_xy[max_y_idx, 1])


def detect_tip_skeleton(current_frame: np.ndarray, previous_frame: np.ndarray,
                        board_center: Tuple[float, float] = None,
                        diff_threshold: float = 25.0,
                        min_contour_area: int = 100) -> Optional[SkeletonTip]:
    """
    Detect dart tip using skeleton-based method.
    
    Args:
        current_frame: Current frame with new dart
        previous_frame: Previous frame (before dart landed)
        board_center: (cx, cy) of dartboard center
        diff_threshold: Threshold for frame differencing
        min_contour_area: Minimum contour area to consider as dart
    
    Returns:
        SkeletonTip with detected position, or None
    """
    # 1. Get dart mask via frame differencing
    mask = find_dart_mask(current_frame, previous_frame, diff_threshold)
    
    # 2. Find largest contour (the dart)
    contour = find_largest_contour(mask)
    if contour is None or cv2.contourArea(contour) < min_contour_area:
        return None
    
    # 3. Create clean mask from contour
    dart_mask = np.zeros_like(mask)
    cv2.drawContours(dart_mask, [contour], -1, 255, -1)
    
    # 4. Skeletonize
    skeleton = zhang_suen_thinning(dart_mask)
    
    # 5. Find lowest point (tip)
    tip = find_lowest_point(skeleton, contour, board_center)
    if tip is None:
        return None
    
    # Confidence based on skeleton quality
    skeleton_points = cv2.countNonZero(skeleton)
    contour_area = cv2.contourArea(contour)
    
    # Good skeleton should be thin relative to contour area
    # Typical dart: area ~500-2000px, skeleton ~50-200 points
    if skeleton_points > 0 and contour_area > 0:
        thinness_ratio = skeleton_points / contour_area
        # Ideal ratio ~0.05-0.2, very thin skeleton relative to area
        if 0.02 < thinness_ratio < 0.5:
            confidence = 0.8
        else:
            confidence = 0.5
    else:
        confidence = 0.3
    
    return SkeletonTip(x=tip[0], y=tip[1], confidence=confidence)


def detect_tips_skeleton_multi(current_frames: List[np.ndarray], 
                               previous_frames: List[np.ndarray],
                               board_centers: List[Tuple[float, float]] = None,
                               diff_threshold: float = 25.0) -> List[Optional[SkeletonTip]]:
    """
    Detect tips from multiple cameras using skeleton method.
    
    Args:
        current_frames: List of current frames from each camera
        previous_frames: List of previous frames from each camera
        board_centers: List of (cx, cy) for each camera's view of board center
        diff_threshold: Threshold for frame differencing
    
    Returns:
        List of SkeletonTip (or None) for each camera
    """
    results = []
    
    for i, (current, previous) in enumerate(zip(current_frames, previous_frames)):
        center = board_centers[i] if board_centers and i < len(board_centers) else None
        tip = detect_tip_skeleton(current, previous, center, diff_threshold)
        results.append(tip)
    
    return results


# Debug visualization
def visualize_skeleton_detection(current_frame: np.ndarray, previous_frame: np.ndarray,
                                 board_center: Tuple[float, float] = None,
                                 diff_threshold: float = 25.0) -> Tuple[np.ndarray, Optional[SkeletonTip]]:
    """
    Detect tip and create debug visualization.
    
    Returns:
        (debug_image, detected_tip)
    """
    # Get mask
    mask = find_dart_mask(current_frame, previous_frame, diff_threshold)
    
    # Find contour
    contour = find_largest_contour(mask)
    
    # Create debug image
    debug = current_frame.copy() if len(current_frame.shape) == 3 else cv2.cvtColor(current_frame, cv2.COLOR_GRAY2BGR)
    
    if contour is None:
        cv2.putText(debug, "No dart detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return debug, None
    
    # Draw contour
    cv2.drawContours(debug, [contour], -1, (0, 255, 0), 2)
    
    # Skeletonize
    dart_mask = np.zeros_like(mask)
    cv2.drawContours(dart_mask, [contour], -1, 255, -1)
    skeleton = zhang_suen_thinning(dart_mask)
    
    # Draw skeleton in blue
    skeleton_points = np.column_stack(np.where(skeleton > 0))
    for pt in skeleton_points:
        cv2.circle(debug, (pt[1], pt[0]), 1, (255, 0, 0), -1)
    
    # Find and draw tip
    tip = find_lowest_point(skeleton, contour, board_center)
    if tip:
        cv2.circle(debug, (int(tip[0]), int(tip[1])), 8, (0, 0, 255), -1)
        cv2.circle(debug, (int(tip[0]), int(tip[1])), 10, (255, 255, 255), 2)
        cv2.putText(debug, f"Tip: ({int(tip[0])}, {int(tip[1])})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return debug, SkeletonTip(x=tip[0], y=tip[1], confidence=0.8)
    
    return debug, None
