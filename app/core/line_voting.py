"""
Line-direction voting for multi-camera dart detection.

Key insight from Autodarts: Instead of scoring each camera's tip position
separately (which varies due to perspective), use the LINE DIRECTION from
each camera to vote on which segment the dart is in.

The dart shaft POINTS toward the segment it's stuck in. The line direction
is more consistent across cameras than the tip pixel position.
"""

import numpy as np
import math


# Standard dartboard segment order (clockwise from top, segment 20 at top)
SEGMENT_ORDER = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]


def angle_to_segment(angle_deg: float) -> int:
    """
    Convert angle (0-360, 0=right, CCW) to dartboard segment.
    
    The dartboard has 20 segments, each 18 degrees wide.
    Segment 20 is at the top (90 degrees).
    """
    # Normalize to 0-360
    angle_deg = angle_deg % 360
    
    # Convert from math convention (0=right, CCW) to dartboard (0=top, CW)
    # Math: 0° = right (3 o'clock), goes counter-clockwise
    # Dartboard: 20 is at top, goes clockwise
    
    # Offset so 0 degrees points to the middle of segment 20 (at top)
    # Top is 90 degrees in math convention
    dart_angle = (90 - angle_deg) % 360
    
    # Each segment is 18 degrees (360/20)
    # Segment boundaries are at 9, 27, 45, 63... (offset by 9 degrees)
    segment_idx = int((dart_angle + 9) / 18) % 20
    
    return SEGMENT_ORDER[segment_idx]


def line_to_segment_angle(vx: float, vy: float, tip_x: float, tip_y: float, 
                          center_x: float, center_y: float) -> float:
    """
    Calculate the angle from the tip to the center.
    
    This gives us the direction the dart is pointing, which indicates
    which segment it's in (regardless of exact tip position).
    
    Returns: angle in degrees (0=right, CCW)
    """
    # Vector from tip to center
    dx = center_x - tip_x
    dy = center_y - tip_y
    
    # Calculate angle (atan2 gives -180 to 180)
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    
    # Normalize to 0-360
    return (angle_deg + 360) % 360


def vote_on_line_directions(camera_results: list, calibrations: dict) -> dict:
    """
    Use line directions from multiple cameras to vote on segment.
    
    camera_results: list of dicts with:
        - camera_id: str
        - tip: (x, y) tip position  
        - line: (vx, vy, x0, y0) line parameters (optional)
        - confidence: float
        
    calibrations: dict of camera calibration data with 'center' key
    
    Returns: dict with:
        - segment: winning segment number
        - angles: list of angles from each camera
        - avg_angle: averaged angle
        - method: 'line_voting' or 'angle_averaging'
    """
    angles = []
    weights = []
    
    for cam_result in camera_results:
        cam_id = cam_result.get("camera_id")
        tip = cam_result.get("tip")
        line = cam_result.get("line")
        confidence = cam_result.get("confidence", 1.0)
        
        if not tip:
            continue
        
        cal = calibrations.get(cam_id, {})
        center = cal.get("center", [320, 240])  # Default center
        cx, cy = center[0], center[1]
        
        # Calculate angle from tip to center
        angle = line_to_segment_angle(
            line[0] if line else 0, 
            line[1] if line else 0,
            tip[0], tip[1],
            cx, cy
        )
        
        angles.append(angle)
        weights.append(confidence)
    
    if not angles:
        return None
    
    # Circular mean of angles (handles wraparound at 360)
    sin_sum = sum(w * math.sin(math.radians(a)) for a, w in zip(angles, weights))
    cos_sum = sum(w * math.cos(math.radians(a)) for a, w in zip(angles, weights))
    avg_angle_rad = math.atan2(sin_sum, cos_sum)
    avg_angle = (math.degrees(avg_angle_rad) + 360) % 360
    
    # Convert averaged angle to segment
    segment = angle_to_segment(avg_angle)
    
    return {
        "segment": segment,
        "angles": angles,
        "avg_angle": avg_angle,
        "method": "angle_averaging"
    }


def estimate_distance_from_center(camera_results: list, calibrations: dict) -> float:
    """
    Estimate normalized distance from center (for determining multiplier zone).
    
    This averages the tip positions across cameras and returns a normalized
    distance (0 = center, 1 = edge of board).
    
    Returns: normalized distance (0-1ish)
    """
    distances = []
    
    for cam_result in camera_results:
        cam_id = cam_result.get("camera_id")
        tip = cam_result.get("tip")
        
        if not tip:
            continue
        
        cal = calibrations.get(cam_id, {})
        center = cal.get("center", [320, 240])
        cx, cy = center[0], center[1]
        
        # Get outer double ellipse for board edge reference
        outer_double = cal.get("outer_double_ellipse")
        if outer_double:
            # Ellipse format: [[cx, cy], [width, height], angle]
            board_radius = max(outer_double[1][0], outer_double[1][1]) / 2
        else:
            board_radius = 200  # Default fallback
        
        # Distance from center in pixels
        dist_px = math.sqrt((tip[0] - cx)**2 + (tip[1] - cy)**2)
        
        # Normalize by board radius
        norm_dist = dist_px / board_radius
        distances.append(norm_dist)
    
    if not distances:
        return 0.5
    
    return sum(distances) / len(distances)
