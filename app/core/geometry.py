"""
Dartboard Geometry Constants

Standard dartboard dimensions in millimeters.
All measurements are radii from the center (bullseye).
"""
import math
from typing import List, Tuple

# Segment order clockwise from top (20 at 12 o'clock)
DARTBOARD_SEGMENTS: List[int] = [
    20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 
    3, 19, 7, 16, 8, 11, 14, 9, 12, 5
]

# Radii in millimeters (standard dartboard)
BULL_RADIUS_MM = 6.35           # Inner bull (50 points)
OUTER_BULL_RADIUS_MM = 15.9     # Outer bull (25 points)
TRIPLE_INNER_RADIUS_MM = 99.0   # Inner edge of triple ring
TRIPLE_OUTER_RADIUS_MM = 107.0  # Outer edge of triple ring
DOUBLE_INNER_RADIUS_MM = 162.0  # Inner edge of double ring
DOUBLE_OUTER_RADIUS_MM = 170.0  # Outer edge of double ring (board edge)

# Total dartboard diameter
DARTBOARD_DIAMETER_MM = DOUBLE_OUTER_RADIUS_MM * 2  # 340mm

# Angle offset: segment 20 is at top (12 o'clock = -90 degrees = -π/2 radians)
SEGMENT_ANGLE_OFFSET = -math.pi / 2

# Degrees per segment
DEGREES_PER_SEGMENT = 18.0  # 360 / 20


def get_segment_from_angle(angle_radians: float, rotation_offset: float = 0.0) -> int:
    """
    Get the segment number from an angle (in radians).
    
    Args:
        angle_radians: Angle from center, where 0 = right (3 o'clock)
        rotation_offset: Additional rotation in radians
        
    Returns:
        Segment number (1-20)
    """
    # Adjust for segment offset and any rotation
    adjusted = angle_radians - SEGMENT_ANGLE_OFFSET + rotation_offset
    
    # Normalize to 0-2π
    while adjusted < 0:
        adjusted += 2 * math.pi
    while adjusted >= 2 * math.pi:
        adjusted -= 2 * math.pi
    
    # Calculate segment index
    segment_index = int((adjusted / (2 * math.pi)) * 20) % 20
    
    return DARTBOARD_SEGMENTS[segment_index]


def get_zone_from_distance(distance_mm: float) -> Tuple[str, int]:
    """
    Get the zone name and multiplier from distance to center.
    
    Args:
        distance_mm: Distance from center in millimeters
        
    Returns:
        Tuple of (zone_name, multiplier)
    """
    if distance_mm <= BULL_RADIUS_MM:
        return ("inner_bull", 1)
    elif distance_mm <= OUTER_BULL_RADIUS_MM:
        return ("outer_bull", 1)
    elif distance_mm <= TRIPLE_INNER_RADIUS_MM:
        return ("single_inner", 1)
    elif distance_mm <= TRIPLE_OUTER_RADIUS_MM:
        return ("triple", 3)
    elif distance_mm <= DOUBLE_INNER_RADIUS_MM:
        return ("single_outer", 1)
    elif distance_mm <= DOUBLE_OUTER_RADIUS_MM:
        return ("double", 2)
    else:
        return ("miss", 0)


def calculate_score(distance_mm: float, angle_radians: float, rotation_offset: float = 0.0) -> dict:
    """
    Calculate the complete score for a dart.
    
    Args:
        distance_mm: Distance from center in millimeters
        angle_radians: Angle from center in radians
        rotation_offset: Board rotation offset in radians
        
    Returns:
        Dictionary with score, multiplier, segment, zone
    """
    zone, multiplier = get_zone_from_distance(distance_mm)
    
    # Bulls don't have segments
    if zone in ("inner_bull", "outer_bull"):
        segment = 0
        score = 50 if zone == "inner_bull" else 25
    elif zone == "miss":
        segment = 0
        score = 0
    else:
        segment = get_segment_from_angle(angle_radians, rotation_offset)
        score = segment * multiplier
    
    return {
        "score": score,
        "multiplier": multiplier,
        "segment": segment,
        "zone": zone
    }
