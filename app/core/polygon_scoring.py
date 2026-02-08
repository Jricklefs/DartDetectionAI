"""
Polygon-based dart scoring using Autodarts-style 20-point calibration.

Key insight: The 20 polygon points are at segment BOUNDARIES, starting from 
segment 20 and going clockwise. Point[0] is between segments 5 and 20,
Point[1] is between segments 20 and 1, etc.
"""

import numpy as np
import cv2


def score_with_polygons(tip_x, tip_y, cal_data):
    """
    Score a dart tip using polygon boundaries.
    
    cal_data should have:
    - center: [cx, cy]
    - polygons:
        - double_outer: 20 points
        - double_inner: 20 points
        - triple_outer: 20 points
        - triple_inner: 20 points
    - segment_order: [20, 1, 18, 4, ...]
    
    Returns: dict with segment, multiplier, score
    """
    center = cal_data['center']
    polygons = cal_data['polygons']
    segment_order = cal_data['segment_order']
    
    cx, cy = center
    
    # Convert polygon lists to numpy arrays
    double_outer = np.array(polygons['double_outer'], dtype=np.float32)
    double_inner = np.array(polygons['double_inner'], dtype=np.float32)
    triple_outer = np.array(polygons['triple_outer'], dtype=np.float32)
    triple_inner = np.array(polygons['triple_inner'], dtype=np.float32)
    
    point = (float(tip_x), float(tip_y))
    
    # Check which zone the point is in
    in_double_outer = cv2.pointPolygonTest(double_outer, point, False) >= 0
    in_double_inner = cv2.pointPolygonTest(double_inner, point, False) >= 0
    in_triple_outer = cv2.pointPolygonTest(triple_outer, point, False) >= 0
    in_triple_inner = cv2.pointPolygonTest(triple_inner, point, False) >= 0
    
    # Distance from center for bull check
    dist_to_center = np.sqrt((tip_x - cx)**2 + (tip_y - cy)**2)
    
    # Determine multiplier
    if not in_double_outer:
        # Outside board - miss
        return {'segment': 0, 'multiplier': 0, 'score': 0, 'zone': 'miss'}
    elif not in_double_inner:
        # In double ring
        multiplier = 2
        zone = 'double'
    elif not in_triple_outer:
        # In single (outer)
        multiplier = 1
        zone = 'single_outer'
    elif not in_triple_inner:
        # In triple ring
        multiplier = 3
        zone = 'triple'
    else:
        # Inside triple ring - check for bull
        # Approximate bull radii based on polygon scale
        # Inner bull ~ 6% of board, outer bull ~ 16% of board
        avg_double_dist = np.mean([np.sqrt((p[0]-cx)**2 + (p[1]-cy)**2) for p in double_outer])
        inner_bull_radius = avg_double_dist * 0.038  # ~6.35mm / 170mm
        outer_bull_radius = avg_double_dist * 0.095  # ~16mm / 170mm
        
        if dist_to_center <= inner_bull_radius:
            return {'segment': 25, 'multiplier': 2, 'score': 50, 'zone': 'double_bull'}
        elif dist_to_center <= outer_bull_radius:
            return {'segment': 25, 'multiplier': 1, 'score': 25, 'zone': 'single_bull'}
        else:
            multiplier = 1
            zone = 'single_inner'
    
    # Determine segment by angle from center
    angle = np.arctan2(tip_y - cy, tip_x - cx)
    
    # The polygon points define segment boundaries
    # Point[i] is the boundary between segment_order[i-1] and segment_order[i]
    # So we need to find which two adjacent points our angle falls between
    
    segment = find_segment_from_angle(angle, double_outer, center, segment_order)
    score = segment * multiplier
    
    return {
        'segment': segment,
        'multiplier': multiplier,
        'score': score,
        'zone': zone
    }


def find_segment_from_angle(angle, boundary_points, center, segment_order):
    """
    Find which segment a given angle falls into.
    
    boundary_points: 20 points defining segment boundaries
    center: [cx, cy]
    segment_order: [20, 1, 18, 4, ...] - segments in clockwise order starting from 20
    """
    cx, cy = center
    
    # Calculate angle to each boundary point
    boundary_angles = []
    for px, py in boundary_points:
        ba = np.arctan2(py - cy, px - cx)
        boundary_angles.append(ba)
    
    # Find which segment the angle falls into
    # Segment i is between boundary i and boundary (i+1)%20
    for i in range(20):
        left = boundary_angles[i]
        right = boundary_angles[(i + 1) % 20]
        
        # Handle angle wrap-around
        if right < left:
            # Boundary crosses the -pi/+pi line
            if angle >= left or angle < right:
                return segment_order[i]
        else:
            if left <= angle < right:
                return segment_order[i]
    
    # Fallback - find closest boundary
    min_diff = float('inf')
    closest_idx = 0
    for i, ba in enumerate(boundary_angles):
        diff = abs(angle - ba)
        if diff > np.pi:
            diff = 2 * np.pi - diff
        if diff < min_diff:
            min_diff = diff
            closest_idx = i
    
    return segment_order[closest_idx]


# Test
if __name__ == "__main__":
    import json
    
    with open(r"C:\Users\clawd\DartDetectionAI\autodarts_calibration.json", "r") as f:
        cal = json.load(f)
    
    # Test with cam0 calibration
    cam0_cal = cal['cam0']
    
    # Test a few points
    test_cases = [
        (670, 288, "center - should be bull"),
        (670, 200, "above center - should be segment near 20"),
        (800, 288, "right of center - should be segment 6 area"),
        (540, 288, "left of center - should be segment 11 area"),
    ]
    
    for tx, ty, desc in test_cases:
        result = score_with_polygons(tx, ty, cam0_cal)
        print(f"Point ({tx}, {ty}): {desc}")
        print(f"  -> segment={result['segment']}, mult={result['multiplier']}, zone={result['zone']}")
        print()
