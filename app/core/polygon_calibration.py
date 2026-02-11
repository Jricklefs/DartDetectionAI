"""
Polygon-based Dartboard Calibration Module

Autodarts-style 20-point polygon calibration for precise segment boundaries.
This runs parallel to the existing ellipse calibration - both can coexist.

Calibration structure (per camera):
- bull: (x, y) center point
- double_outers: 20 points on outer edge of double ring
- double_inners: 20 points on inner edge of double ring  
- treble_outers: 20 points on outer edge of triple ring
- treble_inners: 20 points on inner edge of triple ring

Points are ordered starting from the 20/1 boundary, going clockwise.
"""

import math
import toml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

# Dartboard segment order (clockwise from 20/1 boundary)
SEGMENT_ORDER = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]


@dataclass
class PolygonCalibration:
    """Stores polygon-based calibration for a single camera."""
    camera_id: str
    bull: Tuple[float, float]  # Center point (x, y) in pixels
    double_outers: List[Tuple[float, float]]  # 20 points
    double_inners: List[Tuple[float, float]]  # 20 points
    treble_outers: List[Tuple[float, float]]  # 20 points
    treble_inners: List[Tuple[float, float]]  # 20 points
    image_width: int = 1280
    image_height: int = 720
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        return {
            "camera_id": self.camera_id,
            "bull": list(self.bull),
            "double_outers": [list(p) for p in self.double_outers],
            "double_inners": [list(p) for p in self.double_inners],
            "treble_outers": [list(p) for p in self.treble_outers],
            "treble_inners": [list(p) for p in self.treble_inners],
            "image_width": self.image_width,
            "image_height": self.image_height,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolygonCalibration":
        """Create from dictionary."""
        return cls(
            camera_id=data["camera_id"],
            bull=tuple(data["bull"]),
            double_outers=[tuple(p) for p in data["double_outers"]],
            double_inners=[tuple(p) for p in data["double_inners"]],
            treble_outers=[tuple(p) for p in data["treble_outers"]],
            treble_inners=[tuple(p) for p in data["treble_inners"]],
            image_width=data.get("image_width", 1280),
            image_height=data.get("image_height", 720),
        )


def import_autodarts_config(config_path: str) -> Dict[str, PolygonCalibration]:
    """
    Import calibration from Autodarts config.toml file.
    
    Args:
        config_path: Path to Autodarts config.toml
        
    Returns:
        Dict mapping camera_id to PolygonCalibration
    """
    config = toml.load(config_path)
    
    # Get camera config
    cam_config = config.get("cam", {})
    width = cam_config.get("width", 1280)
    height = cam_config.get("height", 720)
    
    calibrations = {}
    dartboard = config.get("dartboard", {})
    
    for cam_key, cam_data in dartboard.items():
        camera_id = cam_key  # "0", "1", "2"
        
        bull = cam_data.get("bull", [width // 2, height // 2])
        
        # Autodarts stores as normalized [0-1] in [calibration] section
        # but pixel coords in [dartboard] section
        # The dartboard section has raw pixel coordinates
        double_outers = cam_data.get("double_outers", [])
        double_inners = cam_data.get("double_inners", [])
        treble_outers = cam_data.get("treble_outers", [])
        treble_inners = cam_data.get("treble_inners", [])
        
        # Validate we have 20 points per ring
        if len(double_outers) != 20:
            print(f"Warning: Camera {camera_id} has {len(double_outers)} double_outers (expected 20)")
        if len(double_inners) != 20:
            print(f"Warning: Camera {camera_id} has {len(double_inners)} double_inners (expected 20)")
        if len(treble_outers) != 20:
            print(f"Warning: Camera {camera_id} has {len(treble_outers)} treble_outers (expected 20)")
        if len(treble_inners) != 20:
            print(f"Warning: Camera {camera_id} has {len(treble_inners)} treble_inners (expected 20)")
        
        calibrations[camera_id] = PolygonCalibration(
            camera_id=camera_id,
            bull=tuple(bull),
            double_outers=[tuple(p) for p in double_outers],
            double_inners=[tuple(p) for p in double_inners],
            treble_outers=[tuple(p) for p in treble_outers],
            treble_inners=[tuple(p) for p in treble_inners],
            image_width=width,
            image_height=height,
        )
    
    return calibrations


def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    """
    Check if a point is inside a polygon using ray casting algorithm.
    
    Args:
        point: (x, y) coordinate
        polygon: List of (x, y) vertices
        
    Returns:
        True if point is inside polygon
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    
    return inside


def shrink_polygon(polygon: List[Tuple[float, float]], margin_px: float = 2.0) -> List[Tuple[float, float]]:
    """
    Shrink a polygon inward by a margin (in pixels).
    Used to add robustness to boundary detection.
    
    Args:
        polygon: List of (x, y) vertices
        margin_px: Pixels to shrink inward (default 2px)
        
    Returns:
        Shrunk polygon
    """
    if len(polygon) < 3:
        return polygon
    
    # Find centroid
    cx = sum(p[0] for p in polygon) / len(polygon)
    cy = sum(p[1] for p in polygon) / len(polygon)
    
    # Shrink each point toward centroid
    shrunk = []
    for px, py in polygon:
        dx = px - cx
        dy = py - cy
        dist = math.sqrt(dx*dx + dy*dy)
        if dist > margin_px:
            factor = (dist - margin_px) / dist
            shrunk.append((cx + dx * factor, cy + dy * factor))
        else:
            shrunk.append((cx, cy))
    
    return shrunk


def validate_polygon(polygon: List[Tuple[float, float]]) -> bool:
    """
    Validate that a polygon is properly formed.
    
    Returns:
        True if polygon is valid (at least 3 points, forms closed shape)
    """
    if len(polygon) < 3:
        return False
    
    # Check that points are not all collinear
    # Use cross product of first two edges
    if len(polygon) >= 3:
        x1, y1 = polygon[0]
        x2, y2 = polygon[1]
        x3, y3 = polygon[2]
        cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
        if abs(cross) < 1e-10:
            # First 3 points collinear, check more
            for i in range(3, len(polygon)):
                x3, y3 = polygon[i]
                cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
                if abs(cross) > 1e-10:
                    break
            else:
                return False  # All points collinear
    
    return True


def point_in_ring_segment(
    point: Tuple[float, float],
    segment_idx: int,
    outer_ring: List[Tuple[float, float]],
    inner_ring: List[Tuple[float, float]]
) -> bool:
    """
    Check if point is in a specific ring segment (wedge between two rings).
    
    Args:
        point: (x, y) coordinate
        segment_idx: 0-19 index into ring points
        outer_ring: 20 points for outer boundary
        inner_ring: 20 points for inner boundary
        
    Returns:
        True if point is in the segment
    """
    if len(outer_ring) != 20 or len(inner_ring) != 20:
        return False
    
    # Build polygon for this segment (4 corners)
    next_idx = (segment_idx + 1) % 20
    
    # Quad: outer[i], outer[i+1], inner[i+1], inner[i]
    quad = [
        outer_ring[segment_idx],
        outer_ring[next_idx],
        inner_ring[next_idx],
        inner_ring[segment_idx],
    ]
    
    return point_in_polygon(point, quad)


def score_from_polygon_calibration(
    tip_px: Tuple[float, float],
    calibration: PolygonCalibration
) -> Dict[str, Any]:
    """
    Calculate dart score from tip position using polygon calibration.
    
    Args:
        tip_px: (x, y) tip position in pixels
        calibration: PolygonCalibration for this camera
        
    Returns:
        Dict with segment, zone, score, multiplier
    """
    x, y = tip_px
    bull = calibration.bull
    
    # Calculate distance from bull for bull detection
    dist_from_bull = math.sqrt((x - bull[0])**2 + (y - bull[1])**2)
    
    # Estimate bull radius from treble inner ring
    # Bull is roughly 1/6 the radius of the treble ring
    if calibration.treble_inners:
        # Average distance of treble_inners from bull
        treble_dists = [math.sqrt((p[0] - bull[0])**2 + (p[1] - bull[1])**2) 
                        for p in calibration.treble_inners]
        avg_treble_dist = sum(treble_dists) / len(treble_dists)
        
        # Standard dartboard: bullseye=6.35mm, bull=15.9mm, inner_triple=99mm
        bullseye_radius = avg_treble_dist * (6.35 / 99)
        bull_radius = avg_treble_dist * (15.9 / 99)
    else:
        # Fallback estimates
        bullseye_radius = 15
        bull_radius = 35
    
    # Check if OUTSIDE the double ring first
    # Instead of returning miss, check how far outside and use angle-based scoring
    # The tip may be slightly outside the polygon due to detection noise
    is_outside_polygon = False
    if calibration.double_outers and len(calibration.double_outers) >= 3:
        polygon = list(calibration.double_outers)
        if not point_in_polygon((x, y), polygon):
            # Check distance from bull vs max polygon radius
            # If way too far out, it's truly a miss
            if calibration.double_outers:
                max_poly_dist = max(
                    math.sqrt((p[0] - bull[0])**2 + (p[1] - bull[1])**2) 
                    for p in calibration.double_outers
                )
                if dist_from_bull > max_poly_dist * 1.3:  # 30% tolerance
                    return {
                        "segment": 0,
                        "zone": "miss",
                        "score": 0,
                        "multiplier": 0,
                        "ring": "miss"
                    }
            is_outside_polygon = True
    
    # Check bullseye first
    if dist_from_bull <= bullseye_radius:
        return {
            "segment": 50,
            "zone": "inner_bull",
            "score": 50,
            "multiplier": 1,
            "ring": "bullseye"
        }
    
    # Check outer bull
    if dist_from_bull <= bull_radius:
        return {
            "segment": 25,
            "zone": "outer_bull", 
            "score": 25,
            "multiplier": 1,
            "ring": "bull"
        }
    
    # Check each segment for each ring
    for seg_idx in range(20):
        segment_value = SEGMENT_ORDER[seg_idx]
        
        # Check double ring (outer_double to inner_double)
        if point_in_ring_segment(tip_px, seg_idx, 
                                  calibration.double_outers, 
                                  calibration.double_inners):
            return {
                "segment": segment_value,
                "zone": "double",
                "score": segment_value * 2,
                "multiplier": 2,
                "ring": "double"
            }
        
        # Check triple ring (outer_triple to inner_triple)
        if point_in_ring_segment(tip_px, seg_idx,
                                  calibration.treble_outers,
                                  calibration.treble_inners):
            return {
                "segment": segment_value,
                "zone": "triple",
                "score": segment_value * 3,
                "multiplier": 3,
                "ring": "triple"
            }
        
        # Check outer single (inner_double to outer_triple)
        if point_in_ring_segment(tip_px, seg_idx,
                                  calibration.double_inners,
                                  calibration.treble_outers):
            return {
                "segment": segment_value,
                "zone": "single_outer",
                "score": segment_value,
                "multiplier": 1,
                "ring": "single"
            }
        
        # Check inner single (inner_triple to bull)
        # Need to create a polygon from inner_triple to bull
        # For now, use angle-based detection for inner single
    
    # Fallback: determine segment by which polygon boundary points the tip falls between
    # This accounts for camera perspective distortion (unlike standard angle math)
    angle = math.atan2(y - bull[1], x - bull[0])
    angle_deg = math.degrees(angle)
    
    # Get angles of all double_outer boundary points from bull
    if calibration.double_outers and len(calibration.double_outers) == 20:
        boundary_angles = []
        for p in calibration.double_outers:
            ba = math.degrees(math.atan2(p[1] - bull[1], p[0] - bull[0]))
            boundary_angles.append(ba)
        
        # Find which two boundary points the tip angle falls between
        # Boundary i is between SEGMENT_ORDER[i] and SEGMENT_ORDER[(i+1)%20]
        # So if tip is between boundary[i] and boundary[(i+1)%20], it's in SEGMENT_ORDER[(i+1)%20]
        best_seg_idx = 0
        min_angular_dist = 999
        for i in range(20):
            ba1 = boundary_angles[i]
            ba2 = boundary_angles[(i + 1) % 20]
            
            # Check if angle_deg is between ba1 and ba2 (handling wraparound)
            # Normalize angles relative to ba1
            diff1 = ((angle_deg - ba1 + 180) % 360) - 180  # -180 to 180
            diff2 = ((ba2 - ba1 + 180) % 360) - 180
            
            if diff2 > 0:
                # Normal case: ba1 < ba2
                if 0 <= diff1 <= diff2:
                    best_seg_idx = (i + 1) % 20
                    break
            else:
                # Wraparound case: ba2 < ba1
                if diff1 >= 0 or diff1 <= diff2:
                    best_seg_idx = (i + 1) % 20
                    break
        
        segment_value = SEGMENT_ORDER[best_seg_idx]
    else:
        # No polygon data, use standard angle math as last resort
        if angle_deg < 0:
            angle_deg += 360
        adjusted_angle = (angle_deg + 90 + 9) % 360
        seg_idx = int(adjusted_angle / 18) % 20
        segment_value = SEGMENT_ORDER[seg_idx]
    
    # Check if inside inner triple ring (inner single)
    if calibration.treble_inners:
        treble_inner_dists = [math.sqrt((p[0] - bull[0])**2 + (p[1] - bull[1])**2)
                              for p in calibration.treble_inners]
        avg_treble_inner = sum(treble_inner_dists) / len(treble_inner_dists)
        
        if dist_from_bull <= avg_treble_inner and dist_from_bull > bull_radius:
            return {
                "segment": segment_value,
                "zone": "single_inner",
                "score": segment_value,
                "multiplier": 1,
                "ring": "single"
            }
    
    # Default: inner single
    return {
        "segment": segment_value,
        "zone": "single_inner",
        "score": segment_value,
        "multiplier": 1,
        "ring": "single"
    }


# Global storage for polygon calibrations (parallel to ellipse calibrations)
_polygon_calibrations: Dict[str, PolygonCalibration] = {}
_calibration_mode: str = "ellipse"  # "ellipse" or "polygon"


def set_calibration_mode(mode: str) -> bool:
    """Set the active calibration mode."""
    global _calibration_mode
    if mode not in ("ellipse", "polygon"):
        return False
    _calibration_mode = mode
    print(f"[CALIBRATION] Mode set to: {mode}")
    return True


def get_calibration_mode() -> str:
    """Get the current calibration mode."""
    return _calibration_mode


def set_polygon_calibration(camera_id: str, calibration: PolygonCalibration):
    """Store polygon calibration for a camera."""
    global _polygon_calibrations
    _polygon_calibrations[camera_id] = calibration


def get_polygon_calibration(camera_id: str) -> Optional[PolygonCalibration]:
    """Get polygon calibration for a camera."""
    return _polygon_calibrations.get(camera_id)


def get_all_polygon_calibrations() -> Dict[str, PolygonCalibration]:
    """Get all polygon calibrations."""
    return _polygon_calibrations.copy()


def load_polygon_calibrations_from_autodarts(config_path: str) -> int:
    """
    Load polygon calibrations from Autodarts config.
    
    Returns number of cameras loaded.
    """
    global _polygon_calibrations
    try:
        calibrations = import_autodarts_config(config_path)
        _polygon_calibrations.update(calibrations)
        print(f"[CALIBRATION] Loaded {len(calibrations)} polygon calibrations from Autodarts")
        return len(calibrations)
    except Exception as e:
        print(f"[CALIBRATION] Error loading Autodarts config: {e}")
        return 0


def save_polygon_calibrations(file_path: str) -> bool:
    """Save polygon calibrations to JSON file."""
    import json
    try:
        data = {
            cam_id: cal.to_dict() 
            for cam_id, cal in _polygon_calibrations.items()
        }
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"[CALIBRATION] Error saving calibrations: {e}")
        return False


def load_polygon_calibrations(file_path: str) -> int:
    """Load polygon calibrations from JSON file."""
    import json
    global _polygon_calibrations
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        for cam_id, cal_data in data.items():
            _polygon_calibrations[cam_id] = PolygonCalibration.from_dict(cal_data)
        return len(data)
    except Exception as e:
        print(f"[CALIBRATION] Error loading calibrations: {e}")
        return 0
"""
Convert YOLO-detected calibration points to ordered 20-point polygons.

YOLO detects ring intersection points, but they come in random order.
This module sorts them into the correct clockwise order starting from
the 20/1 segment boundary, matching the Autodarts format.
"""

import math
from typing import List, Tuple, Dict, Any, Optional

# Dartboard segment order (clockwise from top, starting at 20)
SEGMENT_ORDER = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]


def points_to_ordered_polygon(
    points: List[Tuple[float, float, float]],  # (x, y, confidence)
    center: Tuple[float, float],
    twenty_angle_rad: float
) -> List[Tuple[float, float]]:
    """
    Convert unordered ring points to ordered 20-point polygon.
    
    Args:
        points: Detected points (x, y, confidence) - should have ~20 points
        center: Dartboard center (x, y)
        twenty_angle_rad: Angle to segment 20 center in radians
        
    Returns:
        List of 20 (x, y) points, ordered clockwise from 20/1 boundary
    """
    if len(points) < 15:
        # Not enough points - can't make a reliable polygon
        return []
    
    # Calculate angle from center for each point
    point_angles = []
    for p in points:
        x, y = p[0], p[1]
        angle = math.atan2(y - center[1], x - center[0])
        point_angles.append((angle, x, y, p[2]))  # (angle, x, y, conf)
    
    # Sort by angle
    point_angles.sort(key=lambda pa: pa[0])
    
    # The 20/1 boundary is 9 degrees CCW from the 20 center
    # (each segment is 18 degrees, so boundary is at segment_center - 9°)
    boundary_20_1_rad = twenty_angle_rad - math.radians(9)
    
    # Normalize to [-pi, pi]
    while boundary_20_1_rad > math.pi:
        boundary_20_1_rad -= 2 * math.pi
    while boundary_20_1_rad < -math.pi:
        boundary_20_1_rad += 2 * math.pi
    
    # Find the point closest to the 20/1 boundary angle - this is our starting point
    def angle_diff(a1, a2):
        diff = abs(a1 - a2)
        if diff > math.pi:
            diff = 2 * math.pi - diff
        return diff
    
    start_idx = 0
    min_diff = float('inf')
    for i, pa in enumerate(point_angles):
        diff = angle_diff(pa[0], boundary_20_1_rad)
        if diff < min_diff:
            min_diff = diff
            start_idx = i
    
    # Reorder starting from the 20/1 boundary point
    ordered = point_angles[start_idx:] + point_angles[:start_idx]
    
    # If we have exactly 20 points, we're done
    if len(ordered) == 20:
        return [(p[1], p[2]) for p in ordered]
    
    # If we have more than 20, we need to pick the best 20
    # Use angular spacing - expect points every 18 degrees
    if len(ordered) > 20:
        selected = []
        expected_angles = []
        for i in range(20):
            expected = boundary_20_1_rad + math.radians(i * 18)
            # Normalize
            while expected > math.pi:
                expected -= 2 * math.pi
            while expected < -math.pi:
                expected += 2 * math.pi
            expected_angles.append(expected)
        
        # For each expected angle, find the closest detected point
        used = set()
        for exp_angle in expected_angles:
            best_idx = -1
            best_diff = float('inf')
            for i, pa in enumerate(point_angles):
                if i in used:
                    continue
                diff = angle_diff(pa[0], exp_angle)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i
            if best_idx >= 0:
                used.add(best_idx)
                selected.append((point_angles[best_idx][1], point_angles[best_idx][2]))
        
        return selected
    
    # If we have fewer than 20, interpolate missing points
    # For now, just return what we have with a warning
    print(f"Warning: Only {len(ordered)} points detected, expected 20")
    return [(p[1], p[2]) for p in ordered]


def generate_polygon_calibration_from_yolo(
    cal_points: List[Tuple[float, float, float]],   # Outer double (board edge)
    cal1_points: List[Tuple[float, float, float]],  # Outer triple
    cal2_points: List[Tuple[float, float, float]],  # Inner double
    cal3_points: List[Tuple[float, float, float]],  # Inner triple
    center: Tuple[float, float],
    twenty_angle_rad: float,
    image_width: int = 1280,
    image_height: int = 720
) -> Dict[str, Any]:
    """
    Generate polygon calibration from YOLO-detected points.
    
    Maps YOLO classes to Autodarts polygon format:
    - cal (class 3)  -> double_outers (board edge)
    - cal1 (class 4) -> treble_outers (outer triple)
    - cal2 (class 5) -> double_inners (inner double)
    - cal3 (class 6) -> treble_inners (inner triple)
    
    Returns dict matching PolygonCalibration format.
    """
    double_outers = points_to_ordered_polygon(cal_points, center, twenty_angle_rad)
    treble_outers = points_to_ordered_polygon(cal1_points, center, twenty_angle_rad)
    double_inners = points_to_ordered_polygon(cal2_points, center, twenty_angle_rad)
    treble_inners = points_to_ordered_polygon(cal3_points, center, twenty_angle_rad)
    
    return {
        "bull": center,
        "double_outers": double_outers,
        "double_inners": double_inners,
        "treble_outers": treble_outers,
        "treble_inners": treble_inners,
        "image_width": image_width,
        "image_height": image_height,
        "valid": (
            len(double_outers) >= 18 and 
            len(double_inners) >= 18 and
            len(treble_outers) >= 18 and
            len(treble_inners) >= 18
        )
    }
