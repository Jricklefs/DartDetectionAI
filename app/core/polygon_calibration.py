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
    
    # Fallback: use angle to determine segment for inner single
    angle = math.atan2(y - bull[1], x - bull[0])
    angle_deg = math.degrees(angle)
    if angle_deg < 0:
        angle_deg += 360
    
    # Each segment is 18 degrees, starting at 20
    # 20 is at the top (-90 degrees in image coords, or 270 degrees)
    # Adjust for segment boundaries
    adjusted_angle = (angle_deg + 90 + 9) % 360  # +9 to center on segment
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
    
    # Check if outside double ring (miss)
    if calibration.double_outers:
        double_outer_dists = [math.sqrt((p[0] - bull[0])**2 + (p[1] - bull[1])**2)
                              for p in calibration.double_outers]
        avg_double_outer = sum(double_outer_dists) / len(double_outer_dists)
        
        if dist_from_bull > avg_double_outer:
            return {
                "segment": 0,
                "zone": "miss",
                "score": 0,
                "multiplier": 0,
                "ring": "miss"
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
