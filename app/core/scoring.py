"""
Scoring module for dart detection.

Calculates scores from detected dart positions using calibration data.
"""
import math
from typing import Dict, Any, Tuple, Optional

from app.core.geometry import (
    DARTBOARD_SEGMENTS,
    BULL_RADIUS_MM,
    OUTER_BULL_RADIUS_MM,
    TRIPLE_INNER_RADIUS_MM,
    TRIPLE_OUTER_RADIUS_MM,
    DOUBLE_INNER_RADIUS_MM,
    DOUBLE_OUTER_RADIUS_MM,
    calculate_score
)


class ScoringSystem:
    """
    Calculate dart scores from positions in dartboard coordinates.
    """
    
    def __init__(self):
        pass
    
    def score_from_dartboard_coords(
        self, 
        x_mm: float, 
        y_mm: float,
        rotation_offset: float = 0.0
    ) -> Dict[str, Any]:
        """
        Calculate score from coordinates in dartboard space.
        
        Args:
            x_mm: X coordinate in mm (0 = center)
            y_mm: Y coordinate in mm (0 = center)
            rotation_offset: Board rotation offset in radians
            
        Returns:
            Score dictionary
        """
        # Calculate polar coordinates
        distance = math.sqrt(x_mm**2 + y_mm**2)
        angle = math.atan2(y_mm, x_mm)
        
        return calculate_score(distance, angle, rotation_offset)
    
    def score_from_pixel_coords(
        self,
        x_px: float,
        y_px: float,
        center_px: Tuple[float, float],
        pixels_per_mm: float,
        rotation_offset: float = 0.0
    ) -> Dict[str, Any]:
        """
        Calculate score from pixel coordinates.
        
        Args:
            x_px: X pixel coordinate
            y_px: Y pixel coordinate  
            center_px: (x, y) pixel coordinates of dartboard center
            pixels_per_mm: Scale factor
            rotation_offset: Board rotation offset in radians
            
        Returns:
            Score dictionary
        """
        # Convert to mm relative to center
        x_mm = (x_px - center_px[0]) / pixels_per_mm
        y_mm = (y_px - center_px[1]) / pixels_per_mm
        
        return self.score_from_dartboard_coords(x_mm, y_mm, rotation_offset)


# Global instance
scoring_system = ScoringSystem()
