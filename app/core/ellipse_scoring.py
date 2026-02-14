"""
Ellipse-based dart scoring using YOLO ring calibration data.
Replaces polygon scoring with accurate curved ring boundaries.
"""
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

SEGMENT_ORDER = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]


def ellipse_radius_at_angle(ellipse: list, angle_rad: float) -> float:
    """
    Calculate the radius of an ellipse at a given angle from its center.
    ellipse: [[cx, cy], [width, height], rotation_deg]  (OpenCV format)
    angle_rad: angle from center in image coordinates
    Returns: distance from ellipse center to edge at that angle.
    """
    (cx, cy), (w, h), rot_deg = ellipse[0], ellipse[1], ellipse[2]
    a = w / 2.0  # semi-major
    b = h / 2.0  # semi-minor
    rot_rad = np.radians(rot_deg)

    # Angle relative to ellipse axes
    theta = angle_rad - rot_rad

    # Polar form of ellipse: r = ab / sqrt((b*cos(t))^2 + (a*sin(t))^2)
    denom = np.sqrt((b * np.cos(theta)) ** 2 + (a * np.sin(theta)) ** 2)
    if denom < 1e-6:
        return 0.0
    return (a * b) / denom


def score_from_ellipse_calibration(
    tip_x: float, tip_y: float,
    calibration: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Score a dart tip position using ellipse calibration.
    
    Args:
        tip_x, tip_y: pixel coordinates of the dart tip
        calibration: dict with center, segment_angles, seg20_idx, and ellipse data
    
    Returns:
        dict with segment, multiplier, zone, score, confidence
    """
    center = calibration["center"]
    cx, cy = center[0], center[1]
    segment_angles = calibration["segment_angles"]  # radians, 20 boundary angles
    seg20_idx = calibration["segment_20_index"]

    # Vector from center to tip
    dx = tip_x - cx
    dy = tip_y - cy
    dist = np.sqrt(dx * dx + dy * dy)
    angle = np.arctan2(dy, dx)  # radians, same coordinate system as segment_angles

    # --- Determine segment ---
    # segment_angles are boundaries; segment i is between boundary i and boundary (i+1)
    segment_idx = None
    for i in range(20):
        a1 = segment_angles[i]
        a2 = segment_angles[(i + 1) % 20]

        # Handle wraparound at ±pi
        if a2 < a1:
            # Boundary crosses the -pi/+pi line
            if angle >= a1 or angle < a2:
                segment_idx = i
                break
        else:
            if a1 <= angle < a2:
                segment_idx = i
                break

    if segment_idx is None:
        # Fallback: find closest boundary midpoint
        best_i = 0
        best_diff = float('inf')
        for i in range(20):
            a1 = segment_angles[i]
            a2 = segment_angles[(i + 1) % 20]
            mid = (a1 + a2) / 2
            diff = abs(np.arctan2(np.sin(angle - mid), np.cos(angle - mid)))
            if diff < best_diff:
                best_diff = diff
                best_i = i
        segment_idx = best_i

    segment_number = SEGMENT_ORDER[(segment_idx - seg20_idx) % 20]

    # --- Determine zone using ellipse radii at this angle ---
    zones = []  # (max_radius, zone_name, multiplier)

    # Build zone boundaries from inside out
    bullseye_ell = calibration.get("bullseye_ellipse")
    bull_ell = calibration.get("bull_ellipse")
    inner_triple_ell = calibration.get("inner_triple_ellipse")
    outer_triple_ell = calibration.get("outer_triple_ellipse")
    inner_double_ell = calibration.get("inner_double_ellipse")
    outer_double_ell = calibration.get("outer_double_ellipse")

    if bullseye_ell:
        r = ellipse_radius_at_angle(bullseye_ell, angle)
        zones.append((r, "bullseye", 2, 25))
    if bull_ell:
        r = ellipse_radius_at_angle(bull_ell, angle)
        zones.append((r, "bull", 1, 25))
    if inner_triple_ell:
        r = ellipse_radius_at_angle(inner_triple_ell, angle)
        zones.append((r, "single_inner", 1, segment_number))
    if outer_triple_ell:
        r = ellipse_radius_at_angle(outer_triple_ell, angle)
        zones.append((r, "triple", 3, segment_number))
    if inner_double_ell:
        r = ellipse_radius_at_angle(inner_double_ell, angle)
        zones.append((r, "single_outer", 1, segment_number))
    if outer_double_ell:
        r = ellipse_radius_at_angle(outer_double_ell, angle)
        zones.append((r, "double", 2, segment_number))

    # Sort by radius (should already be in order, but be safe)
    zones.sort(key=lambda z: z[0])

    # Find which zone the tip falls in
    zone_name = "miss"
    multiplier = 0
    score = 0

    for max_r, zname, mult, seg_score in zones:
        if dist <= max_r:
            zone_name = zname
            multiplier = mult
            score = mult * seg_score
            break

    # Confidence: higher when further from zone boundary
    confidence = 0.8  # default
    for i, (max_r, zname, mult, seg_score) in enumerate(zones):
        if dist <= max_r:
            prev_r = zones[i - 1][0] if i > 0 else 0
            zone_width = max_r - prev_r
            if zone_width > 0:
                dist_from_boundary = min(dist - prev_r, max_r - dist)
                confidence = min(0.95, 0.5 + 0.45 * (dist_from_boundary / (zone_width / 2)))
            break

    return {
        "segment": segment_number,
        "multiplier": multiplier,
        "zone": zone_name,
        "score": score,
        "confidence": round(confidence, 3),
        "angle_deg": round(np.degrees(angle), 1),
        "dist_px": round(dist, 1),
    }


# --- Convenience: score from DB calibration JSON ---
def score_tip_pixel(tip_x: float, tip_y: float, calibration_json: Dict) -> Dict[str, Any]:
    """Score a tip using calibration data as stored in the Calibrations DB table."""
    return score_from_ellipse_calibration(tip_x, tip_y, calibration_json)


if __name__ == "__main__":
    # Quick test with cam0 calibration from benchmark
    import json
    with open(r"C:\Users\clawd\DartBenchmark\default\1a3f8338-f554-4e75-8176-0f7079246bd4\round_1_Player_1\dart_1\metadata.json") as f:
        meta = json.load(f)

    for cam_id in ["cam0", "cam1", "cam2"]:
        cal = meta["calibrations"][cam_id]
        tip = meta["pipeline"][cam_id]["selected_tip"]
        result = score_from_ellipse_calibration(tip["x_px"], tip["y_px"], cal)
        print(f"{cam_id}: tip=({tip['x_px']:.0f},{tip['y_px']:.0f}) -> {result}")
