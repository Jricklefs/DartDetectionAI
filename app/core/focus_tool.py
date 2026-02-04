"""
Camera focus measurement tool using Siemens star metrics.
Helps users achieve optimal camera focus before calibration.
"""
import cv2
import numpy as np
from typing import Dict, Tuple, Optional


def laplacian_variance(gray: np.ndarray) -> float:
    """Calculate Laplacian variance (edge sharpness)."""
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    return float(lap.var())


def tenengrad(gray: np.ndarray) -> float:
    """Calculate Tenengrad (gradient magnitude)."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    t = gx * gx + gy * gy
    return float(t.mean())


def brenner(gray: np.ndarray) -> float:
    """Calculate Brenner focus measure."""
    diff = gray[:, 2:].astype(np.float64) - gray[:, :-2].astype(np.float64)
    return float((diff * diff).mean())


def siemens_center_contrast_radius(gray: np.ndarray, center: Tuple[int, int], 
                                    max_check_radius: int, min_radius: int = 80) -> float:
    """
    Siemens-star-inspired metric using polar unwrap.
    Returns the radius where spokes become resolvable.
    Smaller radius = better focus (sharper lines visible closer to center).
    """
    h, w = gray.shape[:2]
    cx, cy = center
    
    # Calculate max radius that fits in image
    max_r = int(min(max_check_radius, 
                    min(cx, w - 1 - cx, cy, h - 1 - cy)))
    
    if max_r <= min_radius + 5:
        return float('nan')
    
    # Polar transform - unwrap image around center
    flags = cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS
    polar = cv2.warpPolar(gray, (360, max_r), (cx, cy), max_r, flags)
    polar = polar.astype(np.float32)
    
    # Normalize each row (radius level)
    mean = polar.mean(axis=1, keepdims=True)
    std = polar.std(axis=1, keepdims=True) + 1e-6
    polar = (polar - mean) / std
    
    # Calculate angular standard deviation at each radius
    ang_std = polar.std(axis=1)
    
    # Find baseline in outer region (where lines should be visible)
    band_start = max(int(0.6 * max_r), min_radius + 10)
    if band_start < len(ang_std):
        baseline = np.median(ang_std[band_start:])
    else:
        baseline = np.median(ang_std)
    
    # Threshold for detecting resolved spokes
    thr = max(0.35 * baseline, 0.08)
    
    # Find smallest radius where spokes are resolved
    for r in range(min_radius, max_r):
        if ang_std[r] > thr:
            return float(r)
    
    return float(max_r)


def normalize_score(x: float, lo: float, hi: float, clip: bool = True) -> float:
    """Normalize value to 0-1 range."""
    if hi == lo:
        return 0.0
    s = (x - lo) / (hi - lo)
    if clip:
        s = max(0.0, min(1.0, s))
    return s


def calculate_focus_score(frame: np.ndarray, center: Optional[Tuple[int, int]] = None,
                          roi_size: int = 300) -> Dict:
    """
    Calculate comprehensive focus score for a frame.
    
    Args:
        frame: BGR image
        center: Center point for Siemens metric (defaults to image center)
        roi_size: Size of ROI for classic metrics
        
    Returns:
        Dictionary with metrics and combined focus score (0-100)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    
    # Default to image center
    if center is None:
        center = (w // 2, h // 2)
    
    cx, cy = center
    
    # Extract ROI around center for classic metrics
    half = roi_size // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)
    roi = gray[y1:y2, x1:x2]
    
    # Calculate classic focus metrics on ROI
    lap_var = laplacian_variance(roi)
    tgrad = tenengrad(roi)
    bren = brenner(roi)
    
    # Calculate Siemens star metric (smaller = better)
    max_radius = min(h, w) // 3
    siemens_r = siemens_center_contrast_radius(gray, center, max_radius, min_radius=20)
    
    # Normalize metrics to 0-1 scale
    # Thresholds tuned for 640x480 dartboard camera images
    # Poor focus: lap~5000, tgrad~10000, bren~200, siemens~120
    # Good focus: lap~25000, tgrad~40000, bren~1500, siemens~40
    lap_norm = normalize_score(lap_var, 5000, 30000)
    tgrad_norm = normalize_score(tgrad, 10000, 50000)
    bren_norm = normalize_score(bren, 200, 2000)
    
    # Siemens: lower radius = better focus, so invert
    if np.isnan(siemens_r):
        siemens_norm = 0.0
    else:
        # For printed Siemens star: range 20 (excellent) to 100 (poor)
        siemens_norm = normalize_score(siemens_r, 100, 20)
    
    # Combined score (weighted average)
    combined = (
        0.25 * lap_norm +
        0.25 * tgrad_norm +
        0.20 * bren_norm +
        0.30 * siemens_norm
    )
    
    # Convert to 0-100 scale
    score = int(combined * 100)
    
    # Determine quality level
    if score >= 80:
        quality = "excellent"
    elif score >= 60:
        quality = "good"
    elif score >= 40:
        quality = "fair"
    else:
        quality = "poor"
    
    return {
        "score": score,
        "quality": quality,
        "metrics": {
            "laplacian_variance": round(lap_var, 2),
            "tenengrad": round(tgrad, 2),
            "brenner": round(bren, 2),
            "siemens_radius": round(siemens_r, 1) if not np.isnan(siemens_r) else None,
        },
        "normalized": {
            "laplacian": round(lap_norm, 3),
            "tenengrad": round(tgrad_norm, 3),
            "brenner": round(bren_norm, 3),
            "siemens": round(siemens_norm, 3),
        },
        "center": center,
    }


__all__ = ['calculate_focus_score', 'laplacian_variance', 'tenengrad', 'brenner', 
           'siemens_center_contrast_radius']
