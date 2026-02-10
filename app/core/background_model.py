"""
Background Modeling for Dart Detection

Instead of simple frame differencing (current - before), we use an
accumulated running average that adapts over time. This provides a 
more stable baseline and better handles lighting changes.

Key concepts:
1. Running average: bg = alpha * current + (1 - alpha) * bg
2. Motion mask: |current - bg| > threshold
3. Adaptive threshold: per-region thresholds based on local variance
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
import time


@dataclass
class BackgroundModel:
    """Per-camera background model with running average."""
    
    # Running average (float32 for precision)
    background: Optional[np.ndarray] = None
    
    # Variance estimate for adaptive thresholding
    variance: Optional[np.ndarray] = None
    
    # Last update time
    last_update: float = 0.0
    
    # Number of frames accumulated
    frame_count: int = 0
    
    # Config
    alpha: float = 0.02  # Learning rate: higher = faster adaptation
    var_alpha: float = 0.01  # Variance learning rate
    min_threshold: float = 15.0  # Minimum difference to detect motion
    var_scale: float = 2.5  # Threshold = min_threshold + var_scale * sqrt(variance)


class BackgroundModelManager:
    """Manages background models for multiple cameras."""
    
    def __init__(self, alpha: float = 0.02, var_alpha: float = 0.01,
                 min_threshold: float = 15.0, var_scale: float = 2.5):
        self.models: Dict[str, BackgroundModel] = {}
        self.alpha = alpha
        self.var_alpha = var_alpha
        self.min_threshold = min_threshold
        self.var_scale = var_scale
    
    def get_or_create(self, camera_id: str) -> BackgroundModel:
        """Get or create background model for a camera."""
        if camera_id not in self.models:
            self.models[camera_id] = BackgroundModel(
                alpha=self.alpha,
                var_alpha=self.var_alpha,
                min_threshold=self.min_threshold,
                var_scale=self.var_scale
            )
        return self.models[camera_id]
    
    def reset(self, camera_id: str = None):
        """Reset background model(s)."""
        if camera_id:
            if camera_id in self.models:
                self.models[camera_id] = BackgroundModel(
                    alpha=self.alpha,
                    var_alpha=self.var_alpha,
                    min_threshold=self.min_threshold,
                    var_scale=self.var_scale
                )
        else:
            self.models.clear()
    
    def initialize(self, camera_id: str, frame: np.ndarray):
        """Initialize background model with a frame."""
        model = self.get_or_create(camera_id)
        
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Initialize with current frame
        model.background = gray.astype(np.float32)
        model.variance = np.ones_like(gray, dtype=np.float32) * 100.0  # Initial variance
        model.frame_count = 1
        model.last_update = time.time()
    
    def update(self, camera_id: str, frame: np.ndarray, 
               mask: np.ndarray = None) -> np.ndarray:
        """
        Update background model with new frame.
        
        Args:
            camera_id: Camera identifier
            frame: Current frame (BGR or grayscale)
            mask: Optional mask where 0 = update background, 255 = foreground (don't update)
        
        Returns:
            Motion mask (foreground regions)
        """
        model = self.get_or_create(camera_id)
        
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = frame.astype(np.float32)
        
        # Initialize if first frame
        if model.background is None:
            self.initialize(camera_id, frame)
            return np.zeros_like(gray, dtype=np.uint8)
        
        # Compute difference from background
        diff = cv2.absdiff(gray, model.background)
        
        # Adaptive threshold based on local variance
        # threshold = min_threshold + var_scale * sqrt(variance)
        adaptive_thresh = model.min_threshold + model.var_scale * np.sqrt(model.variance)
        adaptive_thresh = np.clip(adaptive_thresh, model.min_threshold, 100.0)
        
        # Motion mask
        motion_mask = (diff > adaptive_thresh).astype(np.uint8) * 255
        
        # Update background where there's no motion (and not masked)
        update_mask = (motion_mask == 0).astype(np.float32)
        if mask is not None:
            update_mask = update_mask * (mask == 0).astype(np.float32)
        
        # Running average update: bg = alpha * current + (1-alpha) * bg
        model.background = model.alpha * gray * update_mask + \
                          (1 - model.alpha * update_mask) * model.background
        
        # Update variance estimate
        sq_diff = (diff ** 2)
        model.variance = model.var_alpha * sq_diff * update_mask + \
                        (1 - model.var_alpha * update_mask) * model.variance
        
        model.frame_count += 1
        model.last_update = time.time()
        
        return motion_mask
    
    def get_motion_mask(self, camera_id: str, frame: np.ndarray,
                        update: bool = False) -> Tuple[np.ndarray, float]:
        """
        Get motion mask for a frame without updating the model.
        
        Args:
            camera_id: Camera identifier
            frame: Current frame
            update: If True, also update the background model
        
        Returns:
            (motion_mask, diff_percentage)
        """
        model = self.get_or_create(camera_id)
        
        if model.background is None:
            # Not initialized - return empty mask
            h, w = frame.shape[:2]
            return np.zeros((h, w), dtype=np.uint8), 0.0
        
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = frame.astype(np.float32)
        
        # Compute difference from background
        diff = cv2.absdiff(gray, model.background)
        
        # Adaptive threshold
        adaptive_thresh = model.min_threshold + model.var_scale * np.sqrt(model.variance)
        adaptive_thresh = np.clip(adaptive_thresh, model.min_threshold, 100.0)
        
        # Motion mask
        motion_mask = (diff > adaptive_thresh).astype(np.uint8) * 255
        
        # Calculate percentage of pixels that are motion
        total_pixels = motion_mask.shape[0] * motion_mask.shape[1]
        motion_pixels = np.sum(motion_mask > 0)
        diff_pct = (motion_pixels / total_pixels) * 100.0
        
        if update:
            self.update(camera_id, frame, motion_mask)
        
        return motion_mask, diff_pct
    
    def get_background(self, camera_id: str) -> Optional[np.ndarray]:
        """Get current background image for a camera."""
        if camera_id in self.models and self.models[camera_id].background is not None:
            return self.models[camera_id].background.astype(np.uint8)
        return None
    
    def get_adaptive_diff(self, camera_id: str, current_frame: np.ndarray,
                          previous_frame: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get an improved diff using background model for adaptive thresholding.
        
        If previous_frame is provided, diffs against it (like current behavior).
        Otherwise, diffs against the background model.
        
        Returns:
            (diff_image, motion_mask) - both suitable for skeleton detection
        """
        model = self.get_or_create(camera_id)
        
        # Convert to grayscale
        if len(current_frame.shape) == 3:
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            current_gray = current_frame.astype(np.float32)
        
        # Use previous_frame if provided, otherwise use background
        if previous_frame is not None:
            if len(previous_frame.shape) == 3:
                prev_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            else:
                prev_gray = previous_frame.astype(np.float32)
            base = prev_gray
        elif model.background is not None:
            base = model.background
        else:
            # No reference - return zeros
            h, w = current_gray.shape
            return np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)
        
        # Compute diff
        diff = cv2.absdiff(current_gray, base)
        diff_uint8 = np.clip(diff, 0, 255).astype(np.uint8)
        
        # Adaptive threshold if we have variance data
        if model.variance is not None:
            # Per-pixel threshold based on local variance
            adaptive_thresh = model.min_threshold + model.var_scale * np.sqrt(model.variance)
            adaptive_thresh = np.clip(adaptive_thresh, model.min_threshold, 100.0)
            motion_mask = (diff > adaptive_thresh).astype(np.uint8) * 255
        else:
            # Fall back to fixed threshold
            _, motion_mask = cv2.threshold(diff_uint8, 20, 255, cv2.THRESH_BINARY)
        
        return diff_uint8, motion_mask


# Global instance
_bg_manager: Optional[BackgroundModelManager] = None


def get_background_manager() -> BackgroundModelManager:
    """Get the global background model manager."""
    global _bg_manager
    if _bg_manager is None:
        _bg_manager = BackgroundModelManager()
    return _bg_manager


def reset_background_models():
    """Reset all background models."""
    global _bg_manager
    if _bg_manager is not None:
        _bg_manager.reset()
