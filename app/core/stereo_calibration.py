#!/usr/bin/env python3
"""
Stereo Calibration Module for DartDetect

Provides true 3D triangulation using checkerboard calibration.
Optional upgrade from the default ellipse-based calibration.
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    camera_id: str
    fx: float  # Focal length X
    fy: float  # Focal length Y
    cx: float  # Principal point X
    cy: float  # Principal point Y
    k1: float = 0.0  # Radial distortion
    k2: float = 0.0
    p1: float = 0.0  # Tangential distortion
    p2: float = 0.0
    k3: float = 0.0
    image_width: int = 1920
    image_height: int = 1080
    
    @property
    def camera_matrix(self) -> np.ndarray:
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)
    
    @property
    def dist_coeffs(self) -> np.ndarray:
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float64)


@dataclass 
class CameraExtrinsics:
    """Camera extrinsic parameters (position/rotation relative to dartboard)."""
    camera_id: str
    rvec: List[float]  # Rotation vector (Rodrigues)
    tvec: List[float]  # Translation vector (mm from dartboard center)
    
    @property
    def rotation_matrix(self) -> np.ndarray:
        R, _ = cv2.Rodrigues(np.array(self.rvec))
        return R
    
    @property
    def translation(self) -> np.ndarray:
        return np.array(self.tvec)


@dataclass
class StereoCalibration:
    """Full stereo calibration data for all cameras."""
    board_id: str
    calibration_type: str = "stereo"  # "stereo" vs "ellipse"
    intrinsics: Dict[str, CameraIntrinsics] = None
    extrinsics: Dict[str, CameraExtrinsics] = None
    reprojection_error: float = 0.0
    checkerboard_size: Tuple[int, int] = (9, 6)  # Inner corners
    square_size_mm: float = 25.0
    
    def to_dict(self) -> dict:
        return {
            "board_id": self.board_id,
            "calibration_type": self.calibration_type,
            "intrinsics": {k: asdict(v) for k, v in self.intrinsics.items()} if self.intrinsics else {},
            "extrinsics": {k: asdict(v) for k, v in self.extrinsics.items()} if self.extrinsics else {},
            "reprojection_error": self.reprojection_error,
            "checkerboard_size": list(self.checkerboard_size),
            "square_size_mm": self.square_size_mm
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'StereoCalibration':
        intrinsics = {}
        for k, v in data.get("intrinsics", {}).items():
            intrinsics[k] = CameraIntrinsics(**v)
        
        extrinsics = {}
        for k, v in data.get("extrinsics", {}).items():
            extrinsics[k] = CameraExtrinsics(**v)
        
        return cls(
            board_id=data.get("board_id", "default"),
            calibration_type=data.get("calibration_type", "stereo"),
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            reprojection_error=data.get("reprojection_error", 0.0),
            checkerboard_size=tuple(data.get("checkerboard_size", [9, 6])),
            square_size_mm=data.get("square_size_mm", 25.0)
        )


class StereoCalibrator:
    """
    Handles checkerboard-based stereo camera calibration.
    
    Usage:
    1. Create calibrator with checkerboard parameters
    2. Capture multiple images with checkerboard visible in all cameras
    3. Call calibrate() with the captured images
    4. Save calibration data
    5. Use triangulate() to get 3D positions from pixel coordinates
    """
    
    def __init__(self, checkerboard_size: Tuple[int, int] = (9, 6), 
                 square_size_mm: float = 25.0):
        """
        Args:
            checkerboard_size: Number of inner corners (width, height)
            square_size_mm: Size of each square in mm
        """
        self.checkerboard_size = checkerboard_size
        self.square_size_mm = square_size_mm
        self.calibration: Optional[StereoCalibration] = None
        
        # Prepare object points (checkerboard corners in 3D)
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size_mm
    
    def detect_checkerboard(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect checkerboard corners in image.
        
        Returns:
            Corner points if found, None otherwise
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Find checkerboard corners
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, flags)
        
        if ret:
            # Refine corners to sub-pixel accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            return corners
        
        return None
    
    def draw_checkerboard(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Draw detected checkerboard on image for visualization."""
        vis = image.copy()
        cv2.drawChessboardCorners(vis, self.checkerboard_size, corners, True)
        return vis
    
    def calibrate_single_camera(self, images: List[np.ndarray], 
                                 camera_id: str) -> Optional[CameraIntrinsics]:
        """
        Calibrate a single camera from multiple checkerboard images.
        
        Args:
            images: List of images with checkerboard visible
            camera_id: Camera identifier
            
        Returns:
            CameraIntrinsics if successful
        """
        obj_points = []  # 3D points
        img_points = []  # 2D points
        
        image_size = None
        
        for img in images:
            if image_size is None:
                image_size = (img.shape[1], img.shape[0])
            
            corners = self.detect_checkerboard(img)
            if corners is not None:
                obj_points.append(self.objp)
                img_points.append(corners)
        
        if len(obj_points) < 3:
            logger.error(f"[STEREO] Camera {camera_id}: Need at least 3 valid images, got {len(obj_points)}")
            return None
        
        logger.info(f"[STEREO] Camera {camera_id}: Calibrating with {len(obj_points)} images")
        
        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, image_size, None, None
        )
        
        if not ret:
            logger.error(f"[STEREO] Camera {camera_id}: Calibration failed")
            return None
        
        # Calculate reprojection error
        total_error = 0
        for i in range(len(obj_points)):
            img_points_reproj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(img_points[i], img_points_reproj, cv2.NORM_L2) / len(img_points_reproj)
            total_error += error
        mean_error = total_error / len(obj_points)
        
        logger.info(f"[STEREO] Camera {camera_id}: Reprojection error = {mean_error:.3f} pixels")
        
        return CameraIntrinsics(
            camera_id=camera_id,
            fx=mtx[0, 0],
            fy=mtx[1, 1],
            cx=mtx[0, 2],
            cy=mtx[1, 2],
            k1=dist[0, 0],
            k2=dist[0, 1],
            p1=dist[0, 2],
            p2=dist[0, 3],
            k3=dist[0, 4] if len(dist[0]) > 4 else 0.0,
            image_width=image_size[0],
            image_height=image_size[1]
        )
    
    def calibrate_extrinsics(self, image: np.ndarray, intrinsics: CameraIntrinsics,
                             camera_id: str) -> Optional[CameraExtrinsics]:
        """
        Find camera pose relative to checkerboard (dartboard plane).
        
        The checkerboard should be mounted flat on the dartboard for this step.
        """
        corners = self.detect_checkerboard(image)
        if corners is None:
            logger.error(f"[STEREO] Camera {camera_id}: No checkerboard found for extrinsics")
            return None
        
        # Solve PnP to find camera pose
        ret, rvec, tvec = cv2.solvePnP(
            self.objp, corners,
            intrinsics.camera_matrix, intrinsics.dist_coeffs
        )
        
        if not ret:
            logger.error(f"[STEREO] Camera {camera_id}: solvePnP failed")
            return None
        
        logger.info(f"[STEREO] Camera {camera_id}: Extrinsics found, t={tvec.flatten()}")
        
        return CameraExtrinsics(
            camera_id=camera_id,
            rvec=rvec.flatten().tolist(),
            tvec=tvec.flatten().tolist()
        )
    
    def calibrate(self, camera_images: Dict[str, List[np.ndarray]],
                  board_id: str = "default") -> Optional[StereoCalibration]:
        """
        Full stereo calibration from multiple cameras.
        
        Args:
            camera_images: Dict of camera_id -> list of checkerboard images
            board_id: Board identifier
            
        Returns:
            StereoCalibration if successful
        """
        intrinsics = {}
        extrinsics = {}
        
        # Step 1: Calibrate intrinsics for each camera
        for cam_id, images in camera_images.items():
            intr = self.calibrate_single_camera(images, cam_id)
            if intr is None:
                logger.error(f"[STEREO] Failed to calibrate camera {cam_id}")
                return None
            intrinsics[cam_id] = intr
        
        # Step 2: Find extrinsics using the last image (assumed to have checkerboard on dartboard)
        for cam_id, images in camera_images.items():
            if images:
                extr = self.calibrate_extrinsics(images[-1], intrinsics[cam_id], cam_id)
                if extr is None:
                    logger.error(f"[STEREO] Failed to find extrinsics for camera {cam_id}")
                    return None
                extrinsics[cam_id] = extr
        
        self.calibration = StereoCalibration(
            board_id=board_id,
            calibration_type="stereo",
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            checkerboard_size=self.checkerboard_size,
            square_size_mm=self.square_size_mm
        )
        
        logger.info(f"[STEREO] Calibration complete for {len(intrinsics)} cameras")
        return self.calibration
    
    def pixel_to_ray(self, x_px: float, y_px: float, 
                     camera_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert pixel coordinate to 3D ray in world coordinates.
        
        Returns:
            (origin, direction) - ray origin and direction in dartboard frame
        """
        if self.calibration is None:
            raise ValueError("No calibration loaded")
        
        intr = self.calibration.intrinsics.get(camera_id)
        extr = self.calibration.extrinsics.get(camera_id)
        
        if intr is None or extr is None:
            raise ValueError(f"No calibration for camera {camera_id}")
        
        # Undistort the point
        pts = np.array([[[x_px, y_px]]], dtype=np.float32)
        pts_undist = cv2.undistortPoints(pts, intr.camera_matrix, intr.dist_coeffs)
        x_norm, y_norm = pts_undist[0, 0]
        
        # Ray direction in camera frame (normalized coordinates)
        ray_cam = np.array([x_norm, y_norm, 1.0])
        ray_cam = ray_cam / np.linalg.norm(ray_cam)
        
        # Transform to world frame
        R = extr.rotation_matrix
        t = extr.translation
        
        # Camera position in world frame
        origin = -R.T @ t
        
        # Ray direction in world frame
        direction = R.T @ ray_cam
        
        return origin, direction
    
    def triangulate(self, pixel_coords: Dict[str, Tuple[float, float]]) -> Tuple[float, float, float]:
        """
        Triangulate 3D position from multiple camera observations.
        
        Args:
            pixel_coords: Dict of camera_id -> (x_px, y_px)
            
        Returns:
            (x_mm, y_mm, z_mm) in dartboard coordinates
            z should be ~0 for darts in the board
        """
        if len(pixel_coords) < 2:
            raise ValueError("Need at least 2 cameras for triangulation")
        
        origins = []
        directions = []
        
        for cam_id, (x_px, y_px) in pixel_coords.items():
            origin, direction = self.pixel_to_ray(x_px, y_px, cam_id)
            origins.append(origin)
            directions.append(direction)
        
        # Find point closest to all rays (least squares)
        # Minimize sum of squared distances to each ray
        A = np.zeros((3, 3))
        b = np.zeros(3)
        
        for origin, direction in zip(origins, directions):
            d = direction / np.linalg.norm(direction)
            I = np.eye(3)
            P = I - np.outer(d, d)  # Projection matrix perpendicular to ray
            A += P
            b += P @ origin
        
        # Solve for 3D point
        try:
            point_3d = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            logger.error("[STEREO] Triangulation failed - singular matrix")
            # Fallback: average the ray-plane intersections
            point_3d = np.mean(origins, axis=0)
        
        return float(point_3d[0]), float(point_3d[1]), float(point_3d[2])
    
    def triangulate_to_dartboard(self, pixel_coords: Dict[str, Tuple[float, float]]) -> Tuple[float, float]:
        """
        Triangulate and project to dartboard plane (z=0).
        
        Returns:
            (x_mm, y_mm) on dartboard surface
        """
        x, y, z = self.triangulate(pixel_coords)
        
        # Log if dart is far from board plane
        if abs(z) > 50:
            logger.warning(f"[STEREO] Dart tip {z:.1f}mm from board plane")
        
        return x, y
    
    def save(self, filepath: Path):
        """Save calibration to file."""
        if self.calibration is None:
            raise ValueError("No calibration to save")
        
        with open(filepath, 'w') as f:
            json.dump(self.calibration.to_dict(), f, indent=2)
        
        logger.info(f"[STEREO] Saved calibration to {filepath}")
    
    def load(self, filepath: Path) -> bool:
        """Load calibration from file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.calibration = StereoCalibration.from_dict(data)
            logger.info(f"[STEREO] Loaded calibration from {filepath}")
            return True
        except Exception as e:
            logger.error(f"[STEREO] Failed to load calibration: {e}")
            return False


def generate_checkerboard_pdf(size: Tuple[int, int] = (9, 6), 
                               square_mm: float = 25.0,
                               output_path: str = "checkerboard.png") -> str:
    """
    Generate a checkerboard pattern image for printing.
    
    Args:
        size: Number of inner corners (width, height)
        square_mm: Size of each square in mm
        output_path: Where to save the image
        
    Returns:
        Path to generated image
    """
    # Add 1 to each dimension for squares (inner corners + 1 = squares)
    cols = size[0] + 1
    rows = size[1] + 1
    
    # Create at 300 DPI
    dpi = 300
    mm_per_inch = 25.4
    pixels_per_mm = dpi / mm_per_inch
    
    square_px = int(square_mm * pixels_per_mm)
    
    # Add margin
    margin_px = int(10 * pixels_per_mm)
    
    width = cols * square_px + 2 * margin_px
    height = rows * square_px + 2 * margin_px
    
    # Create white image
    img = np.ones((height, width), dtype=np.uint8) * 255
    
    # Draw black squares
    for row in range(rows):
        for col in range(cols):
            if (row + col) % 2 == 0:
                x1 = margin_px + col * square_px
                y1 = margin_px + row * square_px
                x2 = x1 + square_px
                y2 = y1 + square_px
                img[y1:y2, x1:x2] = 0
    
    cv2.imwrite(output_path, img)
    
    print(f"Generated checkerboard: {cols}x{rows} squares, {square_mm}mm each")
    print(f"Print size: {cols * square_mm}mm x {rows * square_mm}mm")
    print(f"Saved to: {output_path}")
    
    return output_path


# Test
if __name__ == "__main__":
    # Generate a printable checkerboard
    generate_checkerboard_pdf((9, 6), 25.0, "checkerboard_9x6_25mm.png")
