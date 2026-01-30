"""
DartDetect API Routes

Stateless dart detection API with API key authentication.
"""
import os
import time
import uuid
import math
from fastapi import APIRouter, HTTPException, Depends, Header
from typing import List, Optional

from app.models.schemas import (
    CalibrateRequest,
    CalibrateResponse,
    CameraCalibrationResult,
    DetectRequest,
    DetectResponse,
    DetectedTip,
    CameraDetectionResult,
    CalibrationInfo,
    HealthResponse,
)
from app.core.calibration import DartboardCalibrator
from app.core.storage import calibration_store
from app.core.scoring import scoring_system

router = APIRouter()

# Configuration
REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "true").lower() == "true"
API_KEYS = set(os.getenv("API_KEYS", "").split(",")) if os.getenv("API_KEYS") else set()

# Shared calibrator instance
calibrator = DartboardCalibrator()


# === Authentication ===

async def verify_api_key(authorization: Optional[str] = Header(None)) -> str:
    """Verify API key from Authorization header."""
    if not REQUIRE_AUTH:
        return "local"
    
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    
    # Expect "Bearer <key>"
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid Authorization format. Use: Bearer <api_key>")
    
    api_key = parts[1]
    
    if api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return api_key


# === Health ===

@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint (no auth required)."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=calibrator.detector.is_initialized if calibrator.detector else False,
        calibrations_count=len(calibration_store.list_all())
    )


# === Calibration Endpoints ===

@router.post("/v1/calibrate", response_model=CalibrateResponse)
async def calibrate_cameras(
    request: CalibrateRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Calibrate cameras from dartboard images.
    
    - Send images of the dartboard (no darts)
    - Returns overlay images showing detected zones
    - Calibration is stored for future detection requests
    """
    results = []
    
    for camera in request.cameras:
        try:
            result = calibrator.calibrate(
                camera_id=camera.camera_id,
                image_base64=camera.image
            )
            
            # Store successful calibration (keyed by api_key + camera_id)
            if result.success:
                storage_key = f"{api_key}:{camera.camera_id}"
                calibration_store.save(storage_key, result.calibration_data)
            
            results.append(CameraCalibrationResult(
                camera_id=camera.camera_id,
                success=result.success,
                quality=result.quality,
                overlay_image=result.overlay_image,
                segment_at_top=result.segment_at_top,
                error=result.error
            ))
            
        except Exception as e:
            results.append(CameraCalibrationResult(
                camera_id=camera.camera_id,
                success=False,
                error=str(e)
            ))
    
    return CalibrateResponse(results=results)


@router.get("/v1/calibrations", response_model=List[CalibrationInfo])
async def list_calibrations(api_key: str = Depends(verify_api_key)):
    """List calibrations for this API key."""
    all_cals = calibration_store.list_all()
    # Filter to this API key's calibrations
    prefix = f"{api_key}:"
    return [
        CalibrationInfo(
            camera_id=c.camera_id.replace(prefix, ""),
            created_at=c.created_at,
            quality=c.quality,
            segment_at_top=c.segment_at_top
        )
        for c in all_cals
        if c.camera_id.startswith(prefix)
    ]


@router.delete("/v1/calibrations/{camera_id}")
async def delete_calibration(
    camera_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Delete a camera's calibration."""
    storage_key = f"{api_key}:{camera_id}"
    if calibration_store.delete(storage_key):
        return {"message": f"Calibration for '{camera_id}' deleted"}
    raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found")


# === Detection Endpoint (Stateless) ===

@router.post("/v1/detect", response_model=DetectResponse)
async def detect_tips(
    request: DetectRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Detect dart tips in images.
    
    - Send images from calibrated cameras
    - Returns ALL detected dart tips with positions and scores
    - Stateless - no memory of previous requests
    - Caller is responsible for tracking which darts are new
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())[:12]
    
    all_tips = []  # Raw tips from each camera
    camera_results = []
    
    for cam in request.cameras:
        # Get calibration for this camera
        storage_key = f"{api_key}:{cam.camera_id}"
        calibration_data = calibration_store.get(storage_key)
        
        if calibration_data is None:
            camera_results.append(CameraDetectionResult(
                camera_id=cam.camera_id,
                tips_detected=0,
                error=f"Camera not calibrated. Call /v1/calibrate first."
            ))
            continue
        
        try:
            # Detect tips in this image
            tips = calibrator.detect_tips(
                camera_id=cam.camera_id,
                image_base64=cam.image,
                calibration_data=calibration_data
            )
            
            # Add camera_id to each tip
            for tip in tips:
                tip['camera_id'] = cam.camera_id
                all_tips.append(tip)
            
            camera_results.append(CameraDetectionResult(
                camera_id=cam.camera_id,
                tips_detected=len(tips)
            ))
            
        except Exception as e:
            camera_results.append(CameraDetectionResult(
                camera_id=cam.camera_id,
                tips_detected=0,
                error=str(e)
            ))
    
    # Cluster tips from multiple cameras (same dart seen by different cameras)
    clustered_tips = cluster_tips(all_tips)
    
    # Calculate scores for each clustered tip
    detected_tips = []
    for cluster in clustered_tips:
        # Average position from all cameras that saw this tip
        avg_x = sum(t['x_mm'] for t in cluster) / len(cluster)
        avg_y = sum(t['y_mm'] for t in cluster) / len(cluster)
        avg_conf = sum(t['confidence'] for t in cluster) / len(cluster)
        cameras_seen = list(set(t['camera_id'] for t in cluster))
        
        # Calculate score
        score_info = scoring_system.score_from_dartboard_coords(avg_x, avg_y)
        
        detected_tips.append(DetectedTip(
            x_mm=round(avg_x, 2),
            y_mm=round(avg_y, 2),
            segment=score_info.get('segment', 0),
            multiplier=score_info.get('multiplier', 1),
            zone=score_info.get('zone', 'miss'),
            score=score_info.get('score', 0),
            confidence=round(avg_conf, 3),
            cameras_seen=cameras_seen
        ))
    
    processing_ms = int((time.time() - start_time) * 1000)
    
    return DetectResponse(
        request_id=request_id,
        processing_ms=processing_ms,
        tips=detected_tips,
        camera_results=camera_results
    )


def cluster_tips(
    tips: List[dict],
    cluster_threshold_mm: float = 20.0
) -> List[List[dict]]:
    """
    Cluster nearby tips (same dart seen by multiple cameras).
    
    Simple greedy clustering.
    """
    if not tips:
        return []
    
    clusters = []
    used = set()
    
    for i, tip in enumerate(tips):
        if i in used:
            continue
        
        cluster = [tip]
        used.add(i)
        
        for j, other in enumerate(tips):
            if j in used:
                continue
            
            # Check distance
            dist = math.sqrt(
                (tip['x_mm'] - other['x_mm'])**2 +
                (tip['y_mm'] - other['y_mm'])**2
            )
            
            if dist < cluster_threshold_mm:
                cluster.append(other)
                used.add(j)
        
        clusters.append(cluster)
    
    return clusters


# === Legacy routes for backwards compatibility ===

@router.post("/api/calibrate", response_model=CalibrateResponse, include_in_schema=False)
async def legacy_calibrate(request: CalibrateRequest):
    """Legacy endpoint - no auth required."""
    # Use 'local' as the API key for legacy requests
    results = []
    for camera in request.cameras:
        try:
            result = calibrator.calibrate(
                camera_id=camera.camera_id,
                image_base64=camera.image
            )
            if result.success:
                calibration_store.save(camera.camera_id, result.calibration_data)
            results.append(CameraCalibrationResult(
                camera_id=camera.camera_id,
                success=result.success,
                quality=result.quality,
                overlay_image=result.overlay_image,
                segment_at_top=result.segment_at_top,
                error=result.error
            ))
        except Exception as e:
            results.append(CameraCalibrationResult(
                camera_id=camera.camera_id,
                success=False,
                error=str(e)
            ))
    return CalibrateResponse(results=results)


@router.get("/api/calibrations", include_in_schema=False)
async def legacy_list_calibrations():
    """Legacy endpoint."""
    return calibration_store.list_all()
