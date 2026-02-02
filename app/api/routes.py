"""
DartDetect API Routes - Stateless Detection

Calibration + Detection endpoints only. No motion detection, no cameras.
"""
import os
import time
import uuid
import math
import base64
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
REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "false").lower() == "true"
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
        version="3.0.0",
        models_loaded=calibrator.detector.is_initialized if calibrator.detector else False,
        calibrations_count=len(calibration_store.list_all()),
        motion_detection_running=False  # Stateless - no motion detection
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


@router.post("/v1/calibrations/{camera_id}/mark20")
async def mark_segment_20(
    camera_id: str,
    x: float,
    y: float,
    api_key: str = Depends(verify_api_key)
):
    """
    Mark where segment 20 is located on the calibration overlay.
    """
    storage_key = f"{api_key}:{camera_id}"
    calibration_data = calibration_store.get(storage_key)
    
    if calibration_data is None:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not calibrated")
    
    center = calibration_data.get('center')
    segment_angles = calibration_data.get('segment_angles', [])
    image_size = calibration_data.get('image_size', (640, 480))
    
    if not center or len(segment_angles) < 20:
        raise HTTPException(status_code=400, detail="Calibration data incomplete - please recalibrate")
    
    px = x * image_size[0]
    py = y * image_size[1]
    
    dx = px - center[0]
    dy = py - center[1]
    click_angle = math.atan2(dy, dx)
    
    if click_angle < 0:
        click_angle += 2 * math.pi
    
    normalized_angles = sorted([a if a >= 0 else a + 2 * math.pi for a in segment_angles])
    
    segment_20_index = 0
    for i, angle in enumerate(normalized_angles):
        next_angle = normalized_angles[(i + 1) % len(normalized_angles)]
        if next_angle < angle:
            next_angle += 2 * math.pi
        
        check_angle = click_angle
        if check_angle < angle:
            check_angle += 2 * math.pi
        
        if angle <= check_angle < next_angle:
            segment_20_index = i
            break
    
    calibration_data['segment_20_index'] = segment_20_index
    calibration_store.save(storage_key, calibration_data)
    
    twenty_angle_rad = normalized_angles[segment_20_index]
    twenty_angle_deg = math.degrees(twenty_angle_rad)
    
    return {
        "camera_id": camera_id,
        "segment_20_index": segment_20_index,
        "twenty_angle": twenty_angle_deg,
        "message": "Segment 20 marked successfully."
    }


# === Detection Endpoint (Stateless) ===

@router.post("/v1/detect", response_model=DetectResponse)
async def detect_tips(
    request: DetectRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Detect dart tips in images (stateless).
    
    Send images from cameras and get dart positions/scores back.
    Requires cameras to be calibrated first.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())[:12]
    
    all_tips = []
    camera_results = []
    
    for cam in request.cameras:
        storage_key = f"{api_key}:{cam.camera_id}"
        calibration_data = calibration_store.get(storage_key)
        
        # Also try without api_key prefix (legacy)
        if calibration_data is None:
            calibration_data = calibration_store.get(cam.camera_id)
        
        if calibration_data is None:
            camera_results.append(CameraDetectionResult(
                camera_id=cam.camera_id,
                tips_detected=0,
                error=f"Camera not calibrated. Call /v1/calibrate first."
            ))
            continue
        
        try:
            tips = calibrator.detect_tips(
                camera_id=cam.camera_id,
                image_base64=cam.image,
                calibration_data=calibration_data
            )
            
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
    
    clustered_tips = cluster_tips(all_tips)
    
    detected_tips = []
    for cluster in clustered_tips:
        avg_x = sum(t['x_mm'] for t in cluster) / len(cluster)
        avg_y = sum(t['y_mm'] for t in cluster) / len(cluster)
        avg_conf = sum(t['confidence'] for t in cluster) / len(cluster)
        cameras_seen = list(set(t['camera_id'] for t in cluster))
        
        rotation_rad = (request.rotation_offset_degrees or 0.0) * (3.14159265 / 180.0)
        score_info = scoring_system.score_from_dartboard_coords(avg_x, avg_y, rotation_rad)
        
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
    """Cluster nearby tips (same dart seen by multiple cameras)."""
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


@router.post("/api/calibrations/{camera_id}/mark20", include_in_schema=False)
async def legacy_mark_segment_20(camera_id: str, x: float, y: float):
    """Legacy Mark 20 endpoint."""
    calibration_data = calibration_store.get(camera_id)
    
    if calibration_data is None:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not calibrated")
    
    center = calibration_data.get('center')
    segment_angles = calibration_data.get('segment_angles', [])
    image_size = calibration_data.get('image_size', (640, 480))
    
    if not center or len(segment_angles) < 20:
        raise HTTPException(status_code=400, detail="Calibration data incomplete")
    
    px = x * image_size[0]
    py = y * image_size[1]
    
    dx = px - center[0]
    dy = py - center[1]
    click_angle = math.atan2(dy, dx)
    if click_angle < 0:
        click_angle += 2 * math.pi
    
    normalized_angles = sorted([a if a >= 0 else a + 2 * math.pi for a in segment_angles])
    
    segment_20_index = 0
    for i, angle in enumerate(normalized_angles):
        next_angle = normalized_angles[(i + 1) % len(normalized_angles)]
        if next_angle < angle:
            next_angle += 2 * math.pi
        check_angle = click_angle
        if check_angle < angle:
            check_angle += 2 * math.pi
        if angle <= check_angle < next_angle:
            segment_20_index = i
            break
    
    calibration_data['segment_20_index'] = segment_20_index
    calibration_store.save(camera_id, calibration_data)
    
    twenty_angle_rad = normalized_angles[segment_20_index]
    twenty_angle_deg = math.degrees(twenty_angle_rad)
    
    return {
        "camera_id": camera_id,
        "segment_20_index": segment_20_index,
        "twenty_angle": twenty_angle_deg
    }


@router.get("/api/calibrations", include_in_schema=False)
async def legacy_list_calibrations():
    """Legacy endpoint."""
    return calibration_store.list_all()


# === Rebase endpoint (for DartSensor compatibility) ===
# In stateless mode, this just returns success - actual rebasing 
# happens in DartSensor which maintains its own baseline frames

@router.post("/rebase")
async def rebase_noop():
    """
    Rebase endpoint for compatibility.
    
    In stateless mode, DartDetect doesn't maintain baselines.
    DartSensor handles its own baseline management.
    This endpoint exists for API compatibility.
    """
    return {
        "message": "Rebase acknowledged (stateless mode - handled by DartSensor)",
        "cameras": {},
        "known_darts_count": 0,
        "note": "DartDetect v3.0 is stateless. Use DartSensor for motion detection."
    }
