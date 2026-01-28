"""
API Routes for DartDetectionAI
"""
from fastapi import APIRouter, HTTPException
from typing import List

from app.models.schemas import (
    CalibrateRequest,
    CalibrateResponse,
    CameraCalibrationResult,
    DetectRequest,
    DetectResponse,
    CalibrationInfo
)
from app.core.calibration import DartboardCalibrator
from app.core.storage import calibration_store

router = APIRouter()

# Shared calibrator instance
calibrator = DartboardCalibrator()


@router.post("/calibrate", response_model=CalibrateResponse)
async def calibrate_cameras(request: CalibrateRequest):
    """
    Calibrate one or more cameras from dartboard images.
    
    Returns overlay images showing the detected coordinate system.
    """
    results = []
    
    for camera in request.cameras:
        try:
            result = calibrator.calibrate(
                camera_id=camera.camera_id,
                image_base64=camera.image
            )
            
            # Store successful calibration
            if result.success:
                calibration_store.save(camera.camera_id, result.calibration_data)
            
            results.append(result)
            
        except Exception as e:
            results.append(CameraCalibrationResult(
                camera_id=camera.camera_id,
                success=False,
                error=str(e)
            ))
    
    return CalibrateResponse(results=results)


@router.post("/detect", response_model=DetectResponse)
async def detect_dart(request: DetectRequest):
    """
    Detect dart position in an image and calculate the score.
    
    Requires camera to be calibrated first.
    """
    # Check if camera is calibrated
    calibration_data = calibration_store.get(request.camera_id)
    if calibration_data is None:
        raise HTTPException(
            status_code=400,
            detail=f"Camera '{request.camera_id}' is not calibrated. Call /calibrate first."
        )
    
    try:
        result = calibrator.detect_dart(
            camera_id=request.camera_id,
            image_base64=request.image,
            calibration_data=calibration_data
        )
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/calibrations", response_model=List[CalibrationInfo])
async def list_calibrations():
    """List all stored camera calibrations."""
    return calibration_store.list_all()


@router.delete("/calibration/{camera_id}")
async def delete_calibration(camera_id: str):
    """Delete a camera's calibration."""
    if calibration_store.delete(camera_id):
        return {"message": f"Calibration for '{camera_id}' deleted"}
    raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found")
