"""
API Routes for DartDetectionAI

Provides endpoints for:
- Camera calibration
- Multi-camera dart detection with stateful tracking (board-scoped)
- Board management
"""
from fastapi import APIRouter, HTTPException, Query
from typing import List

from app.models.schemas import (
    CalibrateRequest,
    CalibrateResponse,
    CameraCalibrationResult,
    DetectRequest,
    DetectResponse,
    CalibrationInfo,
    MultiDetectRequest,
    MultiDetectResponse,
    DartInfo,
    CameraResult,
    ConsensusResult,
    TrackerState,
    BoardInfo,
    BoardListResponse
)
from app.core.calibration import DartboardCalibrator
from app.core.storage import calibration_store
from app.core.dart_tracker import tracker_manager
from app.core.scoring import scoring_system

router = APIRouter()

# Shared calibrator instance
calibrator = DartboardCalibrator()


# === Calibration Endpoints ===

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


# === Multi-Camera Detection with Board-Scoped Tracking ===

@router.post("/detect/multi", response_model=MultiDetectResponse)
async def detect_multi_camera(request: MultiDetectRequest):
    """
    Detect darts from multiple cameras with stateful tracking.
    
    Each board_id gets its own tracker. State is maintained between calls
    until POST /detect/reset is called.
    
    - Runs YOLO on each camera image to find dart tips
    - Clusters tips from different cameras viewing same dart
    - Compares to known dart positions to identify NEW darts
    - Calculates score using calibration data
    - Returns new dart and full board state
    
    Call POST /detect/reset?board_id=xxx when board is cleared.
    """
    board_id = request.board_id
    tracker = tracker_manager.get_tracker(board_id)
    
    detected_tips = []
    camera_results = []
    
    for cam in request.cameras:
        # Check calibration
        calibration_data = calibration_store.get(cam.camera_id)
        if calibration_data is None:
            camera_results.append(CameraResult(
                camera_id=cam.camera_id,
                tips_detected=0,
                tips=[]
            ))
            continue
        
        try:
            # Run YOLO detection on this camera's image
            tips = calibrator.detect_tips(
                camera_id=cam.camera_id,
                image_base64=cam.image,
                calibration_data=calibration_data
            )
            
            # Convert pixel coords to dartboard mm coords
            for tip in tips:
                tip['camera_id'] = cam.camera_id
                # Transform using calibration
                if 'x_mm' not in tip:
                    tip['x_mm'], tip['y_mm'] = calibrator.pixel_to_dartboard(
                        tip['x_px'], tip['y_px'],
                        calibration_data
                    )
                detected_tips.append(tip)
            
            camera_results.append(CameraResult(
                camera_id=cam.camera_id,
                tips_detected=len(tips),
                tips=tips
            ))
            
        except Exception as e:
            camera_results.append(CameraResult(
                camera_id=cam.camera_id,
                tips_detected=0,
                tips=[{"error": str(e)}]
            ))
    
    # Process through board-specific tracker
    result = tracker.process_detection(
        detected_tips=detected_tips,
        scoring_func=lambda x, y: scoring_system.score_from_dartboard_coords(x, y)
    )
    
    # Build response
    new_dart_info = None
    if result.new_dart:
        new_dart_info = DartInfo(
            dart_id=result.new_dart.dart_id,
            dart_index=result.new_dart.dart_index,
            segment=result.new_dart.segment,
            multiplier=result.new_dart.multiplier,
            score=result.new_dart.score,
            zone=result.new_dart.zone,
            confidence=result.new_dart.confidence,
            x_mm=result.new_dart.x_mm,
            y_mm=result.new_dart.y_mm,
            is_new=True
        )
    
    all_darts_info = [
        DartInfo(
            dart_id=d.dart_id,
            dart_index=d.dart_index,
            segment=d.segment,
            multiplier=d.multiplier,
            score=d.score,
            zone=d.zone,
            confidence=d.confidence,
            x_mm=d.x_mm,
            y_mm=d.y_mm,
            is_new=(result.new_dart and d.dart_id == result.new_dart.dart_id)
        )
        for d in result.all_darts
    ]
    
    consensus = None
    if result.consensus:
        consensus = ConsensusResult(**result.consensus)
    
    return MultiDetectResponse(
        detection_id=result.detection_id,
        board_id=board_id,
        timestamp=result.timestamp,
        new_dart=new_dart_info,
        all_darts=all_darts_info,
        consensus=consensus,
        camera_results=camera_results,
        dart_count=result.dart_count,
        board_cleared=result.board_cleared
    )


@router.post("/detect/reset")
async def reset_tracker(board_id: str = Query(..., description="Board ID to reset")):
    """
    Reset the dart tracker for a board (board cleared).
    
    Call this when darts are removed from the board.
    """
    tracker_manager.reset_board(board_id)
    return {"message": f"Board '{board_id}' reset", "dart_count": 0}


@router.get("/detect/state", response_model=TrackerState)
async def get_tracker_state(board_id: str = Query(..., description="Board ID")):
    """
    Get current dart tracker state for a board.
    
    Returns all darts currently tracked on the board.
    """
    state = tracker_manager.get_state(board_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Board '{board_id}' not found")
    
    return TrackerState(
        board_id=state['board_id'],
        dart_count=state['dart_count'],
        darts=[
            DartInfo(
                dart_id=d['dart_id'],
                dart_index=d['dart_index'],
                segment=d['segment'],
                multiplier=d['multiplier'],
                score=d['score'],
                zone=d['zone'],
                confidence=d['confidence'],
                x_mm=d['x_mm'],
                y_mm=d['y_mm'],
                is_new=False
            )
            for d in state['darts']
        ],
        last_detection_id=state['last_detection_id']
    )


@router.get("/boards", response_model=BoardListResponse)
async def list_boards():
    """
    List all active boards with trackers.
    """
    boards = tracker_manager.list_boards()
    return BoardListResponse(
        boards=[
            BoardInfo(
                board_id=b['board_id'],
                dart_count=b['dart_count'],
                created_at=b['created_at'],
                last_activity=b['last_activity']
            )
            for b in boards
        ]
    )


@router.delete("/board/{board_id}")
async def remove_board(board_id: str):
    """
    Remove a board's tracker entirely.
    """
    if tracker_manager.remove_board(board_id):
        return {"message": f"Board '{board_id}' removed"}
    raise HTTPException(status_code=404, detail=f"Board '{board_id}' not found")


@router.delete("/detect/dart/{dart_id}")
async def remove_dart(
    dart_id: str,
    board_id: str = Query(..., description="Board ID")
):
    """
    Remove a specific dart from a board (e.g., it fell out).
    """
    state = tracker_manager.get_state(board_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Board '{board_id}' not found")
    
    tracker = tracker_manager.get_tracker(board_id)
    if tracker.remove_dart(dart_id):
        return {"message": f"Dart '{dart_id}' removed", "dart_count": tracker.dart_count}
    raise HTTPException(status_code=404, detail=f"Dart '{dart_id}' not found on board '{board_id}'")


# === Legacy Single-Camera Detection ===

@router.post("/detect", response_model=DetectResponse)
async def detect_dart(request: DetectRequest):
    """
    Detect dart position in an image and calculate the score.
    
    Legacy single-camera endpoint. Use /detect/multi for multi-camera with tracking.
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
