
# === Camera Endpoints ===

@router.get("/cameras")
async def list_cameras():
    """List available cameras on this machine."""
    import cv2
    cameras = []
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            cameras.append({
                "index": i,
                "camera_id": f"cam{i}",
                "connected": ret,
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            })
            cap.release()
    return {"cameras": cameras}


@router.get("/cameras/{camera_index}/frame")
async def get_camera_frame(camera_index: int):
    """Capture a frame from the specified camera."""
    import cv2
    from fastapi.responses import Response
    
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise HTTPException(status_code=404, detail=f"Camera {camera_index} not found")
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise HTTPException(status_code=500, detail="Failed to capture frame")
    
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(content=buffer.tobytes(), media_type="image/jpeg")


@router.get("/cameras/{camera_index}/snapshot")
async def get_camera_snapshot_base64(camera_index: int):
    """Capture a frame and return as base64 (for calibration)."""
    import cv2
    import base64
    
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise HTTPException(status_code=404, detail=f"Camera {camera_index} not found")
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise HTTPException(status_code=500, detail="Failed to capture frame")
    
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "camera_id": f"cam{camera_index}",
        "image": b64,
        "width": frame.shape[1],
        "height": frame.shape[0]
    }
