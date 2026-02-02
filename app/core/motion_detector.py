"""
Motion Detection Service for Dart Detection

Runs a continuous loop watching for dart tip changes on the board.
When motion is detected, runs YOLO detection and notifies the DartGame API.
"""
import cv2
import numpy as np
import base64
import threading
import time
import httpx
import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

from app.core.storage import calibration_store
from app.core.detection import DartTipDetector, score_from_ellipse_calibration

logger = logging.getLogger(__name__)


@dataclass
class CameraState:
    """State for a single camera."""
    camera_id: str
    index: int
    baseline_frame: Optional[np.ndarray] = None
    baseline_gray: Optional[np.ndarray] = None
    baseline_captured_at: Optional[datetime] = None
    last_tips: List[dict] = field(default_factory=list)


@dataclass
class DetectionEvent:
    """A dart detection event to send to DartGame API."""
    camera_id: str
    segment: int
    multiplier: int
    score: int
    x_mm: float
    y_mm: float
    confidence: float
    zone: str
    image_base64: Optional[str] = None


class MotionDetectionService:
    """
    Continuous motion detection service.
    
    Watches cameras for changes, runs YOLO detection when motion detected,
    and notifies DartGame API when new darts are found.
    """
    
    def __init__(
        self,
        game_api_url: str = "http://localhost:5000",
        board_id: str = "default",
        motion_threshold: float = 30.0,  # Pixel intensity difference threshold
        min_contour_area: int = 500,  # Minimum contour area to consider as motion
        detection_interval_ms: int = 50,  # 20 FPS
        cooldown_after_detection_ms: int = 500,  # Wait after detecting a dart
        api_key: str = "local"
    ):
        self.game_api_url = game_api_url
        self.board_id = board_id
        self.motion_threshold = motion_threshold
        self.min_contour_area = min_contour_area
        self.detection_interval = detection_interval_ms / 1000.0
        self.cooldown_after_detection = cooldown_after_detection_ms / 1000.0
        self.api_key = api_key
        
        # Camera states
        self.cameras: Dict[str, CameraState] = {}
        self._camera_caps: Dict[str, cv2.VideoCapture] = {}
        
        # Detection service
        self.detector = DartTipDetector()
        
        # Control
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._last_detection_time = 0.0
        
        # Known dart positions (to avoid duplicate notifications)
        self._known_darts: List[dict] = []
        
        # HTTP client for async notifications
        self._http_client: Optional[httpx.AsyncClient] = None
        
        # Event loop for async operations
        self._loop: Optional[asyncio.AbstractEventLoop] = None
    
    def initialize_cameras(self, camera_indices: Optional[List[int]] = None) -> List[str]:
        """
        Initialize camera connections.
        
        Args:
            camera_indices: List of camera indices to use, or None to auto-detect
            
        Returns:
            List of initialized camera IDs
        """
        if camera_indices is None:
            # Auto-detect cameras (check indices 0-4)
            camera_indices = []
            for i in range(5):
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        camera_indices.append(i)
                    cap.release()
        
        initialized = []
        for idx in camera_indices:
            camera_id = f"cam{idx}"
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                # Set resolution if needed
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                self._camera_caps[camera_id] = cap
                self.cameras[camera_id] = CameraState(
                    camera_id=camera_id,
                    index=idx
                )
                initialized.append(camera_id)
                logger.info(f"Initialized camera {camera_id}")
            else:
                logger.warning(f"Failed to open camera {idx}")
        
        return initialized
    
    def release_cameras(self):
        """Release all camera resources."""
        for cap in self._camera_caps.values():
            cap.release()
        self._camera_caps.clear()
        self.cameras.clear()
    
    def capture_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """Capture a frame from a camera."""
        cap = self._camera_caps.get(camera_id)
        if cap is None or not cap.isOpened():
            return None
        
        ret, frame = cap.read()
        return frame if ret else None
    
    def rebase(self) -> Dict[str, bool]:
        """
        Capture new baseline images for all cameras.
        Called when game starts or turn changes.
        
        Returns:
            Dict mapping camera_id to success status
        """
        print(f"[REBASE] Capturing baseline for all cameras...")
        results = {}
        with self._lock:
            self._known_darts.clear()  # Reset known darts on rebase
            print(f"[REBASE] Cleared known darts list")
            
            for camera_id, state in self.cameras.items():
                frame = self.capture_frame(camera_id)
                if frame is not None:
                    state.baseline_frame = frame.copy()
                    state.baseline_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    state.baseline_gray = cv2.GaussianBlur(state.baseline_gray, (21, 21), 0)
                    state.baseline_captured_at = datetime.utcnow()
                    
                    # Detect any existing tips and store as "known"
                    existing_tips = self._detect_tips_in_frame(camera_id, frame)
                    state.last_tips = existing_tips
                    for tip in existing_tips:
                        if not self._is_known_dart(tip):
                            self._known_darts.append(tip)
                    
                    results[camera_id] = True
                    print(f"[REBASE] {camera_id}: baseline captured, {len(existing_tips)} existing tips")
                    logger.info(f"Rebased {camera_id}: {len(existing_tips)} existing tips")
                else:
                    results[camera_id] = False
                    print(f"[REBASE] {camera_id}: FAILED to capture frame")
                    logger.warning(f"Failed to capture baseline for {camera_id}")
        
        return results
    
    def _detect_tips_in_frame(self, camera_id: str, frame: np.ndarray) -> List[dict]:
        """
        Run YOLO detection on a frame and return detected tips with scores.
        """
        if not self.detector.is_initialized:
            return []
        
        # Get calibration data
        storage_key = f"{self.api_key}:{camera_id}"
        calibration_data = calibration_store.get(storage_key)
        
        # Also try without api_key prefix (legacy)
        if calibration_data is None:
            calibration_data = calibration_store.get(camera_id)
        
        if calibration_data is None:
            logger.debug(f"No calibration for {camera_id}")
            return []
        
        # Run detection
        tips = self.detector.detect_tips(frame, confidence_threshold=0.5)
        
        results = []
        for tip in tips:
            # Calculate score using ellipse calibration
            score_info = score_from_ellipse_calibration(
                (tip.x, tip.y),
                calibration_data
            )
            
            # Convert pixel position to mm
            center = calibration_data.get("center", (0, 0))
            outer_double = calibration_data.get("outer_double_ellipse")
            
            x_mm, y_mm = 0.0, 0.0
            if outer_double:
                _, (ew, eh), _ = outer_double
                avg_radius_px = (ew + eh) / 4
                # Outer double is at 170mm from center
                scale = 170.0 / avg_radius_px if avg_radius_px > 0 else 1.0
                x_mm = (tip.x - center[0]) * scale
                y_mm = (tip.y - center[1]) * scale
            
            results.append({
                "camera_id": camera_id,
                "x_px": tip.x,
                "y_px": tip.y,
                "x_mm": x_mm,
                "y_mm": y_mm,
                "segment": score_info.get("segment", 0),
                "multiplier": score_info.get("multiplier", 1),
                "score": score_info.get("score", 0),
                "zone": score_info.get("zone", "unknown"),
                "confidence": tip.confidence
            })
        
        return results
    
    def _detect_motion(self, camera_id: str, current_frame: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        Detect if there's significant motion compared to baseline.
        
        Returns:
            Tuple of (motion_detected, diff_frame)
        """
        state = self.cameras.get(camera_id)
        if state is None or state.baseline_gray is None:
            return False, current_frame
        
        # Convert to grayscale and blur
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Compute absolute difference
        diff = cv2.absdiff(state.baseline_gray, gray)
        
        # Threshold to get binary mask
        _, thresh = cv2.threshold(diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
        
        # Dilate to fill gaps
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check if any contour is large enough
        for contour in contours:
            if cv2.contourArea(contour) > self.min_contour_area:
                return True, diff
        
        return False, diff
    
    def _is_known_dart(self, tip: dict, threshold_mm: float = 20.0) -> bool:
        """Check if a tip matches a known dart position."""
        for known in self._known_darts:
            dist = np.sqrt(
                (tip["x_mm"] - known["x_mm"])**2 + 
                (tip["y_mm"] - known["y_mm"])**2
            )
            if dist < threshold_mm:
                return True
        return False
    
    async def _notify_game_api(self, event: DetectionEvent):
        """Send dart detection to DartGame API."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=5.0)
        
        url = f"{self.game_api_url}/api/games/board/{self.board_id}/dart-detected"
        
        payload = {
            "cameraId": event.camera_id,
            "segment": event.segment,
            "multiplier": event.multiplier,
            "score": event.score,
            "xMm": event.x_mm,
            "yMm": event.y_mm,
            "confidence": event.confidence,
            "zone": event.zone,
            "imageBase64": event.image_base64
        }
        
        try:
            response = await self._http_client.post(url, json=payload)
            if response.status_code == 200:
                print(f"[DART] ✓ Sent to DartGame API: {event.zone} {event.segment}x{event.multiplier}={event.score}")
                logger.info(f"Notified DartGame API: {event.zone} {event.segment}x{event.multiplier}={event.score}")
            else:
                print(f"[DART] ✗ DartGame API error {response.status_code}: {response.text}")
                logger.warning(f"DartGame API returned {response.status_code}: {response.text}")
        except Exception as e:
            print(f"[DART] ✗ Failed to notify DartGame API: {e}")
            logger.error(f"Failed to notify DartGame API: {e}")
    
    def _detection_loop(self):
        """Main detection loop running in background thread."""
        print(f"[DETECTION] Motion detection loop STARTED - watching {len(self.cameras)} cameras")
        logger.info("Motion detection loop started")
        
        # Create event loop for this thread
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        while self._running:
            loop_start = time.time()
            
            try:
                # Check each camera
                for camera_id, state in self.cameras.items():
                    if state.baseline_frame is None:
                        continue
                    
                    # Capture current frame
                    frame = self.capture_frame(camera_id)
                    if frame is None:
                        continue
                    
                    # Check for motion
                    motion_detected, _ = self._detect_motion(camera_id, frame)
                    
                    if motion_detected:
                        # Motion detected - run YOLO detection
                        current_tips = self._detect_tips_in_frame(camera_id, frame)
                        
                        # Find new tips
                        new_tips = []
                        for tip in current_tips:
                            if tip["confidence"] < 0.5:
                                continue
                            if not self._is_known_dart(tip):
                                new_tips.append(tip)
                        
                        if new_tips:
                            print(f"[MOTION] Detected {len(new_tips)} new dart(s) on {camera_id}")
                            logger.info(f"Detected {len(new_tips)} new dart(s) on {camera_id}")
                            
                            for tip in new_tips:
                                # Add to known darts
                                with self._lock:
                                    self._known_darts.append(tip)
                                
                                # Encode frame as base64 for the event
                                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                                image_b64 = base64.b64encode(buffer).decode('utf-8')
                                
                                # Create detection event
                                event = DetectionEvent(
                                    camera_id=tip["camera_id"],
                                    segment=tip["segment"],
                                    multiplier=tip["multiplier"],
                                    score=tip["score"],
                                    x_mm=tip["x_mm"],
                                    y_mm=tip["y_mm"],
                                    confidence=tip["confidence"],
                                    zone=tip["zone"],
                                    image_base64=image_b64
                                )
                                
                                # Notify DartGame API
                                self._loop.run_until_complete(self._notify_game_api(event))
                            
                            # Update baseline after detection
                            state.baseline_frame = frame.copy()
                            state.baseline_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            state.baseline_gray = cv2.GaussianBlur(state.baseline_gray, (21, 21), 0)
                            
                            self._last_detection_time = time.time()
                
            except Exception as e:
                logger.error(f"Error in detection loop: {e}", exc_info=True)
            
            # Wait for next interval
            elapsed = time.time() - loop_start
            sleep_time = max(0, self.detection_interval - elapsed)
            time.sleep(sleep_time)
        
        # Cleanup
        if self._http_client:
            self._loop.run_until_complete(self._http_client.aclose())
        self._loop.close()
        logger.info("Motion detection loop stopped")
    
    def start(self):
        """Start the detection service."""
        if self._running:
            logger.warning("Detection service already running")
            return
        
        # Initialize cameras if not already done
        if not self.cameras:
            self.initialize_cameras()
        
        if not self.cameras:
            logger.error("No cameras available, cannot start detection")
            return
        
        # Capture initial baseline
        self.rebase()
        
        # Start detection thread
        self._running = True
        self._thread = threading.Thread(target=self._detection_loop, daemon=True)
        self._thread.start()
        logger.info("Motion detection service started")
    
    def stop(self):
        """Stop the detection service."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("Motion detection service stopped")
    
    def get_status(self) -> dict:
        """Get current service status."""
        return {
            "running": self._running,
            "board_id": self.board_id,
            "game_api_url": self.game_api_url,
            "cameras": {
                cid: {
                    "index": state.index,
                    "has_baseline": state.baseline_frame is not None,
                    "baseline_captured_at": state.baseline_captured_at.isoformat() if state.baseline_captured_at else None,
                    "last_tips_count": len(state.last_tips)
                }
                for cid, state in self.cameras.items()
            },
            "known_darts_count": len(self._known_darts),
            "motion_threshold": self.motion_threshold,
            "detection_interval_ms": int(self.detection_interval * 1000)
        }


# Global instance
motion_service: Optional[MotionDetectionService] = None


def get_motion_service() -> MotionDetectionService:
    """Get or create the global motion detection service instance."""
    global motion_service
    if motion_service is None:
        motion_service = MotionDetectionService()
    return motion_service
