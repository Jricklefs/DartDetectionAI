"""
Calibration storage - loads from DartGame API on demand.

When a game starts, DartDetect should call load_from_api(board_id)
to fetch fresh calibration data for that board's cameras.
"""
import json
import os
import requests
from typing import Dict, Optional, List, Any
from datetime import datetime
from threading import Lock

from app.models.schemas import CalibrationInfo


class CalibrationStore:
    """Thread-safe in-memory storage for camera calibrations."""
    
    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()
        self._dartgame_api_url = os.environ.get("DARTGAME_API_URL", "http://localhost:5000")
    
    def load_from_api(self, board_id: str = "default") -> dict:
        """
        Load calibrations from DartGame API for a specific board.
        
        Call this when a game starts to get fresh calibration data.
        Clears existing calibrations and loads only the ones for this board's cameras.
        
        Args:
            board_id: The board ID to load calibrations for
            
        Returns:
            Dict with success status, loaded cameras, and any errors
        """
        result = {
            "success": False,
            "board_id": board_id,
            "cameras_loaded": [],
            "cameras_failed": [],
            "errors": []
        }
        
        try:
            # Step 1: Get cameras for this board
            cameras_url = f"{self._dartgame_api_url}/api/boards/{board_id}/cameras"
            print(f"[CalibrationStore] Fetching cameras from {cameras_url}")
            
            cameras_response = requests.get(cameras_url, timeout=10)
            
            if cameras_response.status_code == 404:
                result["errors"].append(f"Board '{board_id}' not found")
                return result
            
            cameras_response.raise_for_status()
            cameras = cameras_response.json()
            
            if not cameras:
                result["errors"].append(f"No cameras registered for board '{board_id}'")
                return result
            
            print(f"[CalibrationStore] Found {len(cameras)} cameras for board {board_id}")
            
            # Step 2: Clear existing calibrations
            with self._lock:
                self._store.clear()
            
            # Step 3: Load calibration for each camera
            for cam in cameras:
                camera_id = cam.get("cameraId") or cam.get("camera_id") or cam.get("CameraId")
                if not camera_id:
                    continue
                
                try:
                    cal_url = f"{self._dartgame_api_url}/api/calibrations/{camera_id}"
                    print(f"[CalibrationStore] Fetching calibration from {cal_url}")
                    
                    cal_response = requests.get(cal_url, timeout=10)
                    
                    if cal_response.status_code == 404:
                        result["cameras_failed"].append({
                            "camera_id": camera_id,
                            "error": "Not calibrated"
                        })
                        continue
                    
                    cal_response.raise_for_status()
                    cal = cal_response.json()
                    
                    # Parse calibration data JSON if it's a string
                    cal_data = cal.get("calibrationData") or cal.get("calibration_data")
                    if isinstance(cal_data, str):
                        try:
                            cal_data = json.loads(cal_data)
                        except json.JSONDecodeError:
                            result["cameras_failed"].append({
                                "camera_id": camera_id,
                                "error": "Invalid calibration data JSON"
                            })
                            continue
                    
                    if not cal_data:
                        result["cameras_failed"].append({
                            "camera_id": camera_id,
                            "error": "Empty calibration data"
                        })
                        continue
                    
                    # Store the calibration
                    with self._lock:
                        self._store[camera_id] = {
                            "data": cal_data,
                            "created_at": datetime.utcnow(),
                            "quality": cal.get("quality", 0.0),
                            "segment_at_top": cal_data.get("segment_at_top")
                        }
                    
                    result["cameras_loaded"].append({
                        "camera_id": camera_id,
                        "quality": cal.get("quality", 0.0)
                    })
                    
                    print(f"[CalibrationStore] Loaded calibration for {camera_id} (quality: {cal.get('quality', 'N/A')})")
                    
                except requests.RequestException as e:
                    result["cameras_failed"].append({
                        "camera_id": camera_id,
                        "error": str(e)
                    })
            
            result["success"] = len(result["cameras_loaded"]) > 0
            print(f"[CalibrationStore] Loaded {len(result['cameras_loaded'])} calibrations for board {board_id}")
            return result
            
        except requests.RequestException as e:
            result["errors"].append(f"Failed to connect to DartGame API: {e}")
            print(f"[CalibrationStore] Failed to load from API: {e}")
            return result
        except Exception as e:
            result["errors"].append(f"Unexpected error: {e}")
            print(f"[CalibrationStore] Error loading calibrations: {e}")
            return result
    
    def save(self, camera_id: str, calibration_data: Dict[str, Any]) -> None:
        """Save calibration data for a camera (memory only)."""
        with self._lock:
            self._store[camera_id] = {
                "data": calibration_data,
                "created_at": datetime.utcnow(),
                "quality": calibration_data.get("quality", 0.0),
                "segment_at_top": calibration_data.get("segment_at_top")
            }
    
    def get(self, camera_id: str) -> Optional[Dict[str, Any]]:
        """Get calibration data for a camera."""
        with self._lock:
            entry = self._store.get(camera_id)
            return entry["data"] if entry else None
    
    def delete(self, camera_id: str) -> bool:
        """Delete calibration for a camera. Returns True if existed."""
        with self._lock:
            if camera_id in self._store:
                del self._store[camera_id]
                return True
            return False
    
    def list_all(self) -> List[CalibrationInfo]:
        """List all stored calibrations."""
        with self._lock:
            return [
                CalibrationInfo(
                    camera_id=camera_id,
                    created_at=entry["created_at"],
                    quality=entry["quality"],
                    segment_at_top=entry.get("segment_at_top")
                )
                for camera_id, entry in self._store.items()
            ]
    
    def clear(self) -> None:
        """Clear all calibrations."""
        with self._lock:
            self._store.clear()


# Global singleton instance
calibration_store = CalibrationStore()
