"""
In-memory calibration storage.

Stores calibration data for each camera so it can be reused
for subsequent dart detection requests.
"""
from typing import Dict, Optional, List, Any
from datetime import datetime
from threading import Lock

from app.models.schemas import CalibrationInfo


class CalibrationStore:
    """Thread-safe in-memory storage for camera calibrations."""
    
    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()
    
    def save(self, camera_id: str, calibration_data: Dict[str, Any]) -> None:
        """Save calibration data for a camera."""
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
