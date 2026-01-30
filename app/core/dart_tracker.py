"""
Dart Tracker - Stateful dart position tracking.

Maintains state of darts currently on the board, identifies new darts,
and calculates scores using calibration data.

Supports multiple boards via BoardTrackerManager.
"""
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from threading import Lock


@dataclass
class DartPosition:
    """Represents a detected dart on the board."""
    dart_id: str
    dart_index: int  # 0, 1, 2 for first, second, third dart
    segment: int  # 1-20 or 0 for bull
    multiplier: int  # 1=single, 2=double, 3=triple
    zone: str  # 'single', 'double', 'triple', 'inner_bull', 'outer_bull', 'miss'
    score: int  # segment * multiplier (or 25/50 for bulls)
    confidence: float  # Average confidence across cameras
    x_mm: float  # Position in dartboard coordinates
    y_mm: float
    timestamp: float = field(default_factory=time.time)
    
    # Per-camera details
    camera_detections: List[Dict[str, Any]] = field(default_factory=list)


@dataclass 
class DetectionResult:
    """Result from a detection request."""
    detection_id: str
    board_id: str
    timestamp: float
    
    # The new dart (if any)
    new_dart: Optional[DartPosition]
    
    # All darts currently on board
    all_darts: List[DartPosition]
    
    # Consensus from cameras for new dart
    consensus: Optional[Dict[str, Any]]
    
    # Per-camera raw detections
    camera_results: List[Dict[str, Any]]
    
    # Any darts that went missing
    missing_darts: List[str]  # dart_ids
    
    # Board state
    dart_count: int
    board_cleared: bool


class DartTracker:
    """
    Stateful dart tracker for a single board.
    
    Maintains list of darts on the board, detects new darts by comparing
    to known positions, and provides scoring.
    """
    
    # Distance threshold in mm to consider two tips as same dart
    POSITION_TOLERANCE_MM = 15.0
    
    def __init__(self, board_id: str):
        self.board_id = board_id
        self._lock = Lock()
        self._darts: List[DartPosition] = []
        self._next_dart_index = 0
        self._last_detection_id: Optional[str] = None
        self._created_at = time.time()
        self._last_activity = time.time()
        
    def reset(self) -> None:
        """Clear all tracked darts (board cleared)."""
        with self._lock:
            self._darts = []
            self._next_dart_index = 0
            self._last_detection_id = None
            self._last_activity = time.time()
    
    @property
    def dart_count(self) -> int:
        """Number of darts currently tracked."""
        with self._lock:
            return len(self._darts)
    
    @property
    def darts(self) -> List[DartPosition]:
        """Get copy of current dart list."""
        with self._lock:
            return list(self._darts)
    
    def process_detection(
        self,
        detected_tips: List[Dict[str, Any]],
        scoring_func: callable
    ) -> DetectionResult:
        """
        Process a new detection from cameras.
        
        Args:
            detected_tips: List of detected tip positions from YOLO
                Each item: {
                    'camera_id': str,
                    'x_mm': float,  # dartboard coordinates
                    'y_mm': float,
                    'confidence': float,
                    'x_px': int,  # pixel coords (for debugging)
                    'y_px': int
                }
            scoring_func: Function to calculate score from (x_mm, y_mm) -> Dict
                
        Returns:
            DetectionResult with new dart info
        """
        detection_id = str(uuid.uuid4())[:8]
        timestamp = time.time()
        
        with self._lock:
            self._last_activity = timestamp
            
            # Group detections by position (cluster nearby tips)
            clusters = self._cluster_detections(detected_tips)
            
            # Match clusters to known darts
            new_dart = None
            missing_darts = []
            matched_dart_ids = set()
            
            for cluster in clusters:
                avg_x = sum(d['x_mm'] for d in cluster) / len(cluster)
                avg_y = sum(d['y_mm'] for d in cluster) / len(cluster)
                avg_conf = sum(d['confidence'] for d in cluster) / len(cluster)
                
                # Check if this matches a known dart
                matched = False
                for dart in self._darts:
                    dist = math.sqrt((avg_x - dart.x_mm)**2 + (avg_y - dart.y_mm)**2)
                    if dist < self.POSITION_TOLERANCE_MM:
                        matched_dart_ids.add(dart.dart_id)
                        matched = True
                        break
                
                if not matched:
                    # New dart!
                    score_info = scoring_func(avg_x, avg_y)
                    
                    new_dart = DartPosition(
                        dart_id=f"dart_{detection_id}_{self._next_dart_index}",
                        dart_index=self._next_dart_index,
                        segment=score_info.get('segment', 0),
                        multiplier=score_info.get('multiplier', 1),
                        zone=score_info.get('zone', 'miss'),
                        score=score_info.get('score', 0),
                        confidence=avg_conf,
                        x_mm=avg_x,
                        y_mm=avg_y,
                        camera_detections=cluster
                    )
                    
                    self._darts.append(new_dart)
                    self._next_dart_index += 1
            
            # Check for missing darts
            for dart in self._darts:
                if dart.dart_id not in matched_dart_ids and new_dart and dart.dart_id != new_dart.dart_id:
                    # This dart wasn't seen - might have fallen out
                    # For now, just track it but don't remove
                    pass
            
            # Build consensus for new dart
            consensus = None
            if new_dart:
                consensus = {
                    'segment': new_dart.segment,
                    'multiplier': new_dart.multiplier,
                    'score': new_dart.score,
                    'zone': new_dart.zone,
                    'confidence': new_dart.confidence
                }
            
            # Camera results summary
            camera_results = []
            cameras_seen = set()
            for tip in detected_tips:
                cam_id = tip.get('camera_id', 'unknown')
                if cam_id not in cameras_seen:
                    cameras_seen.add(cam_id)
                    # Group tips by camera
                    cam_tips = [t for t in detected_tips if t.get('camera_id') == cam_id]
                    camera_results.append({
                        'camera_id': cam_id,
                        'tips_detected': len(cam_tips),
                        'tips': cam_tips
                    })
            
            self._last_detection_id = detection_id
            
            return DetectionResult(
                detection_id=detection_id,
                board_id=self.board_id,
                timestamp=timestamp,
                new_dart=new_dart,
                all_darts=list(self._darts),
                consensus=consensus,
                camera_results=camera_results,
                missing_darts=missing_darts,
                dart_count=len(self._darts),
                board_cleared=len(self._darts) == 0
            )
    
    def _cluster_detections(
        self, 
        detections: List[Dict[str, Any]],
        cluster_threshold_mm: float = 20.0
    ) -> List[List[Dict[str, Any]]]:
        """
        Cluster nearby detections (same dart seen by multiple cameras).
        
        Simple greedy clustering - could be improved with proper clustering algo.
        """
        if not detections:
            return []
        
        clusters = []
        used = set()
        
        for i, det in enumerate(detections):
            if i in used:
                continue
                
            cluster = [det]
            used.add(i)
            
            for j, other in enumerate(detections):
                if j in used:
                    continue
                    
                # Check distance
                dist = math.sqrt(
                    (det['x_mm'] - other['x_mm'])**2 + 
                    (det['y_mm'] - other['y_mm'])**2
                )
                
                if dist < cluster_threshold_mm:
                    cluster.append(other)
                    used.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def remove_dart(self, dart_id: str) -> bool:
        """Remove a specific dart (e.g., fell out)."""
        with self._lock:
            for i, dart in enumerate(self._darts):
                if dart.dart_id == dart_id:
                    self._darts.pop(i)
                    self._last_activity = time.time()
                    return True
            return False
    
    def get_state(self) -> Dict[str, Any]:
        """Get current tracker state for API response."""
        with self._lock:
            return {
                'board_id': self.board_id,
                'dart_count': len(self._darts),
                'darts': [
                    {
                        'dart_id': d.dart_id,
                        'dart_index': d.dart_index,
                        'segment': d.segment,
                        'multiplier': d.multiplier,
                        'score': d.score,
                        'zone': d.zone,
                        'confidence': d.confidence,
                        'x_mm': d.x_mm,
                        'y_mm': d.y_mm
                    }
                    for d in self._darts
                ],
                'last_detection_id': self._last_detection_id
            }


class BoardTrackerManager:
    """
    Manages DartTracker instances for multiple boards.
    
    Each physical dartboard gets its own tracker instance.
    Trackers are created on-demand and can be cleaned up after inactivity.
    """
    
    # Clean up trackers after 1 hour of inactivity
    INACTIVE_TIMEOUT_SECONDS = 3600
    
    def __init__(self):
        self._lock = Lock()
        self._trackers: Dict[str, DartTracker] = {}
    
    def get_tracker(self, board_id: str) -> DartTracker:
        """Get or create a tracker for the given board."""
        with self._lock:
            if board_id not in self._trackers:
                self._trackers[board_id] = DartTracker(board_id)
            return self._trackers[board_id]
    
    def reset_board(self, board_id: str) -> bool:
        """Reset the tracker for a board (clear darts)."""
        with self._lock:
            if board_id in self._trackers:
                self._trackers[board_id].reset()
                return True
            return False
    
    def remove_board(self, board_id: str) -> bool:
        """Remove a board's tracker entirely."""
        with self._lock:
            if board_id in self._trackers:
                del self._trackers[board_id]
                return True
            return False
    
    def list_boards(self) -> List[Dict[str, Any]]:
        """List all active boards."""
        with self._lock:
            return [
                {
                    'board_id': board_id,
                    'dart_count': tracker.dart_count,
                    'created_at': tracker._created_at,
                    'last_activity': tracker._last_activity
                }
                for board_id, tracker in self._trackers.items()
            ]
    
    def cleanup_inactive(self) -> List[str]:
        """Remove trackers that have been inactive too long."""
        now = time.time()
        removed = []
        
        with self._lock:
            inactive_boards = [
                board_id
                for board_id, tracker in self._trackers.items()
                if now - tracker._last_activity > self.INACTIVE_TIMEOUT_SECONDS
            ]
            
            for board_id in inactive_boards:
                del self._trackers[board_id]
                removed.append(board_id)
        
        return removed
    
    def get_state(self, board_id: str) -> Optional[Dict[str, Any]]:
        """Get state for a specific board."""
        with self._lock:
            if board_id in self._trackers:
                return self._trackers[board_id].get_state()
            return None


# Global manager instance
tracker_manager = BoardTrackerManager()
