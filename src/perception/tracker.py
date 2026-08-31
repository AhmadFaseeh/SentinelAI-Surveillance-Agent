"""
SentinelAI: Multi-Object Tracking & Trajectory Engine
Maintains persistent IDs across video frames and analyzes dwell/loitering time.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from ultralytics import YOLO

from ..core.types import TrackedObject, BoundingBox
from ..core.config import PerceptionConfig, default_config
from ..core.logger import logger


class SurveillanceTracker:
    def __init__(self, config: Optional[PerceptionConfig] = None):
        self.config = config or default_config.perception
        logger.info("Initializing ByteTrack Tracker Engine...")
        self.model = YOLO(self.config.yolo_model_name)
        # Stores track histories: track_id -> List[TrackedObject]
        self.tracks_history: Dict[int, List[TrackedObject]] = {}

    def track_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp_sec: float
    ) -> List[TrackedObject]:
        """
        Runs tracking on a frame and updates internal track trajectories.
        """
        results = self.model.track(
            source=frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            classes=self.config.track_classes,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        current_objects: List[TrackedObject] = []
        if not results or len(results) == 0:
            return current_objects

        res = results[0]
        if res.boxes is None or res.boxes.id is None:
            return current_objects

        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
        track_ids = res.boxes.id.cpu().numpy().astype(int)

        for xyxy, conf, cls_id, track_id in zip(boxes, confs, cls_ids, track_ids):
            cls_name = self.model.names.get(cls_id, f"class_{cls_id}")
            bbox = BoundingBox(
                x1=float(xyxy[0]),
                y1=float(xyxy[1]),
                x2=float(xyxy[2]),
                y2=float(xyxy[3]),
                confidence=float(conf),
                class_id=int(cls_id),
                class_name=cls_name
            )

            obj = TrackedObject(
                track_id=int(track_id),
                class_name=cls_name,
                class_id=int(cls_id),
                bbox=bbox,
                timestamp_sec=timestamp_sec,
                frame_index=frame_idx
            )
            current_objects.append(obj)

            if track_id not in self.tracks_history:
                self.tracks_history[track_id] = []
            self.tracks_history[track_id].append(obj)

        return current_objects

    def get_track_summary(self, track_id: int) -> Optional[Dict[str, Any]]:
        """
        Computes analytical summary for a given track ID:
        - Class
        - First seen timestamp
        - Last seen timestamp
        - Total duration (dwell time)
        - Spatial trajectory points
        """
        history = self.tracks_history.get(track_id)
        if not history:
            return None

        first_seen = history[0].timestamp_sec
        last_seen = history[-1].timestamp_sec
        duration = max(0.0, last_seen - first_seen)
        class_name = history[0].class_name

        trajectory = [
            (
                (obj.bbox.x1 + obj.bbox.x2) / 2.0,
                (obj.bbox.y1 + obj.bbox.y2) / 2.0,
                obj.timestamp_sec
            )
            for obj in history
        ]

        return {
            "track_id": track_id,
            "class_name": class_name,
            "first_seen_sec": first_seen,
            "last_seen_sec": last_seen,
            "duration_sec": duration,
            "total_detections": len(history),
            "trajectory_points": trajectory,
            "is_loitering": duration > 60.0  # Alert if dwelling > 60 seconds
        }

    def get_all_tracks_summary(self) -> List[Dict[str, Any]]:
        """Returns summary for all recorded tracks."""
        return [self.get_track_summary(tid) for tid in self.tracks_history.keys() if tid is not None]

    def reset(self):
        """Clears all tracking history."""
        self.tracks_history.clear()
