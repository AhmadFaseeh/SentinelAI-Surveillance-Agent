import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from ultralytics import YOLO
from ..core.types import BoundingBox, KeyframeMetadata
from ..core.config import PerceptionConfig, default_config
from ..core.logger import logger


class SurveillanceDetector:
    COCO_CLASSES = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        4: "airplane",
        5: "bus",
        6: "train",
        7: "truck",
        8: "boat",
        24: "backpack",
        25: "umbrella",
        26: "handbag",
        28: "suitcase",
        43: "knife",
    }

    def __init__(self, config: Optional[PerceptionConfig] = None):
        self.config = config or default_config.perception
        logger.info(f"Loading YOLO Model checkpoint: [bold cyan]{self.config.yolo_model_name}[/bold cyan]...")
        self.model = YOLO(self.config.yolo_model_name)

    def detect_in_frame(self, frame: np.ndarray) -> List[BoundingBox]:
        results = self.model.predict(
            source=frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            classes=self.config.track_classes,
            verbose=False
        )
        boxes: List[BoundingBox] = []
        if not results or len(results) == 0:
            return boxes

        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            return boxes

        for box in res.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = self.model.names.get(cls_id, f"class_{cls_id}")

            bbox = BoundingBox(
                x1=float(xyxy[0]),
                y1=float(xyxy[1]),
                x2=float(xyxy[2]),
                y2=float(xyxy[3]),
                confidence=conf,
                class_id=cls_id,
                class_name=cls_name
            )
            boxes.append(bbox)

        return boxes

    def annotate_frame(self, frame: np.ndarray, boxes: List[BoundingBox]) -> np.ndarray:
        """Draws bounding boxes and labels onto the image frame."""
        annotated = frame.copy()
        for b in boxes:
            x1, y1, x2, y2 = map(int, b.to_xyxy())
            color = (0, 255, 0) if b.class_name == "person" else (255, 100, 0)
            if "knife" in b.class_name or "weapon" in b.class_name:
                color = (0, 0, 255)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{b.class_name} {b.confidence:.2f}"
            cv2.putText(
                annotated,
                label,
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
        return annotated
