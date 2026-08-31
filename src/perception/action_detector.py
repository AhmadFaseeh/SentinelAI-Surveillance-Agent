"""
SentinelAI: Live Action & Posture Recognition Engine
Analyzes human motion dynamics, posture (sitting/standing), and object interactions in real-time.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2
from .detector import SurveillanceDetector
from ..core.types import BoundingBox


class ActionDetector:
    def __init__(self, detector: Optional[SurveillanceDetector] = None):
        self.detector = detector or SurveillanceDetector()
        self.prev_centroid = None
        self.movement_history: List[str] = []

    def analyze_live_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Processes a live camera frame:
        - Detects persons and interaction objects (phones, cups, bags, laptops)
        - Computes movement velocity and direction (Moving Left/Right/Stationary)
        - Estimates posture (Sitting vs. Standing based on aspect ratio)
        - Returns annotated frame and structured action telemetry.
        """
        boxes = self.detector.detect_in_frame(frame)
        h, w = frame.shape[:2]

        person_box = None
        detected_items = []

        for b in boxes:
            if b.class_name == "person":
                if person_box is None or b.area() > person_box.area():
                    person_box = b
            else:
                detected_items.append(b.class_name)

        action_state = {
            "is_person_present": person_box is not None,
            "posture": "Unknown",
            "movement": "No Person",
            "interacting_with": detected_items,
            "confidence": 0.0,
            "status_text": "No Person in View"
        }

        annotated = frame.copy()

        if person_box:
            action_state["confidence"] = person_box.confidence
            bx1, by1, bx2, by2 = map(int, person_box.to_xyxy())
            pw = bx2 - bx1
            ph = by2 - by1
            cx = int((bx1 + bx2) / 2)
            cy = int((by1 + by2) / 2)

            # 1. Posture Estimation (Aspect Ratio & Height relative to frame)
            aspect_ratio = ph / max(1, pw)
            if aspect_ratio < 1.3:
                action_state["posture"] = "Sitting / Crouched"
            else:
                action_state["posture"] = "Standing"

            # 2. Movement & Direction Analysis
            if self.prev_centroid is not None:
                dx = cx - self.prev_centroid[0]
                dy = cy - self.prev_centroid[1]
                distance = np.sqrt(dx**2 + dy**2)

                if distance > 18:
                    if abs(dx) > abs(dy):
                        direction = "Moving Right ➔" if dx > 0 else "Moving Left ⬅"
                    else:
                        direction = "Moving Down ⬇" if dy > 0 else "Moving Up ⬆"
                    action_state["movement"] = f"Dynamic ({direction})"
                else:
                    action_state["movement"] = "Stationary / Subtle Movement"
            else:
                action_state["movement"] = "Tracking Centroid..."

            self.prev_centroid = (cx, cy)

            # 3. Overall Activity Summary
            items_str = f" with {', '.join(detected_items)}" if detected_items else ""
            status_text = f"{action_state['posture']} | {action_state['movement']}{items_str}"
            action_state["status_text"] = status_text

            # Draw HUD Overlays on live frame
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)

            # Top HUD banner
            cv2.rectangle(annotated, (0, 0), (w, 45), (20, 20, 25), -1)
            cv2.putText(
                annotated,
                f"ACTION: {status_text}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 200),
                2
            )

        # Draw any secondary detected objects (phones, cups, laptops)
        for b in boxes:
            if b.class_name != "person":
                x1, y1, x2, y2 = map(int, b.to_xyxy())
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 165, 0), 2)
                cv2.putText(
                    annotated,
                    b.class_name,
                    (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 165, 0),
                    2
                )

        return annotated, action_state
