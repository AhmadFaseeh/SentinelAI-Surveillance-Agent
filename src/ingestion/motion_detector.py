import cv2
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from ..core.config import IngestionConfig, default_config
from ..core.logger import logger


class MotionDetector:
    def __init__(self, config: Optional[IngestionConfig] = None):
        self.config = config or default_config.ingestion
        # MOG2 background subtractor with shadow detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300,
            varThreshold=25,
            detectShadows=True
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.prev_gray = None

    def analyze_frame(self, frame: np.ndarray) -> Tuple[bool, float, List[Tuple[int, int, int, int]]]:
        h, w = frame.shape[:2]
        total_pixels = h * w

        fg_mask = self.bg_subtractor.apply(frame)

        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, self.kernel, iterations=2)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_boxes = []
        motion_pixel_count = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.config.min_motion_area:
                x, y, cw, ch = cv2.boundingRect(cnt)
                motion_boxes.append((x, y, cw, ch))
                motion_pixel_count += area

        motion_score = min(1.0, motion_pixel_count / total_pixels)
        is_motion = motion_score >= self.config.motion_threshold or len(motion_boxes) > 0

        return is_motion, motion_score, motion_boxes

    def reset(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300,
            varThreshold=25,
            detectShadows=True
        )
        self.prev_gray = None
