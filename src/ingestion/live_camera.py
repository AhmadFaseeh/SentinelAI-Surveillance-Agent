"""
SentinelAI: Live Webcam Ingestion & Recording Engine
Streams frames from webcam/RTSP feeds and records high-frame-rate forensic session clips.
Dynamically matches webcam resolution to avoid 0-byte video drops.
"""

import cv2
import time
import numpy as np
from pathlib import Path
from typing import Optional, Generator, Tuple
from ..core.logger import logger


class LiveCameraRecorder:
    def __init__(self, camera_index: int = 0, fps: int = 15):
        self.camera_index = camera_index
        self.fps = fps
        self.cap = None
        self.writer = None
        self.is_recording = False
        self.output_file = None
        self.frames_count = 0
        self.actual_width = 640
        self.actual_height = 480

    def start_camera(self) -> bool:
        """Opens camera capture device with DirectShow fallback on Windows."""
        if self.cap is None or not self.cap.isOpened():
            # Try DirectShow first on Windows
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.camera_index)

            if self.cap.isOpened():
                # Read a test frame to get exact native dimensions
                ret, test_frame = self.cap.read()
                if ret and test_frame is not None:
                    self.actual_height, self.actual_width = test_frame.shape[:2]
                    logger.info(f"Camera opened successfully. Native Resolution: {self.actual_width}x{self.actual_height}")
                return True
            else:
                logger.error(f"Failed to open camera with index {self.camera_index}")
                return False

        return True

    def start_recording(self, output_path: str):
        """Starts saving incoming frames into an MP4 video file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.output_file = str(out.resolve())
        self.frames_count = 0
        self.is_recording = True
        logger.info(f"Started live recording to: {self.output_file}")

    def get_frame(self) -> Optional[np.ndarray]:
        """Reads a single frame and writes to MP4 if recording."""
        if self.cap is None or not self.cap.isOpened():
            if not self.start_camera():
                return None

        ret, frame = self.cap.read()
        if ret and frame is not None and frame.size > 0:
            h, w = frame.shape[:2]
            self.actual_height, self.actual_width = h, w

            if self.is_recording:
                # Lazy initialize writer with exact frame dimensions to prevent dropping
                if self.writer is None and self.output_file:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    self.writer = cv2.VideoWriter(self.output_file, fourcc, self.fps, (w, h))

                if self.writer is not None:
                    self.writer.write(frame)
                    self.frames_count += 1

            return frame
        return None

    def stop_recording(self) -> Optional[str]:
        """Stops recording and finalizes MP4 file."""
        self.is_recording = False
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        return self.output_file

    def release(self):
        """Closes camera and releases all handles."""
        self.stop_recording()
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass
