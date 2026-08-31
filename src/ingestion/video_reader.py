"""
SentinelAI: Video Ingestion & Frame Reader Engine
Handles video metadata extraction, frame streaming, timestamp seeking, and forensic snapshotting.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Tuple, Optional, Dict, Any
from PIL import Image
from ..core.logger import logger


class VideoReader:
    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found at: {self.video_path}")

        file_size = self.video_path.stat().st_size
        if file_size < 100:
            raise ValueError(f"Video file is empty or corrupted ({file_size} bytes): {self.video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video source: {self.video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_sec = self.total_frames / self.fps if self.fps > 0 else 0.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(
            f"[bold green]Loaded Video:[/bold green] {self.video_path.name} | "
            f"Resolution: {self.width}x{self.height} | "
            f"FPS: {self.fps:.2f} | Duration: {self.format_time(self.duration_sec)}"
        )

    @staticmethod
    def format_time(seconds: float) -> str:
        """Converts seconds into HH:MM:SS format."""
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        if hrs > 0:
            return f"{hrs:02d}:{mins:02d}:{secs:02d}.{millis:03d}"
        return f"{mins:02d}:{secs:02d}.{millis:03d}"

    def get_metadata(self) -> Dict[str, Any]:
        """Returns video metadata summary."""
        return {
            "filename": self.video_path.name,
            "path": str(self.video_path.resolve()),
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "duration_sec": self.duration_sec,
            "duration_formatted": self.format_time(self.duration_sec),
        }

    def stream_sampled_frames(
        self, target_fps: float = 2.0
    ) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """
        Streams frames at a reduced sampling rate (e.g. 2 FPS instead of 30 FPS).
        Yields (frame_index, timestamp_sec, bgr_frame).
        """
        step = max(1, int(round(self.fps / target_fps)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                timestamp_sec = frame_idx / self.fps
                yield frame_idx, timestamp_sec, frame

            frame_idx += 1

    def get_frame_at_timestamp(self, timestamp_sec: float) -> Optional[np.ndarray]:
        """Seeks and extracts a single frame at an exact timestamp."""
        target_frame = int(timestamp_sec * self.fps)
        target_frame = min(max(0, target_frame), self.total_frames - 1)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None

    def save_snapshot(
        self,
        frame: np.ndarray,
        output_path: str,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        crop: bool = False
    ) -> str:
        """
        Saves a frame snapshot or cropped object region to disk for forensic evidence.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        if bbox is not None and crop:
            x1, y1, x2, y2 = map(int, bbox)
            x1 = max(0, min(x1, self.width - 1))
            y1 = max(0, min(y1, self.height - 1))
            x2 = max(x1 + 1, min(x2, self.width))
            y2 = max(y1 + 1, min(y2, self.height))
            frame_to_save = frame[y1:y2, x1:x2]
        else:
            frame_to_save = frame.copy()
            if bbox is not None:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame_to_save, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imwrite(str(out), frame_to_save)
        return str(out.resolve())

    @staticmethod
    def bgr_to_pil(frame: np.ndarray) -> Image.Image:
        """Converts OpenCV BGR image to PIL RGB Image."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def close(self):
        """Releases the video capture device."""
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
