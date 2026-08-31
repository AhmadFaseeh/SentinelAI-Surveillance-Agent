"""
Unit Tests for Video Ingestion & Motion Detection
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile

from src.ingestion.video_reader import VideoReader
from src.ingestion.motion_detector import MotionDetector
from src.ingestion.scene_segmenter import SceneSegmenter


def create_synthetic_cctv_video(file_path: str, duration_sec: int = 4, fps: int = 10):
    """Creates a dummy CCTV video with static background and a moving white box."""
    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))

    total_frames = duration_sec * fps
    for i in range(total_frames):
        # Black background
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # In seconds 1 to 3 (frames 10 to 30), draw a moving square (simulating moving object)
        if 10 <= i <= 30:
            x_pos = int(20 + (i - 10) * 8)
            cv2.rectangle(frame, (x_pos, 80), (x_pos + 40, 140), (255, 255, 255), -1)

        out.write(frame)

    out.release()


def test_video_reader():
    with tempfile.TemporaryDirectory() as tmpdir:
        vid_path = str(Path(tmpdir) / "test_cctv.mp4")
        create_synthetic_cctv_video(vid_path, duration_sec=3, fps=10)

        reader = VideoReader(vid_path)
        meta = reader.get_metadata()
        assert meta["total_frames"] == 30
        assert meta["fps"] == 10.0
        assert meta["width"] == 320

        frame = reader.get_frame_at_timestamp(1.5)
        assert frame is not None
        assert frame.shape == (240, 320, 3)
        reader.close()


def test_motion_detector():
    detector = MotionDetector()
    static_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    
    # Run through static frames to let background model settle
    for _ in range(5):
        detector.analyze_frame(static_frame)

    # Frame with new object
    moving_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.rectangle(moving_frame, (50, 50), (150, 150), (255, 255, 255), -1)

    is_motion, score, boxes = detector.analyze_frame(moving_frame)
    assert is_motion is True
    assert score > 0.0
    assert len(boxes) > 0
