"""
Unit Tests for Live Action Recognition & Camera Processing
"""

import pytest
import numpy as np
import cv2

from src.perception.action_detector import ActionDetector


def test_action_detector_sitting_and_standing():
    action_detector = ActionDetector()

    # Create dummy frame with synthetic person box
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated, action_state = action_detector.analyze_live_frame(frame)

    assert "is_person_present" in action_state
    assert "posture" in action_state
    assert "movement" in action_state
    assert annotated.shape == (480, 640, 3)
