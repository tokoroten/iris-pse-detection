# Copyright (c) 2026 iris_py contributors
# SPDX-License-Identifier: MIT

"""Tests for FlashDetection class."""

import numpy as np
import pytest

from iris_py.flash_detection import FlashDetection
from iris_py.configuration import Configuration
from iris_py.result import FlashResult


class TestFlashDetection:
    """Test FlashDetection class."""

    @pytest.fixture
    def detection(self):
        """Create a flash detection instance."""
        config = Configuration()
        return FlashDetection(
            fps=30,
            frame_size=(100, 100),
            config=config,
        )

    def test_initialization(self, detection):
        """Test flash detection initialization."""
        assert detection.fps == 30
        assert detection.luminance is not None
        assert detection.red_saturation is not None
        assert detection.transition_tracker is not None

    def test_set_current_frame(self, detection):
        """Test setting current frame."""
        bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        detection.set_current_frame(bgr)

        assert detection.luminance.current_frame is not None
        assert detection.red_saturation.current_frame is not None

    def test_analyse_frame_first_frame(self, detection):
        """Test analysing first frame."""
        bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        detection.set_current_frame(bgr)

        data = detection.analyse_frame(frame_pos=0)

        assert data.frame == 0
        assert data.luminance_frame_result == FlashResult.Pass
        assert data.red_frame_result == FlashResult.Pass

    def test_analyse_frame_no_flash(self, detection):
        """Test analysing frames with no flash."""
        # Two identical frames - no flash
        bgr = np.full((100, 100, 3), 128, dtype=np.uint8)

        detection.set_current_frame(bgr)
        detection.analyse_frame(frame_pos=0)

        detection.set_current_frame(bgr)
        data = detection.analyse_frame(frame_pos=1)

        assert data.luminance_frame_result == FlashResult.Pass

    def test_analyse_frame_with_flash(self, detection):
        """Test analysing frames with significant luminance change."""
        # Black frame
        black = np.zeros((100, 100, 3), dtype=np.uint8)
        detection.set_current_frame(black)
        detection.analyse_frame(frame_pos=0)

        # Simulate enough flashes to trigger failure
        for i in range(8):
            # Alternate black/white to create transitions
            if i % 2 == 0:
                frame = np.full((100, 100, 3), 255, dtype=np.uint8)
            else:
                frame = np.zeros((100, 100, 3), dtype=np.uint8)

            detection.set_current_frame(frame)
            data = detection.analyse_frame(frame_pos=i + 1)

        # Should eventually trigger a flash fail
        assert detection.flash_fail or data.luminance_transitions > 0

    def test_flash_fail_property(self, detection):
        """Test flash_fail property."""
        assert detection.flash_fail is False

    def test_extended_failure_property(self, detection):
        """Test extended_failure property."""
        assert detection.extended_failure is False

    def test_luminance_incidents_property(self, detection):
        """Test luminance_incidents property."""
        incidents = detection.luminance_incidents
        assert incidents.flash_fail_frames == 0
        assert incidents.extended_fail_frames == 0

    def test_red_incidents_property(self, detection):
        """Test red_incidents property."""
        incidents = detection.red_incidents
        assert incidents.flash_fail_frames == 0
        assert incidents.extended_fail_frames == 0


class TestFlashDetectionIntegration:
    """Integration tests for FlashDetection."""

    def test_rapid_flash_sequence(self):
        """Test detection of rapid flash sequence."""
        config = Configuration()
        detection = FlashDetection(
            fps=30,
            frame_size=(100, 100),
            config=config,
        )

        black = np.zeros((100, 100, 3), dtype=np.uint8)
        white = np.full((100, 100, 3), 255, dtype=np.uint8)

        # Initial frame
        detection.set_current_frame(black)
        detection.analyse_frame(frame_pos=0)

        # Rapid flashing - should eventually trigger failure
        transitions_detected = False
        for i in range(30):  # 1 second of rapid flashing
            frame = white if i % 2 == 0 else black
            detection.set_current_frame(frame)
            data = detection.analyse_frame(frame_pos=i + 1)

            if data.luminance_transitions > 0:
                transitions_detected = True

        assert transitions_detected

    def test_gradual_change_no_flash(self):
        """Test that gradual changes don't trigger flash."""
        config = Configuration()
        detection = FlashDetection(
            fps=30,
            frame_size=(100, 100),
            config=config,
        )

        # Gradual brightness increase over many frames
        for i in range(30):
            brightness = int(i * 8)  # 0 to 232 over 30 frames
            frame = np.full((100, 100, 3), brightness, dtype=np.uint8)
            detection.set_current_frame(frame)
            data = detection.analyse_frame(frame_pos=i)

        # Should not trigger flash fail
        assert detection.flash_fail is False
