# Copyright (c) 2026 iris_py contributors
# SPDX-License-Identifier: MIT

"""Tests for Flash classes (RelativeLuminance and RedSaturation)."""

import numpy as np
import pytest

from iris_py.flash import RelativeLuminance, RedSaturation, Flash
from iris_py.configuration import FlashParams


class TestRelativeLuminance:
    """Test RelativeLuminance class."""

    @pytest.fixture
    def luminance_detector(self):
        """Create a luminance detector."""
        params = FlashParams(
            flash_threshold=0.1,
            area_proportion=0.25,
            dark_threshold=0.8,
        )
        return RelativeLuminance(fps=30, frame_size=(100, 100), flash_params=params)

    def test_set_current_frame_black(self, luminance_detector):
        """Test processing a black frame."""
        # Black sRGB frame
        srgb = np.zeros((100, 100, 3), dtype=np.float32)
        luminance_detector.set_current_frame(srgb)

        assert luminance_detector.current_frame is not None
        assert luminance_detector.avg_current_frame == 0.0

    def test_set_current_frame_white(self, luminance_detector):
        """Test processing a white frame."""
        # White sRGB frame
        srgb = np.ones((100, 100, 3), dtype=np.float32)
        luminance_detector.set_current_frame(srgb)

        # Luminance = 0.0722*B + 0.7152*G + 0.2126*R = 1.0
        assert np.isclose(luminance_detector.avg_current_frame, 1.0)

    def test_luminance_weights(self, luminance_detector):
        """Test that luminance uses correct RGB weights."""
        # Pure red frame
        red_frame = np.zeros((100, 100, 3), dtype=np.float32)
        red_frame[:, :, 2] = 1.0  # R channel (BGR order)
        luminance_detector.set_current_frame(red_frame)
        assert np.isclose(luminance_detector.avg_current_frame, 0.2126, atol=0.001)

        # Pure green frame
        green_frame = np.zeros((100, 100, 3), dtype=np.float32)
        green_frame[:, :, 1] = 1.0  # G channel
        luminance_detector.set_current_frame(green_frame)
        assert np.isclose(luminance_detector.avg_current_frame, 0.7152, atol=0.001)

        # Pure blue frame
        blue_frame = np.zeros((100, 100, 3), dtype=np.float32)
        blue_frame[:, :, 0] = 1.0  # B channel
        luminance_detector.set_current_frame(blue_frame)
        assert np.isclose(luminance_detector.avg_current_frame, 0.0722, atol=0.001)

    def test_frame_difference(self, luminance_detector):
        """Test frame difference calculation."""
        # First frame - black
        srgb1 = np.zeros((100, 100, 3), dtype=np.float32)
        luminance_detector.set_current_frame(srgb1)

        # Second frame - white
        srgb2 = np.ones((100, 100, 3), dtype=np.float32)
        luminance_detector.set_current_frame(srgb2)

        diff = luminance_detector.frame_difference()
        assert diff is not None
        assert np.all(diff == 1.0)  # 1.0 - 0.0 = 1.0

    def test_add_frame_manager(self, luminance_detector):
        """Test frame manager state updates."""
        assert luminance_detector._current_frames == 0

        luminance_detector.add_frame()
        assert luminance_detector._current_frames == 1
        assert luminance_detector._frames_to_remove == 0

        # Fill up the window (30 fps * 1 second = 30 frames)
        for _ in range(29):
            luminance_detector.add_frame()

        assert luminance_detector._current_frames == 30

        # Next add should start removing
        luminance_detector.add_frame()
        assert luminance_detector._frames_to_remove == 1


class TestRedSaturation:
    """Test RedSaturation class."""

    @pytest.fixture
    def red_detector(self):
        """Create a red saturation detector."""
        params = FlashParams(
            flash_threshold=20.0,
            area_proportion=0.25,
            dark_threshold=321.0,
        )
        return RedSaturation(fps=30, frame_size=(100, 100), flash_params=params)

    def test_non_red_frame(self, red_detector):
        """Test that non-red frames have zero saturation."""
        # Green frame - not red saturated
        srgb = np.zeros((100, 100, 3), dtype=np.float32)
        srgb[:, :, 1] = 1.0  # Green channel
        red_detector.set_current_frame(srgb)

        assert red_detector.avg_current_frame == 0.0

    def test_saturated_red_frame(self, red_detector):
        """Test that saturated red frames are detected."""
        # Saturated red: R/(R+G+B) >= 0.8
        # R=1.0, G=0.1, B=0.1 -> ratio = 1.0/1.2 = 0.833
        srgb = np.zeros((100, 100, 3), dtype=np.float32)
        srgb[:, :, 2] = 1.0   # R = 1.0
        srgb[:, :, 1] = 0.1   # G = 0.1
        srgb[:, :, 0] = 0.1   # B = 0.1
        red_detector.set_current_frame(srgb)

        # Should have positive red saturation value
        assert red_detector.avg_current_frame > 0

    def test_non_saturated_red_frame(self, red_detector):
        """Test that non-saturated red is not detected."""
        # R/(R+G+B) = 0.5/1.5 = 0.33 < 0.8
        srgb = np.full((100, 100, 3), 0.5, dtype=np.float32)
        red_detector.set_current_frame(srgb)

        assert red_detector.avg_current_frame == 0.0


class TestFlashBase:
    """Test Flash base class methods."""

    @pytest.fixture
    def flash_detector(self):
        """Create a luminance detector for testing base methods."""
        params = FlashParams(
            flash_threshold=0.1,
            area_proportion=0.25,
            dark_threshold=0.8,
        )
        return RelativeLuminance(fps=30, frame_size=(100, 100), flash_params=params)

    def test_check_safe_area_below_threshold(self, flash_detector):
        """Test safe area check when below threshold."""
        # Set up two similar frames
        srgb1 = np.full((100, 100, 3), 0.5, dtype=np.float32)
        flash_detector.set_current_frame(srgb1)

        # Very small change
        srgb2 = srgb1.copy()
        srgb2[0, 0, :] = 0.6  # Change only 1 pixel
        flash_detector.set_current_frame(srgb2)

        diff = flash_detector.frame_difference()
        result = flash_detector.check_safe_area(diff)

        # Should return 0 because change is below safe area threshold
        assert result == 0.0

    def test_check_safe_area_above_threshold(self, flash_detector):
        """Test safe area check when above threshold."""
        # Black frame
        srgb1 = np.zeros((100, 100, 3), dtype=np.float32)
        flash_detector.set_current_frame(srgb1)

        # White frame - 100% change
        srgb2 = np.ones((100, 100, 3), dtype=np.float32)
        flash_detector.set_current_frame(srgb2)

        diff = flash_detector.frame_difference()
        result = flash_detector.check_safe_area(diff)

        # Should return the average difference
        assert result != 0.0

    def test_same_sign(self, flash_detector):
        """Test same_sign helper method."""
        assert flash_detector._same_sign(1.0, 2.0) is True
        assert flash_detector._same_sign(-1.0, -2.0) is True
        assert flash_detector._same_sign(0.0, 1.0) is True
        assert flash_detector._same_sign(0.0, -1.0) is True
        assert flash_detector._same_sign(1.0, -1.0) is False

    def test_roundoff(self):
        """Test roundoff static method."""
        assert Flash.roundoff(0.123456789) == 0.123457
        assert Flash.roundoff(0.123456789, 2) == 0.12
