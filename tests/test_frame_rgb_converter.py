# Copyright (c) 2026 iris_py contributors
# SPDX-License-Identifier: MIT

"""Tests for FrameRgbConverter class."""

import numpy as np
import pytest

from iris_py.frame_rgb_converter import FrameRgbConverter
from iris_py.configuration import DEFAULT_SRGB_VALUES


class TestFrameRgbConverter:
    """Test FrameRgbConverter class."""

    @pytest.fixture
    def converter(self):
        """Create a converter with default sRGB values."""
        srgb = np.array(DEFAULT_SRGB_VALUES, dtype=np.float32)
        return FrameRgbConverter(srgb)

    def test_convert_black_frame(self, converter):
        """Test converting an all-black frame."""
        # Black BGR frame (all zeros)
        bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        srgb = converter.convert(bgr)

        assert srgb.shape == (10, 10, 3)
        assert srgb.dtype == np.float32
        assert np.all(srgb == 0.0)

    def test_convert_white_frame(self, converter):
        """Test converting an all-white frame."""
        # White BGR frame (all 255)
        bgr = np.full((10, 10, 3), 255, dtype=np.uint8)
        srgb = converter.convert(bgr)

        assert srgb.shape == (10, 10, 3)
        assert np.all(srgb == 1.0)

    def test_convert_preserves_shape(self, converter):
        """Test that conversion preserves frame dimensions."""
        bgr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        srgb = converter.convert(bgr)

        assert srgb.shape == bgr.shape

    def test_convert_output_range(self, converter):
        """Test that output values are in [0, 1] range."""
        bgr = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        srgb = converter.convert(bgr)

        assert np.all(srgb >= 0.0)
        assert np.all(srgb <= 1.0)

    def test_convert_monotonic(self, converter):
        """Test that higher input values produce higher output values."""
        # Create gradient image
        gradient = np.arange(256, dtype=np.uint8).reshape(1, 256, 1)
        gradient = np.tile(gradient, (1, 1, 3))
        srgb = converter.convert(gradient)

        # Check monotonicity along the gradient
        for i in range(1, 256):
            assert srgb[0, i, 0] >= srgb[0, i - 1, 0]

    def test_convert_specific_values(self, converter):
        """Test specific input/output value pairs."""
        # Create frame with specific values
        bgr = np.array([[[0, 0, 0], [128, 128, 128], [255, 255, 255]]], dtype=np.uint8)
        srgb = converter.convert(bgr)

        # Check endpoints
        assert srgb[0, 0, 0] == 0.0  # Black
        assert srgb[0, 2, 0] == 1.0  # White

        # Mid-gray should be around 0.2 (sRGB gamma)
        assert 0.1 < srgb[0, 1, 0] < 0.3
