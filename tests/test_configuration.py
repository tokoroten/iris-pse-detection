# Copyright (c) 2026 iris_py contributors
# SPDX-License-Identifier: MIT

"""Tests for Configuration class."""

import numpy as np
import pytest

from iris_py.configuration import (
    Configuration,
    FlashParams,
    TransitionTrackerParams,
    PatternDetectionParams,
    DEFAULT_SRGB_VALUES,
)


class TestConfiguration:
    """Test Configuration class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = Configuration()

        # Luminance defaults
        assert config.luminance_flash_threshold == 0.1
        assert config.luminance_dark_threshold == 0.8

        # Red saturation defaults
        assert config.red_flash_threshold == 20.0
        assert config.red_dark_threshold == 321.0

        # Area proportion
        assert config.area_proportion == 0.25

        # Transition tracker defaults
        assert config.max_transitions == 6
        assert config.min_transitions == 4
        assert config.extended_fail_seconds == 4
        assert config.extended_fail_window == 5
        assert config.warning_transitions == 4

        # Pattern detection defaults
        assert config.pattern_detection_enabled is False

    def test_srgb_values_loaded(self):
        """Test sRGB lookup table is loaded correctly."""
        config = Configuration()

        assert len(config.srgb_values) == 256
        assert config.srgb_values[0] == 0.0
        assert config.srgb_values[255] == 1.0
        assert config.srgb_values.dtype == np.float32

    def test_get_luminance_params(self):
        """Test getting luminance parameters."""
        config = Configuration()
        params = config.get_luminance_params()

        assert isinstance(params, FlashParams)
        assert params.flash_threshold == 0.1
        assert params.area_proportion == 0.25
        assert params.dark_threshold == 0.8

    def test_get_red_saturation_params(self):
        """Test getting red saturation parameters."""
        config = Configuration()
        params = config.get_red_saturation_params()

        assert isinstance(params, FlashParams)
        assert params.flash_threshold == 20.0
        assert params.area_proportion == 0.25
        assert params.dark_threshold == 321.0

    def test_get_transition_tracker_params(self):
        """Test getting transition tracker parameters."""
        config = Configuration()
        params = config.get_transition_tracker_params()

        assert isinstance(params, TransitionTrackerParams)
        assert params.max_transitions == 6
        assert params.min_transitions == 4

    def test_get_pattern_detection_params(self):
        """Test getting pattern detection parameters."""
        config = Configuration()
        params = config.get_pattern_detection_params()

        assert isinstance(params, PatternDetectionParams)
        assert params.min_stripes == 6


class TestDefaultSrgbValues:
    """Test DEFAULT_SRGB_VALUES constant."""

    def test_length(self):
        """Test sRGB LUT has 256 entries."""
        assert len(DEFAULT_SRGB_VALUES) == 256

    def test_range(self):
        """Test sRGB values are in valid range."""
        assert all(0 <= v <= 1 for v in DEFAULT_SRGB_VALUES)

    def test_monotonic(self):
        """Test sRGB values are monotonically increasing."""
        for i in range(1, len(DEFAULT_SRGB_VALUES)):
            assert DEFAULT_SRGB_VALUES[i] >= DEFAULT_SRGB_VALUES[i - 1]

    def test_endpoints(self):
        """Test sRGB LUT endpoints."""
        assert DEFAULT_SRGB_VALUES[0] == 0
        assert DEFAULT_SRGB_VALUES[255] == 1
