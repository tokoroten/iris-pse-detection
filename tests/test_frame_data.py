# Copyright (c) 2026 iris_pse_detection contributors
# SPDX-License-Identifier: MIT

"""Tests for FrameData class."""

import pytest

from iris_pse_detection.frame_data import FrameData, ms_to_timespan, proportion_to_percentage
from iris_pse_detection.result import FlashResult, PatternResult


class TestMsToTimespan:
    """Test ms_to_timespan function."""

    def test_zero(self):
        """Test zero milliseconds."""
        assert ms_to_timespan(0) == "00:00:00.000000"

    def test_one_second(self):
        """Test one second."""
        result = ms_to_timespan(1000)
        assert result.startswith("00:00:01")

    def test_one_minute(self):
        """Test one minute."""
        result = ms_to_timespan(60000)
        assert result.startswith("00:01:00")

    def test_one_hour(self):
        """Test one hour."""
        result = ms_to_timespan(3600000)
        assert result.startswith("01:00:00")

    def test_complex_time(self):
        """Test complex time value."""
        # 1 hour, 30 minutes, 45.123 seconds
        ms = 1 * 3600000 + 30 * 60000 + 45123
        result = ms_to_timespan(ms)
        assert result.startswith("01:30:45")


class TestProportionToPercentage:
    """Test proportion_to_percentage function."""

    def test_zero(self):
        """Test zero proportion."""
        assert proportion_to_percentage(0.0) == "0.00%"

    def test_half(self):
        """Test 50%."""
        assert proportion_to_percentage(0.5) == "50.00%"

    def test_full(self):
        """Test 100%."""
        assert proportion_to_percentage(1.0) == "100.00%"

    def test_quarter(self):
        """Test 25%."""
        assert proportion_to_percentage(0.25) == "25.00%"


class TestFrameData:
    """Test FrameData class."""

    def test_default_values(self):
        """Test default values."""
        data = FrameData()

        assert data.frame == 0
        assert data.luminance_average == 0.0
        assert data.red_average == 0.0
        assert data.luminance_transitions == 0
        assert data.red_transitions == 0
        assert data.luminance_frame_result == FlashResult.Pass
        assert data.red_frame_result == FlashResult.Pass
        assert data.pattern_frame_result == PatternResult.Pass

    def test_create(self):
        """Test create factory method."""
        data = FrameData.create(frame=10, time_ms=5000)

        assert data.frame == 10
        assert data.time_stamp_val == 5000
        assert "00:00:05" in data.time_stamp_ms

    def test_csv_columns(self):
        """Test CSV column headers."""
        columns = FrameData.csv_columns()

        assert "Frame" in columns
        assert "AverageLuminance" in columns
        assert "LuminanceTransitions" in columns
        assert "RedTransitions" in columns

    def test_to_csv(self):
        """Test CSV row generation."""
        data = FrameData.create(frame=1, time_ms=0)
        data.luminance_average = 0.5
        data.luminance_transitions = 3

        csv = data.to_csv()
        parts = csv.split(",")

        assert parts[0] == "1"  # frame
        assert "0.5" in parts[2]  # luminance_average

    def test_to_dict(self):
        """Test dictionary conversion."""
        data = FrameData.create(frame=5, time_ms=1000)
        data.luminance_average = 0.75
        data.luminance_transitions = 2

        d = data.to_dict()

        assert d["Frame"] == 5
        assert d["AverageLuminance"] == 0.75
        assert d["LuminanceTransitions"] == 2
