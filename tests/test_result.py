# Copyright (c) 2026 iris_py contributors
# SPDX-License-Identifier: MIT

"""Tests for Result types and enums."""

import pytest

from iris_py.result import (
    FlashResult,
    PatternResult,
    AnalysisResult,
    TotalFlashIncidents,
    Result,
)


class TestFlashResult:
    """Test FlashResult enum."""

    def test_values(self):
        """Test enum values."""
        assert FlashResult.Pass == 0
        assert FlashResult.PassWithWarning == 1
        assert FlashResult.ExtendedFail == 2
        assert FlashResult.FlashFail == 3

    def test_comparison(self):
        """Test enum comparison."""
        assert FlashResult.Pass < FlashResult.FlashFail
        assert FlashResult.FlashFail > FlashResult.Pass


class TestPatternResult:
    """Test PatternResult enum."""

    def test_values(self):
        """Test enum values."""
        assert PatternResult.Pass == 0
        assert PatternResult.Fail == 1


class TestAnalysisResult:
    """Test AnalysisResult enum."""

    def test_values(self):
        """Test enum values."""
        assert AnalysisResult.Fail == 0
        assert AnalysisResult.Pass == 1
        assert AnalysisResult.PassWithWarning == 2
        assert AnalysisResult.LuminanceFlashFailure == 3
        assert AnalysisResult.LuminanceExtendedFlashFailure == 4
        assert AnalysisResult.RedFlashFailure == 5
        assert AnalysisResult.RedExtendedFlashFailure == 6
        assert AnalysisResult.PatternFailure == 7


class TestTotalFlashIncidents:
    """Test TotalFlashIncidents class."""

    def test_default_values(self):
        """Test default values."""
        incidents = TotalFlashIncidents()
        assert incidents.extended_fail_frames == 0
        assert incidents.flash_fail_frames == 0
        assert incidents.pass_with_warning_frames == 0

    def test_total_failed_frames(self):
        """Test total_failed_frames property."""
        incidents = TotalFlashIncidents(
            extended_fail_frames=5,
            flash_fail_frames=10,
        )
        assert incidents.total_failed_frames == 15

    def test_to_dict(self):
        """Test to_dict method."""
        incidents = TotalFlashIncidents(
            extended_fail_frames=5,
            flash_fail_frames=10,
            pass_with_warning_frames=3,
        )
        d = incidents.to_dict()

        assert d["ExtendedFailFrames"] == 5
        assert d["FlashFailFrames"] == 10
        assert d["PassWithWarningFrames"] == 3
        assert d["TotalFailedFrames"] == 15


class TestResult:
    """Test Result class."""

    def test_default_values(self):
        """Test default values."""
        result = Result()
        assert result.video_len == 0.0
        assert result.analysis_time == 0
        assert result.total_frames == 0
        assert result.overall_result == AnalysisResult.Pass
        assert result.results == []
        assert result.pattern_fail_frames == 0

    def test_with_values(self):
        """Test with custom values."""
        result = Result(
            video_len=10000.0,
            analysis_time=5000,
            total_frames=300,
            overall_result=AnalysisResult.Fail,
            results=[AnalysisResult.LuminanceFlashFailure],
        )

        assert result.video_len == 10000.0
        assert result.analysis_time == 5000
        assert result.total_frames == 300
        assert result.overall_result == AnalysisResult.Fail

    def test_to_dict(self):
        """Test to_dict method."""
        result = Result(
            video_len=10000.0,
            analysis_time=5000,
            total_frames=300,
            overall_result=AnalysisResult.Fail,
            results=[AnalysisResult.LuminanceFlashFailure],
        )
        d = result.to_dict()

        assert d["VideoLen"] == 10000.0
        assert d["AnalysisTime"] == 5000
        assert d["TotalFrames"] == 300
        assert d["OverallResult"] == "Fail"
        assert "LuminanceFlashFailure" in d["Results"]
