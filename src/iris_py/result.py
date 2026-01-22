# Copyright (c) 2026 iris_py contributors
# SPDX-License-Identifier: MIT
#
# Based on IRIS by Electronic Arts Inc.
# https://github.com/electronicarts/IRIS

"""Result types and enums for IRIS analysis."""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List


class FlashResult(IntEnum):
    """Result of flash detection for a single frame."""
    Pass = 0
    PassWithWarning = 1
    ExtendedFail = 2
    FlashFail = 3


class PatternResult(IntEnum):
    """Result of pattern detection for a single frame."""
    Pass = 0
    Fail = 1


class AnalysisResult(IntEnum):
    """Overall analysis result types."""
    Fail = 0
    Pass = 1
    PassWithWarning = 2
    LuminanceFlashFailure = 3
    LuminanceExtendedFlashFailure = 4
    RedFlashFailure = 5
    RedExtendedFlashFailure = 6
    PatternFailure = 7


@dataclass
class TotalFlashIncidents:
    """Counts of frames for each incident type."""
    extended_fail_frames: int = 0
    flash_fail_frames: int = 0
    pass_with_warning_frames: int = 0

    @property
    def total_failed_frames(self) -> int:
        return self.extended_fail_frames + self.flash_fail_frames

    def to_dict(self) -> dict:
        return {
            "ExtendedFailFrames": self.extended_fail_frames,
            "FlashFailFrames": self.flash_fail_frames,
            "PassWithWarningFrames": self.pass_with_warning_frames,
            "TotalFailedFrames": self.total_failed_frames,
        }


@dataclass
class Result:
    """Overall result of video analysis."""
    video_len: float = 0.0
    analysis_time: int = 0  # milliseconds
    total_frames: int = 0
    overall_result: AnalysisResult = AnalysisResult.Pass
    results: List[AnalysisResult] = field(default_factory=list)
    total_luminance_incidents: TotalFlashIncidents = field(default_factory=TotalFlashIncidents)
    total_red_incidents: TotalFlashIncidents = field(default_factory=TotalFlashIncidents)
    pattern_fail_frames: int = 0

    def to_dict(self) -> dict:
        return {
            "VideoLen": self.video_len,
            "AnalysisTime": self.analysis_time,
            "TotalFrames": self.total_frames,
            "OverallResult": self.overall_result.name,
            "Results": [r.name for r in self.results],
            "TotalLuminanceIncidents": self.total_luminance_incidents.to_dict(),
            "TotalRedIncidents": self.total_red_incidents.to_dict(),
            "PatternFailFrames": self.pattern_fail_frames,
        }
