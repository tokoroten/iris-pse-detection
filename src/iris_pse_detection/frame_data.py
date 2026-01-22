# Copyright (c) 2026 iris_pse_detection contributors
# SPDX-License-Identifier: MIT
#
# Based on IRIS by Electronic Arts Inc.
# https://github.com/electronicarts/IRIS

"""Frame data structures for IRIS analysis."""

from dataclasses import dataclass
from typing import List

from iris_pse_detection.result import FlashResult, PatternResult


def ms_to_timespan(ms: int) -> str:
    """Convert milliseconds to HH:MM:SS.ffffff format."""
    seconds = (ms / 1000.0) % 60
    minutes = int((ms / (1000 * 60)) % 60)
    hours = int((ms / (1000 * 60 * 60)) % 24)
    return f"{hours:02d}:{minutes:02d}:{seconds:09.6f}"


def proportion_to_percentage(proportion: float) -> str:
    """Convert proportion (0-1) to percentage string."""
    return f"{proportion * 100:.2f}%"


@dataclass
class FrameData:
    """Data for a single analyzed frame."""
    frame: int = 0
    time_stamp_ms: str = "00:00:00.000000"
    time_stamp_val: int = 0  # milliseconds

    # Luminance data
    luminance_average: float = 0.0
    luminance_flash_area: str = "0.00%"
    average_luminance_diff: float = 0.0
    average_luminance_diff_acc: float = 0.0

    # Red saturation data
    red_average: float = 0.0
    red_flash_area: str = "0.00%"
    average_red_diff: float = 0.0
    average_red_diff_acc: float = 0.0

    # Transition counts
    luminance_transitions: int = 0
    red_transitions: int = 0

    # Extended fail counts
    luminance_extended_fail_count: int = 0
    red_extended_fail_count: int = 0

    # Results
    luminance_frame_result: FlashResult = FlashResult.Pass
    red_frame_result: FlashResult = FlashResult.Pass

    # Pattern data
    pattern_area: str = "0.00%"
    pattern_detected_lines: int = 0
    pattern_frame_result: PatternResult = PatternResult.Pass
    pattern_risk: float = 0.0

    @classmethod
    def create(cls, frame: int, time_ms: int) -> "FrameData":
        """Create a new FrameData with frame number and timestamp."""
        return cls(
            frame=frame,
            time_stamp_val=time_ms,
            time_stamp_ms=ms_to_timespan(time_ms),
        )

    def to_csv(self) -> str:
        """Convert to CSV row string."""
        return ",".join([
            str(self.frame),
            self.time_stamp_ms,
            str(self.luminance_average),
            self.luminance_flash_area,
            str(self.average_luminance_diff),
            str(self.average_luminance_diff_acc),
            str(self.red_average),
            self.red_flash_area,
            str(self.average_red_diff),
            str(self.average_red_diff_acc),
            str(self.luminance_transitions),
            str(self.red_transitions),
            str(self.luminance_extended_fail_count),
            str(self.red_extended_fail_count),
            str(int(self.luminance_frame_result)),
            str(int(self.red_frame_result)),
            self.pattern_area,
            str(self.pattern_detected_lines),
            str(int(self.pattern_frame_result)),
        ])

    @staticmethod
    def csv_columns() -> str:
        """Get CSV header row."""
        columns = [
            "Frame",
            "TimeStamp",
            "AverageLuminance",
            "FlashAreaLuminance",
            "AverageLuminanceDiff",
            "AverageLuminanceDiffAcc",
            "AverageRed",
            "FlashAreaRed",
            "AverageRedDiff",
            "AverageRedDiffAcc",
            "LuminanceTransitions",
            "RedTransitions",
            "LuminanceExtendedFailCount",
            "RedExtendedFailCount",
            "LuminanceFrameResult",
            "RedFrameResult",
            "PatternArea",
            "PatternDetectedLines",
            "PatternFrameResult",
        ]
        return ",".join(columns)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "Frame": self.frame,
            "TimeStampString": self.time_stamp_ms,
            "AverageLuminance": self.luminance_average,
            "FlashAreaLuminance": self.luminance_flash_area,
            "AverageLuminanceDiff": self.average_luminance_diff,
            "AverageLuminanceDiffAcc": self.average_luminance_diff_acc,
            "AverageRed": self.red_average,
            "FlashAreaRed": self.red_flash_area,
            "AverageRedDiff": self.average_red_diff,
            "AverageRedDiffAcc": self.average_red_diff_acc,
            "LuminanceTransitions": self.luminance_transitions,
            "RedTransitions": self.red_transitions,
            "LuminanceExtendedFailCount": self.luminance_extended_fail_count,
            "RedExtendedFailCount": self.red_extended_fail_count,
            "LuminanceFrameResult": int(self.luminance_frame_result),
            "RedFrameResult": int(self.red_frame_result),
            "PatternArea": self.pattern_area,
            "PatternDetectedLines": self.pattern_detected_lines,
            "PatternFrameResult": int(self.pattern_frame_result),
        }
