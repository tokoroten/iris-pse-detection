# Copyright (c) 2026 iris_pse_detection contributors
# SPDX-License-Identifier: MIT
#
# Based on IRIS by Electronic Arts Inc.
# https://github.com/electronicarts/IRIS

"""
IRIS-Py: Photosensitive epilepsy risk detection for video content.

A Python port of EA's IRIS library for detecting:
- Luminance flashes
- Red saturation flashes
- Spatial patterns

that could potentially cause photosensitive epileptic risks.
"""

from iris_pse_detection.video_analyser import VideoAnalyser
from iris_pse_detection.configuration import Configuration
from iris_pse_detection.frame_data import FrameData
from iris_pse_detection.result import Result, AnalysisResult, FlashResult, PatternResult

__version__ = "1.0.0"
__all__ = ["VideoAnalyser", "Configuration", "FrameData", "AnalysisResult", "Result"]
