# Copyright (c) 2026 iris_py contributors
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

from iris_py.video_analyser import VideoAnalyser
from iris_py.configuration import Configuration
from iris_py.frame_data import FrameData
from iris_py.result import Result, AnalysisResult, FlashResult, PatternResult

__version__ = "1.0.0"
__all__ = ["VideoAnalyser", "Configuration", "FrameData", "AnalysisResult", "Result"]
