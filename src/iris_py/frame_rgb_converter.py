# Copyright (c) 2026 iris_py contributors
# SPDX-License-Identifier: MIT
#
# Based on IRIS by Electronic Arts Inc.
# https://github.com/electronicarts/IRIS

"""Frame RGB conversion for IRIS analysis."""

import cv2
import numpy as np


class FrameRgbConverter:
    """Converts BGR frames to sRGB color space using a lookup table."""

    def __init__(self, srgb_values: np.ndarray):
        """
        Initialize the converter with sRGB lookup table.

        Args:
            srgb_values: Array of 256 sRGB values for lookup
        """
        # Create lookup table for cv2.LUT (must be 1x256 or 256x1)
        self.srgb_lut = srgb_values.astype(np.float32).reshape(1, 256)

    def convert(self, bgr_frame: np.ndarray) -> np.ndarray:
        """
        Convert BGR frame (uint8) to sRGB frame (float32).

        Uses cv2.LUT for optimized SIMD conversion, matching the
        original C++ IRIS implementation.

        Args:
            bgr_frame: Input frame in BGR format (H, W, 3) uint8

        Returns:
            sRGB frame (H, W, 3) float32
        """
        return cv2.LUT(bgr_frame, self.srgb_lut)
