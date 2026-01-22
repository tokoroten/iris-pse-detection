# Copyright (c) 2026 iris_pse_detection contributors
# SPDX-License-Identifier: MIT
#
# Based on IRIS by Electronic Arts Inc.
# https://github.com/electronicarts/IRIS

"""Flash detection base class and implementations."""

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from iris_pse_detection.configuration import FlashParams


@dataclass
class CheckTransitionResult:
    """Result of checking for a flash transition."""
    check_result: bool = False
    last_avg_diff_acc: float = 0.0


class Flash(ABC):
    """Abstract base class for flash detection (luminance and red saturation)."""

    TIME_WINDOW = 1  # 1 second window

    def __init__(
        self,
        fps: int,
        frame_size: Tuple[int, int],
        flash_params: FlashParams,
    ):
        """
        Initialize flash detector.

        Args:
            fps: Video frames per second
            frame_size: (height, width) of frames
            flash_params: Configuration parameters for flash detection
        """
        self.fps = fps
        self.params = flash_params
        self.frame_size = frame_size[0] * frame_size[1]  # total pixels
        self.safe_area = int(self.frame_size * flash_params.area_proportion)

        self.current_frame: Optional[np.ndarray] = None
        self.last_frame: Optional[np.ndarray] = None

        self.avg_current_frame: float = 0.0
        self.avg_last_frame: float = 0.0
        self.flash_area: float = 0.0

        # Track average differences within current second window
        self._avg_diff_in_second: deque = deque([0.0])

        # Frame manager state (like FpsFrameManager in C++)
        self._max_frames: int = fps * self.TIME_WINDOW  # fps frames = 1 second
        self._current_frames: int = 0
        self._frames_to_remove: int = 0

    @abstractmethod
    def set_current_frame(self, srgb_frame: np.ndarray) -> None:
        """
        Process a new frame and set it as current.

        Args:
            srgb_frame: Frame in sRGB color space (H, W, 3) float32
        """
        pass

    def _update_frames(self, flash_values_frame: np.ndarray) -> None:
        """Update frame buffers with new flash values frame."""
        self.last_frame = self.current_frame
        self.current_frame = flash_values_frame

        self.avg_last_frame = self.avg_current_frame
        self.avg_current_frame = float(np.mean(self.current_frame))

    def add_frame(self) -> None:
        """
        Update frame manager state (called once per frame).

        This mimics FpsFrameManager::AddFrame in C++.
        """
        if self._current_frames >= self._max_frames:
            self._frames_to_remove = 1
        else:
            self._frames_to_remove = 0
            self._current_frames += 1

    def _get_frames_to_remove(self) -> int:
        """Get number of frames to remove from window."""
        return self._frames_to_remove

    def _reset_manager(self) -> None:
        """Reset frame manager state (called when sign changes)."""
        self._frames_to_remove = 0
        self._current_frames = 1  # Assume we're always adding a frame when resetting

    def frame_difference(self) -> Optional[np.ndarray]:
        """Calculate difference between current and last frame."""
        if self.current_frame is None or self.last_frame is None:
            return None
        return self.current_frame - self.last_frame

    def frame_mean(self) -> float:
        """Get mean value of current frame."""
        if self.current_frame is None:
            return 0.0
        return float(np.mean(self.current_frame))

    def check_safe_area(self, frame_difference: np.ndarray) -> float:
        """
        Check if flash affects safe area threshold.

        Args:
            frame_difference: Difference between current and last frame

        Returns:
            Average difference if safe area exceeded, 0 otherwise
        """
        # Count non-zero pixels (where change occurred)
        variation = np.count_nonzero(frame_difference)
        self.flash_area = variation / self.frame_size

        if variation >= self.safe_area:
            return self.avg_current_frame - self.avg_last_frame

        return 0.0

    def _same_sign(self, num1: float, num2: float) -> bool:
        """Check if both numbers have the same sign (0 is both)."""
        return (num1 <= 0 and num2 <= 0) or (num1 >= 0 and num2 >= 0)

    def _is_flash_transition(
        self,
        last_avg_diff_acc: float,
        avg_diff_acc: float,
        threshold: float,
    ) -> bool:
        """
        Determine if a flash transition has occurred.

        Args:
            last_avg_diff_acc: Previous accumulated average difference
            avg_diff_acc: Current accumulated average difference
            threshold: Flash threshold

        Returns:
            True if a new transition occurred
        """
        # Check darker frame luminance (not applicable to red saturation)
        darker_diff = min(self.avg_last_frame, self.avg_current_frame)

        # If tendency hasn't changed and last avg was a transition
        if self._same_sign(last_avg_diff_acc, avg_diff_acc) and abs(last_avg_diff_acc) >= threshold:
            return False
        # Check if current value indicates a transition
        elif abs(avg_diff_acc) >= threshold and darker_diff < self.params.dark_threshold:
            return True

        return False

    def check_transition(
        self,
        avg_diff: float,
        last_avg_diff_acc: float,
    ) -> CheckTransitionResult:
        """
        Check for flash transition and accumulate differences.

        Args:
            avg_diff: Current frame average difference
            last_avg_diff_acc: Last accumulated average difference

        Returns:
            CheckTransitionResult with transition status
        """
        result = CheckTransitionResult(last_avg_diff_acc=last_avg_diff_acc)

        if self._same_sign(last_avg_diff_acc, avg_diff):
            # Remove excess frames from window (0 or 1 frame)
            frames_to_remove = self._get_frames_to_remove()
            while frames_to_remove > 0 and len(self._avg_diff_in_second) > 0:
                last_avg_diff_acc -= self._avg_diff_in_second[0]
                self._avg_diff_in_second.popleft()  # O(1) with deque
                frames_to_remove -= 1

            self._avg_diff_in_second.append(avg_diff)
            avg_diff += last_avg_diff_acc  # Accumulate value
        else:
            # Reset window when sign changes
            self._reset_manager()
            self._avg_diff_in_second.clear()
            self._avg_diff_in_second.append(avg_diff)

        result.check_result = self._is_flash_transition(
            result.last_avg_diff_acc, avg_diff, self.params.flash_threshold
        )
        result.last_avg_diff_acc = avg_diff

        return result

    @staticmethod
    def roundoff(value: float, precision: int = 6) -> float:
        """Round value to specified precision."""
        return round(value, precision)


class RelativeLuminance(Flash):
    """
    Relative luminance flash detection.

    Calculates relative luminance using:
    Y = 0.0722 * B + 0.7152 * G + 0.2126 * R
    """

    # sRGB to relative luminance coefficients (BGR order for OpenCV compatibility)
    # Shape (1, 3) for cv2.transform
    RGB_WEIGHTS = np.array([[0.0722, 0.7152, 0.2126]], dtype=np.float32)

    def set_current_frame(self, srgb_frame: np.ndarray) -> None:
        """
        Convert sRGB frame to relative luminance and set as current.

        Uses cv2.transform for SIMD-optimized weighted sum.

        Args:
            srgb_frame: Frame in sRGB color space (H, W, 3) float32, BGR order
        """
        # Calculate relative luminance: Y = 0.0722*B + 0.7152*G + 0.2126*R
        # cv2.transform applies matrix multiplication per pixel (SIMD optimized)
        luminance_frame = cv2.transform(srgb_frame, self.RGB_WEIGHTS)
        # Result may be (H, W) or (H, W, 1) depending on OpenCV version
        if luminance_frame.ndim == 3:
            luminance_frame = luminance_frame[:, :, 0]
        self._update_frames(luminance_frame)


class RedSaturation(Flash):
    """
    Red saturation flash detection.

    Detects saturated red pixels where R/(R+G+B) >= 0.8
    and calculates (R - G - B) * 320 as the flash value.
    """

    def set_current_frame(self, srgb_frame: np.ndarray) -> None:
        """
        Calculate red saturation values and set as current frame.

        Args:
            srgb_frame: Frame in sRGB color space (H, W, 3) float32, BGR order
        """
        # Split channels (BGR order)
        b = srgb_frame[:, :, 0]
        g = srgb_frame[:, :, 1]
        r = srgb_frame[:, :, 2]

        # Calculate red ratio: R / (R + G + B)
        total = r + g + b
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            red_ratio = np.where(total > 0, r / total, 0)

        # Create red saturation frame (default 0)
        red_sat_frame = np.zeros_like(r, dtype=np.float32)

        # Where R/(R+G+B) >= 0.8, calculate (R - G - B) * 320
        saturated_mask = red_ratio >= 0.8
        red_value = (r - g - b) * 320

        # Only positive values where saturated
        red_sat_frame = np.where(
            saturated_mask & (red_value > 0),
            red_value,
            0
        ).astype(np.float32)

        self._update_frames(red_sat_frame)
