# Copyright (c) 2026 iris_pse_detection contributors
# SPDX-License-Identifier: MIT
#
# Based on IRIS by Electronic Arts Inc.
# https://github.com/electronicarts/IRIS

"""Flash detection orchestration for IRIS analysis."""

from typing import Tuple
import numpy as np

from iris_pse_detection.configuration import Configuration
from iris_pse_detection.flash import RelativeLuminance, RedSaturation, Flash
from iris_pse_detection.frame_data import FrameData, proportion_to_percentage
from iris_pse_detection.frame_rgb_converter import FrameRgbConverter
from iris_pse_detection.transition_tracker import TransitionTracker


class FlashDetection:
    """
    Orchestrates flash detection analysis.

    Combines relative luminance and red saturation detection
    with transition tracking to evaluate photosensitivity risks.
    """

    def __init__(
        self,
        fps: int,
        frame_size: Tuple[int, int],
        config: Configuration,
    ):
        """
        Initialize flash detection.

        Args:
            fps: Video frames per second
            frame_size: (height, width) of frames
            config: Configuration parameters
        """
        self.fps = fps
        self.frame_size = frame_size
        self.config = config

        # Frame converter
        self.frame_converter = FrameRgbConverter(config.srgb_values)

        # Flash detectors
        self.luminance = RelativeLuminance(
            fps=fps,
            frame_size=frame_size,
            flash_params=config.get_luminance_params(),
        )
        self.red_saturation = RedSaturation(
            fps=fps,
            frame_size=frame_size,
            flash_params=config.get_red_saturation_params(),
        )

        # Transition tracker
        self.transition_tracker = TransitionTracker(
            fps=fps,
            params=config.get_transition_tracker_params(),
        )

        # State for accumulating differences
        self._last_lum_avg_diff_acc: float = 0.0
        self._last_red_avg_diff_acc: float = 0.0
        self._frame_count: int = 0

    def set_current_frame(self, bgr_frame: np.ndarray) -> None:
        """
        Process a new frame for flash detection.

        Args:
            bgr_frame: Input frame in BGR format (H, W, 3) uint8
        """
        # Convert BGR to sRGB
        srgb_frame = self.frame_converter.convert(bgr_frame)

        # Set frame for both detectors
        self.luminance.set_current_frame(srgb_frame)
        self.red_saturation.set_current_frame(srgb_frame)

    def analyse_frame(self, frame_pos: int = 0) -> FrameData:
        """
        Analyze current frame for flash transitions.

        Args:
            frame_pos: Current frame position in video

        Returns:
            FrameData with analysis results
        """
        self._frame_count += 1

        # Calculate time in milliseconds
        time_ms = int((frame_pos / self.fps) * 1000) if self.fps > 0 else 0
        data = FrameData.create(frame_pos, time_ms)

        # Update frame managers (must be called before check_transition)
        # This mimics FpsFrameManager::AddFrame which updates all managers at once
        self.luminance.add_frame()
        self.red_saturation.add_frame()
        self.transition_tracker.add_frame()

        # Analyze luminance
        lum_diff = self.luminance.frame_difference()
        if lum_diff is not None:
            avg_lum_diff = self.luminance.check_safe_area(lum_diff)
            data.average_luminance_diff = Flash.roundoff(avg_lum_diff)
            data.luminance_flash_area = proportion_to_percentage(self.luminance.flash_area)

            lum_result = self.luminance.check_transition(
                avg_lum_diff,
                self._last_lum_avg_diff_acc,
            )
            self._last_lum_avg_diff_acc = lum_result.last_avg_diff_acc
            lum_transition = lum_result.check_result
        else:
            lum_transition = False

        data.luminance_average = Flash.roundoff(self.luminance.avg_current_frame)
        data.average_luminance_diff_acc = Flash.roundoff(self._last_lum_avg_diff_acc)

        # Analyze red saturation
        red_diff = self.red_saturation.frame_difference()
        if red_diff is not None:
            avg_red_diff = self.red_saturation.check_safe_area(red_diff)
            data.average_red_diff = Flash.roundoff(avg_red_diff)
            data.red_flash_area = proportion_to_percentage(self.red_saturation.flash_area)

            red_result = self.red_saturation.check_transition(
                avg_red_diff,
                self._last_red_avg_diff_acc,
            )
            self._last_red_avg_diff_acc = red_result.last_avg_diff_acc
            red_transition = red_result.check_result
        else:
            red_transition = False

        data.red_average = Flash.roundoff(self.red_saturation.avg_current_frame)
        data.average_red_diff_acc = Flash.roundoff(self._last_red_avg_diff_acc)

        # Only update transition tracker for frame_pos != 0 (matches C++ behavior)
        # Frame 0 has no previous frame to compare against
        if frame_pos != 0:
            # Update transition tracker
            self.transition_tracker.set_transitions(
                lum_transition,
                red_transition,
                data,
                frame_pos,
            )

            # Evaluate frame moment for failures
            self.transition_tracker.evaluate_frame_moment(data)

        return data

    @property
    def flash_fail(self) -> bool:
        """Check if any flash failure occurred."""
        return self.transition_tracker.flash_fail

    @property
    def extended_failure(self) -> bool:
        """Check if any extended failure occurred."""
        return self.transition_tracker.extended_failure

    @property
    def luminance_incidents(self):
        """Get luminance incident counts."""
        return self.transition_tracker.luminance_incidents

    @property
    def red_incidents(self):
        """Get red incident counts."""
        return self.transition_tracker.red_incidents
