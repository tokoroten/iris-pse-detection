# Copyright (c) 2026 iris_py contributors
# SPDX-License-Identifier: MIT
#
# Based on IRIS by Electronic Arts Inc.
# https://github.com/electronicarts/IRIS

"""Transition tracking for flash detection."""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque

from iris_py.frame_data import FrameData
from iris_py.result import FlashResult, TotalFlashIncidents
from iris_py.configuration import TransitionTrackerParams


@dataclass
class Counter:
    """Counter for tracking transitions within a time window."""
    count: Deque[int] = field(default_factory=lambda: deque([0]))
    current: int = 0
    passed: int = 0

    def update_current(self, new_transition: bool) -> int:
        """
        Update current frame's transition count.

        Args:
            new_transition: True if a new transition occurred

        Returns:
            Current transition count
        """
        # Handle empty count (real-time big frame drop)
        if not self.count:
            self.count.append(1 if new_transition else 0)
            self.current = 1 if new_transition else 0
            return self.current

        # Update transition count
        if new_transition:
            self.count.append(self.count[-1] + 1)
        else:
            self.count.append(self.count[-1])

        self.current = self.count[-1] - self.passed
        return self.current

    def update_passed(self) -> None:
        """Update passed count when frame exits the time window."""
        if self.count:
            self.passed = self.count[0]
            self.count.popleft()  # O(1) with deque

        # Handle empty count (real-time big frame drop)
        if not self.count:
            self.passed = 0
            self.current = 0


@dataclass
class FlashResults:
    """Possible flash result states."""
    pass_with_warning: bool = False
    flash_fail: bool = False
    extended_fail: bool = False


class TransitionTracker:
    """Tracks flash transitions and evaluates failure criteria."""

    FAIL_TIME_WINDOW = 1  # seconds
    EXTENDED_FAIL_SECONDS = 4  # seconds
    EXTENDED_FAIL_WINDOW = 5  # seconds

    def __init__(self, fps: int, params: TransitionTrackerParams):
        """
        Initialize transition tracker.

        Args:
            fps: Video frames per second
            params: Configuration parameters
        """
        self.fps = fps
        self.params = params

        # Transition counters
        self.luminance_transition_count = Counter()
        self.red_transition_count = Counter()

        # Extended failure counters
        self.luminance_extended_count = Counter()
        self.red_extended_count = Counter()

        # Results
        self.luminance_results = FlashResults()
        self.red_results = FlashResults()

        # Incident counts
        self.luminance_incidents = TotalFlashIncidents()
        self.red_incidents = TotalFlashIncidents()

        # Frame manager state (like FpsFrameManager in C++)
        # For 1-second fail window
        self._fail_max_frames = fps * self.FAIL_TIME_WINDOW
        self._fail_current_frames = 0
        self._fail_frames_to_remove = 0

        # For 5-second extended window
        self._extended_max_frames = fps * self.EXTENDED_FAIL_WINDOW
        self._extended_current_frames = 0
        self._extended_frames_to_remove = 0

    def add_frame(self) -> None:
        """
        Update frame manager state (called once per frame).

        This mimics FpsFrameManager::AddFrame in C++ - each manager
        tracks whether its window is full and returns 0 or 1 frames to remove.
        """
        # 1-second fail window manager
        if self._fail_current_frames >= self._fail_max_frames:
            self._fail_frames_to_remove = 1
        else:
            self._fail_frames_to_remove = 0
            self._fail_current_frames += 1

        # 5-second extended window manager
        if self._extended_current_frames >= self._extended_max_frames:
            self._extended_frames_to_remove = 1
        else:
            self._extended_frames_to_remove = 0
            self._extended_current_frames += 1

    def set_transitions(
        self,
        lum_transition: bool,
        red_transition: bool,
        data: FrameData,
        frame_pos: int = 0,
    ) -> None:
        """
        Update transition counts for the current frame.

        Args:
            lum_transition: True if a luminance transition occurred
            red_transition: True if a red transition occurred
            data: Frame data to update
            frame_pos: Current frame position
        """
        self._update_counters(frame_pos)

        data.luminance_transitions = self.luminance_transition_count.update_current(lum_transition)
        data.red_transitions = self.red_transition_count.update_current(red_transition)

        # Update extended failure counts
        lum_in_range = (
            self.params.min_transitions <= self.luminance_transition_count.current <= self.params.max_transitions
        )
        data.luminance_extended_fail_count = self.luminance_extended_count.update_current(lum_in_range)

        red_in_range = (
            self.params.min_transitions <= self.red_transition_count.current <= self.params.max_transitions
        )
        data.red_extended_fail_count = self.red_extended_count.update_current(red_in_range)

    def evaluate_frame_moment(self, data: FrameData) -> None:
        """
        Evaluate if the current frame fails flash criteria.

        Args:
            data: Frame data to update with results
        """
        # C++ GetCurrentFrameNum always returns maxFrames
        frames_in_four_seconds = self.fps * self.EXTENDED_FAIL_SECONDS

        # Luminance evaluation
        if self.luminance_transition_count.current > self.params.max_transitions:
            # FAIL: max allowed transitions exceeded
            self.luminance_results.flash_fail = True
            data.luminance_frame_result = FlashResult.FlashFail
            self.luminance_incidents.flash_fail_frames += 1
        elif (
            self.luminance_extended_count.current >= frames_in_four_seconds
            and self.luminance_transition_count.current >= self.params.min_transitions
        ):
            # EXTENDED FAILURE
            self.luminance_results.extended_fail = True
            data.luminance_frame_result = FlashResult.ExtendedFail
            self.luminance_incidents.extended_fail_frames += 1
        elif self.luminance_transition_count.current >= self.params.warning_transitions:
            # WARNING
            self.luminance_results.pass_with_warning = True
            data.luminance_frame_result = FlashResult.PassWithWarning
            self.luminance_incidents.pass_with_warning_frames += 1

        # Red evaluation
        if self.red_transition_count.current > self.params.max_transitions:
            # FAIL: max allowed transitions exceeded
            self.red_results.flash_fail = True
            data.red_frame_result = FlashResult.FlashFail
            self.red_incidents.flash_fail_frames += 1
        elif (
            self.red_extended_count.current >= frames_in_four_seconds
            and self.red_transition_count.current >= self.params.min_transitions
        ):
            # EXTENDED FAILURE
            self.red_results.extended_fail = True
            data.red_frame_result = FlashResult.ExtendedFail
            self.red_incidents.extended_fail_frames += 1
        elif self.red_transition_count.current >= self.params.warning_transitions:
            # WARNING
            self.red_results.pass_with_warning = True
            data.red_frame_result = FlashResult.PassWithWarning
            self.red_incidents.pass_with_warning_frames += 1

    def _update_counters(self, frame_pos: int) -> None:
        """Update counters when frames exit time windows."""
        # Remove frames that exited 1-second window (0 or 1 frame)
        one_sec_to_remove = self._fail_frames_to_remove
        while one_sec_to_remove > 0:
            self.luminance_transition_count.update_passed()
            self.red_transition_count.update_passed()
            one_sec_to_remove -= 1

        # Remove frames that exited 5-second window (0 or 1 frame)
        five_sec_to_remove = self._extended_frames_to_remove
        while five_sec_to_remove > 0:
            self.luminance_extended_count.update_passed()
            self.red_extended_count.update_passed()
            five_sec_to_remove -= 1

    @property
    def flash_fail(self) -> bool:
        return self.luminance_results.flash_fail or self.red_results.flash_fail

    @property
    def extended_failure(self) -> bool:
        return self.luminance_results.extended_fail or self.red_results.extended_fail

    @property
    def lum_pass_with_warning(self) -> bool:
        return self.luminance_results.pass_with_warning

    @property
    def red_pass_with_warning(self) -> bool:
        return self.red_results.pass_with_warning
