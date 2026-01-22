# Copyright (c) 2026 iris_py contributors
# SPDX-License-Identifier: MIT

"""Tests for TransitionTracker class."""

import pytest

from iris_py.transition_tracker import TransitionTracker, Counter, FlashResults
from iris_py.configuration import TransitionTrackerParams
from iris_py.frame_data import FrameData
from iris_py.result import FlashResult


class TestCounter:
    """Test Counter class."""

    def test_initial_state(self):
        """Test counter initial state."""
        counter = Counter()
        assert list(counter.count) == [0]
        assert counter.current == 0
        assert counter.passed == 0

    def test_update_current_no_transition(self):
        """Test updating with no transition."""
        counter = Counter()
        result = counter.update_current(False)

        assert result == 0
        assert list(counter.count) == [0, 0]

    def test_update_current_with_transition(self):
        """Test updating with transition."""
        counter = Counter()
        result = counter.update_current(True)

        assert result == 1
        assert list(counter.count) == [0, 1]

    def test_multiple_transitions(self):
        """Test multiple transitions accumulate."""
        counter = Counter()

        counter.update_current(True)   # 1
        counter.update_current(True)   # 2
        counter.update_current(False)  # still 2
        result = counter.update_current(True)   # 3

        assert result == 3

    def test_update_passed(self):
        """Test update_passed removes old frames."""
        counter = Counter()

        # Add some transitions
        counter.update_current(True)  # [0, 1]
        counter.update_current(True)  # [0, 1, 2]

        # Remove oldest
        counter.update_passed()

        assert counter.passed == 0  # First element was 0
        assert list(counter.count) == [1, 2]

        # Current should be adjusted
        assert counter.current == 2  # 2 - 0 = 2


class TestFlashResults:
    """Test FlashResults class."""

    def test_initial_state(self):
        """Test initial state is all False."""
        results = FlashResults()
        assert results.pass_with_warning is False
        assert results.flash_fail is False
        assert results.extended_fail is False


class TestTransitionTracker:
    """Test TransitionTracker class."""

    @pytest.fixture
    def tracker(self):
        """Create a transition tracker."""
        params = TransitionTrackerParams(
            max_transitions=6,
            min_transitions=4,
            extended_fail_seconds=4,
            extended_fail_window=5,
            warning_transitions=4,
        )
        return TransitionTracker(fps=30, params=params)

    def test_initial_state(self, tracker):
        """Test initial tracker state."""
        assert tracker.flash_fail is False
        assert tracker.extended_failure is False
        assert tracker.lum_pass_with_warning is False
        assert tracker.red_pass_with_warning is False

    def test_add_frame_updates_managers(self, tracker):
        """Test add_frame updates frame managers."""
        assert tracker._fail_current_frames == 0

        tracker.add_frame()
        assert tracker._fail_current_frames == 1
        assert tracker._fail_frames_to_remove == 0

    def test_add_frame_window_full(self, tracker):
        """Test add_frame when window is full."""
        # Fill the 1-second window (30 frames)
        for _ in range(30):
            tracker.add_frame()

        assert tracker._fail_current_frames == 30

        # Next add should set frames_to_remove
        tracker.add_frame()
        assert tracker._fail_frames_to_remove == 1

    def test_set_transitions_no_transition(self, tracker):
        """Test set_transitions with no transitions."""
        tracker.add_frame()
        data = FrameData()

        tracker.set_transitions(False, False, data, frame_pos=1)

        assert data.luminance_transitions == 0
        assert data.red_transitions == 0

    def test_set_transitions_with_transition(self, tracker):
        """Test set_transitions with transitions."""
        tracker.add_frame()
        data = FrameData()

        tracker.set_transitions(True, True, data, frame_pos=1)

        assert data.luminance_transitions == 1
        assert data.red_transitions == 1

    def test_evaluate_flash_fail(self, tracker):
        """Test evaluate_frame_moment detects flash fail."""
        # Simulate 7 transitions (> max_transitions=6)
        for i in range(7):
            tracker.add_frame()
            data = FrameData()
            tracker.set_transitions(True, False, data, frame_pos=i + 1)

        data = FrameData()
        tracker.evaluate_frame_moment(data)

        assert data.luminance_frame_result == FlashResult.FlashFail
        assert tracker.luminance_results.flash_fail is True

    def test_evaluate_warning(self, tracker):
        """Test evaluate_frame_moment detects warning."""
        # Simulate 4 transitions (= warning_transitions)
        for i in range(4):
            tracker.add_frame()
            data = FrameData()
            tracker.set_transitions(True, False, data, frame_pos=i + 1)

        data = FrameData()
        tracker.evaluate_frame_moment(data)

        assert data.luminance_frame_result == FlashResult.PassWithWarning
        assert tracker.luminance_results.pass_with_warning is True

    def test_evaluate_pass(self, tracker):
        """Test evaluate_frame_moment with passing result."""
        # Simulate 2 transitions (< warning_transitions)
        for i in range(2):
            tracker.add_frame()
            data = FrameData()
            tracker.set_transitions(True, False, data, frame_pos=i + 1)

        data = FrameData()
        tracker.evaluate_frame_moment(data)

        assert data.luminance_frame_result == FlashResult.Pass

    def test_incidents_counted(self, tracker):
        """Test that incidents are counted correctly."""
        # Trigger a flash fail
        for i in range(7):
            tracker.add_frame()
            data = FrameData()
            tracker.set_transitions(True, False, data, frame_pos=i + 1)
            tracker.evaluate_frame_moment(data)

        assert tracker.luminance_incidents.flash_fail_frames > 0
