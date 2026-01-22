# Copyright (c) 2026 iris_pse_detection contributors
# SPDX-License-Identifier: MIT
#
# Based on IRIS by Electronic Arts Inc.
# https://github.com/electronicarts/IRIS

"""Video analyser for photosensitivity detection."""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from iris_pse_detection.configuration import Configuration
from iris_pse_detection.flash_detection import FlashDetection
from iris_pse_detection.frame_data import FrameData
from iris_pse_detection.pattern_detection import PatternDetection
from iris_pse_detection.result import AnalysisResult, Result


@dataclass
class VideoInfo:
    """Video metadata."""
    fps: int = 0
    frame_count: int = 0
    duration: float = 0.0
    frame_size: Tuple[int, int] = (0, 0)  # (width, height)


class VideoAnalyser:
    """
    Main video analyser for photosensitivity detection.

    Analyzes video frames for flash and pattern hazards that could
    trigger photosensitive seizures.
    """

    def __init__(self, config: Optional[Configuration] = None):
        """
        Initialize video analyser.

        Args:
            config: Configuration parameters (uses defaults if None)
        """
        print("WARNING: This output report is for informational purposes only "
              "and should not be used as certification or validation of "
              "compliance with any legal, regulatory or other requirements")

        self.config = config or Configuration()
        self.video_info = VideoInfo()

        self._flash_detection: Optional[FlashDetection] = None
        self._pattern_detection: Optional[PatternDetection] = None

        # Output paths
        self._result_json_path: str = ""
        self._frame_data_json_path: str = ""
        self._frame_data_path: str = ""

    def analyse_video(
        self,
        source_video: str,
        output_json: bool = False,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Result:
        """
        Analyze a video file for photosensitivity hazards.

        Args:
            source_video: Path to video file
            output_json: If True, save results to JSON files
            progress_callback: Optional callback(progress: float) for updates

        Returns:
            Result with analysis outcome
        """
        video = cv2.VideoCapture(source_video)

        if not self._video_is_open(source_video, video):
            raise RuntimeError(f"Could not open video: {source_video}")

        try:
            self._init(source_video, output_json)
            self._set_optimal_cv_threads(self.video_info.frame_size)

            # Read first frame
            ret, frame = video.read()
            if not ret:
                raise RuntimeError("Could not read first frame")

            num_frames = 0
            last_percentage = 0

            print("Video analysis started")
            start_time = time.time()

            # Data collection
            all_frame_data: List[FrameData] = []
            non_pass_data: List[FrameData] = []

            # Open CSV file for writing
            csv_path = Path(self._frame_data_path)
            csv_path.parent.mkdir(parents=True, exist_ok=True)

            with open(csv_path, "w") as csv_file:
                csv_file.write(FrameData.csv_columns() + "\n")

                while ret and frame is not None:
                    # Create frame data
                    time_ms = int(1000.0 * num_frames / self.video_info.fps)
                    data = FrameData.create(num_frames + 1, time_ms)

                    # Resize if enabled
                    if self.config.frame_resize_enabled:
                        frame = cv2.resize(frame, self.video_info.frame_size)

                    # Analyze frame
                    self._analyse_frame(frame, num_frames, data)

                    # Write to CSV
                    csv_file.write(data.to_csv() + "\n")

                    # Collect for JSON
                    if output_json:
                        all_frame_data.append(data)
                        if (
                            int(data.luminance_frame_result) > 0
                            or int(data.red_frame_result) > 0
                            or int(data.pattern_frame_result) > 0
                        ):
                            non_pass_data.append(data)

                    # Update progress
                    num_frames += 1
                    progress = num_frames / self.video_info.frame_count * 100
                    if progress_callback:
                        progress_callback(progress)

                    if int(progress) % 10 == 0 and int(progress) != last_percentage:
                        last_percentage = int(progress)
                        print(f"Analysed {last_percentage}%")

                    # Read next frame
                    ret, frame = video.read()

            end_time = time.time()
            elapsed_ms = int((end_time - start_time) * 1000)
            print("Video analysis ended")
            print(f"Elapsed time: {elapsed_ms} ms")

            # Build result
            result = self._build_result(elapsed_ms)

            # Output overall result
            if result.overall_result == AnalysisResult.Fail:
                print("Video Overall Result: FAIL")
            elif result.overall_result == AnalysisResult.PassWithWarning:
                print("Video Overall Result: PASS WITH WARNING")
            else:
                print("Video Overall Result: PASS")

            # Write JSON if requested
            if output_json:
                self._serialize_results(result, all_frame_data, non_pass_data)

            return result

        finally:
            self._deinit()
            video.release()

    def _init(self, video_path: str, output_json: bool) -> None:
        """Initialize detection components."""
        # Apply frame resize if enabled
        frame_size = self.video_info.frame_size
        if self.config.frame_resize_enabled:
            prop = self.config.frame_resize_proportion
            print(f"Resizing frames at: {prop * 100}%")
            frame_size = (
                int(frame_size[0] * prop),
                int(frame_size[1] * prop),
            )
            self.video_info.frame_size = frame_size

        # Initialize flash detection
        # Note: frame_size for detection is (height, width)
        detection_size = (frame_size[1], frame_size[0])
        self._flash_detection = FlashDetection(
            fps=self.video_info.fps,
            frame_size=detection_size,
            config=self.config,
        )

        # Initialize pattern detection if enabled
        if self.config.pattern_detection_enabled:
            self._pattern_detection = PatternDetection(
                fps=self.video_info.fps,
                frame_size=detection_size,
                params=self.config.get_pattern_detection_params(),
            )

        # Set up output paths
        video_filename = Path(video_path).name
        results_dir = Path(self.config.results_path) / video_filename
        results_dir.mkdir(parents=True, exist_ok=True)

        self._frame_data_path = str(results_dir / "framedata.csv")

        if output_json:
            self._result_json_path = str(results_dir / "result.json")
            self._frame_data_json_path = str(results_dir / "frameData.json")

        print(f"Pattern Detection: {self.config.pattern_detection_enabled}")
        print(f"Frame Resize: {self.config.frame_resize_enabled}")
        print(f"Safe Area Proportion: {self.config.area_proportion}")
        print(f"Write json file: {output_json}")

    def _deinit(self) -> None:
        """Clean up detection components."""
        self._flash_detection = None
        self._pattern_detection = None

    def _analyse_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        data: FrameData,
    ) -> None:
        """
        Analyze a single frame.

        Args:
            frame: BGR frame from video
            frame_index: Frame number
            data: FrameData to populate with results
        """
        assert self._flash_detection is not None

        # Flash detection
        self._flash_detection.set_current_frame(frame)
        flash_data = self._flash_detection.analyse_frame(frame_index)

        # Copy flash results to data
        data.luminance_average = flash_data.luminance_average
        data.luminance_flash_area = flash_data.luminance_flash_area
        data.average_luminance_diff = flash_data.average_luminance_diff
        data.average_luminance_diff_acc = flash_data.average_luminance_diff_acc
        data.red_average = flash_data.red_average
        data.red_flash_area = flash_data.red_flash_area
        data.average_red_diff = flash_data.average_red_diff
        data.average_red_diff_acc = flash_data.average_red_diff_acc
        data.luminance_transitions = flash_data.luminance_transitions
        data.red_transitions = flash_data.red_transitions
        data.luminance_extended_fail_count = flash_data.luminance_extended_fail_count
        data.red_extended_fail_count = flash_data.red_extended_fail_count
        data.luminance_frame_result = flash_data.luminance_frame_result
        data.red_frame_result = flash_data.red_frame_result

        # Pattern detection
        if self._pattern_detection is not None and self._flash_detection is not None:
            # Get luminance frame from flash detection
            luminance = self._flash_detection.luminance.current_frame
            if luminance is not None:
                self._pattern_detection.check_frame(luminance, frame_index, data)

    def _video_is_open(self, source_video: str, video: cv2.VideoCapture) -> bool:
        """Check if video is open and get metadata."""
        if not video.isOpened():
            print(f"Error: Video {source_video} could not be opened")
            return False

        self.video_info.fps = int(video.get(cv2.CAP_PROP_FPS))
        self.video_info.frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_info.frame_size = (
            int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        self.video_info.duration = (
            self.video_info.frame_count / self.video_info.fps
            if self.video_info.fps > 0
            else 0
        )

        print(f"Video: {source_video} opened successfully")
        print(f"Video FPS: {self.video_info.fps}")
        print(f"Total frames: {self.video_info.frame_count}")
        print(f"Video resolution: {self.video_info.frame_size[0]}x{self.video_info.frame_size[1]}")
        print(f"Duration: {self.video_info.duration:.2f}s")

        return True

    def _set_optimal_cv_threads(self, frame_size: tuple) -> None:
        """Set optimal OpenCV thread count based on resolution."""
        import os

        max_threads = os.cpu_count() or 1
        height = frame_size[1]

        if height <= 1080:
            num_threads = 6
        else:
            num_threads = min(10, 6 + (height - 1080) // 270)

        threads_used = min(max_threads, num_threads)
        cv2.setNumThreads(threads_used)
        print(f"Number of threads used: {threads_used}")

    def _build_result(self, elapsed_ms: int) -> Result:
        """Build final result from detection components."""
        assert self._flash_detection is not None

        result = Result()
        result.video_len = self.video_info.duration * 1000
        result.analysis_time = elapsed_ms
        result.total_frames = self.video_info.frame_count

        # Flash results
        if self._flash_detection.flash_fail:
            result.overall_result = AnalysisResult.Fail

            # Determine specific failure type
            if self._flash_detection.luminance_incidents.flash_fail_frames > 0:
                result.results.append(AnalysisResult.LuminanceFlashFailure)
            if self._flash_detection.luminance_incidents.extended_fail_frames > 0:
                result.results.append(AnalysisResult.LuminanceExtendedFlashFailure)
            if self._flash_detection.red_incidents.flash_fail_frames > 0:
                result.results.append(AnalysisResult.RedFlashFailure)
            if self._flash_detection.red_incidents.extended_fail_frames > 0:
                result.results.append(AnalysisResult.RedExtendedFlashFailure)

        elif self._flash_detection.extended_failure:
            result.overall_result = AnalysisResult.Fail
            if self._flash_detection.luminance_incidents.extended_fail_frames > 0:
                result.results.append(AnalysisResult.LuminanceExtendedFlashFailure)
            if self._flash_detection.red_incidents.extended_fail_frames > 0:
                result.results.append(AnalysisResult.RedExtendedFlashFailure)

        elif (
            self._flash_detection.transition_tracker.lum_pass_with_warning
            or self._flash_detection.transition_tracker.red_pass_with_warning
        ):
            result.overall_result = AnalysisResult.PassWithWarning
            result.results.append(AnalysisResult.PassWithWarning)

        # Copy incident counts
        result.total_luminance_incidents = self._flash_detection.luminance_incidents
        result.total_red_incidents = self._flash_detection.red_incidents

        # Pattern results
        if self._pattern_detection is not None and self._pattern_detection.is_fail:
            result.overall_result = AnalysisResult.Fail
            result.results.append(AnalysisResult.PatternFailure)
            result.pattern_fail_frames = self._pattern_detection.pattern_fail_frames

        return result

    def _serialize_results(
        self,
        result: Result,
        all_frame_data: List[FrameData],
        non_pass_data: List[FrameData],
    ) -> None:
        """Write results to JSON files."""
        # Result JSON
        result_dict = result.to_dict()
        with open(self._result_json_path, "w") as f:
            json.dump(result_dict, f, indent=2)
        print(f"Results Json written to {self._result_json_path}")

        # Frame data JSON
        frame_data_dict = {
            "NonPassFrameData": [d.to_dict() for d in non_pass_data],
            "LineGraphFrameData": [
                {
                    "Frame": d.frame,
                    "LuminanceTransitions": d.luminance_transitions,
                    "RedTransitions": d.red_transitions,
                    "LuminanceFrameResult": int(d.luminance_frame_result),
                    "RedFrameResult": int(d.red_frame_result),
                    "PatternFrameResult": int(d.pattern_frame_result),
                }
                for d in all_frame_data
            ],
        }
        with open(self._frame_data_json_path, "w") as f:
            json.dump(frame_data_dict, f, indent=2)
        print(f"Frame Data Json written to {self._frame_data_json_path}")

    @property
    def result_json_path(self) -> str:
        """Get result JSON file path."""
        return self._result_json_path

    @property
    def frame_data_json_path(self) -> str:
        """Get frame data JSON file path."""
        return self._frame_data_json_path

    @property
    def frame_data_path(self) -> str:
        """Get frame data CSV file path."""
        return self._frame_data_path
