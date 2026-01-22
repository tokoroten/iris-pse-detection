#!/usr/bin/env python3
"""Basic usage example for IRIS-Py."""

from iris_pse_detection import VideoAnalyser, Configuration
from iris_pse_detection.result import AnalysisResult


def main():
    # Create configuration (uses defaults)
    config = Configuration()

    # Or customize configuration
    # config.pattern_detection_enabled = True
    # config.frame_resize_enabled = True
    # config.frame_resize_proportion = 0.5

    # Create analyser
    analyser = VideoAnalyser(config)

    # Analyze video
    result = analyser.analyse_video(
        source_video="your_video.mp4",
        output_json=True,  # Save results to JSON
    )

    # Check result
    if result.overall_result == AnalysisResult.Pass:
        print("Video passed photosensitivity check")
    elif result.overall_result == AnalysisResult.PassWithWarning:
        print("Video passed with warnings")
    else:
        print("Video FAILED photosensitivity check")

    # Access detailed results
    print(f"\nTotal frames: {result.total_frames}")
    print(f"Analysis time: {result.analysis_time} ms")

    # Luminance flash incidents
    lum = result.total_luminance_incidents
    print(f"\nLuminance incidents:")
    print(f"  Flash failures: {lum.flash_fail_frames}")
    print(f"  Extended failures: {lum.extended_fail_frames}")
    print(f"  Warnings: {lum.pass_with_warning_frames}")

    # Red flash incidents
    red = result.total_red_incidents
    print(f"\nRed flash incidents:")
    print(f"  Flash failures: {red.flash_fail_frames}")
    print(f"  Extended failures: {red.extended_fail_frames}")
    print(f"  Warnings: {red.pass_with_warning_frames}")


if __name__ == "__main__":
    main()
