#!/usr/bin/env python3
"""Example with progress callback and custom configuration."""

import sys
from iris_pse_detection import VideoAnalyser, Configuration


def progress_callback(progress: float) -> None:
    """Print progress bar."""
    bar_length = 40
    filled = int(bar_length * progress / 100)
    bar = "=" * filled + "-" * (bar_length - filled)
    sys.stdout.write(f"\rProgress: [{bar}] {progress:.1f}%")
    sys.stdout.flush()
    if progress >= 100:
        print()


def main():
    # Custom configuration
    config = Configuration()

    # Adjust thresholds (defaults shown)
    config.luminance_flash_threshold = 0.1  # 10% luminance change
    config.red_flash_threshold = 20.0
    config.area_proportion = 0.25  # 25% of screen area

    # Enable pattern detection
    config.pattern_detection_enabled = True

    # Enable frame resizing for faster processing
    config.frame_resize_enabled = True
    config.frame_resize_proportion = 0.5  # 50% of original size

    # Set output directory
    config.results_path = "my_results/"

    # Create analyser
    analyser = VideoAnalyser(config)

    # Analyze with progress callback
    result = analyser.analyse_video(
        source_video="your_video.mp4",
        output_json=True,
        progress_callback=progress_callback,
    )

    print(f"\nResult: {result.overall_result.name}")
    print(f"Output files saved to: {config.results_path}")


if __name__ == "__main__":
    main()
