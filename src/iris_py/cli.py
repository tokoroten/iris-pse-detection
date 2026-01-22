# Copyright (c) 2026 iris_py contributors
# SPDX-License-Identifier: MIT
#
# Based on IRIS by Electronic Arts Inc.
# https://github.com/electronicarts/IRIS

"""Command-line interface for IRIS photosensitivity analysis."""

import argparse
import sys
from pathlib import Path

from iris_py.configuration import Configuration
from iris_py.video_analyser import VideoAnalyser


def main():
    """Main entry point for IRIS CLI."""
    parser = argparse.ArgumentParser(
        prog="iris",
        description="IRIS - Photosensitivity video analysis tool",
        epilog="This output report is for informational purposes only "
               "and should not be used as certification or validation of "
               "compliance with any legal, regulatory or other requirements.",
    )

    parser.add_argument(
        "video",
        help="Path to video file to analyze",
    )

    parser.add_argument(
        "-c", "--config",
        help="Path to directory containing appsettings.json",
        default=".",
    )

    parser.add_argument(
        "-j", "--json",
        action="store_true",
        help="Output results to JSON files",
    )

    parser.add_argument(
        "-o", "--output",
        help="Output directory for results (default: Results/)",
        default=None,
    )

    parser.add_argument(
        "--pattern-detection",
        action="store_true",
        help="Enable pattern detection (disabled by default)",
    )

    parser.add_argument(
        "--resize",
        type=float,
        help="Resize frame proportion (e.g., 0.5 for 50%%)",
        default=None,
    )

    args = parser.parse_args()

    # Validate video file
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    # Load configuration
    config = Configuration.from_json(args.config)

    # Override config with CLI arguments
    if args.output:
        config.results_path = args.output
        if not config.results_path.endswith("/"):
            config.results_path += "/"

    if args.pattern_detection:
        config.pattern_detection_enabled = True

    if args.resize is not None:
        config.frame_resize_enabled = True
        config.frame_resize_proportion = args.resize

    # Create analyser and run
    analyser = VideoAnalyser(config)

    try:
        result = analyser.analyse_video(
            source_video=str(video_path),
            output_json=args.json,
        )

        # Print summary
        print("\n" + "=" * 50)
        print("ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"Overall Result: {result.overall_result.name}")
        print(f"Total Frames: {result.total_frames}")
        print(f"Analysis Time: {result.analysis_time} ms")

        if result.total_luminance_incidents.total_failed_frames > 0:
            print(f"\nLuminance Flash Failures: {result.total_luminance_incidents.flash_fail_frames}")
            print(f"Luminance Extended Failures: {result.total_luminance_incidents.extended_fail_frames}")
            print(f"Luminance Warnings: {result.total_luminance_incidents.pass_with_warning_frames}")

        if result.total_red_incidents.total_failed_frames > 0:
            print(f"\nRed Flash Failures: {result.total_red_incidents.flash_fail_frames}")
            print(f"Red Extended Failures: {result.total_red_incidents.extended_fail_frames}")
            print(f"Red Warnings: {result.total_red_incidents.pass_with_warning_frames}")

        if result.pattern_fail_frames > 0:
            print(f"\nPattern Fail Frames: {result.pattern_fail_frames}")

        print("\nOutput files:")
        print(f"  CSV: {analyser.frame_data_path}")
        if args.json:
            print(f"  Result JSON: {analyser.result_json_path}")
            print(f"  Frame Data JSON: {analyser.frame_data_json_path}")

        # Exit with appropriate code
        from iris_py.result import AnalysisResult
        if result.overall_result == AnalysisResult.Fail:
            sys.exit(1)
        elif result.overall_result == AnalysisResult.PassWithWarning:
            sys.exit(2)
        else:
            sys.exit(0)

    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()
