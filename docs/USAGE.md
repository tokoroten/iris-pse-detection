# IRIS-PSE-Detection Usage Guide

## Basic Usage

### Command Line

```bash
# Basic analysis
iris video.mp4

# With JSON output
iris video.mp4 --json

# With pattern detection
iris video.mp4 --pattern-detection

# With frame resizing (faster processing)
iris video.mp4 --resize 0.5
```

### Python API

```python
from iris_pse_detection import VideoAnalyser, Configuration, AnalysisResult

config = Configuration()
analyser = VideoAnalyser(config)
result = analyser.analyse_video("video.mp4")

if result.overall_result == AnalysisResult.Pass:
    print("Video passed")
elif result.overall_result == AnalysisResult.Fail:
    print("Video failed")
```

---

## Understanding Results

### Overall Result (`AnalysisResult`)

| Value | Meaning |
|-------|---------|
| `Pass` | No issues detected |
| `PassWithWarning` | Minor issues detected, but within acceptable limits |
| `Fail` | Video failed the photosensitivity check |
| `LuminanceFlashFailure` | Failed due to luminance (brightness) flash |
| `LuminanceExtendedFlashFailure` | Luminance flash persisted for 4+ seconds |
| `RedFlashFailure` | Failed due to saturated red flash |
| `RedExtendedFlashFailure` | Red flash persisted for 4+ seconds |
| `PatternFailure` | Failed due to dangerous spatial pattern |

### Flash Result (Per Frame) - `FlashResult`

Used in `luminance_frame_result` and `red_frame_result`.

| Value | Code | Meaning |
|-------|------|---------|
| `Pass` | 0 | Frame is safe |
| `PassWithWarning` | 1 | Frame has minor issues |
| `ExtendedFail` | 2 | Flash failure for 4+ seconds |
| `FlashFail` | 3 | Flash detected in this frame |

### Pattern Result (Per Frame) - `PatternResult`

Used in `pattern_frame_result`.

| Value | Code | Meaning |
|-------|------|---------|
| `Pass` | 0 | No dangerous pattern |
| `Fail` | 1 | Dangerous pattern detected |

---

## Result Object Structure

```python
result = analyser.analyse_video("video.mp4")

# Overall result
result.overall_result        # AnalysisResult enum
result.total_frames          # Total number of frames analyzed
result.video_len             # Video length in seconds
result.analysis_time         # Processing time in milliseconds

# Luminance flash incidents
result.total_luminance_incidents.flash_fail_frames      # Frames with flash failure
result.total_luminance_incidents.extended_fail_frames   # Frames with extended failure
result.total_luminance_incidents.pass_with_warning_frames  # Frames with warnings
result.total_luminance_incidents.total_failed_frames    # Total failed frames

# Red flash incidents (same structure)
result.total_red_incidents.flash_fail_frames
result.total_red_incidents.extended_fail_frames
result.total_red_incidents.pass_with_warning_frames
result.total_red_incidents.total_failed_frames

# Pattern detection
result.pattern_fail_frames   # Frames with pattern failures
```

---

## Output Files

### CSV Output (`Results/video_FrameData.csv`)

Always generated. Contains per-frame analysis data.

| Column | Description |
|--------|-------------|
| `Frame` | Frame number (0-indexed) |
| `TimeStamp` | Timestamp in HH:MM:SS.ffffff format |
| `AverageLuminance` | Average luminance of the frame (0.0-1.0) |
| `FlashAreaLuminance` | Percentage of screen area with luminance flash |
| `AverageLuminanceDiff` | Luminance difference from previous frame |
| `AverageLuminanceDiffAcc` | Accumulated luminance difference |
| `AverageRed` | Average red saturation value |
| `FlashAreaRed` | Percentage of screen area with red flash |
| `AverageRedDiff` | Red difference from previous frame |
| `AverageRedDiffAcc` | Accumulated red difference |
| `LuminanceTransitions` | Number of luminance transitions in 1-second window |
| `RedTransitions` | Number of red transitions in 1-second window |
| `LuminanceExtendedFailCount` | Consecutive extended failure count (luminance) |
| `RedExtendedFailCount` | Consecutive extended failure count (red) |
| `LuminanceFrameResult` | Luminance result code (0-3) |
| `RedFrameResult` | Red result code (0-3) |
| `PatternArea` | Percentage of screen with dangerous pattern |
| `PatternDetectedLines` | Number of detected stripe lines |
| `PatternFrameResult` | Pattern result code (0-1) |

### JSON Output (`Results/video_Result.json`)

Generated when using `--json` flag or `output_json=True`.

```json
{
  "VideoLen": 10.5,
  "AnalysisTime": 1234,
  "TotalFrames": 315,
  "OverallResult": "Pass",
  "Results": ["Pass", "Pass", "PassWithWarning", ...],
  "TotalLuminanceIncidents": {
    "ExtendedFailFrames": 0,
    "FlashFailFrames": 0,
    "PassWithWarningFrames": 5,
    "TotalFailedFrames": 0
  },
  "TotalRedIncidents": {
    "ExtendedFailFrames": 0,
    "FlashFailFrames": 0,
    "PassWithWarningFrames": 0,
    "TotalFailedFrames": 0
  },
  "PatternFailFrames": 0
}
```

---

## Detection Thresholds

Default configuration values:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `luminance_flash_threshold` | 0.1 | 10% luminance change triggers detection |
| `red_flash_threshold` | 20.0 | Red saturation difference threshold |
| `area_proportion` | 0.25 | 25% of screen area must be affected |
| `max_transitions` | 4 | Maximum allowed transitions per second |
| `extended_fail_seconds` | 4 | Seconds for extended failure |

### Customizing Thresholds

```python
from iris_pse_detection import VideoAnalyser, Configuration

config = Configuration()
config.luminance_flash_threshold = 0.15  # More lenient (15%)
config.area_proportion = 0.30            # Require 30% of screen
config.pattern_detection_enabled = True  # Enable pattern detection

analyser = VideoAnalyser(config)
result = analyser.analyse_video("video.mp4")
```

---

## Interpreting Results

### Safe Video
```
Overall Result: Pass
Luminance Flash Failures: 0
Red Flash Failures: 0
```

### Video with Warnings
```
Overall Result: PassWithWarning
Luminance Warnings: 12
```
The video has some frames with borderline issues but is generally acceptable.

### Failed Video
```
Overall Result: LuminanceFlashFailure
Luminance Flash Failures: 45
Luminance Extended Failures: 12
```
The video contains dangerous flash sequences. Consider:
- Reducing brightness contrast
- Slowing down transitions
- Adding a warning before playback

---

## References

- [W3C WCAG 2.1 - Three Flashes or Below Threshold](https://www.w3.org/WAI/WCAG21/Understanding/three-flashes-or-below-threshold.html)
- [ISO 9241-391:2016 - Ergonomics of human-system interaction](https://www.iso.org/standard/56350.html)
- [Original IRIS Project](https://github.com/electronicarts/IRIS)
