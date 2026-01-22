# IRIS-PSE-Detection Examples

## basic_usage.py

Basic example showing how to analyze a video and access results.

```bash
python basic_usage.py
```

## with_progress.py

Example with custom configuration, progress callback, and frame resizing.

```bash
python with_progress.py
```

## CLI Usage

```bash
# Basic analysis
iris video.mp4

# With JSON output
iris video.mp4 --json

# With pattern detection
iris video.mp4 --pattern-detection

# With frame resizing (50%)
iris video.mp4 --resize 0.5

# Custom output directory
iris video.mp4 -o my_results/
```
