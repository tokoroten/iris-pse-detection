# IRIS-PSE-Detection

[![PyPI version](https://badge.fury.io/py/iris-pse-detection.svg)](https://pypi.org/project/iris-pse-detection/)
[![CI](https://github.com/tokoroten/iris-pse-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/tokoroten/iris-pse-detection/actions/workflows/ci.yml)

Python port of [IRIS](https://github.com/electronicarts/IRIS) - Electronic Arts' photosensitive epilepsy risk detection library.

IRIS analyzes video content to detect flash patterns that may trigger seizures in people with photosensitive epilepsy, aiming to implement detection methods described in W3C WCAG and ISO 9241-391.

## Installation

```bash
pip install iris-pse-detection
```

Or for development:

```bash
git clone https://github.com/tokoroten/iris-pse-detection
cd iris-pse-detection
uv sync
```

## Usage

```bash
iris video.mp4
```

Or with Python:

```python
from iris_pse_detection import VideoAnalyser, Configuration

config = Configuration()
analyser = VideoAnalyser(config)
result = analyser.analyse_video("video.mp4")
print(result.overall_result)
```

## Features

- Luminance flash detection
- Red saturation flash detection
- Transition tracking with 1-second sliding window
- Extended failure detection (4+ seconds)
- Pattern detection (optional)

## Note on Accuracy

Due to floating-point precision differences between C++ and Python/NumPy, results may vary slightly from the original IRIS implementation. These differences are minimal and occur at boundary conditions where values are very close to detection thresholds.

## Acknowledgments

This project is a Python port of [IRIS](https://github.com/electronicarts/IRIS) by Electronic Arts Inc., originally released under the BSD 3-Clause License.

## Disclaimer

This software is provided for informational purposes only and should not be used as the sole method for ensuring video content safety. While this implementation is based on the original IRIS library, no guarantee is made regarding accuracy or compliance with any standards. Always consult with accessibility experts and conduct proper testing for production use.

## Links

- [PyPI Package](https://pypi.org/project/iris-pse-detection/)
- [GitHub Repository](https://github.com/tokoroten/iris-pse-detection)
- [Original IRIS (C++)](https://github.com/electronicarts/IRIS)

## License

MIT License - see [LICENSE](LICENSE) for details.
