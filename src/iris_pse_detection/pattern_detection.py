# Copyright (c) 2026 iris_pse_detection contributors
# SPDX-License-Identifier: MIT
#
# Based on IRIS by Electronic Arts Inc.
# https://github.com/electronicarts/IRIS

"""Pattern detection for IRIS analysis using Fourier transform."""

from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np

from iris_pse_detection.configuration import PatternDetectionParams
from iris_pse_detection.frame_data import FrameData, proportion_to_percentage
from iris_pse_detection.result import PatternResult


@dataclass
class Pattern:
    """Detected pattern information."""
    area: int = 0
    n_components: int = 0
    avg_dark_luminance: float = 0.0
    avg_light_luminance: float = 0.0


@dataclass
class PatternCounter:
    """Counter for tracking pattern frames within time window."""
    count: List[int] = field(default_factory=lambda: [0])
    current: int = 0
    passed: int = 0

    def update_current(self, add: bool) -> None:
        """Update current frame's pattern count."""
        if not self.count:
            self.count.append(1 if add else 0)
            self.current = 1 if add else 0
            return

        if add:
            self.count.append(self.count[-1] + 1)
        else:
            self.count.append(self.count[-1])

        self.current = self.count[-1] - self.passed

    def update_passed(self) -> None:
        """Update passed count when frame exits time window."""
        if self.count:
            self.passed = self.count[0]
            self.count.pop(0)

        if not self.count:
            self.passed = 0
            self.current = 0


class FourierTransform:
    """Fourier transform operations for pattern detection."""

    def __init__(self, center_point: Tuple[int, int]):
        """
        Initialize with center point for FFT operations.

        Args:
            center_point: (x, y) center of the frame
        """
        self.center_point = center_point

    def get_psd(self, src: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get power spectral density components.

        Args:
            src: Input grayscale image

        Returns:
            Tuple of (magnitude, phase, power_spectrum)
        """
        dft = self._get_dft(src)
        magnitude, phase = self._get_dft_components(dft)

        # Compute PSD
        power_spectrum = magnitude.copy()
        power_spectrum = self._normalize(power_spectrum, -1.0, 1.0)
        power_spectrum = np.abs(power_spectrum)
        power_spectrum = 1 - power_spectrum
        power_spectrum = np.power(power_spectrum, 2)
        power_spectrum = self._log_transform(power_spectrum)
        power_spectrum = self._normalize(power_spectrum, 0, 255)

        return magnitude, phase, power_spectrum

    def get_peaks(self, psd: np.ndarray) -> np.ndarray:
        """
        Get threshold peaks from power spectrum.

        Args:
            psd: Power spectral density

        Returns:
            Binary thresholded peaks
        """
        thresh_psd = psd.astype(np.uint8)
        _, result = cv2.threshold(thresh_psd, 7, 255, cv2.THRESH_OTSU)
        return result  # type: ignore[return-value]

    def filter_magnitude(self, peaks: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
        """
        Filter magnitude using peaks mask.

        Args:
            peaks: Binary peaks mask
            magnitude: DFT magnitude

        Returns:
            Filtered magnitude
        """
        # Prepare mask
        peaks = self._fft_shift(peaks)
        peaks = peaks.astype(np.uint8)
        cv2.circle(peaks, self.center_point, 5, 0, -1)
        peaks = self._fft_shift(peaks)

        # Apply mask
        magnitude = magnitude.copy()
        magnitude[peaks > 0] = 0
        return magnitude

    def get_ift(self, magnitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
        """
        Get inverse Fourier transform.

        Args:
            magnitude: DFT magnitude
            phase: DFT phase

        Returns:
            Reconstructed image
        """
        # Recover complex DFT from magnitude and phase
        real = magnitude * np.cos(phase)
        imag = magnitude * np.sin(phase)

        dft = np.stack([real, imag], axis=-1).astype(np.float32)

        # Inverse DFT
        ift = cv2.idft(dft, flags=cv2.DFT_REAL_OUTPUT)
        ift = np.clip(ift * 255.0, 0, 255).astype(np.uint8)

        return ift

    def _get_dft(self, src: np.ndarray) -> np.ndarray:
        """Get DFT of source image."""
        # Pad to optimal size
        m = cv2.getOptimalDFTSize(src.shape[0])
        n = cv2.getOptimalDFTSize(src.shape[1])
        padded = cv2.copyMakeBorder(src, 0, m - src.shape[0], 0, n - src.shape[1],
                                     cv2.BORDER_CONSTANT, value=0)

        # Convert to float
        padded = padded.astype(np.float32) / 255.0

        # DFT
        dft = cv2.dft(padded, flags=cv2.DFT_SCALE | cv2.DFT_COMPLEX_OUTPUT)
        return dft

    def _get_dft_components(self, dft: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get magnitude and phase from DFT."""
        real = dft[:, :, 0]
        imag = dft[:, :, 1]
        magnitude, phase = cv2.cartToPolar(real, imag)
        return magnitude, phase

    def _fft_shift(self, src: np.ndarray) -> np.ndarray:
        """Shift FFT quadrants to center."""
        # Crop to even dimensions
        rows, cols = src.shape[:2]
        rows_even = rows & -2
        cols_even = cols & -2
        aux = src[:rows_even, :cols_even].copy()

        cx = cols_even // 2
        cy = rows_even // 2

        # Swap quadrants
        q0 = aux[:cy, :cx].copy()  # Top-Left
        q1 = aux[:cy, cx:].copy()  # Top-Right
        q2 = aux[cy:, :cx].copy()  # Bottom-Left
        q3 = aux[cy:, cx:].copy()  # Bottom-Right

        aux[:cy, :cx] = q3
        aux[cy:, cx:] = q0
        aux[:cy, cx:] = q2
        aux[cy:, :cx] = q1

        result = src.copy()
        result[:rows_even, :cols_even] = aux
        return result

    def _log_transform(self, src: np.ndarray) -> np.ndarray:
        """Apply logarithmic transform."""
        return np.log(src + 1)

    def _normalize(self, src: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        """Normalize to specified range."""
        result = cv2.normalize(src, None, min_val, max_val, cv2.NORM_MINMAX)  # type: ignore[call-overload]
        return result.astype(np.float32)


class PatternDetection:
    """
    Pattern detection using Fourier transform analysis.

    Detects harmful repetitive patterns that could trigger photosensitive
    seizures (e.g., stripes, checkerboards).
    """

    def __init__(
        self,
        fps: int,
        frame_size: Tuple[int, int],
        params: PatternDetectionParams,
    ):
        """
        Initialize pattern detection.

        Args:
            fps: Video frames per second
            frame_size: (height, width) of frames
            params: Pattern detection parameters
        """
        self.fps = fps
        self.params = params

        # Scale down if resolution is high
        height, width = frame_size
        if width > 480:
            scale_percent = 50
            self.scale_size = (
                int(width * scale_percent / 100),
                int(height * scale_percent / 100),
            )
        else:
            self.scale_size = (width, height)

        scaled_area = self.scale_size[0] * self.scale_size[1]
        self.safe_area = int(scaled_area * params.area_proportion)
        self.threshold_area = int(scaled_area * 0.20)
        self.diff_threshold = int(scaled_area * 0.1)
        self.contour_thresh_area = int(scaled_area * 0.00155)
        self.frame_size_scaled = scaled_area

        # Time window tracking
        self.frame_time_thresh = int(fps * params.time_threshold)
        self.pattern_frame_count = PatternCounter()
        self.pattern_frame_count.count = [0]

        # Center point for FFT
        self.center_point = (self.scale_size[0] // 2, self.scale_size[1] // 2)

        # Results
        self._is_fail = False
        self.pattern_fail_frames = 0
        self._frame_count = 0

    def check_frame(
        self,
        luminance_frame: np.ndarray,
        frame_pos: int,
        data: FrameData,
    ) -> None:
        """
        Check a frame for harmful patterns.

        Args:
            luminance_frame: Luminance values (H, W) float32
            frame_pos: Current frame position
            data: FrameData to update with results
        """
        self._frame_count += 1
        pattern = self._detect_pattern(luminance_frame, frame_pos)

        harmful = False
        if (
            pattern.area >= self.safe_area
            and pattern.n_components >= self.params.min_stripes
            and pattern.avg_light_luminance >= 0.25
        ):
            harmful = True
            data.pattern_area = proportion_to_percentage(
                pattern.area / self.frame_size_scaled
            )
            data.pattern_detected_lines = pattern.n_components

        self.pattern_frame_count.update_current(harmful)
        self._check_frame_count(data)

    def _detect_pattern(self, luminance_frame: np.ndarray, frame_pos: int) -> Pattern:
        """Detect pattern in luminance frame."""
        # Resize to scale size
        luminance = cv2.resize(
            luminance_frame,
            self.scale_size,
            interpolation=cv2.INTER_LINEAR,
        )

        # Normalize to 8-bit
        luminance_8uc = cv2.normalize(luminance, None, 0, 255, cv2.NORM_MINMAX)  # type: ignore[call-overload]
        luminance_8uc = luminance_8uc.astype(np.uint8)

        # Check if pattern exists
        has_pattern, ift_thresh = self._has_pattern(luminance_8uc)

        if has_pattern:
            pattern_region, n_components = self._get_pattern_region(
                ift_thresh, luminance_8uc
            )

            if n_components != -1:
                pattern = Pattern()
                pattern.n_components = n_components
                pattern.area = int(np.count_nonzero(pattern_region))
                self._set_pattern_luminance(
                    pattern, pattern_region, luminance_8uc, luminance
                )
                return pattern

        return Pattern()

    def _has_pattern(self, luminance_frame: np.ndarray) -> Tuple[bool, np.ndarray]:
        """Check if frame has a repetitive pattern."""
        ft = FourierTransform(self.center_point)
        magnitude, phase, power_spectrum = ft.get_psd(luminance_frame)
        peaks = ft.get_peaks(power_spectrum)
        magnitude = ft.filter_magnitude(peaks, magnitude)

        # Reconstruct image
        ift = ft.get_ift(magnitude, phase)

        # Highlight pattern area
        ift_thresh = self._highlight_pattern_area(ift, luminance_frame)

        # Check if area threshold reached
        if np.count_nonzero(ift_thresh) < self.diff_threshold:
            return False, ift_thresh

        return True, ift_thresh

    def _highlight_pattern_area(
        self, ift: np.ndarray, luminance_frame: np.ndarray
    ) -> np.ndarray:
        """Highlight pattern area by comparing original and reconstructed."""
        if ift.shape != luminance_frame.shape:
            ift = cv2.resize(ift, (luminance_frame.shape[1], luminance_frame.shape[0]))

        abs_diff = cv2.absdiff(ift, luminance_frame)
        _, thresh = cv2.threshold(abs_diff, 50, 255, cv2.THRESH_BINARY)
        return thresh

    def _get_pattern_region(
        self, thresh_ift: np.ndarray, luminance_frame: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        """Get pattern region mask and component count."""
        # Find contours
        thresh_contours = self._get_contours(thresh_ift)
        contours_mat = self._move_bigger_contours(thresh_contours, thresh_ift)

        # Get contours from processed image
        dilation_contours = self._get_contours(contours_mat)

        if not dilation_contours:
            return np.zeros_like(thresh_ift), -1

        # Get pattern contour
        pattern_contour, pattern_components = self._get_pattern_contour(dilation_contours)

        # Get bounding rect
        rect = cv2.minAreaRect(np.array(pattern_contour))
        box = cv2.boxPoints(rect).astype(np.int32)

        # Create pattern region mask
        pattern_region_mask = np.zeros(thresh_ift.shape, dtype=np.uint8)
        cv2.drawContours(pattern_region_mask, [box], -1, 255, cv2.FILLED)

        # Apply mask to luminance frame (modifies in place like C++ version)
        cv2.bitwise_and(luminance_frame, pattern_region_mask, luminance_frame)

        return pattern_region_mask, pattern_components

    def _set_pattern_luminance(
        self,
        pattern: Pattern,
        pattern_region: np.ndarray,
        luminance_8uc: np.ndarray,
        luminance_frame: np.ndarray,
    ) -> None:
        """Calculate luminance of pattern components."""
        _, light_components = cv2.threshold(
            luminance_8uc, 0, 255, cv2.THRESH_OTSU
        )

        dark_components = cv2.bitwise_not(light_components)
        dark_components = cv2.bitwise_and(dark_components, pattern_region)

        if np.count_nonzero(light_components) > 0:
            pattern.avg_light_luminance = float(
                np.mean(luminance_frame[light_components > 0])
            )

        if np.count_nonzero(dark_components) > 0:
            pattern.avg_dark_luminance = float(
                np.mean(luminance_frame[dark_components > 0])
            )

    def _get_contours(self, src: np.ndarray) -> List[np.ndarray]:
        """Get contours from binary image."""
        contours, _ = cv2.findContours(
            src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return list(contours)

    def _move_bigger_contours(
        self, contours: List[np.ndarray], src: np.ndarray
    ) -> np.ndarray:
        """Keep only contours larger than threshold."""
        contours_mat = np.zeros(src.shape, dtype=np.uint8)

        for contour in contours:
            if cv2.contourArea(contour) > self.contour_thresh_area:
                cv2.fillConvexPoly(contours_mat, contour, 255)

        return contours_mat

    def _get_biggest_contour(self, contours: List[np.ndarray]) -> np.ndarray:
        """Get contour with largest area."""
        if not contours:
            return np.array([])

        biggest = contours[0]
        biggest_area = cv2.contourArea(contours[0])

        for contour in contours[1:]:
            area = cv2.contourArea(contour)
            if area > biggest_area:
                biggest_area = area
                biggest = contour

        return biggest

    def _get_pattern_contour(
        self, contours: List[np.ndarray]
    ) -> Tuple[np.ndarray, int]:
        """Get pattern contour and component count."""
        if len(contours) < 5:
            return self._get_biggest_contour(contours), 0
        else:
            return self._get_similar_contours(contours)

    def _get_similar_contours(
        self, contours: List[np.ndarray]
    ) -> Tuple[np.ndarray, int]:
        """Group contours by shape similarity."""
        similar_contours = []

        for i in range(len(contours)):
            group = [contours[i]]

            for j in range(len(contours)):
                if j != i:
                    similarity = cv2.matchShapes(
                        contours[i], contours[j], cv2.CONTOURS_MATCH_I1, 0
                    )
                    if similarity < 0.7:
                        group.append(contours[j])

            similar_contours.append(group)

        # Sort by group size
        similar_contours.sort(key=len, reverse=True)

        # Merge contours in largest group
        merged = []
        for contour in similar_contours[0]:
            merged.extend(contour.reshape(-1, 2).tolist())

        return np.array(merged), len(similar_contours[0])

    def _check_frame_count(self, data: FrameData) -> None:
        """Check if pattern failure threshold reached."""
        frames_in_window = min(self._frame_count, self.frame_time_thresh)

        if self.pattern_frame_count.current >= frames_in_window:
            data.pattern_frame_result = PatternResult.Fail
            self._is_fail = True
            self.pattern_fail_frames += 1
        else:
            data.pattern_frame_result = PatternResult.Pass

        # Remove old frames from window
        frames_to_remove = max(0, self._frame_count - self.frame_time_thresh)
        while frames_to_remove > 0:
            self.pattern_frame_count.update_passed()
            frames_to_remove -= 1

    @property
    def is_fail(self) -> bool:
        """Check if pattern failure occurred."""
        return self._is_fail
