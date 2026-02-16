"""
qdot/perception/dqc.py
======================
DQCGatekeeper — data quality classifier.

Sits between raw measurements from the Device Adapter and the
Inspection Agent. Every 2D measurement passes through here first.
Low-quality data is stopped before it can poison the classifier.

Unlike the hackathon OOD suppressor, this module surfaces problems
rather than hiding them. A LOW quality flag stops the agent and
triggers HITL. A MODERATE flag lets data through with a warning
that feeds into the Risk Score.

Blueprint §5.2 specification:
    Features:
        SNR          — signal-to-noise ratio (dB)
        Dynamic range — (max - min) / max
        Flatness     — var / mean²  (coefficient of variation squared)
        Plausibility — 0 ≤ G ≤ G_max; no NaN/Inf

    Classification:
        HIGH     — SNR > 20 dB, plausible, dynamic range > 0.3
        MODERATE — SNR 10–20 dB or borderline plausibility
        LOW      — SNR < 10 dB or physically implausible → STOP
"""

from __future__ import annotations

import numpy as np

from qdot.core.types import DQCQuality, DQCResult, Measurement
from uuid import UUID


class DQCGatekeeper:
    """
    Fast, independent quality classifier for raw conductance arrays.

    Designed to run in < 1 ms on any measurement the adapter can produce.
    No ML components — purely analytic. The goal is to catch hardware
    faults, cable disconnects, and saturated amplifiers before they reach
    the CNN classifier.

    Usage:
        gatekeeper = DQCGatekeeper()
        result = gatekeeper.assess(measurement)
        if result.quality == DQCQuality.LOW:
            # STOP — trigger HITL, do not pass to Inspection Agent
            ...
        elif result.quality == DQCQuality.MODERATE:
            # Pass with warning; DQC flag feeds into Risk Score
            ...
    """

    # SNR thresholds (dB)
    SNR_HIGH: float = 20.0
    SNR_LOW: float = 10.0

    # Dynamic range: (max - min) / max after log-preprocessing
    DYNAMIC_RANGE_MIN: float = 0.30

    # Flatness: var / mean²  — very small → featureless / flat noise
    # Note: high flatness (near 1) is OK; low flatness (< 0.001) suggests
    # a nearly-constant array (stuck amplifier or rail saturation).
    FLATNESS_MIN: float = 1e-4

    # Conductance bounds for physical plausibility (normalised [0, 1])
    G_MAX: float = 1.0
    G_MIN: float = 0.0

    def __init__(
        self,
        snr_high: float = SNR_HIGH,
        snr_low: float = SNR_LOW,
        dynamic_range_min: float = DYNAMIC_RANGE_MIN,
        flatness_min: float = FLATNESS_MIN,
    ) -> None:
        self.snr_high = snr_high
        self.snr_low = snr_low
        self.dynamic_range_min = dynamic_range_min
        self.flatness_min = flatness_min

    # -----------------------------------------------------------------------
    # Primary interface
    # -----------------------------------------------------------------------

    def assess(self, measurement: Measurement) -> DQCResult:
        """
        Assess data quality of a single Measurement.

        Args:
            measurement: Any Measurement from the Device Adapter.
                         Works on both 1D (line scan) and 2D arrays,
                         though the Inspection Agent only processes 2D.

        Returns:
            DQCResult with quality flag, sub-scores, and notes.
        """
        arr = np.asarray(measurement.array, dtype=np.float64)
        return self._assess_array(measurement.id, arr)

    def assess_array(self, measurement_id: UUID, array: np.ndarray) -> DQCResult:
        """
        Assess a raw array directly. Useful for unit tests and batch
        evaluation outside of the full Measurement object.
        """
        arr = np.asarray(array, dtype=np.float64)
        return self._assess_array(measurement_id, arr)

    # -----------------------------------------------------------------------
    # Core logic
    # -----------------------------------------------------------------------

    def _assess_array(self, measurement_id: UUID, arr: np.ndarray) -> DQCResult:
        # ---- Physical plausibility ----
        has_nan = bool(np.any(~np.isfinite(arr)))
        out_of_range = bool(np.any(arr < self.G_MIN - 1e-6) or np.any(arr > self.G_MAX + 1e-6))
        physically_plausible = not (has_nan or out_of_range)

        # If completely implausible, short-circuit to LOW immediately
        if has_nan:
            return DQCResult(
                measurement_id=measurement_id,
                quality=DQCQuality.LOW,
                snr_db=0.0,
                dynamic_range=0.0,
                flatness_score=0.0,
                physically_plausible=False,
                notes="Array contains NaN or Inf — hardware fault suspected.",
            )

        # Clip to valid range before feature computation (handles minor noise spill)
        arr = np.clip(arr, 0.0, 1.0)

        # ---- SNR ----
        snr_db = self._compute_snr(arr)

        # ---- Dynamic range ----
        dynamic_range = self._compute_dynamic_range(arr)

        # ---- Flatness ----
        flatness = self._compute_flatness(arr)

        # ---- Classification ----
        quality, notes = self._classify(
            snr_db=snr_db,
            dynamic_range=dynamic_range,
            flatness=flatness,
            physically_plausible=physically_plausible,
            out_of_range=out_of_range,
        )

        return DQCResult(
            measurement_id=measurement_id,
            quality=quality,
            snr_db=float(snr_db),
            dynamic_range=float(dynamic_range),
            flatness_score=float(flatness),
            physically_plausible=physically_plausible,
            notes=notes,
        )

    # -----------------------------------------------------------------------
    # Feature computations
    # -----------------------------------------------------------------------

    def _compute_snr(self, arr: np.ndarray) -> float:
        """
        Signal-to-noise ratio estimate (dB).

        Signal power: variance of a low-pass smoothed version of the array
        (captures the structured conductance features).
        Noise power: variance of the residual (arr - smoothed).

        Uses a simple 3×3 mean filter for smoothing. For 1D arrays,
        uses a 3-point moving average.
        """
        if arr.ndim == 1:
            # 1D line scan
            if len(arr) < 4:
                return 0.0
            kernel = np.ones(3) / 3.0
            smooth = np.convolve(arr, kernel, mode="same")
        else:
            # 2D patch — mean filter via sliding window trick
            smooth = self._mean_filter_2d(arr, size=3)

        signal = smooth
        noise = arr - smooth

        signal_power = float(np.var(signal))
        noise_power = float(np.var(noise))

        if noise_power < 1e-20:
            # Essentially no noise — perfectly synthetic data
            return 60.0   # cap at 60 dB (HIGH)
        if signal_power < 1e-20:
            return 0.0    # no signal

        return float(10.0 * np.log10(signal_power / noise_power))

    def _compute_dynamic_range(self, arr: np.ndarray) -> float:
        """
        (max - min) / max  — measures how much of the available range is used.

        A value close to 0 means the array is nearly constant (stuck rail
        or saturation). A value close to 1 means there are both very low
        and very high conductance regions — expected for a real stability
        diagram with Coulomb peaks.
        """
        lo, hi = float(arr.min()), float(arr.max())
        if hi < 1e-12:
            return 0.0
        return (hi - lo) / hi

    def _compute_flatness(self, arr: np.ndarray) -> float:
        """
        Var / Mean²  — coefficient of variation squared.

        Low flatness (<< 1) with low variance suggests a stuck amplifier
        or an array that's essentially constant. High flatness is fine —
        it means there are large excursions from the mean (peaks and troughs).
        """
        mean = float(arr.mean())
        if abs(mean) < 1e-12:
            # Near-zero mean: just return variance directly
            return float(arr.var())
        return float(arr.var() / (mean ** 2))

    def _classify(
        self,
        snr_db: float,
        dynamic_range: float,
        flatness: float,
        physically_plausible: bool,
        out_of_range: bool,
    ) -> tuple[DQCQuality, str]:
        """
        Decision logic from blueprint §5.2.

        Returns (quality, human-readable notes string).
        """
        notes_parts = []

        # Hard LOW conditions
        if snr_db < self.snr_low:
            return DQCQuality.LOW, f"SNR={snr_db:.1f} dB < {self.snr_low} dB threshold."

        if not physically_plausible:
            reasons = []
            if out_of_range:
                reasons.append("values outside [0,1]")
            return DQCQuality.LOW, f"Physically implausible: {', '.join(reasons)}."

        if flatness < self.flatness_min:
            return DQCQuality.LOW, (
                f"Flatness={flatness:.2e} — array is nearly constant. "
                "Possible stuck amplifier or rail saturation."
            )

        # MODERATE conditions (any single criterion borderline)
        is_moderate = False

        if snr_db < self.snr_high:
            notes_parts.append(f"SNR={snr_db:.1f} dB (borderline)")
            is_moderate = True

        if dynamic_range < self.dynamic_range_min:
            notes_parts.append(f"dynamic_range={dynamic_range:.2f} < {self.dynamic_range_min}")
            is_moderate = True

        if is_moderate:
            return DQCQuality.MODERATE, "; ".join(notes_parts)

        # HIGH — all checks pass
        return DQCQuality.HIGH, (
            f"SNR={snr_db:.1f} dB, dynamic_range={dynamic_range:.2f}, "
            f"flatness={flatness:.3f}. All checks passed."
        )

    # -----------------------------------------------------------------------
    # Internal utilities
    # -----------------------------------------------------------------------

    @staticmethod
    def _mean_filter_2d(arr: np.ndarray, size: int = 3) -> np.ndarray:
        """Fast 2D mean filter using uniform kernel convolution."""
        from scipy.ndimage import uniform_filter
        return uniform_filter(arr.astype(np.float64), size=size, mode="reflect")
