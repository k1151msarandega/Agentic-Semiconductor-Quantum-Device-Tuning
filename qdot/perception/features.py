"""
qdot/perception/features.py
============================
Physics-aware feature extractors for 2D stability diagrams.

Ported from the hackathon and promoted to first-class modules.
In the new architecture these functions are a *validator layer*
that sits after the CNN, not before it. The CNN is the primary
classifier; these features act as physics-consistency checks that
can set the `physics_override` flag on a Classification.

Three extractors:
    log_preprocess(array)     — log₁₀(|G| + ε) normalised to [0,1]
    fft_peak_ratio(array)     — dominant periodicity ratio; > 3.5 → single-dot signal
    diagonal_strength(array)  — Laplacian + gradient angle; high → charge transition lines

All functions accept np.ndarray of any 2D shape.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# 1. Log-space preprocessing
# ---------------------------------------------------------------------------

def log_preprocess(array: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Compress conductance dynamic range via log₁₀ and normalise to [0, 1].

    This is always the *first* transformation applied to raw conductance
    data before any feature extraction or CNN inference. It handles the
    several-orders-of-magnitude variation in conductance across a stability
    diagram by compressing it into a perceptually uniform scale.

    Args:
        array: Raw conductance array, any shape, values ≥ 0.
        eps:   Floor to prevent log(0). Default 1e-9 is safe for typical
               normalised conductance (adapter already clips to ≥ 0).

    Returns:
        Float32 array of the same shape, values ∈ [0, 1].
    """
    arr = np.asarray(array, dtype=np.float64)
    arr = np.clip(arr, 0.0, None)           # guard against negative noise
    log_arr = np.log10(arr + eps)
    lo, hi = log_arr.min(), log_arr.max()
    if hi - lo > 1e-12:
        out = (log_arr - lo) / (hi - lo)
    else:
        out = np.full_like(log_arr, 0.5)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# 2. FFT peak ratio
# ---------------------------------------------------------------------------

def fft_peak_ratio(array: np.ndarray) -> float:
    """
    Measure the dominance of the strongest periodicity in a 2D scan.

    A single quantum dot produces regularly-spaced Coulomb peaks along
    one gate axis — a single dominant frequency. A double dot produces
    a honeycomb with multiple periodicities. A featureless scan produces
    a noisy spectrum with no dominant peak.

    Method:
        1. Apply log-preprocessing (if not already done — idempotent).
        2. Compute 2D FFT magnitude spectrum, zero-DC.
        3. peak_ratio = max_power / mean_power of top-5 frequencies.

    Returns:
        peak_ratio ∈ ℝ.  Empirically, > 3.5 indicates a single dominant
        periodicity (single-dot signature). This threshold is used as a
        physics validator, not a hard classifier.
    """
    arr = log_preprocess(array).astype(np.float64)

    # 2D FFT, shift DC to centre
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(arr))) ** 2

    # Zero out the DC component (centre pixel) to avoid it dominating
    cy, cx = spectrum.shape[0] // 2, spectrum.shape[1] // 2
    spectrum[cy, cx] = 0.0

    flat = spectrum.flatten()
    flat.sort()                         # ascending

    if flat[-1] < 1e-20:                # all-zero spectrum → no signal
        return 0.0

    # Ratio of top peak to mean of top-5 (avoids single outlier artefacts)
    top5 = flat[-5:]
    mean_top5 = top5.mean() if top5.mean() > 1e-20 else 1e-20
    return float(flat[-1] / mean_top5)


# ---------------------------------------------------------------------------
# 3. Diagonal edge detector
# ---------------------------------------------------------------------------

def diagonal_strength(array: np.ndarray) -> float:
    """
    Measure the fraction of image gradient energy aligned with the ±45°
    diagonals characteristic of charge transition lines in stability diagrams.

    Charge state boundaries appear as diagonal lines because a change in
    either gate voltage shifts the electrochemical potential of both dots
    (capacitive coupling). The slope is set by the ratio of the lever arms
    — always diagonal for symmetric devices, and close to diagonal for
    asymmetric ones.

    Method:
        1. Log-preprocess.
        2. Compute gradient magnitude + angle (Sobel operators).
        3. diagonal_strength = fraction of gradient energy with angle ∈
           (±45° ± 22.5°) or (±135° ± 22.5°).

    Returns:
        diagonal_strength ∈ [0, 1]. Values > 0.4 indicate strong diagonal
        structure (consistent with double-dot or single-dot transitions).
    """
    arr = log_preprocess(array).astype(np.float64)

    # Sobel kernels (3×3) for x and y gradients
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

    # Manual 2D convolution (avoids OpenCV dependency here)
    from scipy.signal import convolve2d
    gx = convolve2d(arr, sobel_x, mode="same", boundary="symm")
    gy = convolve2d(arr, sobel_y, mode="same", boundary="symm")

    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    angle_deg = np.degrees(np.arctan2(gy, gx))  # ∈ [-180, 180]

    total_energy = (magnitude ** 2).sum()
    if total_energy < 1e-20:
        return 0.0

    # Diagonals are at ±45° and ±135°.  Accept ±22.5° tolerance.
    angle_mod = np.abs(angle_deg) % 180   # fold to [0, 180)
    is_diagonal = (
        ((angle_mod >= 22.5) & (angle_mod < 67.5)) |   # ~45°
        ((angle_mod >= 112.5) & (angle_mod < 157.5))   # ~135°
    )

    diagonal_energy = (magnitude[is_diagonal] ** 2).sum()
    return float(diagonal_energy / total_energy)


# ---------------------------------------------------------------------------
# 4. Composite physics feature vector
# ---------------------------------------------------------------------------

def physics_features(array: np.ndarray) -> dict:
    """
    Compute all physics validator features for a single 2D patch.

    Returns a dict compatible with Classification.features.

    Keys:
        fft_peak_ratio      — dominance of single periodicity
        diagonal_strength   — fraction of gradient energy on diagonals
        mean_conductance    — mean of log-preprocessed array (proxy for regime)
        conductance_std     — std of log-preprocessed array (proxy for feature contrast)
    """
    preprocessed = log_preprocess(array)
    return {
        "fft_peak_ratio": fft_peak_ratio(array),
        "diagonal_strength": diagonal_strength(array),
        "mean_conductance": float(preprocessed.mean()),
        "conductance_std": float(preprocessed.std()),
    }


# ---------------------------------------------------------------------------
# 5. Physics heuristic override rules
# ---------------------------------------------------------------------------

# Thresholds — these are defaults. InspectionAgent receives device-adaptive
# versions derived from CIM parameters (lever_arm, E_c) at runtime.
PEAK_RATIO_SD_THRESHOLD = 3.5      # above → single-dot signature
DIAGONAL_STRENGTH_MIN = 0.25       # below → no transition lines → likely featureless


def physics_override_label(
    cnn_label: str,
    features: dict,
    peak_ratio_threshold: float = PEAK_RATIO_SD_THRESHOLD,
    diagonal_min: float = DIAGONAL_STRENGTH_MIN,
) -> tuple[str | None, str]:
    """
    Apply physics heuristics to check for inconsistencies with the CNN label.

    Returns:
        (override_label, reason) where override_label is None if no override
        is warranted, or a ChargeLabel string if the heuristics suggest the
        CNN is wrong.

    Override conditions (conservative — only flag clear contradictions):
        - CNN says double-dot but peak_ratio > peak_ratio_threshold AND
          diagonal_strength < diagonal_min  → likely single-dot
        - CNN says anything but diagonal_strength < 0.10 AND
          conductance_std < 0.05            → featureless → misc
    """
    pr = features.get("fft_peak_ratio", 0.0)
    ds = features.get("diagonal_strength", 1.0)
    std = features.get("conductance_std", 1.0)

    # Featureless check — catches SC and Barrier
    if ds < 0.10 and std < 0.05:
        if cnn_label != "misc":
            return "misc", f"Featureless (diag={ds:.2f}, std={std:.3f})"

    # Single-dot signal in a purported double-dot
    if cnn_label == "double-dot" and pr > peak_ratio_threshold and ds < diagonal_min:
        return "single-dot", f"Strong single periodicity (peak_ratio={pr:.1f})"

    return None, ""
