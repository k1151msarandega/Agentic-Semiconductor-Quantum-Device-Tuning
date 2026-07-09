"""
qdot/simulator/physics.py
==========================
Single source of truth for "how big a scan window does the InspectionAgent
need to see a stability diagram," shared by training-data generation and
every inference-time stage that feeds a 2D scan to the CNN.

Background
----------
Before this module existed, three different formulas for "Coulomb period"
were live in the codebase at once:

    qdot/perception/dataset.py   (training ground truth)
        coulomb_period_V = E_c_mean / lever
        delta (half-width) = max(1.5 * coulomb_period_V, 0.5)
        -> full training window = 3.0 x coulomb_period_V

    qdot/agent/executive.py :: _run_navigation()
        _period = 2.0 * E_c_mean / lever          # different definition
        nav_half_width = clip(_period / 2, 1.5, 5.0)
        -> full window = 2.0 x coulomb_period_V   (67% of training window)

    qdot/agent/executive.py :: _run_charge_id()
        half_width = 3.5                          # hardcoded, ignores period
        -> full window = 7.0V regardless of device params

The CNN never saw anything outside the first definition at training time.
Both inference stages were therefore feeding it out-of-distribution windows
(and CHARGE_ID's fixed 7.0V would be wildly wrong for any device params far
from the benchmark's E_c~2.5, lever~0.65 — e.g. ~4.7x oversized for the
E_c=0.5, lever=1.0 "default" params used in earlier conference figures).

Every stage that scans for classification should import from here instead
of recomputing the period locally.
"""

from __future__ import annotations

from typing import Optional, Tuple


def coulomb_period(E_c1: float, E_c2: float, lever_arm: float) -> float:
    """
    One Coulomb period in volts, along a single gate axis, for the
    quadratic Constant Interaction Model in qdot/simulator/cim.py.

    Derivation (single-dot chemical potential, quadratic in n, cross
    term E_cm held fixed):
        mu(n) = 0.5 * E_c * n^2 - alpha * V * n
        Degeneracy mu(n) = mu(n+1) occurs at V = E_c * (n + 0.5) / alpha
        Spacing between consecutive degeneracies = E_c / alpha

    Uses the mean of E_c1, E_c2 to match dataset.py's training-data
    convention (one scalar period shared by both gate axes).
    """
    E_c_mean = (float(E_c1) + float(E_c2)) / 2.0
    lever = max(float(lever_arm), 1e-6)
    return E_c_mean / lever


def coulomb_centre(E_c1: float, E_c2: float, lever_arm: float) -> float:
    """
    Voltage (same value on both gates) of the first charge-degeneracy
    point: dataset.py's V_centre = -E_c_mean / lever.

    Only meaningful as a fallback centre when no better estimate
    (survey peak, refined HYPERSURFACE_SEARCH peak, belief mode) is
    available.
    """
    E_c_mean = (float(E_c1) + float(E_c2)) / 2.0
    lever = max(float(lever_arm), 1e-6)
    return -E_c_mean / lever


def coulomb_window(
    E_c1: float,
    E_c2: float,
    lever_arm: float,
    factor: float = 1.5,
    min_half_width: float = 0.5,
    max_half_width: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Half-width of a 2D scan window sized to match the InspectionAgent's
    training distribution (qdot/perception/dataset.py), plus the period
    it was derived from.

    Args:
        factor: multiplier on one Coulomb period. dataset.py trains with
            factor=1.5 (full window = 3 x period). Keep this at 1.5 for
            any scan that gets classified by the CNN, unless the CNN is
            retrained on a different window size — in which case update
            dataset.py's factor too, so training and inference stay
            locked together.
        min_half_width: floor in volts, mirrors dataset.py's `max(..., 0.5)`.
        max_half_width: optional cap in volts. dataset.py has no cap (it's
            offline data generation); inference-time callers may want one
            to bound measurement-budget cost of a single scan. Pass None
            to match training exactly.

    Returns:
        (half_width, period) in volts. Full window = 2 * half_width.
    """
    period = coulomb_period(E_c1, E_c2, lever_arm)
    half_width = max(factor * period, min_half_width)
    if max_half_width is not None:
        half_width = min(half_width, max_half_width)
    return half_width, period
