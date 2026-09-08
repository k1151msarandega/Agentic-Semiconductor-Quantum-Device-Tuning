"""
gz_tuning.baselines.raster_scan

Deterministic raster-scan baseline for Claim 1's headline comparison
(Primer Section 1): "does active POMDP+SLAM exploration reach convergence
in fewer measurements than a raster-scan baseline". No acquisition
function, no information-gain candidate selection -- just a fixed,
predetermined grid of voltage points, visited in a deterministic order,
each one fed through the SAME belief update (RBParticleFilter.update) and
gated by the SAME ConvergenceMonitor CLASS (control/convergence.py) the
SLAM method uses.

FAIRNESS REQUIREMENT, Primer Section 8, load-bearing for Claim 1's
credibility -- NOT optional: "the raster baseline must use a comparably
rigorous stopping rule ... Give the baseline the same
belief-confidence-threshold stopping rule, fed by raster-order measurements
instead of IG-chosen ones, so the comparison is measurements-to-
equivalent-confidence."

BUG FOUND AND FIXED DURING REVIEW -- read before touching this module's
convergence wiring again: an earlier version passed
`grid[i+1] - vg` (the next SCHEDULED grid point's displacement) as guard
B's actuation-quiescence proxy, with the monitor's default
require_actuation_quiescence=True. This was structurally broken, not just
imprecise: a real raster grid moves by a fixed, non-trivial step between
every consecutive point BY CONSTRUCTION (that is what makes it a raster
scan), and that step size is essentially guaranteed to be much larger than
any sensible epsilon (which should be noise-floor scale, per Section 6a).
So guard B was False on essentially EVERY step except near the very end of
the grid, where the final point's proposed_step is defined as zero --
meaning `converged=True` could only ever fire in the grid's tail,
regardless of how quickly belief actually stabilized. That makes the
baseline structurally incapable of stopping early, which is exactly
backwards from what Section 8's fairness requirement was protecting: it
would inflate the SLAM method's apparent measurements-to-convergence
advantage for a purely mechanical reason unrelated to which method is
actually smarter. (This was specifically masked by a test that revisited
the same grid point on every step, making proposed_step always exactly
zero -- see tests/test_raster_scan.py's history for the corrected test
that exercises a real, moving grid instead.)

FIX ADOPTED: construct the ConvergenceMonitor passed to run_raster_scan
with require_actuation_quiescence=False (see control/convergence.py's
module docstring for the full argument). Guard A (belief confidence) alone
is the fair, apples-to-apples criterion for a fixed, non-adaptive
schedule -- Section 8's own phrasing centers on "the same
belief-confidence-threshold stopping rule," and guard B's physical
justification (an ACTIVE policy's own chosen step signaling live
cross-term perturbation risk) has no coherent operationalization for a
predetermined grid. run_raster_scan below ENFORCES this at the call site
(raises rather than silently accepting a guard-B-enabled monitor), since
silently re-permitting the broken configuration here would be too easy to
reintroduce by accident.

GRID DIMENSIONALITY -- a real design choice, not settled by the primer:
Primer Section 5 contrasts the SLAM method's full-joint-4D acquisition
against "the alternating-navigation-doesn't-scale problem" of scanning one
DQD's 2 gates, then the other's, back and forth -- implying the classical
raster baseline this project should compare against is exactly that
ALTERNATING 2D-then-2D style, not a naive full joint 4D grid (which would
be a much weaker, more expensive strawman, not a genuine "how would someone
actually raster-scan this today" baseline). This module implements the
alternating variant as the default (`mode="alternating"`) for that reason,
with a full joint-grid mode (`mode="joint"`) also available as a stricter
(and, on a 4D grid, much more expensive) point of comparison -- pick
`mode` deliberately per what the paper's Claim 1 actually wants to claim
against, don't default to "joint" just because it sounds more thorough.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from convergence import ConvergenceMonitor, ConvergenceStatus
from noise_model import GaussianReadoutNoise
from particle_filter import RBParticleFilter
from qarray_env import N_GATE, QArrayEnv


def _joint_grid(vg_range: tuple[float, float], n_points_per_gate: int) -> np.ndarray:
    """Every combination of n_points_per_gate evenly-spaced values across
    all N_GATE gates, in itertools.product order (deterministic,
    reproducible without needing to store/shuffle a huge array). Grows as
    n_points_per_gate ** N_GATE -- flagged in the module docstring as the
    expensive, stricter comparison mode."""
    axis = np.linspace(vg_range[0], vg_range[1], n_points_per_gate)
    return np.array(list(itertools.product(axis, repeat=N_GATE)), dtype=np.float64)


def _alternating_grid(
    vg_range: tuple[float, float], n_points_per_gate: int, *, n_alternations: int = 4
) -> np.ndarray:
    """Classical two-DQD raster: scan gates (0,1) [DQD-A] over a full 2D
    grid holding gates (2,3) [DQD-B] fixed at their range midpoint, then
    scan gates (2,3) holding (0,1) fixed at the SAME range midpoint,
    alternating n_alternations times -- the "alternating navigation"
    pattern Primer Section 5 explicitly contrasts the joint method
    against. Each half-pass is a full 2D raster (n_points_per_gate^2
    points), so total cost is n_alternations * 2 * n_points_per_gate**2,
    independent of N_GATE=4 scaling the way the joint mode does.

    Deliberately simple midpoint-holding, not adaptive re-centering on
    each pass's own findings -- a fancier baseline that re-centers based on
    what it already found would blur the line between "classical raster
    baseline" and "a second, competing active method," which is not what
    this module is for.
    """
    axis = np.linspace(vg_range[0], vg_range[1], n_points_per_gate)
    mid = float(np.mean(vg_range))
    grid_2d = np.array(list(itertools.product(axis, repeat=2)), dtype=np.float64)

    points = []
    for _ in range(n_alternations):
        for (v0, v1) in grid_2d:
            points.append([v0, v1, mid, mid])
        for (v2, v3) in grid_2d:
            points.append([mid, mid, v2, v3])
    return np.array(points, dtype=np.float64)


def build_raster_grid(
    vg_range: tuple[float, float] = (0.0, 15.0),
    n_points_per_gate: int = 20,
    *,
    mode: str = "alternating",
    n_alternations: int = 4,
) -> np.ndarray:
    """Deterministic grid of vg points, shape (n_points, N_GATE). See
    module docstring for the `mode` choice."""
    if mode == "joint":
        return _joint_grid(vg_range, n_points_per_gate)
    elif mode == "alternating":
        return _alternating_grid(vg_range, n_points_per_gate, n_alternations=n_alternations)
    else:
        raise ValueError(f"Unknown mode {mode!r} -- expected 'joint' or 'alternating'")


@dataclass
class RasterScanResult:
    """Outcome of a full run_raster_scan() call -- mirrors what a
    SLAM-method run would report, so the two can be compared
    apples-to-apples (Primer Section 1's headline measurements-to-
    convergence metric)."""

    converged: bool
    n_measurements: int
    final_status: ConvergenceStatus | None


def run_raster_scan(
    pf: RBParticleFilter,
    ground_truth: QArrayEnv,
    grid: np.ndarray,
    noise: GaussianReadoutNoise,
    monitor: ConvergenceMonitor,
    *,
    T: float = 0.05,
    rng: np.random.Generator | None = None,
) -> RasterScanResult:
    """Walk `grid` in order, one measurement per point: sample a noisy
    reading from ground_truth at that vg, update pf's belief (SAME
    RBParticleFilter.update() the SLAM method uses -- no separate baseline-
    specific belief machinery), then check monitor.step().

    REQUIRES monitor.require_actuation_quiescence is False -- see module
    docstring's "BUG FOUND AND FIXED" section for why a guard-B-enabled
    monitor is structurally broken against a fixed schedule's step sizes.
    Enforced here with an explicit raise rather than silently overriding
    the caller's monitor, since silent overriding would hide a
    configuration mistake that directly affects Claim 1's reported
    numbers.

    proposed_step fed to monitor.step() is still the next SCHEDULED grid
    point's displacement (or zero at the last point) -- harmless now that
    guard B is disabled for this monitor (its VALUE no longer gates
    anything, per control/convergence.py), but shape-validated on every
    call regardless, so this is kept rather than replaced with a dummy
    zero vector, preserving one real signal (still visible via
    ConvergenceStatus if a caller wants to log it) at no cost.

    Stops the moment monitor reports converged=True, OR the grid is
    exhausted -- whichever comes first. Returns converged=False if the
    grid runs out before convergence, distinguishing "the baseline was
    still converging when we ran out of measurement budget" from "it
    converged," which the caller needs for an honest fairness comparison.
    """
    if monitor.require_actuation_quiescence:
        raise ValueError(
            "run_raster_scan requires a ConvergenceMonitor constructed "
            "with require_actuation_quiescence=False. A fixed raster "
            "grid's step-to-next-scheduled-point distance is essentially "
            "guaranteed to exceed any sensible epsilon, which makes "
            "guard B false on nearly every step and the baseline "
            "structurally incapable of stopping early -- see this "
            "module's docstring ('BUG FOUND AND FIXED') for the full "
            "explanation. Construct the monitor with "
            "require_actuation_quiescence=False (same p_conf/epsilon/N as "
            "whatever monitor the SLAM-method run being compared against "
            "uses) rather than passing this one through unchanged."
        )

    if rng is None:
        rng = np.random.default_rng()
    R = noise.R_matrix()

    status: ConvergenceStatus | None = None
    for i, vg in enumerate(grid):
        true_reading = ground_truth.soft_prediction(vg, T=T)
        measured = noise.sample(true_reading, rng)
        pf.update(measured, vg, R, T=T)

        if i + 1 < len(grid):
            proposed_step = grid[i + 1] - vg
        else:
            proposed_step = np.zeros(N_GATE)

        status = monitor.step(pf, vg, proposed_step)
        if status.converged:
            return RasterScanResult(
                converged=True, n_measurements=i + 1, final_status=status
            )

    return RasterScanResult(
        converged=False, n_measurements=len(grid), final_status=status
    )
