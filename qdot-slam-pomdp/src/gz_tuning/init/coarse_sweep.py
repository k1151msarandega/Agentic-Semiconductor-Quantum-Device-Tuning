"""
gz_tuning.init.coarse_sweep

NEW MODULE -- Phase 0. Implements the primer's own original expert-input
framing (Natalia Ares, quoted directly in the primer's Section 6 "Working
hypothesis from expert input"): "...once enough distinct transitions have
been located -- e.g., via a few 1D probes at different secondary-gate
voltages -- to fit E_c/lever_arm from their spacing..." That framing
specified a coarse, deterministic, belief-independent discovery step. It
never got built; the two-dial acquisition design (bald.py/fim.py/
two_dial.py) specifies how to SCORE a candidate voltage but never how one
gets PROPOSED from a literal blank start. Confirmed as a real, not
hypothetical, gap by run_active_slam.py's own smoke test: with candidates
drawn uniformly over the full 4-gate range, IG_total scored exactly 0.0
on 18/20 steps, because interdot transitions occupy a thin slice of a 4D
15V-per-gate range that blind uniform sampling essentially never lands
near.

Design, and why "just draw more random candidates" doesn't substitute for
this: the probability of a uniform sample landing within tolerance of a
lower-dimensional transition manifold shrinks combinatorially with
dimension; a deterministic sweep that actually crosses every gate's full
range is the only proposal method that's *guaranteed* to cross every
transition line that gate participates in, at least once, regardless of
particle belief (which is exactly what's unreliable at true ground-zero --
Section 2's standing "scale must be earned from measurement" principle,
same instinct as the sign-bug cautionary tale).

Per gate, sweep at >1 background setting (not just one central value) --
closer to Ares's literal "1D probes at different secondary-gate voltages"
than a single midpoint sweep, and reduces the risk that one unlucky
background choice happens to sit in a dead region for that gate.

Every point measured here is a REAL measurement (through the noise model,
never the ground-truth signal directly) and is fed into the particle
filter via the normal pf.update() path -- this isn't a separate
diagnostic pass, it's genuine Phase 0 data collection that directly
improves particle means before Phase 1 (raw two-dial search) even starts.

Transition detection operates on the (noisy) MEASURED occupation, not the
ground truth -- consistent with ground-zero: a real experiment only has
access to what it measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from noise_model import GaussianReadoutNoise
from particle_filter import RBParticleFilter
from qarray_env import N_GATE, QArrayEnv


@dataclass
class CoarseSweepResult:
    seed_points: np.ndarray  # (n_seeds, N_GATE) -- vg locations bracketing a detected jump
    n_measurements: int
    n_transitions_found: int


def run_coarse_sweep(
    pf: RBParticleFilter,
    ground_truth: QArrayEnv,
    noise: GaussianReadoutNoise,
    *,
    vg_range: tuple[float, float] = (0.0, 15.0),
    n_points_per_line: int = 25,
    background_fractions: tuple[float, ...] = (1.0 / 3.0, 2.0 / 3.0),
    T: float = 0.05,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
) -> CoarseSweepResult:
    """Sweep each of the N_GATE gates individually, at len(background_fractions)
    different fixed settings of the other three gates, feeding every point
    into pf.update(). Returns the vg locations where a rounded-occupation
    jump was detected in the measured signal, for use as candidate-search
    seeds (candidate_search.select_next_measurement's `seed_points` arg).
    """
    if rng is None:
        rng = np.random.default_rng()

    lo, hi = vg_range
    background_values = [lo + f * (hi - lo) for f in background_fractions]
    sweep_axis = np.linspace(lo, hi, n_points_per_line)
    R = noise.R_matrix()

    seed_points: list[np.ndarray] = []
    n_measurements = 0

    for gate_idx in range(N_GATE):
        for bg_val in background_values:
            line_vgs = np.full((n_points_per_line, N_GATE), bg_val, dtype=np.float64)
            line_vgs[:, gate_idx] = sweep_axis

            prev_rounded: np.ndarray | None = None
            prev_vg: np.ndarray | None = None
            for vg in line_vgs:
                true_reading = ground_truth.soft_prediction(vg, T=T)
                measured = noise.sample(true_reading, rng)
                pf.update(measured, vg, R, T=T)
                n_measurements += 1

                rounded = np.round(measured)
                if prev_rounded is not None and not np.array_equal(rounded, prev_rounded):
                    midpoint = 0.5 * (vg + prev_vg)
                    seed_points.append(midpoint)
                    if verbose:
                        print(
                            f"  gate {gate_idx} bg={bg_val:.2f}: transition near "
                            f"{np.round(midpoint, 3)} ({prev_rounded} -> {rounded})"
                        )
                prev_rounded = rounded
                prev_vg = vg

    seed_array = np.array(seed_points, dtype=np.float64) if seed_points else np.empty((0, N_GATE))
    return CoarseSweepResult(
        seed_points=seed_array,
        n_measurements=n_measurements,
        n_transitions_found=len(seed_points),
    )
