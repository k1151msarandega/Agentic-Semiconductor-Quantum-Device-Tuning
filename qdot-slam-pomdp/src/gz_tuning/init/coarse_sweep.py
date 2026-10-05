from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from noise_model import GaussianReadoutNoise
from particle_filter import RBParticleFilter
from qarray_env import N_GATE, QArrayEnv


@dataclass
class CoarseSweepResult:
    seed_points: np.ndarray
    n_measurements: int
    n_transitions_found: int
    beta_trace: np.ndarray = field(default_factory=lambda: np.empty(0))
    """pf.last_beta after each Phase-0 measurement, in order. Added to
    compare against Phase 1's acquisition-driven tempering throttle: a
    deterministic per-gate scan (what Phase 0 is) has no mechanism
    connecting candidate choice to particle disagreement the way IG-based
    acquisition does, so if beta here is ALSO pinned near 0, that's
    evidence tempering throttle is a property of the prior/noise width
    alone, independent of acquisition quality -- not a sign that Phase 1's
    acquisition is 'succeeding' at finding disagreement-inducing points.
    """


def run_coarse_sweep(
    pf: RBParticleFilter,
    ground_truth: QArrayEnv,
    noise: GaussianReadoutNoise,
    *,
    vg_range: tuple[float, float] = (0.0, 15.0),
    n_points_per_line: int = 25,
    background_fractions: tuple[float, ...] = (1.0 / 3.0, 2.0 / 3.0),
    T: float = 0.05,
    ess_resample_frac: float = 0.5,
    mu_jitter_scale: float | None = 0.1,
    target_ess_frac: float | None = 0.5,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
) -> CoarseSweepResult:
    if rng is None:
        rng = np.random.default_rng()

    lo, hi = vg_range
    background_values = [lo + f * (hi - lo) for f in background_fractions]
    sweep_axis = np.linspace(lo, hi, n_points_per_line)
    R = noise.R_matrix()
    n_particles = pf.n_particles

    seed_points: list[np.ndarray] = []
    n_measurements = 0
    n_resamples = 0
    beta_trace: list[float] = []

    for gate_idx in range(N_GATE):
        for bg_val in background_values:
            line_vgs = np.full((n_points_per_line, N_GATE), bg_val, dtype=np.float64)
            line_vgs[:, gate_idx] = sweep_axis

            prev_rounded: np.ndarray | None = None
            prev_vg: np.ndarray | None = None
            for vg in line_vgs:
                true_reading = ground_truth.soft_prediction(vg, T=T)
                measured = noise.sample(true_reading, rng)
                pf.update(measured, vg, R, T=T, target_ess_frac=target_ess_frac)
                n_measurements += 1
                beta_trace.append(pf.last_beta)

                if pf.effective_sample_size() < ess_resample_frac * n_particles:
                    pf.resample(rng, mu_jitter_scale=mu_jitter_scale)
                    n_resamples += 1

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

    if verbose:
        print(f"  Phase 0 resampled {n_resamples} times over {n_measurements} measurements")

    seed_array = np.array(seed_points, dtype=np.float64) if seed_points else np.empty((0, N_GATE))
    return CoarseSweepResult(
        seed_points=seed_array,
        n_measurements=n_measurements,
        n_transitions_found=len(seed_points),
        beta_trace=np.array(beta_trace, dtype=np.float64),
    )
