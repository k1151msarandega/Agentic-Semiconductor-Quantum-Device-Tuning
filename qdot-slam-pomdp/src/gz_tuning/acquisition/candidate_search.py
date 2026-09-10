"""
gz_tuning.acquisition.candidate_search

NEW MODULE -- not previously in the repo. two_dial.py computes IG_total for
a single vg; nothing in the repo actually searched the joint 4D voltage
space for the argmax candidate, which is what Primer Section 5's
acquisition design ("evaluated over the full joint 4D voltage space in one
shot") requires an active policy to do at every step. This module is that
missing search.

Also where the bald.py/fim.py envs-reuse fix (found during review, see
those modules' docstrings) actually pays off: build_particle_envs() is
called ONCE per acquisition step, and both the envs and their
self-capacitance diagonals (x0_bases) are reused across every candidate in
the batch AND handed back to the caller so pf.update() (the belief update
for whichever candidate gets chosen) doesn't re-solve a fourth time either.

Candidate proposal strategy: uniform random sampling in vg_range^4, same
pattern noise_model.calibrate_threshold_low already uses for its own
candidate search (a simple, general proxy -- not the closed-form
informative-transition-point formula found during qarray_env.py's
development, which was specific to one symmetric case and doesn't
generalize, per that module's own docstring). Kept deliberately simple
rather than inventing a new search heuristic not covered by the primer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from particle_filter import RBParticleFilter
from qarray_env import N_GATE, QArrayEnv
from two_dial import compute_ig_total


def build_particle_envs(pf: RBParticleFilter) -> tuple[list[QArrayEnv], list[np.ndarray]]:
    """One QArrayEnv per particle at its CURRENT (pre-update) params, plus
    each one's self-capacitance diagonal for x0-warm-starting FD Jacobian
    calls. Call ONCE per acquisition step; reuse the result across every
    candidate vg evaluated in that step's search."""
    envs = [QArrayEnv(p.kalman.params) for p in pf.particles]
    x0_bases = [np.diag(env._Cdd).copy() for env in envs]
    return envs, x0_bases


@dataclass(frozen=True)
class CandidateSearchResult:
    best_vg: np.ndarray
    best_ig: float
    all_vg: np.ndarray
    all_ig: np.ndarray
    envs: list[QArrayEnv]
    x0_bases: list[np.ndarray]


def select_next_measurement(
    pf: RBParticleFilter,
    R: np.ndarray,
    *,
    w_discrete: float,
    w_continuous: float,
    vg_range: tuple[float, float] = (0.0, 15.0),
    n_candidates: int = 50,
    T: float = 0.05,
    seed_points: np.ndarray | None = None,
    seed_fraction: float = 0.5,
    seed_jitter_sigma: float = 0.3,
    rng: np.random.Generator | None = None,
) -> CandidateSearchResult:
    """argmax_v IG_total(v) over a candidate batch. Returns the winner
    plus full diagnostics (every candidate's IG value -- useful for
    logging/detecting the Section 5b "narrow-but-nonzero" trap: a caller
    watching all_ig's spread over successive steps, not just the argmax,
    can tell whether the same one or two candidates keep winning).

    Candidate proposal: without `seed_points`, all n_candidates are drawn
    uniformly from vg_range^N_GATE -- confirmed by a real run to almost
    always score 0.0 (interdot transitions are a thin slice of a 4D
    range; see coarse_sweep.py's module docstring). With `seed_points`
    (e.g. from coarse_sweep.run_coarse_sweep), a `seed_fraction` share of
    the batch is instead drawn as seed + local Gaussian jitter
    (sigma=seed_jitter_sigma per gate), biasing the search toward regions
    already known to contain a real transition, while the remaining
    uniform share preserves some global exploration.
    """
    if rng is None:
        rng = np.random.default_rng()

    envs, x0_bases = build_particle_envs(pf)

    lo, hi = vg_range
    if seed_points is not None and len(seed_points) > 0:
        n_seeded = int(round(n_candidates * seed_fraction))
        n_uniform = n_candidates - n_seeded
        seed_choice_idx = rng.integers(0, len(seed_points), size=n_seeded)
        seeded_candidates = seed_points[seed_choice_idx] + rng.normal(
            0.0, seed_jitter_sigma, size=(n_seeded, N_GATE)
        )
        seeded_candidates = np.clip(seeded_candidates, lo, hi)
        uniform_candidates = rng.uniform(lo, hi, size=(n_uniform, N_GATE))
        candidates = np.concatenate([seeded_candidates, uniform_candidates], axis=0)
    else:
        candidates = rng.uniform(lo, hi, size=(n_candidates, N_GATE))
    ig_values = np.empty(n_candidates, dtype=np.float64)
    for i, vg in enumerate(candidates):
        ig_values[i] = compute_ig_total(
            pf, vg, R,
            w_discrete=w_discrete, w_continuous=w_continuous, T=T,
            envs=envs, x0_bases=x0_bases,
        )

    best_idx = int(np.argmax(ig_values))
    return CandidateSearchResult(
        best_vg=candidates[best_idx],
        best_ig=float(ig_values[best_idx]),
        all_vg=candidates,
        all_ig=ig_values,
        envs=envs,
        x0_bases=x0_bases,
    )
