from __future__ import annotations

import itertools

import numpy as np

from particle_filter import RBParticleFilter
from qarray_env import N_DOT, QArrayEnv


def _local_candidate_states(predictions: np.ndarray, *, epsilon: float = 0.02) -> np.ndarray:
    all_candidates: set[tuple[int, ...]] = set()
    for pred in predictions:
        floor = np.floor(pred)
        frac = pred - floor
        uncertain_mask = (frac > epsilon) & (frac < 1 - epsilon)
        uncertain_idx = np.where(uncertain_mask)[0]
        base = np.round(pred).astype(int)

        if len(uncertain_idx) == 0:
            all_candidates.add(tuple(base))
            continue

        for bits in itertools.product([0, 1], repeat=len(uncertain_idx)):
            candidate = base.copy()
            for idx, bit in zip(uncertain_idx, bits):
                candidate[idx] = int(floor[idx]) + bit
            all_candidates.add(tuple(candidate))

    return np.array(sorted(all_candidates), dtype=np.float64)


def _entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    p = np.clip(p, 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log(p), 0.0)
    return -np.sum(terms, axis=axis)


def compute_ig_bald(
    pf: RBParticleFilter,
    vg: np.ndarray,
    *,
    T: float = 0.05,
    epsilon: float = 0.02,
    envs: list["QArrayEnv"] | None = None,
) -> float:
    """Exact joint-entropy BALD -- same as the repo version, PLUS an
    `envs` parameter.

    BUG FIX (found during review, confirming Primer Section 9's flagged
    open question): the original version built
    `envs = [QArrayEnv(p.kalman.params) for p in pf.particles]` FRESH on
    every call. compute_ig_bald is called once per CANDIDATE voltage point
    during acquisition search (Primer Section 9's ~50-candidate batch), so
    the old version paid a full joint self-capacitance solve
    (n_particles x n_candidates times per acquisition step, no warm-start)
    -- exactly the "does each call reconstruct fresh?" scenario Section 9
    flagged as unconfirmed and potentially much more expensive than the
    197ms/step figure.

    FIX: `envs`, if supplied, must be a list of QArrayEnv, one per
    particle, already constructed at each particle's CURRENT (pre-update)
    kalman.params -- built ONCE per acquisition step by the candidate-
    search loop (acquisition/candidate_search.py) and reused across every
    candidate vg in that step's batch. Falls back to fresh construction
    (old behavior, no warm start) if envs is None, so existing callers/
    tests are unaffected.
    """
    vg = np.asarray(vg, dtype=np.float64)
    weights = np.array([p.weight for p in pf.particles])
    if envs is None:
        envs = [QArrayEnv(p.kalman.params) for p in pf.particles]
    elif len(envs) != len(pf.particles):
        raise ValueError(
            f"envs has {len(envs)} entries but pf has {len(pf.particles)} "
            f"particles -- must be one QArrayEnv per particle, in order."
        )
    predictions = np.stack([env.soft_prediction(vg, T=T) for env in envs])

    candidates = _local_candidate_states(predictions, epsilon=epsilon)
    K = candidates.shape[0]

    if K == 1:
        return 0.0

    P = np.zeros((len(pf.particles), K))
    for i, env in enumerate(envs):
        F = env.free_energy_at_candidates(candidates, vg)
        neg_F_over_T = -F / T
        neg_F_over_T -= neg_F_over_T.max()
        w = np.exp(neg_F_over_T)
        P[i] = w / w.sum()

    mixture = weights @ P
    H_mixture = _entropy(mixture)
    per_particle_H = _entropy(P, axis=1)
    mean_particle_H = weights @ per_particle_H

    return float(H_mixture - mean_particle_H)
