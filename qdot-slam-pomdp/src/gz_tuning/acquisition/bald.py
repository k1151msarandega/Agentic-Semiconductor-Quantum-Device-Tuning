"""
gz_tuning.acquisition.bald

IG_BALD: the discrete-disagreement half of Primer Section 5's two-dial
acquisition function. High when particles' hypotheses disagree about what
a measurement at v would show; correctly goes to zero (not just low) when a
candidate is uninformative for every live hypothesis.

=== REWRITE: fixes a real, quantified bug in the per-dot-marginal-sum
version this replaced ===

The original version summed each dot's own Bernoulli entropy independently,
assuming conditional independence across dots given a particle's own
parameters. That assumption is not just approximately wrong near interdot
transitions, it's PROVABLY wrong there, and the error is exact and
quantifiable: entropy is subadditive (H(X)+H(Y) >= H(X,Y), equality only
under independence). At a clean anti-correlated interdot boundary --
(n0,n1) in {(0,1),(1,0)}, each with probability 0.5, exactly the case
verified during qarray_env.py's development -- the TRUE joint entropy is
ln(2) (one genuine bit: did the charge move or not), but each dot's own
marginal is ALSO ln(2) (each looks 50/50 in isolation), so summing gives
2*ln(2) -- exactly double, not approximately double, in this clean case.

Worse than a generic approximation error: the bias is strictly
one-directional (always over-, by subadditivity) and is largest exactly at
interdot transitions -- which prior work in this project already
established is the ONLY place E_c is observable at all. So the old
approximation wasn't uniformly imprecise, it was systematically most wrong
exactly where the acquisition function's candidate ranking most needs to be
right.

THE FIX: exact joint entropy over a small, locally-enumerated set of nearby
integer occupation states, using QArray's own free-energy quadratic form
(qarray_env.py's free_energy_at_candidates) -- not the hidden internal
softmax weights (never confirmed accessible) and not a full 2^4=16-state
enumeration (unnecessary): for each particle, dots whose soft_prediction is
confidently near an integer are held fixed; only dots with genuine
fractional uncertainty are enumerated over their two nearest integers. This
is cheap (typically 2-4 candidate states, not 16) and exact given that
enumeration -- no sampling, no approximation beyond "candidates outside
this local neighborhood have negligible probability mass," which is the
same locality assumption the softargmin thermal window already relies on.
"""

from __future__ import annotations

import itertools

import numpy as np

from particle_filter import RBParticleFilter
from qarray_env import N_DOT, QArrayEnv


def _local_candidate_states(predictions: np.ndarray, *, epsilon: float = 0.02) -> np.ndarray:
    """Union, across all particles, of each particle's own local candidate
    neighborhood: dots with fractional part within epsilon of 0 or 1 are
    treated as confident (fixed at their rounded value); dots with genuine
    fractional uncertainty are enumerated over {floor, floor+1}.

    predictions: shape (n_particles, N_DOT), each row a soft_prediction.
    Returns: shape (K, N_DOT) integer-valued array, deduplicated, sorted
    for determinism. K is typically small (2-4) in the cases this project
    has actually encountered (single-dot or two-dot transitions) -- can
    grow to 2^N_DOT in the pathological case of every dot being
    simultaneously uncertain, which has not been observed in practice but
    is worth monitoring if candidate counts start looking large.
    """
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
    """Shannon entropy (nats) of a probability vector/array along `axis`.
    Handles p=0 exactly (0*log(0) := 0)."""
    p = np.clip(p, 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log(p), 0.0)
    return -np.sum(terms, axis=axis)


def compute_ig_bald(pf: RBParticleFilter, vg: np.ndarray, *, T: float = 0.05, epsilon: float = 0.02) -> float:
    """Exact joint-entropy BALD: H[mixture] - mean_i[H[p_i]], where the
    mixture and each particle's own distribution are computed over the SAME
    shared local candidate set (the union of every particle's own
    uncertain-dot neighborhood) -- a requirement for the subtraction to be
    meaningful, since entropies over different supports aren't comparable.

    Bounded above by min(log(n_particles), log(K)) where K is the number of
    candidates used -- a genuine information-theoretic fact (this is
    exactly the mutual information between "which particle/hypothesis is
    correct" and "what would be observed", so it's capped by the entropy of
    EITHER side of that mutual information -- see this function's test
    suite for a direct check of both bounds). This corrects an earlier,
    incorrect claim (from this project's own design discussion) that
    IG_BALD is bounded by N_DOT*log(2) -- that bound was specific to (and a
    symptom of) the buggy summed-marginal construction this replaces.
    """
    vg = np.asarray(vg, dtype=np.float64)
    weights = np.array([p.weight for p in pf.particles])
    envs = [QArrayEnv(p.kalman.params) for p in pf.particles]
    predictions = np.stack([env.soft_prediction(vg, T=T) for env in envs])

    candidates = _local_candidate_states(predictions, epsilon=epsilon)
    K = candidates.shape[0]

    if K == 1:
        # Every particle is fully confident and agrees on the same single
        # state -- no uncertainty anywhere, BALD is trivially 0.
        return 0.0

    P = np.zeros((len(pf.particles), K))
    for i, env in enumerate(envs):
        F = env.free_energy_at_candidates(candidates, vg)
        neg_F_over_T = -F / T
        neg_F_over_T -= neg_F_over_T.max()  # numerical stability, doesn't change softmax result
        w = np.exp(neg_F_over_T)
        P[i] = w / w.sum()

    mixture = weights @ P  # shape (K,)
    H_mixture = _entropy(mixture)
    per_particle_H = _entropy(P, axis=1)  # shape (n_particles,)
    mean_particle_H = weights @ per_particle_H

    return float(H_mixture - mean_particle_H)
