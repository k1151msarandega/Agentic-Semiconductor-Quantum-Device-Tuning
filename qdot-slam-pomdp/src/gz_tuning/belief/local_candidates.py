"""
gz_tuning.belief.local_candidates

Extracted from bald.py (its _local_candidate_states/_entropy were exactly
the right machinery for particle_filter.py's mixture-likelihood fix too,
but bald.py already imports particle_filter.py, so importing bald FROM
particle_filter would be circular). Shared here instead.

Used by:
  - bald.py: entropy of the local discrete distribution, per particle
  - particle_filter.py: MIXTURE likelihood over local discrete candidates
    (see particle_filter.py's update() docstring for why a single
    point-prediction likelihood was the real bug behind the ESS=1
    collapse found in a real multi-seed sweep -- ESS=1.0 in 15/15 runs,
    traced to a single ~200+ nat residual penalty on particles that were
    plausible but on the wrong side of a transition boundary).
"""

from __future__ import annotations

import itertools

import numpy as np


def local_candidate_states(predictions: np.ndarray, *, epsilon: float = 0.02) -> np.ndarray:
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


def entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    p = np.clip(p, 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log(p), 0.0)
    return -np.sum(terms, axis=axis)
