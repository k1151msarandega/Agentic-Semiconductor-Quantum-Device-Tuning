from __future__ import annotations

import numpy as np

from bald import compute_ig_bald
from fim import compute_ig_fim
from kalman import DEFAULT_SCALE
from particle_filter import RBParticleFilter
from qarray_env import QArrayEnv


def compute_ig_total(
    pf: RBParticleFilter,
    vg: np.ndarray,
    R: np.ndarray,
    *,
    w_discrete: float,
    w_continuous: float,
    T: float = 0.05,
    scale: np.ndarray = DEFAULT_SCALE,
    envs: list["QArrayEnv"] | None = None,
    x0_bases: list[np.ndarray] | None = None,
) -> float:
    """Same as repo version, PLUS envs/x0_bases pass-through to
    compute_ig_bald/compute_ig_fim -- see those modules' docstrings for
    why (Primer Section 9's flagged-open perf question, confirmed real).
    """
    if scale.shape[0] != next(iter(pf.particles)).kalman.mu.shape[0]:
        raise ValueError(
            f"scale (len {scale.shape[0]}) does not match the tracked "
            f"parameter dimension."
        )

    ig_bald = compute_ig_bald(pf, vg, T=T, envs=envs)
    ig_fim = compute_ig_fim(pf, vg, R, T=T, scale=scale, x0_bases=x0_bases)
    return w_discrete * ig_bald + w_continuous * ig_fim
