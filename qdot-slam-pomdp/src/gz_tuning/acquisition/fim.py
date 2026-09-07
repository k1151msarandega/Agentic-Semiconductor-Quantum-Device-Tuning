"""
gz_tuning.acquisition.fim

IG_FIM: the Fisher-information-based half of Primer Section 5's two-dial
acquisition function. Closed form, no outcome-averaging needed -- the
Kalman covariance update is deterministic given a candidate vg (doesn't
depend on what's actually observed), which is exactly what makes this term
cheap relative to IG_BALD's need to consider outcomes.

PRIMER DISCREPANCY, flagged rather than silently resolved: Section 5 states
two different formulas for IG_FIM in two places:
  - Bullet-point definition: "IG_FIM(v) = Sum_i w_i * [logdet(Sigma_i) -
    logdet(Sigma_i'(v))]" -- no 1/2 factor.
  - Units section: "IG_FIM = 0.5 * [logdet(Sigma_prior,normalized) -
    logdet(Sigma_posterior,normalized)]" -- WITH a 1/2 factor, justified
    explicitly: "the literal differential-entropy-reduction formula (H =
    0.5*logdet(2*pi*e*Sigma), constant cancels in the difference) -- not an
    approximation."
This module implements the units-section version (WITH the 1/2 factor),
since it comes with an explicit, standard derivation (multivariate Gaussian
differential entropy genuinely has a 1/2 coefficient) while the bullet-point
version appears to be an imprecise restatement, not an independently
justified alternative. If you intended the bullet-point version instead,
this is a one-line change (drop the 0.5) -- but do so deliberately, not by
assuming this file made an arbitrary choice.
"""

from __future__ import annotations

import numpy as np

from kalman import DEFAULT_SCALE, N_PARAMS, ParticleKalmanFilter, normalized_logdet
from particle_filter import RBParticleFilter
from qarray_env import observation_jacobian


def _kalman_posterior_covariance(Sigma: np.ndarray, H: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Standard Kalman covariance update (Joseph form, matching kalman.py's
    own update() for consistency), WITHOUT needing an actual innovation --
    this is exactly the property that makes IG_FIM closed-form: the
    posterior covariance depends only on (Sigma, H, R), not on what value
    would actually be measured.
    """
    S = H @ Sigma @ H.T + R
    K = Sigma @ H.T @ np.linalg.inv(S)
    I_KH = np.eye(Sigma.shape[0]) - K @ H
    return I_KH @ Sigma @ I_KH.T + K @ R @ K.T


def compute_ig_fim_particle(
    kalman: ParticleKalmanFilter,
    vg: np.ndarray,
    R: np.ndarray,
    *,
    T: float = 0.05,
    scale: np.ndarray = DEFAULT_SCALE,
) -> float:
    """Single particle's own IG_FIM contribution at candidate vg: 0.5 *
    [logdet(Sigma_prior,normalized) - logdet(Sigma_posterior,normalized)].

    Handles the -inf edge case from normalized_logdet (Sigma already
    exactly singular, e.g. after many informative updates) explicitly:
    if Sigma_prior's logdet is already -inf, there's no further information
    to gain (already at a point estimate in some direction) -- returns 0.0
    rather than propagating a NaN from (-inf - -inf).
    """
    vg = np.asarray(vg, dtype=np.float64)
    H = observation_jacobian(kalman.params, vg, T=T)
    Sigma_posterior = _kalman_posterior_covariance(kalman.Sigma, H, R)

    logdet_prior = normalized_logdet(kalman.Sigma, scale=scale)
    if logdet_prior == -np.inf:
        return 0.0
    logdet_posterior = normalized_logdet(Sigma_posterior, scale=scale)

    return float(0.5 * (logdet_prior - logdet_posterior))


def compute_ig_fim(
    pf: RBParticleFilter,
    vg: np.ndarray,
    R: np.ndarray,
    *,
    T: float = 0.05,
    scale: np.ndarray = DEFAULT_SCALE,
) -> float:
    """Population-level IG_FIM: WEIGHTED MEAN of per-particle terms, per
    Primer Section 5's explicit instruction -- NOT min/max. Between-particle
    disagreement in current parameter MEANS is candidate-independent
    (doesn't vary with vg), so it's a constant offset that drops out of
    argmax_v entirely; weighted mean of within-particle terms is sufficient
    for choosing where to measure next.
    """
    total = 0.0
    for p in pf.particles:
        total += p.weight * compute_ig_fim_particle(p.kalman, vg, R, T=T, scale=scale)
    return total
