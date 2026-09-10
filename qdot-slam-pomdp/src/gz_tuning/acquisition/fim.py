from __future__ import annotations

import numpy as np

from kalman import DEFAULT_SCALE, N_PARAMS, ParticleKalmanFilter, normalized_logdet
from particle_filter import RBParticleFilter
from qarray_env import build_capacitance_matrices, observation_jacobian


def _kalman_posterior_covariance(Sigma: np.ndarray, H: np.ndarray, R: np.ndarray) -> np.ndarray:
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
    x0_base: np.ndarray | None = None,
) -> float:
    """Same as repo version, PLUS an `x0_base` param: the particle's own
    already-solved self-capacitance diagonal (e.g. from a QArrayEnv built
    once per acquisition step by candidate_search.py), passed straight
    through to observation_jacobian to skip its own redundant base solve
    -- same fix pattern as bald.py's `envs` param, for the FIM term.

    BUG FIX (found via a real run): an extremely steep FD Jacobian --
    which happens near a sharp transition when the adaptive step in
    observation_jacobian has shrunk close to its floor -- can blow up the
    Kalman gain enough that Sigma_posterior picks up non-finite entries
    through floating-point overflow in the Joseph-form update. eigvalsh
    then raises LinAlgError('Eigenvalues did not converge'), crashing the
    whole particle filter. Confirmed via a real run (crashed at
    n_particles=20 during acquisition search, not during Phase 0).

    Fix: treat a non-finite H or Sigma_posterior as "this candidate is
    numerically untrustworthy for this particle" and contribute 0 IG,
    rather than letting it propagate into a crash -- the same
    "unobservable rather than broken" posture already used for the
    infeasible-perturbation fallback in qarray_env.py.
    """
    vg = np.asarray(vg, dtype=np.float64)
    H = observation_jacobian(kalman.params, vg, T=T, x0_base=x0_base)
    if not np.all(np.isfinite(H)):
        return 0.0

    Sigma_posterior = _kalman_posterior_covariance(kalman.Sigma, H, R)
    if not np.all(np.isfinite(Sigma_posterior)):
        return 0.0

    # normalized_logdet now floors individual eigenvalues instead of
    # returning -inf for the whole determinant (see kalman.py's docstring
    # for why), so it no longer returns -inf here -- this guard is dead
    # under the fixed version but left as a defensive check in case that
    # invariant changes again.
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
    x0_bases: list[np.ndarray] | None = None,
) -> float:
    total = 0.0
    for i, p in enumerate(pf.particles):
        x0_base = x0_bases[i] if x0_bases is not None else None
        total += p.weight * compute_ig_fim_particle(
            p.kalman, vg, R, T=T, scale=scale, x0_base=x0_base
        )
    return total
