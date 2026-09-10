from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qarray_env import (
    DeviceParams,
    InfeasibleDeviceParamsError,
    QArrayEnv,
    TRACKED_PARAM_FIELDS,
    observation_jacobian,
)

N_PARAMS = len(TRACKED_PARAM_FIELDS)

DEFAULT_SCALE = np.array([1.0, 1.0, 0.05, 0.05, 0.05])


def params_to_vector(params: DeviceParams) -> np.ndarray:
    return np.array([getattr(params, f) for f in TRACKED_PARAM_FIELDS], dtype=np.float64)


def vector_to_params(
    vec: np.ndarray, *, E_cm_intra1: float, E_cm_intra2: float
) -> DeviceParams:
    kwargs = dict(zip(TRACKED_PARAM_FIELDS, vec))
    return DeviceParams(E_cm_intra1=E_cm_intra1, E_cm_intra2=E_cm_intra2, **kwargs)


def normalized_logdet(
    Sigma_raw: np.ndarray, scale: np.ndarray = DEFAULT_SCALE, *, tol: float = 1e-10
) -> float:
    """BUG FIX (found via a real run): the original version returned
    -inf for the WHOLE determinant the instant ANY single eigenvalue fell
    below `tol` -- including a legitimately tiny-but-real eigenvalue (a
    particle that's genuinely become very certain along one direction,
    e.g. after many phase-0 measurements), not just PSD numerical noise.
    Downstream, compute_ig_fim_particle computes
    `0.5*(logdet_prior - logdet_posterior)`, so a -inf posterior logdet
    produces IG_FIM = +inf -- confirmed in a real run (candidates
    scoring "inf" once particle covariances tightened enough). +inf
    breaks argmax comparisons (ties) and any downstream aggregation.

    Fix: floor each eigenvalue individually at `tol` before summing logs,
    rather than discarding the whole determinant when the smallest one is
    small. Still raises on a genuinely negative eigenvalue (real PSD
    violation, a bug elsewhere worth surfacing loudly) -- only the
    near-zero-but-nonnegative case is now floored instead of treated as
    -inf.
    """
    Sigma_raw = np.asarray(Sigma_raw, dtype=np.float64)
    eigvals = np.linalg.eigvalsh(Sigma_raw)
    if np.any(eigvals < -tol):
        raise ValueError(
            f"Sigma_raw is not positive semi-definite (min eigenvalue={eigvals.min():.3e})"
        )
    eigvals_floored = np.maximum(eigvals, tol)
    logdet_raw = np.sum(np.log(eigvals_floored))
    return float(logdet_raw - 2.0 * np.sum(np.log(scale)))


@dataclass
class KalmanUpdateResult:
    accepted: bool
    n_backoffs: int
    innovation: np.ndarray


@dataclass
class ParticleKalmanFilter:
    mu: np.ndarray
    Sigma: np.ndarray
    E_cm_intra1: float
    E_cm_intra2: float
    max_backoffs: int = 5

    def __post_init__(self) -> None:
        self.mu = np.asarray(self.mu, dtype=np.float64)
        self.Sigma = np.asarray(self.Sigma, dtype=np.float64)
        if self.mu.shape != (N_PARAMS,):
            raise ValueError(f"mu must have shape ({N_PARAMS},), got {self.mu.shape}")
        if self.Sigma.shape != (N_PARAMS, N_PARAMS):
            raise ValueError(f"Sigma must have shape ({N_PARAMS},{N_PARAMS}), got {self.Sigma.shape}")

    @property
    def params(self) -> DeviceParams:
        return vector_to_params(self.mu, E_cm_intra1=self.E_cm_intra1, E_cm_intra2=self.E_cm_intra2)

    def predict(self, Q: np.ndarray | None = None) -> None:
        if Q is not None:
            self.Sigma = self.Sigma + Q

    def update(
        self,
        measured_occupation: np.ndarray,
        vg: np.ndarray,
        R: np.ndarray,
        *,
        T: float = 0.05,
        env: "QArrayEnv | None" = None,
    ) -> KalmanUpdateResult:
        measured_occupation = np.asarray(measured_occupation, dtype=np.float64)
        vg = np.asarray(vg, dtype=np.float64)

        current_params = self.params
        current_env = env if env is not None else QArrayEnv(current_params)
        x0_base = np.diag(current_env._Cdd).copy()
        H = observation_jacobian(current_params, vg, T=T, x0_base=x0_base)
        predicted = current_env.soft_prediction(vg, T=T)
        innovation_full = measured_occupation - predicted

        S = H @ self.Sigma @ H.T + R
        K = self.Sigma @ H.T @ np.linalg.inv(S)

        n_backoffs = 0
        scale = 1.0
        while True:
            mu_candidate = self.mu + scale * (K @ innovation_full)
            candidate_params = vector_to_params(
                mu_candidate, E_cm_intra1=self.E_cm_intra1, E_cm_intra2=self.E_cm_intra2
            )
            # BUG FIX (same root cause as fim.py's compute_ig_fim_particle):
            # an extremely steep FD Jacobian can produce a Kalman gain K
            # whose resulting mu_candidate step is huge, or whose
            # posterior Sigma computation (below) overflows to non-finite
            # values. Treat non-finite candidates the same way an
            # infeasible-QArray candidate is already treated -- shrink
            # the step and retry, rather than letting NaN/Inf propagate
            # into the belief and crash a later logdet call.
            if not np.all(np.isfinite(mu_candidate)):
                n_backoffs += 1
                if n_backoffs > self.max_backoffs:
                    return KalmanUpdateResult(
                        accepted=False, n_backoffs=n_backoffs, innovation=innovation_full
                    )
                scale *= 0.5
                continue
            try:
                QArrayEnv(candidate_params)
                break
            except InfeasibleDeviceParamsError:
                n_backoffs += 1
                if n_backoffs > self.max_backoffs:
                    return KalmanUpdateResult(
                        accepted=False, n_backoffs=n_backoffs, innovation=innovation_full
                    )
                scale *= 0.5

        self.mu = mu_candidate
        I_KH = np.eye(N_PARAMS) - K @ H
        Sigma_candidate = I_KH @ self.Sigma @ I_KH.T + K @ R @ K.T
        if not np.all(np.isfinite(Sigma_candidate)):
            # Same non-finite-overflow failure mode, but surfacing after
            # the mu update this time. Keep the (finite, feasibility-
            # checked) mu update but fall back to leaving Sigma
            # unchanged rather than propagating NaN/Inf into the belief.
            return KalmanUpdateResult(
                accepted=True, n_backoffs=n_backoffs, innovation=scale * innovation_full
            )
        self.Sigma = Sigma_candidate

        return KalmanUpdateResult(
            accepted=True, n_backoffs=n_backoffs, innovation=scale * innovation_full
        )
