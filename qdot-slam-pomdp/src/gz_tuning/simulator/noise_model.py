from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from kalman import DEFAULT_SCALE, N_PARAMS, ParticleKalmanFilter, normalized_logdet
from qarray_env import DeviceParams, N_DOT, N_GATE, QArrayEnv, observation_jacobian


@dataclass(frozen=True)
class GaussianReadoutNoise:
    sigma: float

    def sample(self, true_value: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        true_value = np.asarray(true_value, dtype=np.float64)
        return true_value + rng.normal(loc=0.0, scale=self.sigma, size=true_value.shape)

    def R_matrix(self) -> np.ndarray:
        return np.eye(N_DOT) * (self.sigma ** 2)


def per_dot_distance_to_boundary(prediction: np.ndarray) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=np.float64)
    frac = prediction - np.floor(prediction)
    return np.abs(frac - 0.5)


def per_dot_sigma_eff(R: np.ndarray, H: np.ndarray, Sigma_total: np.ndarray) -> np.ndarray:
    predictive_spread = np.diag(H @ Sigma_total @ H.T)
    R_diag = np.diag(R)
    return np.sqrt(R_diag + predictive_spread)


def p_flip_per_dot(distance: np.ndarray, sigma_eff: np.ndarray) -> np.ndarray:
    distance = np.asarray(distance, dtype=np.float64)
    sigma_eff = np.asarray(sigma_eff, dtype=np.float64)
    p = np.zeros_like(distance)
    nonzero = sigma_eff > 0
    z = np.divide(distance[nonzero], sigma_eff[nonzero])
    p[nonzero] = 2.0 * (1.0 - norm.cdf(z))
    return p


def p_flip_total(p_per_dot: np.ndarray) -> float:
    p_per_dot = np.asarray(p_per_dot, dtype=np.float64)
    return float(1.0 - np.prod(1.0 - p_per_dot))


def compute_p_flip(
    prediction: np.ndarray, R: np.ndarray, H: np.ndarray, Sigma_total: np.ndarray
) -> float:
    distance = per_dot_distance_to_boundary(prediction)
    sigma_eff = per_dot_sigma_eff(R, H, Sigma_total)
    p_per_dot = p_flip_per_dot(distance, sigma_eff)
    return p_flip_total(p_per_dot)


def derive_consecutive_count(p_flip: float, *, p_target: float = 0.01, max_N: int = 10_000) -> int:
    if p_flip <= 0.0:
        return 1
    if p_flip >= 1.0:
        raise ValueError(f"p_flip={p_flip} >= 1.0")
    N = math.ceil(math.log(p_target) / math.log(p_flip))
    N = max(N, 1)
    if N > max_N:
        raise ValueError(f"Derived N={N} exceeds max_N={max_N}")
    return N


def calibrate_threshold_low(
    prior_mu: np.ndarray,
    prior_Sigma: np.ndarray,
    ground_truth_params: DeviceParams,
    *,
    n_measurements: int = 300,
    n_candidates_per_step: int = 30,
    R_sigma: float = 0.05,
    T: float = 0.05,
    vg_range: tuple[float, float] = (0.0, 15.0),
    scale: np.ndarray = DEFAULT_SCALE,
    rng: np.random.Generator | None = None,
) -> float:
    if rng is None:
        rng = np.random.default_rng()

    E_cm_intra1 = ground_truth_params.E_cm_intra1
    E_cm_intra2 = ground_truth_params.E_cm_intra2
    pkf = ParticleKalmanFilter(
        mu=prior_mu.copy(), Sigma=prior_Sigma.copy(),
        E_cm_intra1=E_cm_intra1, E_cm_intra2=E_cm_intra2,
    )
    truth_env = QArrayEnv(ground_truth_params)
    noise = GaussianReadoutNoise(sigma=R_sigma)
    R = noise.R_matrix()

    for _ in range(n_measurements):
        candidates = rng.uniform(vg_range[0], vg_range[1], size=(n_candidates_per_step, N_GATE))
        try:
            step_env = QArrayEnv(pkf.params)
            x0_base = np.diag(step_env._Cdd).copy()
        except Exception:
            x0_base = None

        best_score = -np.inf
        best_vg = candidates[0]
        for vg in candidates:
            try:
                H = observation_jacobian(pkf.params, vg, T=T, x0_base=x0_base)
            except Exception:
                continue
            score = np.trace(H @ pkf.Sigma @ H.T)
            if score > best_score:
                best_score = score
                best_vg = vg

        true_reading = truth_env.soft_prediction(best_vg, T=T)
        measured = noise.sample(true_reading, rng)
        pkf.update(measured, best_vg, R, T=T)

    return normalized_logdet(pkf.Sigma, scale=scale)


def derive_phase_thresholds(
    threshold_low: float, *, hysteresis_gap_nats: float = 0.5
) -> tuple[float, float]:
    return threshold_low, threshold_low + hysteresis_gap_nats
