from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from candidate_search import select_next_measurement
from kalman import DEFAULT_SCALE, N_PARAMS, ParticleKalmanFilter, normalized_logdet
from particle_init import CouplingStratum, SharedPriorSpec, init_particle_filter
from qarray_env import DeviceParams, N_DOT, N_GATE, QArrayEnv
from qarray_jax import jax_observation_jacobian


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
                # Analytic JAX Jacobian (matches belief/kalman.py); log_x0=None
                # lets it solve its own warm start if QArrayEnv construction failed.
                H = jax_observation_jacobian(
                    pkf.params, vg, T=T,
                    log_x0=None if x0_base is None else np.log(x0_base),
                )
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


def calibrate_red_threshold(
    shared_prior: SharedPriorSpec,
    coupling_strata: list[CouplingStratum],
    prior_sigma_diag: np.ndarray,
    ground_truth_params: DeviceParams,
    *,
    n_particles: int = 40,
    n_measurements: int = 300,
    n_candidates_per_step: int = 30,
    R_sigma: float = 0.05,
    T: float = 100.0,
    vg_range: tuple[float, float] = (0.0, 15.0),
    ess_resample_frac: float = 0.5,
    mu_jitter_scale: float | None = 0.1,
    target_ess_frac: float | None = 0.5,
    scale: np.ndarray = DEFAULT_SCALE,
    rng: np.random.Generator | None = None,
) -> float:
    """The red-line counterpart calibrate_threshold_low cannot provide:
    that function constructs a single ParticleKalmanFilter, so
    between-particle covariance is mathematically undefined there, not
    just uncalibrated -- see phase_switch.py's module docstring for the
    bug this was found to cause (one shared threshold applied to both
    lines, with red's line structurally unable to ever approach a
    threshold calibrated from a single-particle trajectory).

    Runs a REAL init_particle_filter ensemble -- same prior machinery a
    real run uses, per Section 7's invariant that the prior scheme must
    be identical everywhere it's applied, calibration included -- through
    n_measurements steps of the REAL two-dial acquisition
    (candidate_search.select_next_measurement), not a simplified proxy:
    unlike calibrate_threshold_low's single-Gaussian trace-based
    heuristic (a reasonable stand-in there, since one Gaussian's
    convergence rate is not very sensitive to acquisition sophistication),
    red's trajectory through resampling + adaptive tempering is exactly
    the thing being calibrated, so it needs to be driven by the same
    acquisition function a real run would actually use, not an idealized
    substitute -- otherwise the calibration measures a different system
    than the one whose threshold it's setting.

    T defaults to 100.0 here, NOT calibrate_threshold_low's 0.05 default
    -- that default is a known leftover from before the primer's
    temperature calibration fix; not repeated here deliberately.

    Whether one calibration run is reusable across an entire coupling
    sweep (same prior/particle_count, varying only cross_capacitance) or
    needs recalibrating per sweep point is an open question -- see this
    project's Claim 2 stock-take. Section 6a's blue-line dependency list
    (sigma, particle_count, prior_width -- no ground truth) suggests
    reuse is a reasonable default, but red's convergence RATE plausibly
    depends on how sharp the true transitions are, which cross_capacitance
    itself affects; treat reuse as an assumption to spot-check (compare
    the calibrated value at two different cross_capacitance settings)
    rather than one to take for granted.
    """
    if rng is None:
        rng = np.random.default_rng()

    pf = init_particle_filter(
        n_particles=n_particles,
        shared_prior=shared_prior,
        coupling_strata=coupling_strata,
        prior_sigma_diag=prior_sigma_diag,
        E_cm_intra1=ground_truth_params.E_cm_intra1,
        E_cm_intra2=ground_truth_params.E_cm_intra2,
        rng=rng,
    )
    ground_truth = QArrayEnv(ground_truth_params)
    noise = GaussianReadoutNoise(sigma=R_sigma)
    R = noise.R_matrix()

    for _ in range(n_measurements):
        search = select_next_measurement(
            pf, R, w_discrete=1.0, w_continuous=1.0, vg_range=vg_range,
            n_candidates=n_candidates_per_step, T=T, rng=rng,
        )
        true_reading = ground_truth.soft_prediction(search.best_vg, T=T)
        measured = noise.sample(true_reading, rng)
        pf.update(measured, search.best_vg, R, T=T, envs=search.envs,
                   target_ess_frac=target_ess_frac)
        if pf.effective_sample_size() < ess_resample_frac * n_particles:
            pf.resample(rng, mu_jitter_scale=mu_jitter_scale)

    return pf.between_particle_covariance(scale=scale)


def derive_phase_thresholds(
    threshold_low: float, *, hysteresis_gap_nats: float = 0.5
) -> tuple[float, float]:
    return threshold_low, threshold_low + hysteresis_gap_nats
