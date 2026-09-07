"""
gz_tuning.simulator.noise_model

Simulated measurement noise, and the two principled-threshold derivations
Primer Section 6 (phase-switch hysteresis) and Section 8 (consecutive-
stable-reads count N) call for. This is a full rewrite of an earlier
version -- see git history / prior review conversation for what changed and
why. Summary of what's fixed here, since it's easy to lose track:

  1. distance-to-boundary is now computed from the particle's own current
     MEAN prediction (soft_prediction at the current mu), not conflated
     with predictive uncertainty -- an earlier version used
     sqrt(diag(H @ Sigma @ H^T)) AS IF it were a distance, which is wrong:
     that's an epistemic (uncertainty) quantity, not a positional one. Fixed
     structure: distance comes from the mean; H @ Sigma @ H^T is correctly
     redeployed as an INFLATION to the effective noise variance instead.
  2. sigma_eff is computed PER DOT (not a single scalar), and combined via
     the exact independent-OR formula (not a union bound -- unnecessary
     here since per-dot noise is already modeled as independent, so the
     exact formula costs nothing extra and is strictly tighter).
  3. threshold_low is no longer a placeholder sigma**2 formula (which
     conflated readout-noise units with parameter-covariance units, the
     same category of bug as #1). It's now empirically calibrated by
     running a real Kalman update loop (belief/kalman.py) against a fixed
     ground-truth device and reading off the achieved normalized logdet(Sigma)
     -- per Section 12's resolution: a single-shot closed form is
     impossible here (a single query gives at most 4 independent readout
     dimensions against 5 tracked parameters -- generically rank-deficient,
     verified), so empirical calibration is the correct approach, not a
     fallback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from kalman import DEFAULT_SCALE, N_PARAMS, ParticleKalmanFilter, normalized_logdet
from qarray_env import DeviceParams, N_DOT, N_GATE, QArrayEnv, observation_jacobian


@dataclass(frozen=True)
class GaussianReadoutNoise:
    """i.i.d. Gaussian sensor noise on each dot's scalar occupation readout.
    Represents R's diagonal (R = sigma^2 * I) in the Kalman update sense --
    this is measurement noise, not parameter uncertainty; the two must not
    be conflated (see module docstring point 1's history)."""

    sigma: float

    def sample(self, true_value: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        true_value = np.asarray(true_value, dtype=np.float64)
        return true_value + rng.normal(loc=0.0, scale=self.sigma, size=true_value.shape)

    def R_matrix(self) -> np.ndarray:
        """Observation noise covariance, diagonal, shape (N_DOT, N_DOT)."""
        return np.eye(N_DOT) * (self.sigma ** 2)


def per_dot_distance_to_boundary(prediction: np.ndarray) -> np.ndarray:
    """Distance from each dot's CURRENT MEAN prediction to its nearest
    integer-occupation decision boundary (at half-integers). Formula:
    |frac(prediction) - 0.5| -- zero exactly AT a boundary (frac=0.5),
    maximal (0.5) exactly at an integer (frac=0 or 1, i.e. confidently in a
    stable plateau). This is a positional quantity -- computed from the
    mean, not from any covariance -- deliberately kept separate from
    sigma_eff below.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    frac = prediction - np.floor(prediction)
    return np.abs(frac - 0.5)


def per_dot_sigma_eff(R: np.ndarray, H: np.ndarray, Sigma_total: np.ndarray) -> np.ndarray:
    """Effective per-dot noise std: sensor readout noise inflated by
    parameter-uncertainty-induced predictive spread. R @ diag entries are
    genuine sensor noise; diag(H @ Sigma_total @ H^T) is how much the
    prediction itself wobbles due to not knowing the continuous parameters
    exactly. MUST use Sigma_total (within + between particle covariance,
    Primer Section 6), not a single particle's own Sigma alone -- using
    within-particle Sigma only would reintroduce exactly the overconfidence
    bug Section 6's two-line (red/blue) diagnostic exists to catch.
    """
    predictive_spread = np.diag(H @ Sigma_total @ H.T)
    R_diag = np.diag(R)
    return np.sqrt(R_diag + predictive_spread)


def p_flip_per_dot(distance: np.ndarray, sigma_eff: np.ndarray) -> np.ndarray:
    """P(noise alone flips dot i's reading across its nearest boundary),
    per dot, given that dot's distance-to-boundary and effective noise std.
    Two-sided (symmetric) flip risk, matching Primer Section 8 failure mode
    A (a momentary swing in either direction)."""
    distance = np.asarray(distance, dtype=np.float64)
    sigma_eff = np.asarray(sigma_eff, dtype=np.float64)
    p = np.zeros_like(distance)
    nonzero = sigma_eff > 0
    z = np.divide(distance[nonzero], sigma_eff[nonzero])
    p[nonzero] = 2.0 * (1.0 - norm.cdf(z))
    # sigma_eff == 0 (no noise at all, degenerate) -> p_flip = 0 exactly
    return p


def p_flip_total(p_per_dot: np.ndarray) -> float:
    """Exact P(at least one dot's reading flips), assuming independent
    per-dot noise (matches this file's GaussianReadoutNoise, which is i.i.d.
    across dots by construction). Uses the exact independent-OR formula
    1 - prod(1 - p_i), not a union bound -- the union bound is only needed
    when per-dot events can't be assumed independent or joint computation is
    intractable; neither applies here, so the exact form is used since it's
    free and strictly tighter. If a future noise model introduces cross-dot
    correlation, replace this with the union bound as a documented fallback,
    not silently.
    """
    p_per_dot = np.asarray(p_per_dot, dtype=np.float64)
    return float(1.0 - np.prod(1.0 - p_per_dot))


def compute_p_flip(
    prediction: np.ndarray,
    R: np.ndarray,
    H: np.ndarray,
    Sigma_total: np.ndarray,
) -> float:
    """End-to-end: mean prediction + noise model + Jacobian + total
    parameter covariance -> single scalar flip probability. Convenience
    wrapper chaining the four functions above, since callers (e.g.
    derive_consecutive_count) generally want the final number, not the
    intermediate per-dot arrays -- though those remain available
    individually for diagnostics/logging (Primer Section 6's explicit
    preference for two separately-inspectable lines over one combined
    number applies here too)."""
    distance = per_dot_distance_to_boundary(prediction)
    sigma_eff = per_dot_sigma_eff(R, H, Sigma_total)
    p_per_dot = p_flip_per_dot(distance, sigma_eff)
    return p_flip_total(p_per_dot)


def derive_consecutive_count(
    p_flip: float,
    *,
    p_target: float = 0.01,
    max_N: int = 10_000,
) -> int:
    """Smallest N such that p_flip**N < p_target. Takes a scalar p_flip
    directly (assemble it via compute_p_flip first) -- deliberately
    decoupled from HOW p_flip was computed, so this function's own
    correctness doesn't depend on any physics assumptions.

    Assumes independence across consecutive reads (a real simplification --
    correlated noise, e.g. 1/f, would need a different derivation).
    """
    if p_flip <= 0.0:
        return 1
    if p_flip >= 1.0:
        raise ValueError(
            f"p_flip={p_flip} >= 1.0; no finite N satisfies P(N consecutive) "
            f"< {p_target}."
        )
    N = math.ceil(math.log(p_target) / math.log(p_flip))
    N = max(N, 1)
    if N > max_N:
        raise ValueError(
            f"Derived N={N} exceeds max_N={max_N}; p_flip={p_flip} is too "
            f"small relative to p_target={p_target} for this to be "
            f"achievable in practice. Loosen p_target or check inputs."
        )
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
    """Empirically calibrate threshold_low: run a real Kalman update loop
    (belief/kalman.py) against a fixed ground-truth device, using a
    sequence of greedily-chosen informative measurements, and return the
    achieved normalized logdet(Sigma) after n_measurements steps.

    Per Primer Section 6 ("should come from the noise model") and Section
    12's resolution: NOT a closed form (a single query gives at most 4
    independent readout dimensions against 5 tracked parameters --
    generically rank-deficient, verified in this project's development;
    building an idealized optimal-query-sequence closed form would be a
    real, separate optimization problem, out of scope). This function
    reuses existing simulation machinery instead.

    Candidate selection: at each step, sample n_candidates_per_step random
    vg points uniformly in vg_range^4, evaluate observation_jacobian at the
    CURRENT filter mean for each, and greedily pick the one maximizing
    trace(H @ Sigma @ H^T) (a simple, general information-gain proxy -- NOT
    the closed-form informative-transition-point formula found during
    qarray_env.py's development, since that formula was derived for one
    specific symmetric dot-pair case and does not generalize; this greedy
    proxy is deliberately more general at the cost of being less precisely
    optimal).

    result depends on (sigma, n_measurements, prior_Sigma, candidate search
    budget) -- NOT a universal constant. Recompute per experiment config
    rather than caching across different noise/prior/sweep settings.
    """
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
        # pkf.params is unchanged across every candidate within this step --
        # solve it once and share via x0_base, rather than paying for a
        # redundant base solve inside observation_jacobian on each of the
        # n_candidates_per_step calls (found during review: this was
        # previously re-solving the same params 15-30 times per step).
        try:
            step_env = QArrayEnv(pkf.params)
            x0_base = np.diag(step_env._Cdd).copy()
        except Exception:
            x0_base = None  # fall back to per-call cold-start if pkf.params itself is at a rough spot

        best_score = -np.inf
        best_vg = candidates[0]
        for vg in candidates:
            try:
                H = observation_jacobian(pkf.params, vg, T=T, x0_base=x0_base)
            except Exception:
                continue  # skip candidates that fail (e.g. transient infeasibility during search)
            score = np.trace(H @ pkf.Sigma @ H.T)
            if score > best_score:
                best_score = score
                best_vg = vg

        true_reading = truth_env.soft_prediction(best_vg, T=T)
        measured = noise.sample(true_reading, rng)
        pkf.update(measured, best_vg, R, T=T)

    return normalized_logdet(pkf.Sigma, scale=scale)


def derive_phase_thresholds(
    threshold_low: float,
    *,
    hysteresis_gap_nats: float = 0.5,
) -> tuple[float, float]:
    """(threshold_low, threshold_high) for Primer Section 6's dual-line
    hysteresis phase switch, given an already-calibrated threshold_low
    (e.g. from calibrate_threshold_low). Uses an ADDITIVE nats offset for
    the hysteresis gap, not a multiplicative one -- normalized logdet can
    be negative (Sigma's determinant can be <1 in normalized coordinates),
    so multiplying by (1+gap) would make an already-negative threshold MORE
    negative (i.e. tighter, backwards from the intended looser
    threshold_high). An additive gap in log-space corresponds to a genuine
    multiplicative gap in the underlying determinant/uncertainty-volume,
    which is the physically meaningful framing regardless of sign.
    """
    return threshold_low, threshold_low + hysteresis_gap_nats
