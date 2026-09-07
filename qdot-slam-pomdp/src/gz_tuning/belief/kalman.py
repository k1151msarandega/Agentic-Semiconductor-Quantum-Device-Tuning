"""
gz_tuning.belief.kalman

Per-particle independent Kalman filter over the 5 tracked continuous device
parameters (E_c1, E_c2, alpha1, alpha2, cross_capacitance -- t_c dropped,
see qarray_env.py's module docstring). Primer Section 4: each particle in
the Rao-Blackwellized discrete/continuous belief owns ONE of these filters,
independent of every other particle's -- never shared, never averaged
across particles (Section 2's rejection of a shared/global filter, for the
data-association reasons documented there).

DESIGN CHOICE, not settled by the primer -- flagged rather than assumed:

The EKF update here runs entirely in RAW physical units (the same units
DeviceParams/observation_jacobian use), NOT the normalized/whitened
coordinates Section 5 specifies for IG_FIM's units to be literal nats. This
is deliberate: keeping the core Kalman recursion in raw units avoids mixing
a scaling transform into the update math itself, where a units bug would be
easy to introduce and hard to notice (exactly the failure mode already hit
twice in this project -- the factor-of-2 and the sigma-conflation issues).
Normalization is instead treated as a presentation-layer concern: a
separate `normalized_logdet` function here converts a raw-units covariance
into the whitened logdet Section 5 actually needs, given an explicit scale
vector. This keeps the two concerns (correct EKF math; correct
information-theoretic units) independently checkable. The scale vector
itself (what counts as "one unit of normalized variance" per parameter) is
NOT derived here -- Primer Section 12 lists this as still open ("exact
N/p_conf/epsilon derivation... not yet computed"); pass your own scale
vector once that's decided, or use DEFAULT_SCALE below as an explicitly
flagged placeholder.

INNOVATION MODEL: the update's predicted measurement and Jacobian both come
from qarray_env.py's soft_prediction/observation_jacobian at a fixed
temperature T (default matches observation_jacobian's T=0.05). This means
the filter is comparing a (possibly sensor-noisy) occupation reading
against a THERMALLY-SOFTENED model prediction, not the hard rounded ground
state -- necessary for the Jacobian to be well-defined at all (see
qarray_env.py's docstring for why the hard ground state can't be
differentiated). T is a real, not-yet-independently-calibrated modeling
choice here too -- see qarray_env.py's own flag on this.

FEASIBILITY GUARD: an EKF update to (E_c1, E_c2, alpha1, alpha2,
cross_capacitance) can push the mean estimate into a region
InfeasibleDeviceParamsError would reject (Primer's real, verified
alpha/E_c capacitance-budget constraint -- see qarray_env.py). This filter
does NOT silently allow that: update() checks feasibility of the proposed
new mean and backs off (halves the step) up to a bounded number of times
before giving up and keeping the prior mean unchanged with a raised flag on
the returned KalmanUpdateResult, rather than crashing or silently drifting
into nonsense. This is a real design decision, not a primer-specified one --
revisit if it causes particles to "stick" near the feasibility boundary
more than expected.
"""

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

# Placeholder normalization scale -- see module docstring. These are NOT
# derived from anything; they are round-number stand-ins for "roughly one
# unit of prior uncertainty" per parameter, in the same units DeviceParams
# uses. Replace with real values once Primer Section 12's open item
# (exact N/p_conf/epsilon derivation) is resolved.
DEFAULT_SCALE = np.array([1.0, 1.0, 0.05, 0.05, 0.05])


def params_to_vector(params: DeviceParams) -> np.ndarray:
    """Extract the 5 tracked fields from a DeviceParams into a plain array,
    in TRACKED_PARAM_FIELDS order. E_cm_intra1/E_cm_intra2 (untracked,
    fixed per-particle structural assumptions, not estimated) are not
    included -- callers needing a full DeviceParams back should use
    vector_to_params, which requires supplying those separately.
    """
    return np.array([getattr(params, f) for f in TRACKED_PARAM_FIELDS], dtype=np.float64)


def vector_to_params(
    vec: np.ndarray, *, E_cm_intra1: float, E_cm_intra2: float
) -> DeviceParams:
    """Inverse of params_to_vector. E_cm_intra1/E_cm_intra2 must be supplied
    explicitly since they are not part of the tracked state vector -- a
    particle's Kalman filter has no opinion on them; whatever code
    constructs/owns this filter is responsible for supplying the particle's
    fixed structural assumption for those two fields.
    """
    kwargs = dict(zip(TRACKED_PARAM_FIELDS, vec))
    return DeviceParams(E_cm_intra1=E_cm_intra1, E_cm_intra2=E_cm_intra2, **kwargs)


def normalized_logdet(
    Sigma_raw: np.ndarray, scale: np.ndarray = DEFAULT_SCALE, *, tol: float = 1e-10
) -> float:
    """logdet of Sigma_raw (in raw physical units) converted to the
    normalized/whitened coordinates Primer Section 5 requires for IG_FIM's
    nats-unit formula to be literal. Uses the identity
    logdet(diag(1/s) @ Sigma @ diag(1/s)) = logdet(Sigma) - 2*sum(log(s))
    rather than explicitly forming the whitened matrix -- equivalent, and
    avoids an extra matrix multiply, but the two should agree; see the test
    that checks this explicitly.

    Distinguishes two cases that a plain slogdet(Sigma).sign check CANNOT
    safely distinguish: a matrix with eigenvalues [0, -5, 3] has
    determinant 0 (sign=0) despite having a genuine negative eigenvalue --
    sign=0 alone does NOT imply PSD. This function checks eigenvalues
    directly (via eigvalsh, appropriate for the symmetric matrices this is
    meant for):
      - any eigenvalue < -tol: genuinely broken (not PSD) -> raises. This
        indicates a real problem upstream (e.g. an update producing a
        non-PSD covariance), not something to paper over.
      - all eigenvalues >= -tol but the smallest is ~0 (a genuinely
        singular, valid PSD matrix -- e.g. between_particle_covariance for
        particles with near-identical means): returns -inf. This is a
        CORRECT, meaningful result (zero uncertainty in at least one
        direction), not an error condition -- callers like the phase-switch
        logic (Primer Section 6) should treat -inf as "definitely below any
        finite threshold_low", not crash on it.
    """
    Sigma_raw = np.asarray(Sigma_raw, dtype=np.float64)
    eigvals = np.linalg.eigvalsh(Sigma_raw)
    if np.any(eigvals < -tol):
        raise ValueError(
            f"Sigma_raw is not positive semi-definite (min eigenvalue="
            f"{eigvals.min():.3e}) -- this indicates a real problem "
            f"upstream (e.g. an update that produced a non-PSD "
            f"covariance), not something to paper over here."
        )
    if eigvals.min() < tol:
        return -np.inf
    logdet_raw = np.sum(np.log(eigvals))
    return float(logdet_raw - 2.0 * np.sum(np.log(scale)))


@dataclass
class KalmanUpdateResult:
    """Outcome of a single update() call -- returned rather than just
    mutating in place, so callers (e.g. the phase-switch/convergence logic
    in Section 6/8, which needs to know if a particle's filter is behaving
    normally) can inspect what happened without re-deriving it."""

    accepted: bool  # False if the feasibility guard exhausted its retries and left mu unchanged
    n_backoffs: int  # how many step-halvings were needed (0 = normal update)
    innovation: np.ndarray  # measured - predicted, at whatever step size was ultimately used


@dataclass
class ParticleKalmanFilter:
    """One particle's independent belief over the 5 tracked continuous
    device parameters. mu/Sigma are plain arrays in TRACKED_PARAM_FIELDS
    order and RAW physical units (see module docstring for why raw, not
    normalized). E_cm_intra1/E_cm_intra2 are carried as fixed, non-tracked
    structural parameters -- this filter does not estimate them (Primer:
    not part of the 5-parameter Kalman state).
    """

    mu: np.ndarray  # shape (5,)
    Sigma: np.ndarray  # shape (5,5), must stay positive definite
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
        """Current mean estimate as a full DeviceParams, for constructing a
        QArrayEnv (e.g. to query predictions, or feed into build_capacitance_matrices)."""
        return vector_to_params(self.mu, E_cm_intra1=self.E_cm_intra1, E_cm_intra2=self.E_cm_intra2)

    def predict(self, Q: np.ndarray | None = None) -> None:
        """Process step. Default Q=None means zero process noise -- device
        parameters are physically static during a run (this is parameter
        ESTIMATION of a fixed unknown, not tracking a moving target), so
        the default is a pure static-parameter EKF (mu unchanged, Sigma
        unchanged). An explicit Q can be supplied if you want to guard
        against the filter becoming overconfident and unable to recover
        from a bad early update -- not done by default, since that's a
        real modeling choice with no primer-specified value."""
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
        """Standard EKF update (Joseph form for numerical stability), with
        a feasibility guard: if the proposed new mean is not a constructible
        DeviceParams (InfeasibleDeviceParamsError), halve the innovation
        step and retry, up to max_backoffs times, before giving up and
        leaving mu/Sigma unchanged.

        measured_occupation: length-4 array, e.g. from a noisy sensor
        reading (GaussianReadoutNoise.sample or similar) -- compared
        against this filter's own soft_prediction at its current mean.
        vg: length-4 voltage point the measurement was taken at.
        R: (4,4) observation noise covariance (raw occupation-reading units).
        env: optional pre-solved QArrayEnv for this filter's CURRENT params
        (self.params) -- if the caller already constructed one (e.g.
        particle_filter.py's update(), which needs its own soft_prediction
        for importance weighting before calling this), pass it here to
        avoid re-solving the same joint capacitance problem a second time.
        Caller is responsible for ensuring env actually matches self.params
        -- not checked here (would defeat the point of skipping the solve).
        """
        measured_occupation = np.asarray(measured_occupation, dtype=np.float64)
        vg = np.asarray(vg, dtype=np.float64)

        current_params = self.params
        # Solve current_params ONCE (or reuse caller's), share it between
        # observation_jacobian's warm-start and the predicted-measurement
        # query -- found during review that these were previously two
        # INDEPENDENT solves of the same params.
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
            try:
                QArrayEnv(candidate_params)  # feasibility check via construction
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
        # Joseph form: numerically safer than (I-KH)@Sigma alone, guarantees
        # PSD result even with small numerical errors in K.
        self.Sigma = I_KH @ self.Sigma @ I_KH.T + K @ R @ K.T

        return KalmanUpdateResult(
            accepted=True, n_backoffs=n_backoffs, innovation=scale * innovation_full
        )
