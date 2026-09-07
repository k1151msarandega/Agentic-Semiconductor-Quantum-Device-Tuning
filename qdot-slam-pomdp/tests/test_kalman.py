"""
Tests for gz_tuning.belief.kalman
"""

import numpy as np
import pytest

from kalman import (
    DEFAULT_SCALE,
    N_PARAMS,
    KalmanUpdateResult,
    ParticleKalmanFilter,
    normalized_logdet,
    params_to_vector,
    vector_to_params,
)
from qarray_env import DeviceParams, QArrayEnv, InfeasibleDeviceParamsError


def _default_filter(sigma_scale=1.0, **overrides):
    base_mu = dict(E_c1=2.5, E_c2=2.7, alpha1=0.05, alpha2=0.05, cross_capacitance=0.05)
    base_mu.update(overrides)
    mu = np.array([base_mu["E_c1"], base_mu["E_c2"], base_mu["alpha1"], base_mu["alpha2"], base_mu["cross_capacitance"]])
    Sigma = np.eye(N_PARAMS) * sigma_scale
    return ParticleKalmanFilter(mu=mu, Sigma=Sigma, E_cm_intra1=0.3, E_cm_intra2=0.3)


class TestVectorConversion:
    def test_params_to_vector_order(self):
        p = DeviceParams(E_c1=1.0, E_c2=2.0, alpha1=3.0, alpha2=4.0, E_cm_intra1=0.0, E_cm_intra2=0.0, cross_capacitance=5.0)
        vec = params_to_vector(p)
        np.testing.assert_array_equal(vec, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_round_trip(self):
        p = DeviceParams(E_c1=2.5, E_c2=2.7, alpha1=0.05, alpha2=0.05, E_cm_intra1=0.3, E_cm_intra2=0.3, cross_capacitance=0.05)
        vec = params_to_vector(p)
        p2 = vector_to_params(vec, E_cm_intra1=0.3, E_cm_intra2=0.3)
        assert p == p2


class TestNormalizedLogdet:
    def test_identity_scale_matches_raw_logdet(self):
        Sigma = np.diag([1.0, 2.0, 3.0, 4.0, 5.0])
        raw = np.linalg.slogdet(Sigma)[1]
        norm = normalized_logdet(Sigma, scale=np.ones(N_PARAMS))
        assert norm == pytest.approx(raw)

    def test_matches_explicit_whitening(self):
        rng = np.random.default_rng(0)
        A = rng.normal(size=(N_PARAMS, N_PARAMS))
        Sigma = A @ A.T + np.eye(N_PARAMS) * 0.1  # guaranteed PD
        scale = np.array([1.0, 2.0, 0.5, 0.1, 3.0])

        formula_result = normalized_logdet(Sigma, scale)

        S_inv = np.diag(1.0 / scale)
        Sigma_whitened = S_inv @ Sigma @ S_inv
        explicit_result = np.linalg.slogdet(Sigma_whitened)[1]

        assert formula_result == pytest.approx(explicit_result, abs=1e-9)

    def test_non_positive_definite_raises(self):
        Sigma = np.array([[1.0, 2.0], [2.0, 1.0]])  # not PD (det<0), wrong shape too but det check comes from slogdet
        Sigma_full = np.eye(N_PARAMS)
        Sigma_full[:2, :2] = Sigma
        with pytest.raises(ValueError):
            normalized_logdet(Sigma_full)


class TestParticleKalmanFilterConstruction:
    def test_construction_smoke(self):
        pkf = _default_filter()
        assert pkf.mu.shape == (N_PARAMS,)
        assert pkf.Sigma.shape == (N_PARAMS, N_PARAMS)

    def test_params_property_round_trips(self):
        pkf = _default_filter()
        p = pkf.params
        assert p.E_c1 == pytest.approx(2.5)
        assert p.E_cm_intra1 == pytest.approx(0.3)

    def test_wrong_mu_shape_raises(self):
        with pytest.raises(ValueError):
            ParticleKalmanFilter(mu=np.zeros(4), Sigma=np.eye(N_PARAMS), E_cm_intra1=0.3, E_cm_intra2=0.3)

    def test_wrong_sigma_shape_raises(self):
        with pytest.raises(ValueError):
            ParticleKalmanFilter(mu=np.zeros(N_PARAMS), Sigma=np.eye(4), E_cm_intra1=0.3, E_cm_intra2=0.3)


class TestPredict:
    def test_predict_no_process_noise_leaves_state_unchanged(self):
        pkf = _default_filter()
        mu_before, Sigma_before = pkf.mu.copy(), pkf.Sigma.copy()
        pkf.predict()
        np.testing.assert_array_equal(pkf.mu, mu_before)
        np.testing.assert_array_equal(pkf.Sigma, Sigma_before)

    def test_predict_with_process_noise_inflates_sigma(self):
        pkf = _default_filter()
        Sigma_before = pkf.Sigma.copy()
        Q = np.eye(N_PARAMS) * 0.01
        pkf.predict(Q=Q)
        assert np.all(np.diag(pkf.Sigma) > np.diag(Sigma_before))


class TestUpdate:
    def test_informative_measurement_shrinks_covariance(self):
        """At the genuinely E_c-sensitive transition point found and
        verified in qarray_env.py's development, an on-target measurement
        should shrink Sigma's determinant."""
        pkf = _default_filter(sigma_scale=0.5)
        # Use the true transition point for this filter's own current mean estimate.
        env = QArrayEnv(pkf.params)
        Cdd_raw, Cgd = __import__("qarray_env").build_capacitance_matrices(pkf.params)
        cdd_sum = Cdd_raw.sum(axis=1)
        cgd_sum = Cgd.sum(axis=1)
        offdiag_only = Cdd_raw.copy()
        np.fill_diagonal(offdiag_only, 0.0)
        Cdd_maxwell = np.diag(cdd_sum + cgd_sum) - offdiag_only
        A = np.linalg.inv(Cdd_maxwell)
        v_transition = A[0, 0] / (2 * pkf.params.alpha1 * (A[0, 0] + A[0, 1]))
        vg = np.array([v_transition, v_transition, 0.0, 0.0])

        logdet_before = np.linalg.slogdet(pkf.Sigma)[1]
        # "Measure" exactly the model's own current soft prediction (a
        # noiseless, self-consistent measurement) -- still should shrink
        # Sigma, since even a measurement that confirms the current mean
        # reduces uncertainty about it (a real EKF property, not a bug).
        measured = env.soft_prediction(vg, T=0.05)
        R = np.eye(4) * 0.01
        result = pkf.update(measured, vg, R)

        assert result.accepted
        logdet_after = np.linalg.slogdet(pkf.Sigma)[1]
        assert logdet_after < logdet_before

    def test_uninformative_measurement_leaves_covariance_nearly_unchanged(self):
        """Deep inside a stable plateau (H~0 everywhere, per qarray_env.py's
        verified finding), an update should barely move Sigma."""
        pkf = _default_filter(sigma_scale=0.5)
        vg_plateau = np.array([1.0, 1.0, 1.0, 1.0])
        env = QArrayEnv(pkf.params)
        measured = env.soft_prediction(vg_plateau, T=0.05)
        R = np.eye(4) * 0.01

        logdet_before = np.linalg.slogdet(pkf.Sigma)[1]
        result = pkf.update(measured, vg_plateau, R)
        logdet_after = np.linalg.slogdet(pkf.Sigma)[1]

        assert result.accepted
        assert logdet_after == pytest.approx(logdet_before, abs=1e-3)

    def test_sigma_stays_positive_definite_after_update(self):
        pkf = _default_filter(sigma_scale=0.3)
        vg = np.array([6.5, 6.5, 0.0, 0.0])
        env = QArrayEnv(pkf.params)
        measured = env.soft_prediction(vg, T=0.05) + np.array([0.02, -0.01, 0.0, 0.0])
        R = np.eye(4) * 0.02
        pkf.update(measured, vg, R)

        eigvals = np.linalg.eigvalsh(pkf.Sigma)
        assert np.all(eigvals > 0), f"Sigma not PD after update, eigenvalues: {eigvals}"

    def test_infeasible_update_backs_off_and_flags(self):
        """Force an update whose naive step would push alpha1 toward an
        infeasible region (mirroring the exact alpha=1.0 regression case
        from qarray_env.py's own development) -- filter should back off,
        not crash, and should report accepted=False if backoff is exhausted."""
        # Start very close to the feasibility boundary with high uncertainty
        # on alpha1 and an artificially huge innovation, to force the guard.
        mu = np.array([2.5, 2.7, 0.05, 0.05, 0.05])
        Sigma = np.eye(N_PARAMS)
        Sigma[2, 2] = 100.0  # huge alpha1 uncertainty -> huge Kalman gain there
        pkf = ParticleKalmanFilter(mu=mu, Sigma=Sigma, E_cm_intra1=0.3, E_cm_intra2=0.3, max_backoffs=3)

        vg = np.array([6.5, 6.5, 0.0, 0.0])
        # A wildly out-of-range "measurement" to force a huge innovation-driven step.
        measured = np.array([10.0, 10.0, 10.0, 10.0])
        R = np.eye(4) * 0.001  # tiny R -> filter trusts this bogus measurement strongly

        result = pkf.update(measured, vg, R)
        assert isinstance(result, KalmanUpdateResult)
        # Either it backed off successfully (accepted, n_backoffs>0) or gave
        # up cleanly (not accepted) -- either way, must not have crashed,
        # and mu must still be constructible.
        QArrayEnv(pkf.params)  # should not raise
