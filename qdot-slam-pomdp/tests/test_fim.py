"""Tests for gz_tuning.acquisition.fim"""

import numpy as np
import pytest

from fim import compute_ig_fim, compute_ig_fim_particle, _kalman_posterior_covariance
from kalman import ParticleKalmanFilter, N_PARAMS, DEFAULT_SCALE, normalized_logdet
from particle_filter import OccupationParticle, RBParticleFilter
from qarray_env import DeviceParams, build_capacitance_matrices


def _make_filter(E_c1=2.5, sigma_scale=0.3):
    mu = np.array([E_c1, 2.7, 0.05, 0.05, 0.05])
    Sigma = np.eye(N_PARAMS) * sigma_scale
    return ParticleKalmanFilter(mu=mu, Sigma=Sigma, E_cm_intra1=0.3, E_cm_intra2=0.3)


def _precise_transition_vg(params):
    Cdd_raw, Cgd = build_capacitance_matrices(params)
    cdd_sum = Cdd_raw.sum(axis=1)
    cgd_sum = Cgd.sum(axis=1)
    offdiag_only = Cdd_raw.copy()
    np.fill_diagonal(offdiag_only, 0.0)
    Cdd_maxwell = np.diag(cdd_sum + cgd_sum) - offdiag_only
    A = np.linalg.inv(Cdd_maxwell)
    v = A[0, 0] / (2 * params.alpha1 * (A[0, 0] + A[0, 1]))
    return np.array([v, v, 0.0, 0.0])


class TestKalmanPosteriorCovariance:
    def test_posterior_smaller_determinant_than_prior(self):
        Sigma = np.eye(N_PARAMS) * 0.5
        H = np.random.default_rng(0).normal(size=(4, N_PARAMS))
        R = np.eye(4) * 0.01
        posterior = _kalman_posterior_covariance(Sigma, H, R)
        assert np.linalg.det(posterior) <= np.linalg.det(Sigma)

    def test_zero_jacobian_leaves_covariance_unchanged(self):
        Sigma = np.eye(N_PARAMS) * 0.5
        H = np.zeros((4, N_PARAMS))
        R = np.eye(4) * 0.01
        posterior = _kalman_posterior_covariance(Sigma, H, R)
        np.testing.assert_allclose(posterior, Sigma, atol=1e-10)


class TestComputeIgFimParticle:
    def test_nonnegative_always(self):
        """Real invariant: information gain from a valid Kalman update can
        never be negative (posterior covariance is always <= prior in the
        Loewner order, so logdet(prior)-logdet(posterior) >= 0)."""
        kf = _make_filter()
        rng = np.random.default_rng(1)
        R = np.eye(4) * 0.05
        for _ in range(10):
            vg = rng.uniform(0, 15, size=4)
            ig = compute_ig_fim_particle(kf, vg, R)
            assert ig >= -1e-9, f"Negative IG_FIM at vg={vg}: {ig}"

    def test_large_at_genuinely_informative_point(self):
        kf = _make_filter(sigma_scale=0.5)
        vg = _precise_transition_vg(kf.params)
        R = np.eye(4) * 0.01
        ig = compute_ig_fim_particle(kf, vg, R)
        assert ig > 0.1

    def test_near_zero_deep_in_plateau(self):
        kf = _make_filter(sigma_scale=0.5)
        vg = np.array([1.0, 1.0, 1.0, 1.0])
        R = np.eye(4) * 0.01
        ig = compute_ig_fim_particle(kf, vg, R)
        assert ig < 1e-6

    def test_singular_prior_returns_zero_not_nan(self):
        kf = _make_filter()
        kf.Sigma = np.diag([1.0, 0.0, 1.0, 1.0, 1.0])  # already singular
        vg = _precise_transition_vg(kf.params)
        R = np.eye(4) * 0.01
        ig = compute_ig_fim_particle(kf, vg, R)
        assert ig == 0.0


class TestComputeIgFimPopulation:
    def test_matches_manual_weighted_sum(self):
        kf1 = _make_filter(E_c1=2.4, sigma_scale=0.3)
        kf2 = _make_filter(E_c1=2.6, sigma_scale=0.5)
        p1 = OccupationParticle(kalman=kf1, weight=0.3)
        p2 = OccupationParticle(kalman=kf2, weight=0.7)
        pf = RBParticleFilter(particles=[p1, p2])

        vg = _precise_transition_vg(kf1.params)
        R = np.eye(4) * 0.02

        expected = 0.3 * compute_ig_fim_particle(kf1, vg, R) + 0.7 * compute_ig_fim_particle(kf2, vg, R)
        actual = compute_ig_fim(pf, vg, R)
        assert actual == pytest.approx(expected)
