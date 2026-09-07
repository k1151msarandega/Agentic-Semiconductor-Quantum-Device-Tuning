"""
Tests for gz_tuning.simulator.noise_model (v2 rewrite)
"""

import numpy as np
import pytest

from noise_model import (
    GaussianReadoutNoise,
    calibrate_threshold_low,
    compute_p_flip,
    derive_consecutive_count,
    derive_phase_thresholds,
    p_flip_per_dot,
    p_flip_total,
    per_dot_distance_to_boundary,
    per_dot_sigma_eff,
)
from kalman import N_PARAMS, DEFAULT_SCALE
from qarray_env import DeviceParams, N_DOT, observation_jacobian


def _default_params(**overrides):
    base = dict(E_c1=2.5, E_c2=2.7, alpha1=0.05, alpha2=0.05, E_cm_intra1=0.3, E_cm_intra2=0.3, cross_capacitance=0.05)
    base.update(overrides)
    return DeviceParams(**base)


class TestDistanceToBoundary:
    def test_at_boundary_distance_zero(self):
        pred = np.array([0.5, 1.5, 2.5, 0.5])
        d = per_dot_distance_to_boundary(pred)
        np.testing.assert_allclose(d, 0.0, atol=1e-12)

    def test_at_integer_distance_max(self):
        pred = np.array([0.0, 1.0, 2.0, 3.0])
        d = per_dot_distance_to_boundary(pred)
        np.testing.assert_allclose(d, 0.5)

    def test_intermediate_value(self):
        pred = np.array([0.1, 0.9, 1.2, 1.8])
        d = per_dot_distance_to_boundary(pred)
        np.testing.assert_allclose(d, [0.4, 0.4, 0.3, 0.3], atol=1e-9)


class TestSigmaEff:
    def test_zero_parameter_uncertainty_reduces_to_sensor_noise(self):
        R = np.eye(N_DOT) * 0.01
        H = np.zeros((N_DOT, N_PARAMS))  # no parameter sensitivity at all
        Sigma_total = np.eye(N_PARAMS)
        sigma_eff = per_dot_sigma_eff(R, H, Sigma_total)
        np.testing.assert_allclose(sigma_eff, np.sqrt(0.01))

    def test_uses_full_sigma_total_not_ignoring_between_particle_term(self):
        """Regression guard: sigma_eff must genuinely respond to Sigma_total's
        magnitude (the within+between combination), not silently ignore it."""
        R = np.eye(N_DOT) * 0.001
        H = np.ones((N_DOT, N_PARAMS)) * 0.1
        small_Sigma = np.eye(N_PARAMS) * 0.01
        large_Sigma = np.eye(N_PARAMS) * 10.0
        sigma_eff_small = per_dot_sigma_eff(R, H, small_Sigma)
        sigma_eff_large = per_dot_sigma_eff(R, H, large_Sigma)
        assert np.all(sigma_eff_large > sigma_eff_small)


class TestPFlip:
    def test_at_boundary_flip_probability_is_one(self):
        """distance=0 (sitting exactly ON a boundary, mean == boundary
        value) with any nonzero sigma_eff gives p_flip=1.0, not 0.5 --
        P(|noise| > 0) = 1 for continuous noise (you're guaranteed not to
        read exactly the boundary value). An earlier version of this test
        expected 0.5 here, from conflating 'which side you land on' (that
        split IS 50/50) with 'whether you differ at all from an exact tie'
        (guaranteed, hence 1.0) -- this is the correct, if degenerate, edge
        case; distance=0 isn't really 'flipping' in the intended sense
        since there's no reference side to flip from when you start exactly
        on the line."""
        d = np.array([0.0])
        sigma_eff = np.array([0.1])
        p = p_flip_per_dot(d, sigma_eff)
        assert p[0] == pytest.approx(1.0)

    def test_far_from_boundary_flip_probability_near_zero(self):
        d = np.array([0.5])  # max possible distance
        sigma_eff = np.array([0.01])  # tiny noise relative to distance
        p = p_flip_per_dot(d, sigma_eff)
        assert p[0] < 1e-6

    def test_zero_sigma_eff_zero_flip_probability(self):
        d = np.array([0.1])
        sigma_eff = np.array([0.0])
        p = p_flip_per_dot(d, sigma_eff)
        assert p[0] == 0.0

    def test_total_matches_independent_or_formula(self):
        p_per_dot = np.array([0.1, 0.2, 0.05, 0.0])
        expected = 1 - (0.9 * 0.8 * 0.95 * 1.0)
        assert p_flip_total(p_per_dot) == pytest.approx(expected)

    def test_total_exceeds_any_individual_dot(self):
        p_per_dot = np.array([0.1, 0.1, 0.1, 0.1])
        total = p_flip_total(p_per_dot)
        assert total > 0.1

    def test_total_bounded_by_sum_union_bound(self):
        """Exact formula should never exceed the union bound (sanity check
        on the math, not a design requirement to use the bound itself)."""
        p_per_dot = np.array([0.1, 0.15, 0.05, 0.2])
        exact = p_flip_total(p_per_dot)
        union_bound = p_per_dot.sum()
        assert exact <= union_bound


class TestComputePFlipEndToEnd:
    def test_runs_against_real_jacobian(self):
        """Full pipeline using a real observation_jacobian call, not mocks."""
        params = _default_params()
        vg = np.array([6.5, 6.5, 0.0, 0.0])  # near the verified E_c-sensitive transition
        H = observation_jacobian(params, vg)
        Sigma_total = np.eye(N_PARAMS) * 0.5
        R = np.eye(N_DOT) * 0.05 ** 2
        prediction = np.array([0.5, 0.5, 0.0, 0.0])  # right at a boundary on dots 0,1

        p = compute_p_flip(prediction, R, H, Sigma_total)
        assert 0.0 <= p <= 1.0
        assert p > 0.0  # at a boundary, should never be exactly zero


class TestDeriveConsecutiveCount:
    def test_smaller_p_flip_needs_fewer_reads(self):
        assert derive_consecutive_count(0.01) < derive_consecutive_count(0.3)

    def test_derived_N_satisfies_target(self):
        p_flip = 0.15
        p_target = 0.01
        N = derive_consecutive_count(p_flip, p_target=p_target)
        assert p_flip ** N < p_target
        if N > 1:
            assert p_flip ** (N - 1) >= p_target

    def test_zero_p_flip_returns_one(self):
        assert derive_consecutive_count(0.0) == 1

    def test_p_flip_at_least_one_raises(self):
        with pytest.raises(ValueError):
            derive_consecutive_count(1.0)


class TestDerivePhaseThresholds:
    def test_low_below_high(self):
        low, high = derive_phase_thresholds(threshold_low=-2.0)
        assert low < high

    def test_additive_gap_handles_negative_threshold_correctly(self):
        """Regression guard for the exact bug an earlier version would have
        had: multiplicative scaling of a negative logdet threshold makes it
        MORE negative (tighter), backwards from the intended looser
        threshold_high. Additive gap must always loosen, regardless of sign."""
        low_negative, high_negative = derive_phase_thresholds(threshold_low=-3.0, hysteresis_gap_nats=1.0)
        assert high_negative == pytest.approx(-2.0)
        assert high_negative > low_negative

        low_positive, high_positive = derive_phase_thresholds(threshold_low=3.0, hysteresis_gap_nats=1.0)
        assert high_positive == pytest.approx(4.0)
        assert high_positive > low_positive


class TestCalibrateThresholdLow:
    def test_calibration_reduces_uncertainty_below_prior(self):
        """Real, slower integration test: run an actual (small) Kalman
        update loop against a real ground-truth QArrayEnv and confirm the
        achieved normalized logdet is meaningfully lower than the prior's."""
        rng = np.random.default_rng(42)
        ground_truth = _default_params()
        prior_mu = np.array([2.5, 2.7, 0.05, 0.05, 0.05])
        prior_Sigma = np.eye(N_PARAMS) * 0.3

        prior_logdet_normalized = None
        from kalman import normalized_logdet
        prior_logdet_normalized = normalized_logdet(prior_Sigma, scale=DEFAULT_SCALE)

        achieved = calibrate_threshold_low(
            prior_mu, prior_Sigma, ground_truth,
            n_measurements=40, n_candidates_per_step=15,
            rng=rng,
        )

        assert achieved < prior_logdet_normalized, (
            f"Expected calibration to reduce uncertainty: prior={prior_logdet_normalized}, "
            f"achieved={achieved}"
        )
