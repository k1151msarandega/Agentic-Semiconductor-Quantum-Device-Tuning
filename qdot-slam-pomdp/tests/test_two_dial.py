"""Tests for gz_tuning.acquisition.two_dial"""

import numpy as np
import pytest

from two_dial import compute_ig_total
from bald import compute_ig_bald
from fim import compute_ig_fim
from kalman import ParticleKalmanFilter, N_PARAMS
from particle_filter import OccupationParticle, RBParticleFilter


def _make_pf():
    particles = []
    for E_c1 in [2.3, 2.5, 2.7]:
        mu = np.array([E_c1, 2.7, 0.05, 0.05, 0.05])
        Sigma = np.eye(N_PARAMS) * 0.3
        kf = ParticleKalmanFilter(mu=mu, Sigma=Sigma, E_cm_intra1=0.3, E_cm_intra2=0.3)
        particles.append(OccupationParticle(kalman=kf, weight=1.0))
    return RBParticleFilter(particles=particles)


class TestComputeIgTotal:
    def test_matches_manual_combination(self):
        pf = _make_pf()
        vg = np.array([6.5, 6.5, 0.0, 0.0])
        R = np.eye(4) * 0.02
        w_d, w_c = 0.7, 1.3

        expected = w_d * compute_ig_bald(pf, vg) + w_c * compute_ig_fim(pf, vg, R)
        actual = compute_ig_total(pf, vg, R, w_discrete=w_d, w_continuous=w_c)
        assert actual == pytest.approx(expected)

    def test_zero_weights_gives_zero(self):
        pf = _make_pf()
        vg = np.array([6.5, 6.5, 0.0, 0.0])
        R = np.eye(4) * 0.02
        result = compute_ig_total(pf, vg, R, w_discrete=0.0, w_continuous=0.0)
        assert result == pytest.approx(0.0)

    def test_mismatched_scale_raises(self):
        pf = _make_pf()
        vg = np.array([6.5, 6.5, 0.0, 0.0])
        R = np.eye(4) * 0.02
        bad_scale = np.ones(3)  # wrong length (should be 5)
        with pytest.raises(ValueError):
            compute_ig_total(pf, vg, R, w_discrete=1.0, w_continuous=1.0, scale=bad_scale)

    def test_weights_are_required_not_defaulted(self):
        """No default schedule should exist -- this is deliberate per the
        module's documented open-design-question stance. Calling without
        explicit weights should be a TypeError (missing required kwarg),
        not silently fall back to some invented default."""
        pf = _make_pf()
        vg = np.array([6.5, 6.5, 0.0, 0.0])
        R = np.eye(4) * 0.02
        with pytest.raises(TypeError):
            compute_ig_total(pf, vg, R)
