"""
Tests for gz_tuning.control.phase_switch
"""

import numpy as np
import pytest

from kalman import N_PARAMS, ParticleKalmanFilter, normalized_logdet
from particle_filter import OccupationParticle, RBParticleFilter
from phase_switch import (
    Phase,
    PhaseSwitchController,
    within_particle_mass_fraction_below,
)


def _particle(mu, sigma_scale, weight=1.0):
    mu = np.asarray(mu, dtype=np.float64)
    Sigma = np.eye(N_PARAMS) * sigma_scale
    pkf = ParticleKalmanFilter(mu=mu, Sigma=Sigma, E_cm_intra1=0.3, E_cm_intra2=0.3)
    return OccupationParticle(kalman=pkf, weight=weight)


BASE_MU = np.array([2.5, 2.7, 0.05, 0.05, 0.05])


class TestWithinParticleMassFractionBelow:
    def test_mixed_confidence_fraction(self):
        p_confident_a = _particle(BASE_MU, sigma_scale=1e-6, weight=0.3)
        p_confident_b = _particle(BASE_MU + 0.01, sigma_scale=1e-6, weight=0.2)
        p_diffuse = _particle(BASE_MU - 0.01, sigma_scale=10.0, weight=0.5)
        pf = RBParticleFilter(particles=[p_confident_a, p_confident_b, p_diffuse])

        logdet_confident = normalized_logdet(np.eye(N_PARAMS) * 1e-6)
        logdet_diffuse = normalized_logdet(np.eye(N_PARAMS) * 10.0)
        threshold = (logdet_confident + logdet_diffuse) / 2.0

        fraction = within_particle_mass_fraction_below(pf, threshold)
        assert fraction == pytest.approx(0.5)  # only the two confident particles' weight

    def test_differs_from_matrix_averaged_aggregate(self):
        # A case where the matrix-averaged-first aggregate (particle_filter
        # .py's within_particle_covariance) reads as "converged" while the
        # mass-based fraction correctly shows a real minority still
        # uncertain -- the exact discrepancy this module's docstring flags.
        p_confident = _particle(BASE_MU, sigma_scale=1e-8, weight=0.5)
        p_uncertain = _particle(BASE_MU + 0.02, sigma_scale=1.2, weight=0.5)
        pf = RBParticleFilter(particles=[p_confident, p_uncertain])

        threshold = normalized_logdet(np.eye(N_PARAMS) * 1.0)
        aggregate_logdet = pf.within_particle_covariance()
        assert aggregate_logdet < threshold  # averaged matrix looks converged

        fraction = within_particle_mass_fraction_below(pf, threshold)
        assert fraction == pytest.approx(0.5)  # only half the mass actually is
        assert fraction < 1.0


class TestPhaseSwitchControllerValidation:
    def test_rejects_bad_threshold_order(self):
        with pytest.raises(ValueError):
            PhaseSwitchController(threshold_low=1.0, threshold_high=1.0)

    def test_rejects_bad_mass_fraction(self):
        with pytest.raises(ValueError):
            PhaseSwitchController(threshold_low=-5.0, threshold_high=0.0, mass_fraction_required=0.0)
        with pytest.raises(ValueError):
            PhaseSwitchController(threshold_low=-5.0, threshold_high=0.0, mass_fraction_required=1.5)


class TestPhaseSwitchHysteresis:
    def _population(self, sigma_scale, mu_jitter):
        rng = np.random.default_rng(0)
        particles = []
        for _ in range(8):
            jitter = rng.normal(scale=mu_jitter, size=N_PARAMS) if mu_jitter else 0.0
            particles.append(_particle(BASE_MU + jitter, sigma_scale, weight=1.0))
        return RBParticleFilter(particles=particles)

    def test_starts_in_phase_1(self):
        controller = PhaseSwitchController(threshold_low=-5.0, threshold_high=-2.0)
        assert controller.phase is Phase.PHASE_1_RAW_SIGNAL

    def test_switches_to_phase_2_when_both_lines_clear_low(self):
        controller = PhaseSwitchController(
            threshold_low=-2.0, threshold_high=1.0, mass_fraction_required=0.9
        )
        pf = self._population(sigma_scale=1e-6, mu_jitter=1e-6)
        diag = controller.step(pf)
        assert controller.phase is Phase.PHASE_2_FEATURE_BASED
        assert diag.blue_mass_fraction_below_low >= 0.9
        assert diag.red_below_low

    def test_stays_in_phase_1_if_red_line_fails(self):
        # Individually confident particles (small own-Sigma) that
        # substantially DISAGREE with each other -- the false-convergence
        # scenario Section 6's red line exists to catch.
        controller = PhaseSwitchController(
            threshold_low=-2.0, threshold_high=1.0, mass_fraction_required=0.9
        )
        pf = self._population(sigma_scale=1e-6, mu_jitter=2.0)
        diag = controller.step(pf)
        assert controller.phase is Phase.PHASE_1_RAW_SIGNAL
        assert diag.blue_mass_fraction_below_low >= 0.9
        assert not diag.red_below_low

    def test_reverts_to_phase_1_when_either_line_climbs_above_high(self):
        controller = PhaseSwitchController(
            threshold_low=-2.0, threshold_high=1.0, mass_fraction_required=0.9
        )
        converged_pf = self._population(sigma_scale=1e-6, mu_jitter=1e-6)
        controller.step(converged_pf)
        assert controller.phase is Phase.PHASE_2_FEATURE_BASED

        regressed_pf = self._population(sigma_scale=50.0, mu_jitter=1e-6)
        diag = controller.step(regressed_pf)
        assert controller.phase is Phase.PHASE_1_RAW_SIGNAL
        assert diag.blue_mass_fraction_below_high < 0.9
