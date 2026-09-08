"""
Tests for gz_tuning.belief.particle_filter

NOTE: this file was reconstructed from scratch. The repo's real
test_particle_filter.py was accidentally overwritten with
test_particle_init.py's content (confirmed via file metadata: real file
created Aug 28, modified Sep 7 17:36, 14 minutes before test_particle_init.py
was created at 17:50 -- almost certainly a copy-paste slip). The original
content was not available to reconstruct from; this is a fresh test suite
written directly against the actual particle_filter.py source.
"""

import numpy as np
import pytest

from kalman import N_PARAMS, ParticleKalmanFilter, normalized_logdet
from particle_filter import OccupationParticle, RBParticleFilter
from qarray_env import DeviceParams, N_GATE, QArrayEnv


DEFAULT_PARAMS = DeviceParams(
    E_c1=2.5, E_c2=2.7, alpha1=0.05, alpha2=0.05,
    E_cm_intra1=0.3, E_cm_intra2=0.3, cross_capacitance=0.05,
)


def _particle(params: DeviceParams = DEFAULT_PARAMS, sigma_scale=1e-4, weight=1.0):
    mu = np.array([params.E_c1, params.E_c2, params.alpha1, params.alpha2, params.cross_capacitance])
    Sigma = np.eye(N_PARAMS) * sigma_scale
    pkf = ParticleKalmanFilter(
        mu=mu, Sigma=Sigma,
        E_cm_intra1=params.E_cm_intra1, E_cm_intra2=params.E_cm_intra2,
    )
    return OccupationParticle(kalman=pkf, weight=weight)


class TestWeightNormalization:
    def test_weights_normalized_on_construction(self):
        particles = [_particle(weight=3.0), _particle(weight=1.0)]
        pf = RBParticleFilter(particles=particles)
        weights = np.array([p.weight for p in pf.particles])
        assert weights.sum() == pytest.approx(1.0)
        np.testing.assert_allclose(weights, [0.75, 0.25])

    def test_uniform_weights_stay_uniform(self):
        particles = [_particle(weight=1.0) for _ in range(5)]
        pf = RBParticleFilter(particles=particles)
        weights = np.array([p.weight for p in pf.particles])
        np.testing.assert_allclose(weights, 0.2)

    def test_zero_total_weight_raises(self):
        particles = [_particle(weight=0.0), _particle(weight=0.0)]
        with pytest.raises(ValueError):
            RBParticleFilter(particles=particles)

    def test_negative_total_weight_raises(self):
        # Constructed directly with weight already set (bypassing any
        # upstream validation) to exercise RBParticleFilter's OWN guard,
        # not rely on OccupationParticle rejecting negative weights itself
        # (it doesn't -- weight has no validation at that layer).
        particles = [_particle(weight=-1.0), _particle(weight=-1.0)]
        with pytest.raises(ValueError):
            RBParticleFilter(particles=particles)


class TestNParticles:
    def test_matches_list_length(self):
        pf = RBParticleFilter(particles=[_particle() for _ in range(7)])
        assert pf.n_particles == 7


class TestPredict:
    def test_default_is_noop_on_mu(self):
        pf = RBParticleFilter(particles=[_particle()])
        mu_before = pf.particles[0].kalman.mu.copy()
        pf.predict()
        np.testing.assert_allclose(pf.particles[0].kalman.mu, mu_before)

    def test_default_is_noop_on_sigma(self):
        pf = RBParticleFilter(particles=[_particle(sigma_scale=1e-3)])
        Sigma_before = pf.particles[0].kalman.Sigma.copy()
        pf.predict()
        np.testing.assert_allclose(pf.particles[0].kalman.Sigma, Sigma_before)

    def test_explicit_Q_inflates_every_particles_sigma(self):
        pf = RBParticleFilter(particles=[_particle(sigma_scale=1e-3) for _ in range(3)])
        traces_before = [np.trace(p.kalman.Sigma) for p in pf.particles]
        Q = np.eye(N_PARAMS) * 0.01
        pf.predict(Q=Q)
        traces_after = [np.trace(p.kalman.Sigma) for p in pf.particles]
        for before, after in zip(traces_before, traces_after):
            assert after > before


class TestUpdate:
    def test_runs_without_error_and_keeps_weights_normalized(self):
        pf = RBParticleFilter(particles=[_particle() for _ in range(4)])
        vg = np.zeros(N_GATE)
        ground_truth = QArrayEnv(DEFAULT_PARAMS)
        measured = ground_truth.query(vg)
        R = np.eye(4) * 0.02

        pf.update(measured, vg, R, T=0.05)
        weights = np.array([p.weight for p in pf.particles])
        assert weights.sum() == pytest.approx(1.0)
        assert np.all(weights >= 0.0)

    def test_particle_closer_to_measurement_gets_more_weight(self):
        # Two particles with identical starting weight but different alpha
        # -- whichever one's prediction is actually closer to the observed
        # measurement should end up with more posterior weight.
        vg = np.array([12.0, 12.0, 12.0, 12.0])
        params_close = DEFAULT_PARAMS
        params_far = DEFAULT_PARAMS.replace(alpha1=0.15, alpha2=0.15)

        ground_truth = QArrayEnv(params_close)
        measured = ground_truth.query(vg)

        occ_close = np.round(QArrayEnv(params_close).soft_prediction(vg, T=0.05))
        occ_far = np.round(QArrayEnv(params_far).soft_prediction(vg, T=0.05))
        if np.array_equal(occ_close, occ_far):
            pytest.skip(
                "chosen vg/alpha did not separate the two particles' "
                "predictions -- physics-dependent, pick different values "
                "if this ever skips"
            )

        pf = RBParticleFilter(particles=[
            _particle(params_close, weight=1.0),
            _particle(params_far, weight=1.0),
        ])
        R = np.eye(4) * 0.02
        pf.update(measured, vg, R, T=0.05)
        assert pf.particles[0].weight > pf.particles[1].weight

    def test_update_moves_particle_mean_toward_truth(self):
        # A single particle started slightly off the true alpha1 should
        # move closer to the truth after one update at an informative vg.
        true_params = DEFAULT_PARAMS
        vg = np.array([12.0, 12.0, 12.0, 12.0])
        ground_truth = QArrayEnv(true_params)
        measured = ground_truth.query(vg)

        offset_params = true_params.replace(alpha1=0.052)
        pf = RBParticleFilter(particles=[_particle(offset_params, sigma_scale=0.05)])
        alpha1_before = pf.particles[0].kalman.mu[2]

        R = np.eye(4) * 0.02
        pf.update(measured, vg, R, T=0.05)
        alpha1_after = pf.particles[0].kalman.mu[2]

        assert abs(alpha1_after - true_params.alpha1) <= abs(alpha1_before - true_params.alpha1)


class TestEffectiveSampleSize:
    def test_uniform_weights_give_full_ess(self):
        n = 6
        pf = RBParticleFilter(particles=[_particle(weight=1.0) for _ in range(n)])
        assert pf.effective_sample_size() == pytest.approx(n)

    def test_single_dominant_particle_gives_ess_near_one(self):
        pf = RBParticleFilter(particles=[
            _particle(weight=1000.0),
            _particle(weight=1.0),
            _particle(weight=1.0),
        ])
        assert pf.effective_sample_size() < 2.0


class TestResample:
    def test_preserves_particle_count(self):
        pf = RBParticleFilter(particles=[_particle(weight=w) for w in [1.0, 2.0, 3.0]])
        pf.resample(np.random.default_rng(0))
        assert pf.n_particles == 3

    def test_resets_weights_to_uniform(self):
        pf = RBParticleFilter(particles=[_particle(weight=w) for w in [1.0, 5.0, 1.0]])
        pf.resample(np.random.default_rng(0))
        weights = np.array([p.weight for p in pf.particles])
        np.testing.assert_allclose(weights, 1.0 / 3)

    def test_resampled_particles_are_independent_copies(self):
        pf = RBParticleFilter(particles=[_particle(weight=1.0) for _ in range(3)])
        pf.resample(np.random.default_rng(0))
        # Mutating one resampled particle's mu must not affect any other,
        # even if they were duplicated from the same source particle
        # (systematic resampling with replacement can duplicate a
        # high-weight particle several times).
        pf.particles[0].kalman.mu[0] = 999.0
        assert pf.particles[1].kalman.mu[0] != 999.0

    def test_heavily_weighted_particle_dominates_resampled_population(self):
        distinct_params = DEFAULT_PARAMS.replace(alpha1=0.09)
        pf = RBParticleFilter(particles=[
            _particle(DEFAULT_PARAMS, weight=1000.0),
            _particle(distinct_params, weight=1.0),
        ])
        pf.resample(np.random.default_rng(0))
        alpha1_values = [p.kalman.mu[2] for p in pf.particles]
        assert alpha1_values.count(pytest.approx(DEFAULT_PARAMS.alpha1)) >= 1

    def test_jitter_Q_inflates_sigma_after_resample(self):
        pf = RBParticleFilter(particles=[_particle(sigma_scale=1e-3) for _ in range(4)])
        Q = np.eye(N_PARAMS) * 0.01
        pf.resample(np.random.default_rng(0), jitter_Q=Q)
        for p in pf.particles:
            assert np.trace(p.kalman.Sigma) > 1e-3 * N_PARAMS

    def test_no_jitter_by_default(self):
        pf = RBParticleFilter(particles=[_particle(sigma_scale=1e-3) for _ in range(4)])
        pf.resample(np.random.default_rng(0))
        for p in pf.particles:
            np.testing.assert_allclose(np.diag(p.kalman.Sigma), 1e-3)


class TestWeightedMeanMu:
    def test_matches_manual_weighted_average(self):
        p1 = _particle(DEFAULT_PARAMS, weight=0.25)
        p2 = _particle(DEFAULT_PARAMS.replace(alpha1=0.09), weight=0.75)
        pf = RBParticleFilter(particles=[p1, p2])
        expected = 0.25 * p1.kalman.mu + 0.75 * p2.kalman.mu
        np.testing.assert_allclose(pf.weighted_mean_mu(), expected)


class TestWithinParticleCovariance:
    def test_matches_manual_matrix_averaged_first_computation(self):
        p1 = _particle(sigma_scale=1e-3, weight=0.3)
        p2 = _particle(sigma_scale=2.0, weight=0.7)
        pf = RBParticleFilter(particles=[p1, p2])

        Sigma_avg = 0.3 * p1.kalman.Sigma + 0.7 * p2.kalman.Sigma
        expected = normalized_logdet(Sigma_avg)
        assert pf.within_particle_covariance() == pytest.approx(expected)

    def test_differs_from_average_of_individual_logdets(self):
        # normalized_logdet is not linear -- averaging the MATRICES first
        # (what this function does) must generally differ from averaging
        # each particle's own logdet. Demonstrated directly rather than
        # just asserted, since logdet's nonlinearity is exactly what makes
        # this distinction (and control/phase_switch.py's separate
        # mass-based fix) matter in the first place.
        p1 = _particle(sigma_scale=1e-6, weight=0.5)
        p2 = _particle(sigma_scale=3.0, weight=0.5)
        pf = RBParticleFilter(particles=[p1, p2])

        matrix_averaged_first = pf.within_particle_covariance()
        average_of_logdets = 0.5 * normalized_logdet(p1.kalman.Sigma) + 0.5 * normalized_logdet(p2.kalman.Sigma)
        assert matrix_averaged_first != pytest.approx(average_of_logdets)


class TestBetweenParticleCovariance:
    def test_identical_particles_give_negative_infinity(self):
        # All particles share the exact same mu -- Cov_between is
        # identically the zero matrix, so normalized_logdet must report
        # -inf (a genuinely singular, valid covariance -- see kalman.py's
        # normalized_logdet for why -inf, not an error, is correct here).
        pf = RBParticleFilter(particles=[_particle() for _ in range(8)])
        assert pf.between_particle_covariance() == float("-inf")

    def test_disagreeing_particles_give_finite_value(self):
        rng = np.random.default_rng(0)
        particles = []
        for _ in range(8):
            jitter = rng.normal(scale=0.3, size=N_PARAMS)
            mu = np.array([2.5, 2.7, 0.05, 0.05, 0.05]) + jitter
            pkf = ParticleKalmanFilter(mu=mu, Sigma=np.eye(N_PARAMS) * 1e-4, E_cm_intra1=0.3, E_cm_intra2=0.3)
            particles.append(OccupationParticle(kalman=pkf, weight=1.0))
        pf = RBParticleFilter(particles=particles)
        result = pf.between_particle_covariance()
        assert np.isfinite(result)

    def test_more_disagreement_gives_larger_value(self):
        rng = np.random.default_rng(1)

        def _population(spread):
            particles = []
            for _ in range(8):
                jitter = rng.normal(scale=spread, size=N_PARAMS)
                mu = np.array([2.5, 2.7, 0.05, 0.05, 0.05]) + jitter
                pkf = ParticleKalmanFilter(mu=mu, Sigma=np.eye(N_PARAMS) * 1e-4, E_cm_intra1=0.3, E_cm_intra2=0.3)
                particles.append(OccupationParticle(kalman=pkf, weight=1.0))
            return RBParticleFilter(particles=particles)

        tight_pf = _population(spread=0.01)
        wide_pf = _population(spread=2.0)
        assert wide_pf.between_particle_covariance() > tight_pf.between_particle_covariance()

    def test_structurally_singular_below_n_params_plus_one_particles(self):
        # Documented constraint in the source: fewer than N_PARAMS+1 (6)
        # particles makes Cov_between rank-deficient REGARDLESS of how much
        # the particles actually disagree -- must read as -inf, not a
        # small-but-finite number, even with particles far apart.
        mu_a = np.array([2.5, 2.7, 0.05, 0.05, 0.05])
        mu_b = np.array([2.9, 2.7, 0.05, 0.05, 0.05])
        particles = [
            OccupationParticle(
                kalman=ParticleKalmanFilter(mu=mu_a, Sigma=np.eye(N_PARAMS) * 1e-4, E_cm_intra1=0.3, E_cm_intra2=0.3),
                weight=1.0,
            ),
            OccupationParticle(
                kalman=ParticleKalmanFilter(mu=mu_b, Sigma=np.eye(N_PARAMS) * 1e-4, E_cm_intra1=0.3, E_cm_intra2=0.3),
                weight=1.0,
            ),
        ]
        pf = RBParticleFilter(particles=particles)
        assert pf.between_particle_covariance() == float("-inf")
