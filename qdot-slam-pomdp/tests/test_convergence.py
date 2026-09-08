"""
Tests for gz_tuning.control.convergence
"""

import numpy as np
import pytest

from kalman import N_PARAMS, ParticleKalmanFilter, params_to_vector
from particle_filter import OccupationParticle, RBParticleFilter
from qarray_env import DeviceParams, N_DOT, N_GATE, QArrayEnv
from convergence import (
    ConvergenceMonitor,
    actuation_quiescent,
    belief_stable,
    target_occupation_mass,
    target_occupation_probability,
)


DEFAULT_PARAMS = DeviceParams(
    E_c1=2.5, E_c2=2.7, alpha1=0.05, alpha2=0.05,
    E_cm_intra1=0.3, E_cm_intra2=0.3, cross_capacitance=0.05,
)


def _particle(params: DeviceParams, sigma_scale=1e-6, weight=1.0):
    mu = params_to_vector(params)
    Sigma = np.eye(N_PARAMS) * sigma_scale
    pkf = ParticleKalmanFilter(
        mu=mu, Sigma=Sigma,
        E_cm_intra1=params.E_cm_intra1, E_cm_intra2=params.E_cm_intra2,
    )
    return OccupationParticle(kalman=pkf, weight=weight)


class TestTargetOccupationProbability:
    def test_confident_particle_on_target_gives_near_one(self):
        vg = np.array([6.5, 6.5, 6.5, 6.5])
        ground_truth = QArrayEnv(DEFAULT_PARAMS)
        true_occ = np.round(ground_truth.soft_prediction(vg, T=0.05))

        p = _particle(DEFAULT_PARAMS, sigma_scale=1e-8)
        prob = target_occupation_probability(p.kalman, vg, true_occ)
        assert prob == pytest.approx(1.0, abs=1e-3)

    def test_confident_particle_off_target_gives_near_zero(self):
        vg = np.array([6.5, 6.5, 6.5, 6.5])
        ground_truth = QArrayEnv(DEFAULT_PARAMS)
        true_occ = np.round(ground_truth.soft_prediction(vg, T=0.05))
        wrong_target = true_occ + 3.0

        p = _particle(DEFAULT_PARAMS, sigma_scale=1e-8)
        prob = target_occupation_probability(p.kalman, vg, wrong_target)
        assert prob == pytest.approx(0.0, abs=1e-3)

    def test_larger_sigma_reduces_probability_even_with_same_mean(self):
        # THE FIX, demonstrated directly: two particles with the IDENTICAL
        # mean estimate (both rounding to the target occupation) but
        # different Sigma must get DIFFERENT probabilities -- the old
        # hard-round version would have scored both as 1.0, which is
        # exactly the point-estimate-masks-uncertainty bug this replaces.
        #
        # sigma_scale values recalibrated after fixing observation_jacobian's
        # finite-difference saturation bug (see qarray_env.py's
        # _adaptive_fd_column docstring) -- the corrected Jacobian at this
        # point is ~20x steeper than the old saturated value, so the
        # 'tight' particle needs a correspondingly smaller Sigma to still
        # register as confident.
        # vg CHOSEN DELIBERATELY, not arbitrarily: this must be a point
        # where the observation Jacobian H is actually nonzero, i.e. near
        # a real transition (Section 4c -- H is genuinely zero in a stable
        # plateau, verified directly: an earlier version of this test used
        # vg=[6.5]*4, which sits in a plateau with H identically zero,
        # making sigma_eff zero regardless of Sigma and giving p=1.0 for
        # BOTH particles -- not a bug in target_occupation_probability,
        # a bad test point. This vg was found by scanning
        # QArrayEnv.soft_prediction directly for a genuinely
        # thermally-softened (partial-occupation) point, then confirmed
        # to give a nonzero H via observation_jacobian before use here.
        vg = np.array([9.992498124531133] * 4)
        ground_truth = QArrayEnv(DEFAULT_PARAMS)
        true_occ = np.round(ground_truth.soft_prediction(vg, T=0.05))

        tight = _particle(DEFAULT_PARAMS, sigma_scale=1e-14)
        loose = _particle(DEFAULT_PARAMS, sigma_scale=1e-11)

        p_tight = target_occupation_probability(tight.kalman, vg, true_occ)
        p_loose = target_occupation_probability(loose.kalman, vg, true_occ)

        assert p_tight == pytest.approx(1.0, abs=1e-3)
        assert p_loose < p_tight

    def test_rejects_wrong_target_shape(self):
        p = _particle(DEFAULT_PARAMS)
        with pytest.raises(ValueError):
            target_occupation_probability(p.kalman, np.zeros(N_GATE), np.zeros(N_DOT + 1))


class TestTargetOccupationMass:
    def test_matching_particle_gets_full_mass(self):
        vg = np.zeros(N_GATE)
        ground_truth = QArrayEnv(DEFAULT_PARAMS)
        true_occ = ground_truth.query(vg)

        pf = RBParticleFilter(particles=[_particle(DEFAULT_PARAMS, weight=1.0)])
        mass = target_occupation_mass(pf, vg, true_occ)
        assert mass == pytest.approx(1.0, abs=1e-3)

    def test_mismatched_target_gets_near_zero_mass(self):
        vg = np.zeros(N_GATE)
        ground_truth = QArrayEnv(DEFAULT_PARAMS)
        true_occ = ground_truth.query(vg)
        wrong_target = true_occ + 3.0  # a definitely-different occupation

        pf = RBParticleFilter(particles=[_particle(DEFAULT_PARAMS, weight=1.0)])
        mass = target_occupation_mass(pf, vg, wrong_target)
        assert mass == pytest.approx(0.0, abs=1e-3)

    def test_mixed_population_partial_mass(self):
        # Two particles whose lever arms differ enough to (probably) round
        # to different occupations at a moderately large vg. Verified via
        # the real simulator rather than hand-derived (per this project's
        # own posture: don't trust closed-form physics claims without
        # checking) -- skips gracefully if this particular vg/alpha choice
        # doesn't happen to separate the two, rather than asserting a
        # brittle hardcoded outcome. Both particles are confident (tiny
        # default Sigma), so this exercises the mass-summing logic, not
        # the Sigma-aware probability fix (see TestTargetOccupationProbability
        # for that).
        vg = np.array([12.0, 12.0, 12.0, 12.0])
        params_a = DEFAULT_PARAMS
        params_b = DEFAULT_PARAMS.replace(alpha1=0.08, alpha2=0.08)

        occ_a = np.round(QArrayEnv(params_a).soft_prediction(vg, T=0.05))
        occ_b = np.round(QArrayEnv(params_b).soft_prediction(vg, T=0.05))
        if np.array_equal(occ_a, occ_b):
            pytest.skip(
                "chosen vg/alpha did not separate the two particles' "
                "rounded predictions -- physics-dependent, pick a "
                "different vg/alpha if this ever skips"
            )

        pf = RBParticleFilter(particles=[
            _particle(params_a, weight=0.7),
            _particle(params_b, weight=0.3),
        ])
        mass = target_occupation_mass(pf, vg, occ_a)
        assert mass == pytest.approx(0.7, abs=1e-3)

    def test_rejects_wrong_shape(self):
        pf = RBParticleFilter(particles=[_particle(DEFAULT_PARAMS)])
        with pytest.raises(ValueError):
            target_occupation_mass(pf, np.zeros(N_GATE), np.zeros(N_DOT + 1))


class TestBeliefStable:
    def test_true_when_mass_exceeds_p_conf(self):
        vg = np.zeros(N_GATE)
        ground_truth = QArrayEnv(DEFAULT_PARAMS)
        true_occ = ground_truth.query(vg)
        pf = RBParticleFilter(particles=[_particle(DEFAULT_PARAMS)])
        assert belief_stable(pf, vg, true_occ, p_conf=0.9)

    def test_false_when_mass_below_p_conf(self):
        vg = np.zeros(N_GATE)
        ground_truth = QArrayEnv(DEFAULT_PARAMS)
        true_occ = ground_truth.query(vg)
        wrong_target = true_occ + 3.0
        pf = RBParticleFilter(particles=[_particle(DEFAULT_PARAMS)])
        assert not belief_stable(pf, vg, wrong_target, p_conf=0.9)


class TestActuationQuiescent:
    def test_true_when_all_below_epsilon(self):
        step = np.array([1e-4, -1e-4, 0.0, 5e-5])
        assert actuation_quiescent(step, epsilon=1e-3)

    def test_false_if_any_gate_exceeds_epsilon(self):
        step = np.array([1e-4, -1e-4, 0.5, 5e-5])
        assert not actuation_quiescent(step, epsilon=1e-3)

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError):
            actuation_quiescent(np.zeros(N_GATE + 1), epsilon=1e-3)


class TestConvergenceMonitorActivePolicy:
    """require_actuation_quiescence=True (default) -- the SLAM-method
    configuration, both guards evaluated."""

    def _stable_setup(self):
        vg = np.zeros(N_GATE)
        ground_truth = QArrayEnv(DEFAULT_PARAMS)
        true_occ = ground_truth.query(vg)
        pf = RBParticleFilter(particles=[_particle(DEFAULT_PARAMS)])
        return pf, vg, true_occ

    def test_converges_after_N_consecutive_good_steps(self):
        pf, vg, true_occ = self._stable_setup()
        monitor = ConvergenceMonitor(
            p_conf=0.9, epsilon=1e-3, N=3, target_occupation=true_occ
        )
        good_step = np.zeros(N_GATE)
        statuses = [monitor.step(pf, vg, good_step) for _ in range(3)]
        assert not statuses[0].converged
        assert not statuses[1].converged
        assert statuses[2].converged
        assert statuses[2].consecutive_count == 3
        assert all(s.actuation_quiescence_checked for s in statuses)

    def test_counter_resets_on_actuation_failure(self):
        pf, vg, true_occ = self._stable_setup()
        monitor = ConvergenceMonitor(
            p_conf=0.9, epsilon=1e-3, N=3, target_occupation=true_occ
        )
        good_step = np.zeros(N_GATE)
        bad_step = np.array([1.0, 0.0, 0.0, 0.0])

        monitor.step(pf, vg, good_step)
        monitor.step(pf, vg, good_step)
        status = monitor.step(pf, vg, bad_step)  # breaks the streak
        assert not status.converged
        assert status.consecutive_count == 0
        assert not status.actuation_quiescent

        status2 = monitor.step(pf, vg, good_step)
        assert status2.consecutive_count == 1

    def test_counter_resets_on_belief_instability(self):
        vg = np.zeros(N_GATE)
        ground_truth = QArrayEnv(DEFAULT_PARAMS)
        true_occ = ground_truth.query(vg)
        wrong_target = true_occ + 3.0
        pf = RBParticleFilter(particles=[_particle(DEFAULT_PARAMS)])
        monitor = ConvergenceMonitor(
            p_conf=0.9, epsilon=1e-3, N=2, target_occupation=wrong_target
        )
        good_step = np.zeros(N_GATE)
        status = monitor.step(pf, vg, good_step)
        assert not status.belief_stable
        assert status.consecutive_count == 0

    def test_reset_clears_counter(self):
        pf, vg, true_occ = self._stable_setup()
        monitor = ConvergenceMonitor(
            p_conf=0.9, epsilon=1e-3, N=5, target_occupation=true_occ
        )
        good_step = np.zeros(N_GATE)
        monitor.step(pf, vg, good_step)
        monitor.step(pf, vg, good_step)
        monitor.reset()
        status = monitor.step(pf, vg, good_step)
        assert status.consecutive_count == 1

    def test_rejects_invalid_N(self):
        with pytest.raises(ValueError):
            ConvergenceMonitor(p_conf=0.9, epsilon=1e-3, N=0, target_occupation=np.zeros(N_DOT))

    def test_rejects_wrong_target_shape(self):
        with pytest.raises(ValueError):
            ConvergenceMonitor(p_conf=0.9, epsilon=1e-3, N=1, target_occupation=np.zeros(N_DOT + 1))


class TestConvergenceMonitorNonAdaptiveBaseline:
    """require_actuation_quiescence=False -- the raster-baseline
    configuration. Guard B's VALUE must never gate convergence, but its
    shape is still checked (a caller passing garbage should still be
    caught)."""

    def _stable_setup(self):
        vg = np.zeros(N_GATE)
        ground_truth = QArrayEnv(DEFAULT_PARAMS)
        true_occ = ground_truth.query(vg)
        pf = RBParticleFilter(particles=[_particle(DEFAULT_PARAMS)])
        return pf, vg, true_occ

    def test_large_step_does_not_block_convergence(self):
        pf, vg, true_occ = self._stable_setup()
        monitor = ConvergenceMonitor(
            p_conf=0.9, epsilon=1e-9, N=2, target_occupation=true_occ,
            require_actuation_quiescence=False,
        )
        # A step size that would DEFINITELY fail guard A's active-policy
        # sibling test above (way bigger than epsilon) -- must not matter
        # here, since guard B is disabled.
        huge_step = np.array([5.0, 5.0, 5.0, 5.0])
        statuses = [monitor.step(pf, vg, huge_step) for _ in range(2)]
        assert statuses[1].converged
        assert statuses[1].actuation_quiescent  # reported True by convention
        assert not statuses[1].actuation_quiescence_checked  # but flagged as not meaningful

    def test_still_validates_proposed_step_shape(self):
        pf, vg, true_occ = self._stable_setup()
        monitor = ConvergenceMonitor(
            p_conf=0.9, epsilon=1e-9, N=1, target_occupation=true_occ,
            require_actuation_quiescence=False,
        )
        with pytest.raises(ValueError):
            monitor.step(pf, vg, np.zeros(N_GATE + 1))

    def test_still_gated_by_belief_stability(self):
        vg = np.zeros(N_GATE)
        ground_truth = QArrayEnv(DEFAULT_PARAMS)
        true_occ = ground_truth.query(vg)
        wrong_target = true_occ + 3.0
        pf = RBParticleFilter(particles=[_particle(DEFAULT_PARAMS)])
        monitor = ConvergenceMonitor(
            p_conf=0.9, epsilon=1e-9, N=1, target_occupation=wrong_target,
            require_actuation_quiescence=False,
        )
        status = monitor.step(pf, vg, np.zeros(N_GATE))
        assert not status.belief_stable
        assert not status.converged
