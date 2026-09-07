"""Tests for gz_tuning.acquisition.bald (rewrite: exact joint entropy)"""

import numpy as np
import pytest

from bald import _entropy, _local_candidate_states, compute_ig_bald
from kalman import ParticleKalmanFilter, N_PARAMS
from particle_filter import OccupationParticle, RBParticleFilter
from qarray_env import DeviceParams, QArrayEnv, build_capacitance_matrices


def _make_particle(E_c1=2.5, E_c2=2.7, alpha1=0.05, alpha2=0.05, cross=0.05, sigma_scale=0.3, weight=1.0):
    mu = np.array([E_c1, E_c2, alpha1, alpha2, cross])
    Sigma = np.eye(N_PARAMS) * sigma_scale
    kf = ParticleKalmanFilter(mu=mu, Sigma=Sigma, E_cm_intra1=0.3, E_cm_intra2=0.3)
    return OccupationParticle(kalman=kf, weight=weight)


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


class TestEntropy:
    def test_zero_at_certainty(self):
        assert _entropy(np.array([1.0, 0.0, 0.0])) == pytest.approx(0.0)

    def test_max_at_uniform(self):
        p = np.array([0.25, 0.25, 0.25, 0.25])
        assert _entropy(p) == pytest.approx(np.log(4))

    def test_handles_exact_zero_probability(self):
        p = np.array([0.5, 0.5, 0.0])
        assert _entropy(p) == pytest.approx(np.log(2))


class TestLocalCandidateStates:
    def test_confident_particle_gives_single_candidate(self):
        pred = np.array([[0.01, 0.99, 2.0, 0.0]])  # all dots confidently at integers
        candidates = _local_candidate_states(pred)
        assert candidates.shape == (1, 4)
        np.testing.assert_array_equal(candidates[0], [0, 1, 2, 0])

    def test_one_uncertain_dot_gives_two_candidates(self):
        pred = np.array([[0.5, 0.0, 0.0, 0.0]])
        candidates = _local_candidate_states(pred)
        assert candidates.shape == (2, 4)

    def test_two_uncertain_dots_gives_four_candidates(self):
        pred = np.array([[0.5, 0.5, 0.0, 0.0]])
        candidates = _local_candidate_states(pred)
        assert candidates.shape == (4, 4)

    def test_union_across_particles(self):
        pred = np.array([
            [0.5, 0.0, 0.0, 0.0],   # uncertain on dot 0 only
            [0.0, 0.5, 0.0, 0.0],   # uncertain on dot 1 only
        ])
        candidates = _local_candidate_states(pred)
        # union of {(0,0,0,0),(1,0,0,0)} and {(0,0,0,0),(0,1,0,0)} = 3 unique states
        assert candidates.shape[0] == 3


class TestAntiCorrelationFix:
    def test_triple_point_joint_entropy_is_log3(self):
        """The 'precise transition voltage' formula derived during
        qarray_env.py's development turns out to locate a TRIPLE
        degeneracy point -- (0,0), (1,0), AND (0,1) all simultaneously
        tied, not a clean 2-way split (verified directly: soft_prediction
        there is (1/3, 1/3, 0, 0), not (0.5, 0.5, 0, 0)). True joint
        entropy there is ln(3), confirmed exactly. This test exists so the
        triple-point behavior is documented and checked, separate from the
        genuine clean 2-way case below (an earlier version of this test
        incorrectly conflated the two)."""
        params = DeviceParams(E_c1=2.5, E_c2=2.7, alpha1=0.05, alpha2=0.05,
                                E_cm_intra1=0.3, E_cm_intra2=0.3, cross_capacitance=0.05)
        env = QArrayEnv(params)
        vg = _precise_transition_vg(params)
        pred = env.soft_prediction(vg, T=0.05)
        assert pred[0] == pytest.approx(1 / 3, abs=0.01)

        candidates = _local_candidate_states(pred.reshape(1, -1))
        F = env.free_energy_at_candidates(candidates, vg)
        w = np.exp(-F / 0.05)
        w /= w.sum()
        joint_entropy = _entropy(w)
        assert joint_entropy == pytest.approx(np.log(3), abs=0.01)

    def test_clean_two_way_point_joint_entropy_is_log2_not_2log2(self):
        """THE critical regression test for the fix: moving further along
        the vg0=vg1 line (away from the triple point at v~6.514, to v=8.0)
        gives a genuinely clean (1,0)<->(0,1) 50/50 split -- (0,0) becomes
        disfavored while (1,0)/(0,1) stay exactly tied (proven earlier:
        that tie holds identically along the whole vg0=vg1 line, regardless
        of vg). True joint entropy here must be ln(2) (one genuine bit),
        NOT 2*ln(2) (what summing independent per-dot marginal entropies
        would give, since each dot looks 50/50 in isolation despite being
        perfectly anti-correlated) -- this is the exact, provable
        double-counting bug the rewrite exists to fix."""
        params = DeviceParams(E_c1=2.5, E_c2=2.7, alpha1=0.05, alpha2=0.05,
                                E_cm_intra1=0.3, E_cm_intra2=0.3, cross_capacitance=0.05)
        env = QArrayEnv(params)
        vg = np.array([8.0, 8.0, 0.0, 0.0])
        pred = env.soft_prediction(vg, T=0.05)
        assert pred[0] == pytest.approx(0.5, abs=0.001)
        assert pred[1] == pytest.approx(0.5, abs=0.001)

        candidates = _local_candidate_states(pred.reshape(1, -1))
        F = env.free_energy_at_candidates(candidates, vg)
        w = np.exp(-F / 0.05)
        w /= w.sum()
        joint_entropy = _entropy(w)

        assert joint_entropy == pytest.approx(np.log(2), abs=0.01), (
            f"Expected true joint entropy ~ln(2)={np.log(2):.4f}, got "
            f"{joint_entropy:.4f} -- if this is close to 2*ln(2)="
            f"{2*np.log(2):.4f} instead, the double-counting bug has "
            f"regressed."
        )

    def test_single_particle_bald_is_zero_regardless_of_which_point(self):
        """Whether at the triple point or the clean 2-way point, a SINGLE
        particle's BALD must be exactly zero (mixture == that particle's
        own distribution when there's only one particle) -- this holds
        regardless of how the underlying joint entropy is structured."""
        particle = _make_particle(E_c1=2.5)
        pf = RBParticleFilter(particles=[particle])
        for vg in [_precise_transition_vg(particle.kalman.params), np.array([8.0, 8.0, 0.0, 0.0])]:
            ig = compute_ig_bald(pf, vg)
            assert ig == pytest.approx(0.0, abs=1e-9)


class TestComputeIgBald:
    def test_nonnegative_always(self):
        particles = [_make_particle(E_c1=e) for e in [2.3, 2.5, 2.7, 2.9]]
        pf = RBParticleFilter(particles=particles)
        rng = np.random.default_rng(0)
        for _ in range(10):
            vg = rng.uniform(0, 15, size=4)
            ig = compute_ig_bald(pf, vg)
            assert ig >= -1e-9, f"Negative BALD at vg={vg}: {ig}"

    def test_zero_when_all_particles_identical(self):
        particles = [_make_particle(E_c1=2.5), _make_particle(E_c1=2.5), _make_particle(E_c1=2.5)]
        pf = RBParticleFilter(particles=particles)
        vg = _precise_transition_vg(particles[0].kalman.params)
        ig = compute_ig_bald(pf, vg)
        assert ig == pytest.approx(0.0, abs=1e-9)

    def test_positive_when_particles_disagree(self):
        p_low = _make_particle(E_c1=1.5)
        p_high = _make_particle(E_c1=3.2)
        pf = RBParticleFilter(particles=[p_low, p_high])
        vg = _precise_transition_vg(p_low.kalman.params)
        ig = compute_ig_bald(pf, vg)
        assert ig > 0.01

    def test_bounded_by_log_n_particles(self):
        """Genuine information-theoretic bound: BALD is mutual information
        between which-particle-is-correct and the observation, so it can
        never exceed the entropy of the particle distribution itself,
        log(n_particles) for equal weights."""
        particles = [_make_particle(E_c1=e) for e in [1.0, 1.5, 2.0, 2.5, 3.0]]
        pf = RBParticleFilter(particles=particles)
        rng = np.random.default_rng(1)
        for _ in range(15):
            vg = rng.uniform(0, 20, size=4)
            ig = compute_ig_bald(pf, vg)
            assert ig <= np.log(5) + 1e-6

    def test_bounded_by_log_k_candidates(self):
        """The other genuine bound: BALD also can't exceed the entropy of
        the observation side, log(K) for K candidate states."""
        particles = [_make_particle(E_c1=1.5), _make_particle(E_c1=3.2)]
        pf = RBParticleFilter(particles=particles)
        vg = _precise_transition_vg(particles[0].kalman.params)
        from bald import _local_candidate_states
        envs = [QArrayEnv(p.kalman.params) for p in particles]
        predictions = np.stack([env.soft_prediction(vg, T=0.05) for env in envs])
        K = _local_candidate_states(predictions).shape[0]

        ig = compute_ig_bald(pf, vg)
        assert ig <= np.log(K) + 1e-6

    def test_single_particle_always_zero(self):
        """With only one hypothesis, there's nothing to disagree about --
        BALD must be exactly zero everywhere, regardless of how uncertain
        that single particle's own prediction is."""
        particle = _make_particle(E_c1=2.5)
        pf = RBParticleFilter(particles=[particle])
        rng = np.random.default_rng(2)
        for _ in range(10):
            vg = rng.uniform(0, 15, size=4)
            ig = compute_ig_bald(pf, vg)
            assert ig == pytest.approx(0.0, abs=1e-9)
