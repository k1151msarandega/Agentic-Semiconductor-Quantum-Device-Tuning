"""
Tests for gz_tuning.control.goal_directed_policy
"""

import numpy as np
import pytest

from convergence import target_occupation_probability
from goal_directed_policy import select_goal_directed_measurement
from kalman import N_PARAMS, ParticleKalmanFilter
from particle_filter import OccupationParticle
from qarray_env import N_DOT, N_GATE


def _champion(mu, sigma_scale=1e-4):
    mu = np.asarray(mu, dtype=np.float64)
    Sigma = np.eye(N_PARAMS) * sigma_scale
    pkf = ParticleKalmanFilter(mu=mu, Sigma=Sigma, E_cm_intra1=0.3, E_cm_intra2=0.3)
    return OccupationParticle(kalman=pkf, weight=1.0)


BASE_MU = np.array([2.5, 2.7, 0.05, 0.05, 0.05])


class TestSelectGoalDirectedMeasurement:
    def test_never_worse_than_staying_put(self):
        # Candidate 0 is always v_cur itself (see module docstring), so
        # the winner's score can never fall below staying put -- unlike a
        # pure-random local batch, which has no such guarantee.
        champion = _champion(BASE_MU)
        target_occupation = np.zeros(N_DOT)
        v_cur = np.full(N_GATE, 5.0)
        rng = np.random.default_rng(0)

        p_at_v_cur = target_occupation_probability(champion.kalman, v_cur, target_occupation)
        result = select_goal_directed_measurement(
            champion, target_occupation, v_cur=v_cur,
            vg_range=(0.0, 15.0), n_candidates=30, local_radius=1.0, rng=rng,
        )
        assert result.best_p_target >= p_at_v_cur - 1e-12

    def test_first_candidate_is_incumbent(self):
        champion = _champion(BASE_MU)
        target_occupation = np.zeros(N_DOT)
        v_cur = np.full(N_GATE, 5.0)
        rng = np.random.default_rng(0)

        result = select_goal_directed_measurement(
            champion, target_occupation, v_cur=v_cur,
            vg_range=(0.0, 15.0), n_candidates=30, rng=rng,
        )
        assert np.allclose(result.all_vg[0], v_cur)

    def test_candidates_respect_vg_range(self):
        champion = _champion(BASE_MU)
        target_occupation = np.zeros(N_DOT)
        v_cur = np.full(N_GATE, 0.5)
        rng = np.random.default_rng(0)

        result = select_goal_directed_measurement(
            champion, target_occupation, v_cur=v_cur,
            vg_range=(0.0, 15.0), n_candidates=30, local_radius=5.0, rng=rng,
        )
        assert np.all(result.all_vg >= 0.0)
        assert np.all(result.all_vg <= 15.0)

    def test_best_is_argmax_of_all_scored_candidates(self):
        champion = _champion(BASE_MU)
        target_occupation = np.zeros(N_DOT)
        v_cur = np.full(N_GATE, 5.0)
        rng = np.random.default_rng(0)

        result = select_goal_directed_measurement(
            champion, target_occupation, v_cur=v_cur,
            vg_range=(0.0, 15.0), n_candidates=30, rng=rng,
        )
        best_idx = int(np.argmax(result.all_p_target))
        assert np.allclose(result.best_vg, result.all_vg[best_idx])
        assert result.best_p_target == pytest.approx(result.all_p_target[best_idx])

    def test_rejects_wrong_shape_target(self):
        champion = _champion(BASE_MU)
        v_cur = np.full(N_GATE, 5.0)
        with pytest.raises(ValueError):
            select_goal_directed_measurement(
                champion, np.zeros(N_DOT + 1), v_cur=v_cur,
            )
