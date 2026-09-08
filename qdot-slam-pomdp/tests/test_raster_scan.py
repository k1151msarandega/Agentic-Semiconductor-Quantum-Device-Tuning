"""
Tests for gz_tuning.baselines.raster_scan
"""

import numpy as np
import pytest

from convergence import ConvergenceMonitor
from kalman import N_PARAMS, ParticleKalmanFilter, params_to_vector
from noise_model import GaussianReadoutNoise
from particle_filter import OccupationParticle, RBParticleFilter
from qarray_env import DeviceParams, N_GATE, QArrayEnv
from raster_scan import build_raster_grid, run_raster_scan


DEFAULT_PARAMS = DeviceParams(
    E_c1=2.5, E_c2=2.7, alpha1=0.05, alpha2=0.05,
    E_cm_intra1=0.3, E_cm_intra2=0.3, cross_capacitance=0.05,
)


def _particle(params, sigma_scale=1e-4, weight=1.0):
    mu = params_to_vector(params)
    Sigma = np.eye(N_PARAMS) * sigma_scale
    pkf = ParticleKalmanFilter(
        mu=mu, Sigma=Sigma,
        E_cm_intra1=params.E_cm_intra1, E_cm_intra2=params.E_cm_intra2,
    )
    return OccupationParticle(kalman=pkf, weight=weight)


class TestBuildRasterGrid:
    def test_joint_mode_shape(self):
        grid = build_raster_grid((0.0, 1.0), n_points_per_gate=3, mode="joint")
        assert grid.shape == (3 ** N_GATE, N_GATE)

    def test_alternating_mode_point_count(self):
        grid = build_raster_grid(
            (0.0, 1.0), n_points_per_gate=4, mode="alternating", n_alternations=2
        )
        expected = 2 * 2 * (4 ** 2)  # n_alternations * 2 half-passes * grid_2d size
        assert grid.shape == (expected, N_GATE)

    def test_alternating_mode_holds_other_dqd_fixed(self):
        vg_range = (0.0, 1.0)
        mid = float(np.mean(vg_range))
        grid = build_raster_grid(
            vg_range, n_points_per_gate=3, mode="alternating", n_alternations=1
        )
        n_half = 3 ** 2
        first_half = grid[:n_half]
        second_half = grid[n_half:2 * n_half]
        np.testing.assert_allclose(first_half[:, 2], mid)
        np.testing.assert_allclose(first_half[:, 3], mid)
        np.testing.assert_allclose(second_half[:, 0], mid)
        np.testing.assert_allclose(second_half[:, 1], mid)

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError):
            build_raster_grid((0.0, 1.0), 3, mode="nonsense")

    def test_grid_step_size_exceeds_typical_epsilon(self):
        # Documents WHY require_actuation_quiescence must be False for
        # this baseline (see raster_scan.py's module docstring "BUG FOUND
        # AND FIXED"): a real grid's between-point step is large relative
        # to any noise-floor-scale epsilon, by construction. This test
        # exists so that if build_raster_grid's defaults ever change to
        # produce a tiny step size (which would silently make the old,
        # broken guard-B wiring look like it worked by accident), it gets
        # caught here rather than resurfacing as a mysterious "baseline
        # never converges early" regression.
        grid = build_raster_grid((0.0, 15.0), n_points_per_gate=20, mode="joint")
        typical_epsilon = 1e-3  # noise-floor scale, per Section 6a
        step_sizes = np.linalg.norm(np.diff(grid, axis=0), axis=1)
        assert np.median(step_sizes) > typical_epsilon


class TestRunRasterScanEnforcesGuardAOnly:
    def test_rejects_monitor_with_actuation_quiescence_enabled(self):
        target_vg = np.array([0.25, 0.25, 0.25, 0.25])
        grid = np.stack([target_vg] * 3)
        ground_truth = QArrayEnv(DEFAULT_PARAMS)
        true_occ = ground_truth.query(target_vg)

        pf = RBParticleFilter(particles=[_particle(DEFAULT_PARAMS)])
        noise = GaussianReadoutNoise(sigma=0.01)
        bad_monitor = ConvergenceMonitor(
            p_conf=0.5, epsilon=1e-3, N=1, target_occupation=true_occ,
            require_actuation_quiescence=True,  # the mistake this should catch
        )

        with pytest.raises(ValueError, match="require_actuation_quiescence"):
            run_raster_scan(pf, ground_truth, grid, noise, bad_monitor)


class TestRunRasterScan:
    def test_converges_before_grid_exhausted_on_repeated_point(self):
        target_vg = np.array([0.25, 0.25, 0.25, 0.25])
        grid = np.stack([target_vg] * 5)
        ground_truth = QArrayEnv(DEFAULT_PARAMS)
        true_occ = ground_truth.query(target_vg)

        pf = RBParticleFilter(particles=[_particle(DEFAULT_PARAMS)])
        noise = GaussianReadoutNoise(sigma=0.01)
        monitor = ConvergenceMonitor(
            p_conf=0.5, epsilon=10.0, N=1, target_occupation=true_occ,
            require_actuation_quiescence=False,
        )

        result = run_raster_scan(
            pf, ground_truth, grid, noise, monitor, rng=np.random.default_rng(0)
        )
        assert result.converged
        assert result.n_measurements <= len(grid)

    def test_converges_early_on_a_real_moving_grid(self):
        # THE FIX, demonstrated directly: a grid that actually MOVES by a
        # non-trivial step every point (unlike the repeated-point test
        # above, which happened to mask the original bug entirely since
        # its step was always exactly zero). All points here map to the
        # SAME true occupation (a small voltage range well inside one
        # stable charge region), so belief should stabilize and the run
        # should stop well before the grid of 400 points is exhausted --
        # which the OLD guard-B wiring made structurally impossible
        # (see raster_scan.py's docstring).
        small_range = (0.0, 0.05)
        grid = build_raster_grid(small_range, n_points_per_gate=20, mode="joint")
        step_sizes = np.linalg.norm(np.diff(grid, axis=0), axis=1)
        assert np.all(step_sizes > 0)  # confirm this really is a moving grid

        ground_truth = QArrayEnv(DEFAULT_PARAMS)
        occupations = np.array([ground_truth.query(vg) for vg in grid])
        if not np.all(occupations == occupations[0]):
            pytest.skip(
                "chosen small_range crosses a charge-state boundary -- "
                "physics-dependent, narrow the range further if this "
                "ever skips"
            )
        true_occ = occupations[0]

        pf = RBParticleFilter(particles=[_particle(DEFAULT_PARAMS, sigma_scale=1e-4)])
        noise = GaussianReadoutNoise(sigma=0.01)
        monitor = ConvergenceMonitor(
            p_conf=0.9, epsilon=1e-9, N=3, target_occupation=true_occ,
            require_actuation_quiescence=False,
        )

        result = run_raster_scan(
            pf, ground_truth, grid, noise, monitor, rng=np.random.default_rng(1)
        )
        assert result.converged
        assert result.n_measurements < len(grid)

    def test_reports_not_converged_when_grid_exhausted(self):
        vg_range = (0.0, 0.5)
        grid = build_raster_grid(vg_range, n_points_per_gate=2, mode="joint")
        ground_truth = QArrayEnv(DEFAULT_PARAMS)
        impossible_target = np.array([9.0, 9.0, 9.0, 9.0])  # unreachable in this range

        pf = RBParticleFilter(particles=[_particle(DEFAULT_PARAMS)])
        noise = GaussianReadoutNoise(sigma=0.01)
        monitor = ConvergenceMonitor(
            p_conf=0.99, epsilon=1e-9, N=2, target_occupation=impossible_target,
            require_actuation_quiescence=False,
        )

        result = run_raster_scan(
            pf, ground_truth, grid, noise, monitor, rng=np.random.default_rng(0)
        )
        assert not result.converged
        assert result.n_measurements == len(grid)
