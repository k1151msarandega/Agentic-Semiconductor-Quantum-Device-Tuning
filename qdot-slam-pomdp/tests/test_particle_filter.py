"""
Tests for gz_tuning.init.particle_init
"""

import numpy as np
import pytest

from particle_init import (
    CouplingStratum,
    SharedPriorSpec,
    _allocate_stratum_counts,
    build_coupling_sweep,
    init_particle_filter,
    validate_prior_consistency_across_sweep,
)
from qarray_env import DeviceParams, InfeasibleDeviceParamsError


DEFAULT_SHARED_PRIOR = SharedPriorSpec(
    mean=dict(E_c1=2.5, E_c2=2.7, alpha1=0.05, alpha2=0.05),
    std=dict(E_c1=0.1, E_c2=0.1, alpha1=0.005, alpha2=0.005),
)

DEFAULT_STRATA = [
    CouplingStratum(mean=0.0, std=0.01, weight=0.4),
    CouplingStratum(mean=0.3, std=0.05, weight=0.3),
    CouplingStratum(mean=0.8, std=0.05, weight=0.3),
]

BASE_PARAMS = DeviceParams(
    E_c1=2.5, E_c2=2.7, alpha1=0.05, alpha2=0.05,
    E_cm_intra1=0.3, E_cm_intra2=0.3, cross_capacitance=0.05,
)


class TestSharedPriorSpecValidation:
    def test_rejects_missing_key(self):
        with pytest.raises(ValueError):
            SharedPriorSpec(
                mean=dict(E_c1=2.5, E_c2=2.7, alpha1=0.05),  # missing alpha2
                std=dict(E_c1=0.1, E_c2=0.1, alpha1=0.005, alpha2=0.005),
            )

    def test_rejects_extra_key(self):
        with pytest.raises(ValueError):
            SharedPriorSpec(
                mean=dict(E_c1=2.5, E_c2=2.7, alpha1=0.05, alpha2=0.05, cross_capacitance=0.0),
                std=dict(E_c1=0.1, E_c2=0.1, alpha1=0.005, alpha2=0.005),
            )

    def test_rejects_nonpositive_std(self):
        with pytest.raises(ValueError):
            SharedPriorSpec(
                mean=dict(E_c1=2.5, E_c2=2.7, alpha1=0.05, alpha2=0.05),
                std=dict(E_c1=0.0, E_c2=0.1, alpha1=0.005, alpha2=0.005),
            )


class TestCouplingStratumValidation:
    def test_rejects_nonpositive_std(self):
        with pytest.raises(ValueError):
            CouplingStratum(mean=0.0, std=0.0, weight=0.5)

    def test_rejects_bad_weight(self):
        with pytest.raises(ValueError):
            CouplingStratum(mean=0.0, std=0.01, weight=0.0)
        with pytest.raises(ValueError):
            CouplingStratum(mean=0.0, std=0.01, weight=1.5)


class TestAllocateStratumCounts:
    def test_counts_sum_exactly_to_n_particles(self):
        for n in (7, 10, 13, 100):
            counts = _allocate_stratum_counts(n, DEFAULT_STRATA)
            assert sum(counts) == n
            assert len(counts) == len(DEFAULT_STRATA)

    def test_rejects_weights_not_summing_to_one(self):
        bad_strata = [
            CouplingStratum(mean=0.0, std=0.01, weight=0.5),
            CouplingStratum(mean=0.3, std=0.01, weight=0.3),
        ]
        with pytest.raises(ValueError):
            _allocate_stratum_counts(10, bad_strata)


class TestInitParticleFilter:
    def test_particle_count_and_weight_normalization(self):
        pf = init_particle_filter(
            20, DEFAULT_SHARED_PRIOR, DEFAULT_STRATA,
            prior_sigma_diag=np.array([0.05, 0.05, 0.001, 0.001, 0.01]),
            E_cm_intra1=0.3, E_cm_intra2=0.3,
            rng=np.random.default_rng(42),
        )
        assert pf.n_particles == 20
        weights = np.array([p.weight for p in pf.particles])
        assert weights.sum() == pytest.approx(1.0)

    def test_particle_sigma_matches_prior(self):
        prior_sigma_diag = np.array([0.05, 0.05, 0.001, 0.001, 0.01])
        pf = init_particle_filter(
            10, DEFAULT_SHARED_PRIOR, DEFAULT_STRATA,
            prior_sigma_diag=prior_sigma_diag,
            E_cm_intra1=0.3, E_cm_intra2=0.3,
            rng=np.random.default_rng(1),
        )
        for p in pf.particles:
            np.testing.assert_allclose(np.diag(p.kalman.Sigma), prior_sigma_diag)

    def test_coupling_draws_show_stratified_spread(self):
        # Not asserting exact values (implementation-order-dependent,
        # brittle) -- only that the population shows spread consistent
        # with genuine stratification across 0.0/0.3/0.8, not a single
        # shared Gaussian.
        pf = init_particle_filter(
            300, DEFAULT_SHARED_PRIOR, DEFAULT_STRATA,
            prior_sigma_diag=np.array([0.05, 0.05, 0.001, 0.001, 0.01]),
            E_cm_intra1=0.3, E_cm_intra2=0.3,
            rng=np.random.default_rng(7),
        )
        couplings = np.array([p.kalman.mu[4] for p in pf.particles])
        assert couplings.std() > 0.15

    def test_shared_fields_are_not_stratified(self):
        # E_c1 should show variation consistent with SharedPriorSpec's std
        # regardless of which coupling stratum a particle belongs to.
        pf = init_particle_filter(
            300, DEFAULT_SHARED_PRIOR, DEFAULT_STRATA,
            prior_sigma_diag=np.array([0.05, 0.05, 0.001, 0.001, 0.01]),
            E_cm_intra1=0.3, E_cm_intra2=0.3,
            rng=np.random.default_rng(3),
        )
        e_c1_values = np.array([p.kalman.mu[0] for p in pf.particles])
        assert e_c1_values.std() == pytest.approx(
            DEFAULT_SHARED_PRIOR.std["E_c1"], rel=0.5
        )

    def test_raises_after_exhausting_resample_attempts_for_infeasible_prior(self):
        # alpha near 1.0 is confirmed infeasible at these E_c targets per
        # qarray_env.py's module docstring -- a prior centered there should
        # never successfully draw a feasible particle.
        bad_prior = SharedPriorSpec(
            mean=dict(E_c1=2.5, E_c2=2.7, alpha1=0.99, alpha2=0.99),
            std=dict(E_c1=0.01, E_c2=0.01, alpha1=0.001, alpha2=0.001),
        )
        with pytest.raises(InfeasibleDeviceParamsError):
            init_particle_filter(
                1, bad_prior, DEFAULT_STRATA,
                prior_sigma_diag=np.array([0.05, 0.05, 0.001, 0.001, 0.01]),
                E_cm_intra1=0.3, E_cm_intra2=0.3,
                rng=np.random.default_rng(0),
                max_resample_attempts=3,
            )


class TestBuildCouplingSweep:
    def test_requires_zero_control_point(self):
        with pytest.raises(ValueError):
            build_coupling_sweep(BASE_PARAMS, [0.1, 0.5, 1.0])

    def test_builds_expected_params(self):
        sweep = build_coupling_sweep(BASE_PARAMS, [0.0, 0.5, 1.0])
        assert len(sweep) == 3
        values = [point.true_cross_capacitance for point, _ in sweep]
        assert values == [0.0, 0.5, 1.0]
        for point, params in sweep:
            assert params.cross_capacitance == point.true_cross_capacitance
            assert params.E_c1 == BASE_PARAMS.E_c1
            assert params.alpha1 == BASE_PARAMS.alpha1

    def test_rejects_mismatched_labels(self):
        with pytest.raises(ValueError):
            build_coupling_sweep(BASE_PARAMS, [0.0, 0.5], labels=["only_one"])


class TestValidatePriorConsistencyAcrossSweep:
    def test_passes_for_identical_specs(self):
        specs = [(DEFAULT_SHARED_PRIOR, DEFAULT_STRATA) for _ in range(4)]
        validate_prior_consistency_across_sweep(specs)  # should not raise

    def test_raises_for_differing_shared_prior(self):
        drifted = SharedPriorSpec(
            mean=dict(E_c1=2.6, E_c2=2.7, alpha1=0.05, alpha2=0.05),  # E_c1 changed
            std=dict(E_c1=0.1, E_c2=0.1, alpha1=0.005, alpha2=0.005),
        )
        specs = [
            (DEFAULT_SHARED_PRIOR, DEFAULT_STRATA),
            (drifted, DEFAULT_STRATA),
        ]
        with pytest.raises(ValueError):
            validate_prior_consistency_across_sweep(specs)

    def test_raises_for_differing_strata(self):
        drifted_strata = [
            CouplingStratum(mean=0.0, std=0.01, weight=0.4),
            CouplingStratum(mean=0.3, std=0.05, weight=0.3),
            CouplingStratum(mean=0.9, std=0.05, weight=0.3),  # mean changed
        ]
        specs = [
            (DEFAULT_SHARED_PRIOR, DEFAULT_STRATA),
            (DEFAULT_SHARED_PRIOR, drifted_strata),
        ]
        with pytest.raises(ValueError):
            validate_prior_consistency_across_sweep(specs)
