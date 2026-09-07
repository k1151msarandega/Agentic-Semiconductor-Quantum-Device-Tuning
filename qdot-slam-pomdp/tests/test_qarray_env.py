"""
Tests for gz_tuning.simulator.qarray_env (v2 -- corrected E_c wiring, t_c dropped)
"""

import numpy as np
import pytest

from qarray_env import (
    N_DOT,
    N_GATE,
    DeviceParams,
    InfeasibleDeviceParamsError,
    QArrayEnv,
    build_capacitance_matrices,
    make_default_ground_truth,
)


def _default_params(**overrides) -> DeviceParams:
    base = dict(
        E_c1=2.5,
        E_c2=2.7,
        alpha1=0.05,
        alpha2=0.05,
        E_cm_intra1=0.3,
        E_cm_intra2=0.3,
        cross_capacitance=0.05,
    )
    base.update(overrides)
    return DeviceParams(**base)


class TestBuildCapacitanceMatrices:
    def test_shapes(self):
        Cdd, Cgd = build_capacitance_matrices(_default_params())
        assert Cdd.shape == (N_DOT, N_DOT)
        assert Cgd.shape == (N_GATE, N_DOT)

    def test_cdd_symmetric(self):
        Cdd, _ = build_capacitance_matrices(_default_params())
        np.testing.assert_allclose(Cdd, Cdd.T)

    def test_no_negative_entries(self):
        """QArray's PositiveValuedMatrix constraint -- this is the exact
        failure mode that broke the v1 energy-matrix-inversion attempt."""
        Cdd, Cgd = build_capacitance_matrices(_default_params())
        assert np.all(Cdd >= 0), f"Cdd has negative entries:\n{Cdd}"
        assert np.all(Cgd >= 0), f"Cgd has negative entries:\n{Cgd}"

    def test_e_c_actually_affects_output(self):
        """The original bug this whole rewrite exists to fix: confirm E_c1
        actually changes something now, unlike v1 where it was inert."""
        Cdd_a, Cgd_a = build_capacitance_matrices(_default_params(E_c1=2.0))
        Cdd_b, Cgd_b = build_capacitance_matrices(_default_params(E_c1=3.0))
        assert not np.allclose(Cdd_a, Cdd_b), "E_c1 change had no effect on Cdd -- regression to the original bug"

    def test_achieved_e_c_matches_target_exactly(self):
        """Joint-solve sanity check: reconstruct the achieved Maxwell-form
        E_c from the returned raw matrices and confirm it matches target to
        numerical precision. Tight tolerance is intentional -- the joint
        (simultaneous 4-equation) solve replaced an earlier sequential
        per-dot approximation that was biased by up to 18% at this file's
        alpha=0.05 regime; this test exists specifically to catch a
        regression back to that approximation."""
        params = _default_params(E_c1=2.5, E_c2=2.7)
        Cdd_raw, Cgd = build_capacitance_matrices(params)

        cdd_sum = Cdd_raw.sum(axis=1)
        cgd_sum = Cgd.sum(axis=1)
        Cdd_offdiag_only = Cdd_raw.copy()
        np.fill_diagonal(Cdd_offdiag_only, 0.0)
        Cdd_maxwell = np.diag(cdd_sum + cgd_sum) - Cdd_offdiag_only
        achieved_E_c = np.diag(np.linalg.inv(Cdd_maxwell))

        np.testing.assert_allclose(achieved_E_c, [2.5, 2.5, 2.7, 2.7], rtol=1e-8)

    def test_infeasible_e_c_raises_clear_error(self):
        """The exact regression case from this conversation: E_c=2.5 at
        alpha=1.0 is not achievable given E_cm_intra=0.3, cross=0.05 --
        confirmed empirically (ceiling ~0.81) before this fix existed."""
        params = _default_params(E_c1=2.5, alpha1=1.0)
        with pytest.raises(InfeasibleDeviceParamsError):
            build_capacitance_matrices(params)

    def test_cross_capacitance_zero_control(self):
        """Primer Section 7a: exact-zero cross-capacitance must remain
        representable and feasible."""
        params = _default_params(cross_capacitance=0.0)
        Cdd, Cgd = build_capacitance_matrices(params)
        assert Cdd[1, 2] == 0.0
        assert np.all(np.isfinite(Cdd))
        assert np.all(np.isfinite(Cgd))

    def test_no_t_c_field_exists(self):
        """Regression guard: t_c1/t_c2 should not be reintroduced as
        DeviceParams fields (Primer decision: dropped, unobservable under
        QArray's constant-interaction model)."""
        assert not hasattr(_default_params(), "t_c1")
        assert not hasattr(_default_params(), "t_c2")
        assert hasattr(_default_params(), "E_cm_intra1")
        assert hasattr(_default_params(), "E_cm_intra2")


class TestQArrayEnv:
    def test_construction_smoke(self):
        env = make_default_ground_truth()
        assert env.n_dot == N_DOT
        assert env.n_gate == N_GATE

    def test_construction_with_v1_style_alpha_raises(self):
        """The exact bug from this conversation: alpha=1.0 (the v1 default)
        should now fail loudly at construction time, not silently produce
        an inert E_c or a cryptic linear-algebra error deep in QArray."""
        with pytest.raises(InfeasibleDeviceParamsError):
            QArrayEnv(_default_params(alpha1=1.0, alpha2=1.0))

    def test_query_single_point(self):
        env = make_default_ground_truth()
        vg = np.array([2.0, 2.0, 2.0, 2.0])
        out = env.query(vg)
        assert out.shape == (N_DOT,)

    def test_query_batch(self):
        env = make_default_ground_truth()
        rng = np.random.default_rng(0)
        vg = rng.uniform(-5, 5, size=(25, N_GATE))
        out = env.query(vg)
        assert out.shape == (25, N_DOT)

    def test_query_wrong_last_dim_raises(self):
        env = make_default_ground_truth()
        with pytest.raises(ValueError):
            env.query(np.array([0.1, 0.2, 0.3]))

    def test_update_params_changes_output(self):
        """cross_capacitance bumped to 0.3 -- a genuine 'strong coupling'
        point (equal to E_cm_intra, not just a token perturbation).
        Verified feasible against the CORRECT joint solver: the true
        ceiling at these E_c/alpha/E_cm_intra values is ~0.716, not the
        ~1.0 an earlier (buggy, sequential-solver-based) check suggested --
        that discrepancy was caught by this exact test failing, and is why
        this value is 0.3, not 1.0."""
        env = make_default_ground_truth(cross_capacitance=0.0)
        rng = np.random.default_rng(1)
        vg = rng.uniform(-8, 8, size=(300, N_GATE))
        out_before = np.asarray(env.query(vg))

        env.update_params(env.params.replace(cross_capacitance=0.3))
        out_after = np.asarray(env.query(vg))

        assert not np.array_equal(out_before, out_after)

    def test_update_params_preserves_params_object(self):
        env = make_default_ground_truth()
        new_params = env.params.replace(E_cm_intra1=0.5)
        env.update_params(new_params)
        assert env.params.E_cm_intra1 == pytest.approx(0.5)

    def test_update_params_infeasible_raises(self):
        """alpha1=1.0 reproduces the exact regression case from this
        conversation -- E_c=2.5 was never achievable at alpha=1.0, at any
        self-capacitance, given the default coupling values."""
        env = make_default_ground_truth()
        with pytest.raises(InfeasibleDeviceParamsError):
            env.update_params(env.params.replace(alpha1=1.0))

    def test_zero_coupling_reduces_to_decoupled_structure(self):
        env = QArrayEnv(_default_params(cross_capacitance=0.0))
        rng = np.random.default_rng(2)
        dqd_b_sweep = rng.uniform(-5, 5, size=(15, 2))
        vg = np.zeros((15, N_GATE))
        vg[:, 0] = 1.0
        vg[:, 1] = 1.0
        vg[:, 2:] = dqd_b_sweep

        out = np.asarray(env.query(vg))
        dqd_a_occupation = out[:, :2]
        expected = np.broadcast_to(dqd_a_occupation[0], dqd_a_occupation.shape)
        np.testing.assert_allclose(dqd_a_occupation, expected, atol=1e-9)

    def test_default_ground_truth_e_c_asymmetry_preserved(self):
        """E_c1=2.5 != E_c2=2.7 in the default -- confirm both dots of each
        DQD actually end up distinguishable from the other DQD, i.e. the
        per-dot solve didn't accidentally collapse everything to one value."""
        env = make_default_ground_truth()
        Cdd_raw, Cgd = build_capacitance_matrices(env.params)
        cdd_sum = Cdd_raw.sum(axis=1)
        cgd_sum = Cgd.sum(axis=1)
        Cdd_offdiag_only = Cdd_raw.copy()
        np.fill_diagonal(Cdd_offdiag_only, 0.0)
        Cdd_maxwell = np.diag(cdd_sum + cgd_sum) - Cdd_offdiag_only
        achieved_E_c = np.diag(np.linalg.inv(Cdd_maxwell))
        assert achieved_E_c[0] < achieved_E_c[2]  # DQD-A (E_c=2.5) < DQD-B (E_c=2.7)


class TestObservationJacobian:
    def test_zero_sensitivity_on_symmetry_line_is_real_not_a_bug(self):
        """Regression guard for a genuine, verified structural finding: at
        vg0=vg1 with dot0/dot1 sharing identical target E_c, the (1,0,..)
        vs (0,1,..) comparison is identically E_c-independent -- confirmed
        algebraically and numerically. H's E_c1 column should be ~0 here."""
        from qarray_env import DeviceParams, observation_jacobian
        params = DeviceParams(E_c1=2.5, E_c2=2.7, alpha1=0.05, alpha2=0.05,
                               E_cm_intra1=0.3, E_cm_intra2=0.3, cross_capacitance=0.05)
        vg_symmetric = np.array([6.8, 6.8, 0.0, 0.0])
        H = observation_jacobian(params, vg_symmetric)
        assert abs(H[0, 0]) < 1e-6, "Expected near-zero E_c1 sensitivity on the symmetry line"

    def test_real_sensitivity_at_correctly_located_transition(self):
        """The genuinely E_c-sensitive nearby transition, at the precisely
        derived location v = A00/(2*alpha1*(A00+A01)) -- should show large,
        real sensitivity, confirming the symmetry-line result above is a
        structural fact about THAT point, not a general Jacobian failure."""
        from qarray_env import DeviceParams, build_capacitance_matrices, observation_jacobian
        params = DeviceParams(E_c1=2.5, E_c2=2.7, alpha1=0.05, alpha2=0.05,
                               E_cm_intra1=0.3, E_cm_intra2=0.3, cross_capacitance=0.05)
        Cdd_raw, Cgd = build_capacitance_matrices(params)
        cdd_sum = Cdd_raw.sum(axis=1)
        cgd_sum = Cgd.sum(axis=1)
        offdiag_only = Cdd_raw.copy()
        np.fill_diagonal(offdiag_only, 0.0)
        Cdd_maxwell = np.diag(cdd_sum + cgd_sum) - offdiag_only
        A = np.linalg.inv(Cdd_maxwell)
        v_transition = A[0, 0] / (2 * params.alpha1 * (A[0, 0] + A[0, 1]))

        H = observation_jacobian(params, np.array([v_transition, v_transition, 0.0, 0.0]))
        assert abs(H[0, 0]) > 100, f"Expected large E_c1 sensitivity at the true transition, got {H[0,0]}"
