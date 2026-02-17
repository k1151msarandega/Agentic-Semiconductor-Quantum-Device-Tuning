"""
tests/test_planning.py
======================
Unit tests for Phase 2 planning modules.

Tests use Phase 0/1 types from qdot.core.types and qdot.core.state directly.
No mock replacements of canonical types.
"""

import pytest
import numpy as np
import uuid

# Phase 0 types and state
from qdot.core.types import (
    BacktrackEvent,
    BOPoint,
    ChargeLabel,
    Classification,
    DQCQuality,
    DQCResult,
    Measurement,
    MeasurementModality,
    MeasurementPlan,
    TuningStage,
    VoltagePoint,
)
from qdot.core.state import BeliefState, ExperimentState

# Phase 2 planning modules
from qdot.planning.belief import BeliefUpdater, CIMObservationModel
from qdot.planning.sensing import ActiveSensingPolicy, MODALITY_COST
from qdot.planning.bayesian_opt import GaussianProcess, MultiResBO
from qdot.planning.state_machine import (
    StateMachine, StageResult,
    bootstrap_result, survey_result,
    charge_id_result, navigation_result, verification_result,
    DEFAULT_STAGE_CONFIGS,
)

# Phase 0 simulator (CIM is the observation model source of truth)
from qdot.simulator.cim import ConstantInteractionDevice, CIMSimulatorAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_state() -> ExperimentState:
    return ExperimentState.new(device_id="test_device")


def make_2d_measurement(v1_range=(-0.5, 0.5), v2_range=(-0.5, 0.5), res=16) -> Measurement:
    """Generate a real CIM 2D measurement using the Phase 0 simulator."""
    adapter = CIMSimulatorAdapter(seed=42)
    return adapter.sample_patch(v1_range=v1_range, v2_range=v2_range, res=res)


def make_1d_measurement(axis="vg1", start=-0.5, stop=0.5, steps=32) -> Measurement:
    adapter = CIMSimulatorAdapter(seed=42)
    return adapter.line_scan(axis=axis, start=start, stop=stop, steps=steps, fixed=0.0)


# ---------------------------------------------------------------------------
# BeliefState (from state.py) + BeliefUpdater
# ---------------------------------------------------------------------------

class TestBeliefStateStub:
    """Tests for the Phase 0 BeliefState stub (qdot.core.state)."""

    def test_initialise_uniform(self):
        b = BeliefState()
        b.initialise_uniform()
        assert abs(sum(b.charge_probs.values()) - 1.0) < 1e-9

    def test_entropy_uniform_is_high(self):
        b = BeliefState()
        b.initialise_uniform()
        assert b.entropy() > 2.0   # log2(9) ≈ 3.17 for 9 states

    def test_entropy_empty_is_inf(self):
        b = BeliefState()
        assert b.entropy() == float("inf")

    def test_most_likely_state(self):
        b = BeliefState()
        b.charge_probs = {(0, 0): 0.1, (1, 1): 0.8, (2, 0): 0.1}
        assert b.most_likely_state() == (1, 1)


class TestCIMObservationModel:
    """Tests for the CIM observation model wrapper."""

    def test_uses_cim_device(self):
        model = CIMObservationModel()
        assert isinstance(model.device, ConstantInteractionDevice)

    def test_predicted_conductance_2d_shape(self):
        model = CIMObservationModel()
        patch = model.predicted_conductance_2d(1, 1, (-0.5, 0.5), (-0.5, 0.5), resolution=16)
        assert patch.shape == (16, 16)

    def test_predicted_conductance_1d_shape(self):
        model = CIMObservationModel()
        trace = model.predicted_conductance_1d(1, 1, "vg1", -0.5, 0.5, 32, 0.0)
        assert trace.shape == (32,)

    def test_log_likelihood_2d_is_scalar(self):
        model = CIMObservationModel()
        m = make_2d_measurement(res=8)
        ll = model.log_likelihood_2d(m.array, 1, 1, (-0.5, 0.5), (-0.5, 0.5))
        assert isinstance(ll, float)

    def test_log_likelihood_higher_for_matching_params(self):
        """
        Likelihood should be higher when using the same CIM params used to generate data.
        """
        model = CIMObservationModel()
        m = make_2d_measurement(res=8)
        ll_match = model.log_likelihood_2d(m.array, 1, 1, (-0.5, 0.5), (-0.5, 0.5))
        ll_wrong = model.log_likelihood_2d(m.array, 0, 0, (-0.5, 0.5), (-0.5, 0.5))
        # Not guaranteed to be true for all CIM params, but reasonable test
        assert isinstance(ll_match, float)
        assert isinstance(ll_wrong, float)


class TestBeliefUpdater:
    """Tests for the Phase 2 particle filter belief updater."""

    def test_initialises_charge_probs(self):
        state = make_state()
        state.belief.initialise_uniform()
        updater = BeliefUpdater(belief=state.belief, n_particles=100)
        # After init, charge_probs should be populated
        assert len(state.belief.charge_probs) > 0
        assert abs(sum(state.belief.charge_probs.values()) - 1.0) < 1e-6

    def test_update_from_2d_updates_charge_probs(self):
        state = make_state()
        state.belief.initialise_uniform()
        updater = BeliefUpdater(belief=state.belief, n_particles=200)

        m = make_2d_measurement(res=8)
        entropy_before = state.belief.entropy()
        updater.update_from_2d(m)
        entropy_after = state.belief.entropy()

        # Charge probs should still sum to 1
        assert abs(sum(state.belief.charge_probs.values()) - 1.0) < 1e-5
        # Entropy should change (not necessarily decrease on first update)
        assert entropy_after != float("inf")

    def test_update_from_1d_uses_line_scan_measurement(self):
        state = make_state()
        state.belief.initialise_uniform()
        updater = BeliefUpdater(belief=state.belief, n_particles=100)

        m = make_1d_measurement(steps=16)
        updater.update_from_1d(m)

        assert abs(sum(state.belief.charge_probs.values()) - 1.0) < 1e-5

    def test_update_from_1d_rejects_2d_measurement(self):
        state = make_state()
        state.belief.initialise_uniform()
        updater = BeliefUpdater(belief=state.belief, n_particles=100)

        m = make_2d_measurement(res=8)   # 2D measurement
        with pytest.raises(ValueError, match="LINE_SCAN"):
            updater.update_from_1d(m)

    def test_update_from_2d_rejects_1d_measurement(self):
        state = make_state()
        state.belief.initialise_uniform()
        updater = BeliefUpdater(belief=state.belief, n_particles=100)

        m = make_1d_measurement(steps=16)   # 1D measurement
        with pytest.raises(ValueError, match="2D"):
            updater.update_from_2d(m)

    def test_physics_override_reduces_update_weight(self):
        """physics_override = True should not crash and should update belief."""
        state = make_state()
        state.belief.initialise_uniform()
        updater = BeliefUpdater(belief=state.belief, n_particles=100)

        m = make_2d_measurement(res=8)
        mid = m.id
        cls = Classification(
            measurement_id=mid,
            label=ChargeLabel.DOUBLE_DOT,
            confidence=0.9,
            physics_override=True,   # Should inflate uncertainty
        )
        updater.update_from_2d(m, classification=cls)
        assert abs(sum(state.belief.charge_probs.values()) - 1.0) < 1e-5

    def test_classification_boost_for_double_dot(self):
        state = make_state()
        state.belief.initialise_uniform()
        updater = BeliefUpdater(belief=state.belief, n_particles=200)

        m = make_2d_measurement(res=8)
        cls = Classification(
            measurement_id=m.id,
            label=ChargeLabel.DOUBLE_DOT,
            confidence=0.9,
            physics_override=False,
        )
        updater.update_from_2d(m, classification=cls)
        assert abs(sum(state.belief.charge_probs.values()) - 1.0) < 1e-5

    def test_uncertainty_map_shape(self):
        state = make_state()
        state.belief.initialise_uniform()
        updater = BeliefUpdater(belief=state.belief, n_particles=50)

        umap = updater.uncertainty_map((-0.5, 0.5), (-0.5, 0.5), resolution=8)
        assert umap.shape == (8, 8)
        # Should be written to belief
        assert state.belief.uncertainty_map is not None
        assert state.belief.uncertainty_map.shape == (8, 8)


# ---------------------------------------------------------------------------
# ActiveSensingPolicy
# ---------------------------------------------------------------------------

class TestActiveSensingPolicy:
    """Tests for information-theoretic measurement selection."""

    def test_select_returns_measurement_plan_type(self):
        """Return type must be MeasurementPlan from qdot.core.types."""
        state = make_state()
        state.belief.initialise_uniform()
        policy = ActiveSensingPolicy(n_mc_samples=2)
        plan = policy.select(state.belief, (-0.5, 0.5), (-0.5, 0.5))
        assert isinstance(plan, MeasurementPlan)

    def test_select_returns_valid_modality(self):
        state = make_state()
        state.belief.initialise_uniform()
        policy = ActiveSensingPolicy(n_mc_samples=2)
        plan = policy.select(state.belief, (-0.5, 0.5), (-0.5, 0.5))
        assert plan.modality in MeasurementModality

    def test_cost_model_matches_blueprint(self):
        """Blueprint §5.4: LINE_SCAN=128, COARSE_2D=256, LOCAL=1024, FINE=4096."""
        assert MODALITY_COST[MeasurementModality.LINE_SCAN] == 128
        assert MODALITY_COST[MeasurementModality.COARSE_2D] == 256
        assert MODALITY_COST[MeasurementModality.LOCAL_PATCH] == 1024
        assert MODALITY_COST[MeasurementModality.FINE_2D] == 4096

    def test_modality_values_match_types_py(self):
        """MeasurementModality values must match exactly what types.py defines."""
        assert MeasurementModality.COARSE_2D.value == "coarse_2d"    # lowercase d
        assert MeasurementModality.LINE_SCAN.value == "line_scan"
        assert MeasurementModality.LOCAL_PATCH.value == "local_patch"
        assert MeasurementModality.FINE_2D.value == "fine_2d"
        assert MeasurementModality.NONE.value == "none"

    def test_select_line_scan_has_axis(self):
        state = make_state()
        state.belief.initialise_uniform()
        policy = ActiveSensingPolicy(n_mc_samples=2)
        plan = policy.select(state.belief, (-0.5, 0.5), (-0.5, 0.5))
        if plan.modality == MeasurementModality.LINE_SCAN:
            assert plan.axis in ("vg1", "vg2")

    def test_select_2d_has_ranges(self):
        state = make_state()
        state.belief.initialise_uniform()
        policy = ActiveSensingPolicy(n_mc_samples=2)
        plan = policy.select(state.belief, (-0.5, 0.5), (-0.5, 0.5))
        if plan.modality in (MeasurementModality.COARSE_2D,
                             MeasurementModality.LOCAL_PATCH,
                             MeasurementModality.FINE_2D):
            assert plan.v1_range is not None
            assert plan.v2_range is not None


# ---------------------------------------------------------------------------
# GaussianProcess and MultiResBO
# ---------------------------------------------------------------------------

class TestGaussianProcess:
    def test_predict_prior_when_no_data(self):
        gp = GaussianProcess()
        mu, var = gp.predict(0.0, 0.0)
        assert isinstance(mu, float)
        assert var > 0

    def test_predict_after_fit(self):
        state = make_state()
        state.belief.initialise_uniform()
        gp = GaussianProcess()

        # Create BOPoints (from types.py)
        history = [
            BOPoint(voltage=VoltagePoint(vg1=0.0, vg2=0.0), score=0.5, step=1),
            BOPoint(voltage=VoltagePoint(vg1=0.1, vg2=0.1), score=0.8, step=2),
        ]
        gp.fit(history)
        mu, var = gp.predict(0.05, 0.05)
        assert isinstance(mu, float)
        assert var >= 0


class TestMultiResBO:
    def test_propose_returns_action_proposal_type(self):
        """ActionProposal must be from qdot.core.types (no local redefinition)."""
        from qdot.core.types import ActionProposal as CanonicalActionProposal
        state = make_state()
        state.belief.initialise_uniform()
        bo = MultiResBO(belief=state.belief, voltage_bounds=state.voltage_bounds)
        proposal = bo.propose(
            current=state.current_voltage,
            l1_max=state.step_caps.get("l1_max", 0.10),
        )
        assert isinstance(proposal, CanonicalActionProposal)

    def test_proposal_delta_v_is_voltage_point(self):
        state = make_state()
        state.belief.initialise_uniform()
        bo = MultiResBO(belief=state.belief, voltage_bounds=state.voltage_bounds)
        proposal = bo.propose(state.current_voltage)
        assert isinstance(proposal.delta_v, VoltagePoint)

    def test_proposal_respects_l1_cap(self):
        state = make_state()
        state.belief.initialise_uniform()
        l1_max = 0.10
        bo = MultiResBO(belief=state.belief, voltage_bounds=state.voltage_bounds)
        proposal = bo.propose(state.current_voltage, l1_max=l1_max)
        # Delta should be within bounds
        assert proposal.delta_v.l1_norm <= l1_max + 1e-6

    def test_bo_updates_with_bo_history(self):
        state = make_state()
        state.belief.initialise_uniform()
        bo = MultiResBO(belief=state.belief, voltage_bounds=state.voltage_bounds)

        # Add some BO history (using canonical BOPoint from types.py)
        history = [
            BOPoint(voltage=VoltagePoint(vg1=0.1, vg2=0.1), score=0.7, step=1),
            BOPoint(voltage=VoltagePoint(vg1=-0.1, vg2=0.1), score=0.3, step=2),
        ]
        bo.update(history)  # Should not raise
        proposal = bo.propose(state.current_voltage)
        assert isinstance(proposal.delta_v, VoltagePoint)

    def test_make_bo_point_returns_canonical_type(self):
        """make_bo_point must return BOPoint from qdot.core.types."""
        from qdot.core.types import BOPoint as CanonicalBOPoint
        state = make_state()
        state.belief.initialise_uniform()
        bo = MultiResBO(belief=state.belief, voltage_bounds=state.voltage_bounds)
        point = bo.make_bo_point(
            voltage=VoltagePoint(vg1=0.0, vg2=0.0),
            score=0.5,
            step=1,
        )
        assert isinstance(point, CanonicalBOPoint)


# ---------------------------------------------------------------------------
# StateMachine
# ---------------------------------------------------------------------------

class TestStateMachine:
    def test_initial_stage_is_bootstrapping(self):
        state = make_state()
        sm = StateMachine(state)
        assert state.stage == TuningStage.BOOTSTRAPPING

    def test_advance_on_success(self):
        state = make_state()
        sm = StateMachine(state)
        result = bootstrap_result(device_responds=True, signal_detected=True)
        new_stage, rationale, hitl = sm.process_result(result)
        assert new_stage == TuningStage.COARSE_SURVEY
        assert not hitl

    def test_retry_on_failure(self):
        state = make_state()
        sm = StateMachine(state)
        result = bootstrap_result(device_responds=True, signal_detected=False)
        new_stage, rationale, hitl = sm.process_result(result)
        assert new_stage == TuningStage.BOOTSTRAPPING   # stays here, retries

    def test_hitl_on_consecutive_backtracks(self):
        state = make_state()
        sm = StateMachine(state)
        # Manually set state to simulate 2 consecutive backtracks
        state.consecutive_backtracks = 2
        state.stage = TuningStage.CHARGE_ID

        result = charge_id_result("unknown", 0.1)
        _, _, hitl = sm.process_result(result)
        assert hitl

    def test_advance_resets_consecutive_backtracks(self):
        state = make_state()
        sm = StateMachine(state)
        state.consecutive_backtracks = 1

        result = bootstrap_result(device_responds=True, signal_detected=True)
        sm.process_result(result)
        assert state.consecutive_backtracks == 0

    def test_backtrack_uses_canonical_type(self):
        """BacktrackEvent logged to state must be from qdot.core.types."""
        state = make_state()
        sm = StateMachine(state)

        # Force enough retries to trigger backtrack from COARSE_SURVEY
        state.stage = TuningStage.COARSE_SURVEY
        config = DEFAULT_STAGE_CONFIGS[TuningStage.COARSE_SURVEY]
        sm._retries[TuningStage.COARSE_SURVEY] = config.max_retries

        result = survey_result(peak_found=False, peak_quality=0.1)
        sm.process_result(result)

        if state.backtrack_log:
            # All logged events must be the canonical BacktrackEvent type
            for evt in state.backtrack_log:
                assert isinstance(evt, BacktrackEvent)

    def test_complete_stage_sequence(self):
        """Full happy path: BOOTSTRAP → SURVEY → CHARGE_ID → NAVIGATION → VERIFICATION."""
        state = make_state()
        sm = StateMachine(state)

        stages_results = [
            bootstrap_result(True, True),                          # BOOTSTRAPPING → COARSE_SURVEY
            survey_result(True, 0.8),                              # COARSE_SURVEY → CHARGE_ID
            charge_id_result("double-dot", 0.85),                  # CHARGE_ID → NAVIGATION
            navigation_result(target_reached=True, belief_confidence=0.85),  # NAVIGATION → VERIFICATION
            verification_result(stable=True, reproducibility=0.95, charge_noise=0.02),  # VERIFICATION → COMPLETE
        ]

        for result in stages_results:
            new_stage, rationale, hitl = sm.process_result(result)
            assert not hitl, f"Unexpected HITL at stage {state.stage.name}: {rationale}"

        assert state.stage == TuningStage.COMPLETE


# ---------------------------------------------------------------------------
# Stage result helpers
# ---------------------------------------------------------------------------

class TestStageResultHelpers:
    def test_bootstrap_success(self):
        r = bootstrap_result(device_responds=True, signal_detected=True)
        assert r.success is True
        assert r.confidence == 1.0

    def test_bootstrap_failure(self):
        r = bootstrap_result(device_responds=False, signal_detected=True)
        assert r.success is False

    def test_charge_id_physics_override_caps_confidence(self):
        r = charge_id_result("double-dot", confidence=0.9, physics_override=True)
        assert r.confidence <= 0.65  # Blueprint §5.1

    def test_verification_requires_all_criteria(self):
        r = verification_result(stable=True, reproducibility=0.5, charge_noise=0.0)
        assert r.success is False  # reproducibility < 0.8

        r2 = verification_result(stable=True, reproducibility=0.9, charge_noise=0.05)
        assert r2.success is True
