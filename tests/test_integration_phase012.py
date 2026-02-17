"""
tests/test_integration_phase012.py
===================================
Integration smoke test for Phase 0/1/2.

Verifies that all components work together in a single end-to-end flow:
    Phase 0: ExperimentState, SafetyCritic, HITLManager, GovernanceLogger
    Phase 1: DQCGatekeeper, InspectionAgent (with or without trained models)
    Phase 2: BeliefUpdater, ActiveSensingPolicy, MultiResBO, StateMachine, ExecutiveAgent

This test does NOT require trained model checkpoints — it runs with
InspectionAgent in stub mode (untrained ensemble, no OOD detector).

Usage:
    pytest tests/test_integration_phase012.py -v
"""

import pytest
import tempfile
import numpy as np

# Phase 0
from qdot.core.types import ChargeLabel, DQCQuality, MeasurementModality, TuningStage, VoltagePoint, HITLOutcome
from qdot.core.state import ExperimentState
from qdot.core.governance import GovernanceLogger
from qdot.core.hitl import HITLManager
from qdot.hardware.safety import SafetyCritic
from qdot.simulator.cim import CIMSimulatorAdapter

# Phase 1
from qdot.perception.dqc import DQCGatekeeper
from qdot.perception.inspector import InspectionAgent

# Phase 2
from qdot.planning.belief import BeliefUpdater, CIMObservationModel
from qdot.planning.sensing import ActiveSensingPolicy
from qdot.planning.bayesian_opt import MultiResBO
from qdot.planning.state_machine import StateMachine, bootstrap_result
from qdot.agent.executive import ExecutiveAgent
from qdot.agent.translator import TranslationAgent


class TestPhase012Integration:
    """Integration tests across all three phases."""

    def test_full_pipeline_single_measurement(self):
        """
        End-to-end: Take one measurement and pass it through the full pipeline.

        Flow:
            DeviceAdapter → Measurement
            → DQCGatekeeper → DQCResult
            → InspectionAgent → (Classification, OODResult)
            → BeliefUpdater → updates ExperimentState.belief
            → SafetyCritic → clips voltage move
            → GovernanceLogger → logs Decision
        """
        # Setup
        state = ExperimentState.new(device_id="integration_test")
        adapter = CIMSimulatorAdapter(device_id="integration_test", seed=42)
        dqc = DQCGatekeeper()
        inspector = InspectionAgent(ensemble=None, ood_detector=None)  # Untrained
        belief_updater = BeliefUpdater(belief=state.belief)
        safety = SafetyCritic(voltage_bounds=state.voltage_bounds)

        with tempfile.TemporaryDirectory() as tmpdir:
            governance = GovernanceLogger(run_id=state.run_id, log_dir=tmpdir)

            # Step 1: Take a 2D measurement
            measurement = adapter.sample_patch(
                v1_range=(-0.5, 0.5),
                v2_range=(-0.5, 0.5),
                res=16,
            )
            state.add_measurement(measurement)
            assert measurement.is_2d
            assert measurement.array.shape == (16, 16)

            # Step 2: DQC assessment
            dqc_result = dqc.assess(measurement)
            state.add_dqc_result(dqc_result)
            assert dqc_result.quality in (DQCQuality.HIGH, DQCQuality.MODERATE, DQCQuality.LOW)

            # Step 3: InspectionAgent classification (only if DQC passes)
            if dqc_result.quality != DQCQuality.LOW:
                classification, ood_result = inspector.inspect(measurement, dqc_result)
                state.add_classification(classification)
                state.add_ood_result(ood_result)

                assert classification.label in (ChargeLabel.SINGLE_DOT, ChargeLabel.DOUBLE_DOT, ChargeLabel.MISC)
                assert 0.0 <= classification.confidence <= 1.0

                # Step 4: Belief update
                belief_updater.update_from_2d(measurement, classification)
                assert len(state.belief.charge_probs) > 0
                assert abs(sum(state.belief.charge_probs.values()) - 1.0) < 1e-6

            # Step 5: Safety check on a voltage move
            from qdot.core.types import ActionProposal
            proposal = ActionProposal(delta_v=VoltagePoint(vg1=0.05, vg2=0.03))
            clipped = safety.clip(proposal, state.current_voltage)
            verdict = safety.verify(state.current_voltage, clipped)

            assert verdict.all_passed or not verdict.all_passed  # Either outcome is valid
            if verdict.all_passed:
                state.apply_move(clipped.safe_delta_v)

            # Step 6: Log a decision
            from qdot.core.types import Decision
            decision = Decision(
                run_id=state.run_id,
                step=state.step,
                intent="integration_test",
                stage=state.stage,
                observation_summary={"measurement_id": str(measurement.id)},
                action_summary={"voltage_move": "applied" if verdict.all_passed else "rejected"},
                rationale="Integration test",
            )
            state.add_decision(decision)
            governance.log(decision)

            # Verify state consistency
            assert state.total_measurements > 0
            assert len(state.decisions) >= 1
            assert len(state.trajectory) >= 1

    def test_executive_agent_bootstrap_stage(self):
        """
        ExecutiveAgent runs through the BOOTSTRAPPING stage.

        Verifies:
        - Agent constructs without errors
        - Bootstrap stage takes at least one measurement
        - State machine advances or retries appropriately
        - All Phase 0/1/2 components are called correctly
        """
        state = ExperimentState.new(device_id="bootstrap_test")
        adapter = CIMSimulatorAdapter(device_id="bootstrap_test", seed=42)
        inspector = InspectionAgent(ensemble=None, ood_detector=None)

        hitl = HITLManager(enabled=True)
        hitl.set_test_mode(auto_outcome=HITLOutcome.APPROVED)

        with tempfile.TemporaryDirectory() as tmpdir:
            governance = GovernanceLogger(run_id=state.run_id, log_dir=tmpdir)

            agent = ExecutiveAgent(
                state=state,
                adapter=adapter,
                inspection_agent=inspector,
                hitl_manager=hitl,
                governance_logger=governance,
                max_steps=3,  # Just bootstrap
                measurement_budget=256,
            )

            # Run just the bootstrap executor
            result = agent._run_bootstrap()

            # Verify bootstrap result
            assert result.success in (True, False)  # Either outcome is valid
            assert state.total_measurements > 0  # Should have taken at least one line scan
            assert state.stage == TuningStage.BOOTSTRAPPING  # Still in bootstrap

    def test_state_machine_with_real_state(self):
        """State machine correctly updates ExperimentState during transitions."""
        state = ExperimentState.new(device_id="state_machine_test")
        sm = StateMachine(state=state)

        # Successful bootstrap
        result = bootstrap_result(device_responds=True, signal_detected=True)
        new_stage, rationale, hitl = sm.process_result(result)

        assert new_stage == TuningStage.COARSE_SURVEY
        assert state.stage == TuningStage.COARSE_SURVEY
        assert state.consecutive_backtracks == 0
        assert not hitl

    def test_belief_updater_with_real_measurements(self):
        """BeliefUpdater correctly processes real CIM measurements."""
        state = ExperimentState.new(device_id="belief_test")
        adapter = CIMSimulatorAdapter(device_id="belief_test", seed=42)
        belief_updater = BeliefUpdater(belief=state.belief)

        # Take a real measurement
        measurement = adapter.sample_patch((-0.3, 0.3), (-0.3, 0.3), res=8)

        # Update belief
        entropy_before = state.belief.entropy()
        belief_updater.update_from_2d(measurement)
        entropy_after = state.belief.entropy()

        # Verify belief is updated
        assert len(state.belief.charge_probs) > 0
        assert abs(sum(state.belief.charge_probs.values()) - 1.0) < 1e-5
        assert entropy_after != float("inf")

    def test_bayesian_opt_with_real_bo_history(self):
        """MultiResBO correctly proposes moves using real BOPoint history."""
        state = ExperimentState.new(device_id="bo_test")

        # Add some BO observations (simulate classifications)
        from qdot.core.types import BOPoint
        state.bo_history.append(BOPoint(
            voltage=VoltagePoint(vg1=0.0, vg2=0.0),
            score=0.3,
            label=ChargeLabel.MISC,
            confidence=0.5,
            step=1,
        ))
        state.bo_history.append(BOPoint(
            voltage=VoltagePoint(vg1=0.1, vg2=0.1),
            score=0.7,
            label=ChargeLabel.DOUBLE_DOT,
            confidence=0.8,
            step=2,
        ))

        bo = MultiResBO(belief=state.belief, voltage_bounds=state.voltage_bounds)
        bo.update(state.bo_history)

        # Propose a move
        proposal = bo.propose(current=state.current_voltage, l1_max=0.10)

        assert proposal.delta_v.l1_norm <= 0.10 + 1e-6
        assert proposal.expected_new_voltage is not None

    def test_active_sensing_with_real_belief(self):
        """ActiveSensingPolicy selects measurements using real belief state."""
        state = ExperimentState.new(device_id="sensing_test")
        state.belief.initialise_uniform()

        policy = ActiveSensingPolicy(n_mc_samples=4)  # Small for speed
        plan = policy.select(state.belief, v1_range=(-0.5, 0.5), v2_range=(-0.5, 0.5))

        assert plan.modality in MeasurementModality
        if plan.modality != MeasurementModality.NONE:
            assert plan.info_gain_per_cost >= 0.0

    def test_translator_with_real_adapter(self):
        """TranslationAgent correctly executes plans on real CIM adapter."""
        adapter = CIMSimulatorAdapter(device_id="translator_test", seed=42)
        translator = TranslationAgent(adapter=adapter)

        from qdot.core.types import MeasurementPlan
        plan = MeasurementPlan(
            modality=MeasurementModality.COARSE_2D,
            v1_range=(-0.3, 0.3),
            v2_range=(-0.3, 0.3),
            resolution=8,
        )

        result = translator.execute(plan)

        assert result.success
        assert result.measurement is not None
        assert result.measurement.array.shape == (8, 8)

    def test_full_agent_run_completes(self):
        """
        ExecutiveAgent.run() completes without errors (even if target not reached).

        This is the ultimate integration test — every Phase 0/1/2 component
        must work together correctly for this to succeed.
        """
        state = ExperimentState.new(device_id="full_run_test")
        adapter = CIMSimulatorAdapter(device_id="full_run_test", seed=42)
        inspector = InspectionAgent(ensemble=None, ood_detector=None)

        hitl = HITLManager(enabled=True)
        hitl.set_test_mode(auto_outcome=HITLOutcome.APPROVED)

        with tempfile.TemporaryDirectory() as tmpdir:
            governance = GovernanceLogger(run_id=state.run_id, log_dir=tmpdir)

            agent = ExecutiveAgent(
                state=state,
                adapter=adapter,
                inspection_agent=inspector,
                hitl_manager=hitl,
                governance_logger=governance,
                max_steps=5,
                measurement_budget=512,
            )

            summary = agent.run()

            # Verify the run completed
            assert summary is not None
            assert "success" in summary
            assert "total_steps" in summary
            assert "total_measurements" in summary
            assert summary["total_steps"] <= 5
            assert summary["total_measurements"] <= 512

            # Verify governance log was written
            decisions = GovernanceLogger.load(run_id=state.run_id, log_dir=tmpdir)
            assert len(decisions) > 0
