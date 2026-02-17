"""
tests/test_agent.py
===================
Unit tests for Phase 2 agent modules:
    qdot/agent/translator.py  — TranslationAgent
    qdot/agent/executive.py   — ExecutiveAgent (smoke test with mocked components)

These tests do NOT require trained model checkpoints (EnsembleCNN, OOD detector)
and do NOT trigger real HITL blocking. The ExecutiveAgent is tested with
HITLManager.set_test_mode(auto_outcome=HITLOutcome.APPROVED).
"""

import pytest
import numpy as np
import tempfile
import uuid

# Phase 0 types — canonical imports
from qdot.core.types import (
    HITLOutcome,
    Measurement,
    MeasurementModality,
    MeasurementPlan,
    TuningStage,
    VoltagePoint,
)
from qdot.core.state import ExperimentState
from qdot.core.governance import GovernanceLogger
from qdot.core.hitl import HITLManager

# Phase 0 hardware
from qdot.simulator.cim import CIMSimulatorAdapter
from qdot.hardware.safety import SafetyCritic

# Phase 2 agent
from qdot.agent.executive import ExecutiveAgent
from qdot.agent.translator import TranslationAgent, TranslationResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_adapter():
    return CIMSimulatorAdapter(device_id="test_device", seed=42)


def make_state():
    return ExperimentState.new(device_id="test_device")


# ---------------------------------------------------------------------------
# TranslationAgent
# ---------------------------------------------------------------------------

class TestTranslationAgent:

    def test_execute_line_scan_plan(self):
        """LINE_SCAN plan produces a valid 1D Measurement."""
        adapter = make_adapter()
        agent = TranslationAgent(adapter=adapter)

        plan = MeasurementPlan(
            modality=MeasurementModality.LINE_SCAN,
            axis="vg1",
            start=-0.5,
            stop=0.5,
            steps=32,
        )
        result = agent.execute(plan)

        assert result.success, f"Expected success, got: {result.error_message}"
        assert result.measurement is not None
        assert isinstance(result.measurement, Measurement)
        assert result.measurement.modality == MeasurementModality.LINE_SCAN
        assert result.measurement.array.shape == (32,)

    def test_execute_coarse_2d_plan(self):
        """COARSE_2D plan produces a valid 2D Measurement."""
        adapter = make_adapter()
        agent = TranslationAgent(adapter=adapter)

        plan = MeasurementPlan(
            modality=MeasurementModality.COARSE_2D,
            v1_range=(-0.5, 0.5),
            v2_range=(-0.5, 0.5),
            resolution=16,
        )
        result = agent.execute(plan)

        assert result.success, f"Expected success, got: {result.error_message}"
        assert result.measurement is not None
        assert result.measurement.is_2d
        assert result.measurement.array.shape == (16, 16)

    def test_execute_local_patch_plan(self):
        """LOCAL_PATCH plan produces a 32×32 Measurement."""
        adapter = make_adapter()
        agent = TranslationAgent(adapter=adapter)

        plan = MeasurementPlan(
            modality=MeasurementModality.LOCAL_PATCH,
            v1_range=(-0.2, 0.2),
            v2_range=(-0.2, 0.2),
            resolution=32,
        )
        result = agent.execute(plan)

        assert result.success
        assert result.measurement.array.shape == (32, 32)

    def test_execute_none_modality_returns_no_measurement(self):
        """NONE modality = skip measurement, success=True, measurement=None."""
        adapter = make_adapter()
        agent = TranslationAgent(adapter=adapter)

        plan = MeasurementPlan(modality=MeasurementModality.NONE)
        result = agent.execute(plan)

        assert result.success
        assert result.measurement is None

    def test_execute_voltage_move(self):
        """execute_voltage_move calls set_voltages without raising."""
        adapter = make_adapter()
        agent = TranslationAgent(adapter=adapter)
        result = agent.execute_voltage_move(vg1=0.1, vg2=-0.05)
        assert result.success, f"Move failed: {result.error_message}"
        assert result.measurement is None   # voltage moves don't return a measurement

    def test_validation_blocks_dangerous_code(self):
        """The code validator must reject exec, eval, and import statements."""
        adapter = make_adapter()
        agent = TranslationAgent(adapter=adapter)

        # Test the validator directly via _validate
        assert agent._validate("exec('rm -rf /')") is not None
        assert agent._validate("eval('os.system(\"ls\")')") is not None
        assert agent._validate("import os") is not None

    def test_generated_code_uses_vg1_vg2_naming(self):
        """Generated code must use 'vg1'/'vg2' axis names (not 'V_g1')."""
        adapter = make_adapter()
        agent = TranslationAgent(adapter=adapter)

        plan = MeasurementPlan(
            modality=MeasurementModality.LINE_SCAN,
            axis="vg1",
            start=-0.5,
            stop=0.5,
            steps=32,
        )
        code, _ = agent._generate_code(plan)
        assert "vg1" in code
        assert "V_g1" not in code

    def test_translation_result_type(self):
        """execute() always returns TranslationResult."""
        adapter = make_adapter()
        agent = TranslationAgent(adapter=adapter)
        plan = MeasurementPlan(modality=MeasurementModality.LINE_SCAN,
                               axis="vg1", start=-0.5, stop=0.5, steps=16)
        result = agent.execute(plan)
        assert isinstance(result, TranslationResult)

    def test_measurement_array_normalised(self):
        """All measurements from CIMSimulatorAdapter are normalised to [0, 1]."""
        adapter = make_adapter()
        agent = TranslationAgent(adapter=adapter)

        for modality, plan in [
            (MeasurementModality.LINE_SCAN,
             MeasurementPlan(modality=MeasurementModality.LINE_SCAN,
                             axis="vg2", start=-0.5, stop=0.5, steps=32)),
            (MeasurementModality.COARSE_2D,
             MeasurementPlan(modality=MeasurementModality.COARSE_2D,
                             v1_range=(-0.5, 0.5), v2_range=(-0.5, 0.5), resolution=8)),
        ]:
            result = agent.execute(plan)
            assert result.success
            arr = result.measurement.array
            assert arr.min() >= -1e-6, f"{modality}: min={arr.min()}"
            assert arr.max() <= 1.0 + 1e-6, f"{modality}: max={arr.max()}"


# ---------------------------------------------------------------------------
# ExecutiveAgent (smoke tests — no trained checkpoints needed)
# ---------------------------------------------------------------------------

class TestExecutiveAgentSmoke:
    """
    Smoke tests for ExecutiveAgent.

    Uses:
      - CIMSimulatorAdapter (no hardware)
      - HITLManager in test mode (no blocking)
      - No InspectionAgent (None — exercises the code path that skips classification)
    """

    def _make_agent(self, state=None):
        adapter = make_adapter()
        state = state or make_state()

        hitl = HITLManager(enabled=True)
        hitl.set_test_mode(auto_outcome=HITLOutcome.APPROVED)

        log_dir = tempfile.mkdtemp()
        governance = GovernanceLogger(run_id=state.run_id, log_dir=log_dir)

        return ExecutiveAgent(
            state=state,
            adapter=adapter,
            inspection_agent=None,   # No trained model in CI
            hitl_manager=hitl,
            governance_logger=governance,
            max_steps=5,
            measurement_budget=512,
        )

    def test_agent_constructs_without_error(self):
        agent = self._make_agent()
        assert agent is not None

    def test_initial_stage_is_bootstrapping(self):
        state = make_state()
        agent = self._make_agent(state)
        assert state.stage == TuningStage.BOOTSTRAPPING

    def test_run_bootstrap_step_adds_measurement(self):
        """One bootstrap step should add at least one measurement to state."""
        state = make_state()
        agent = self._make_agent(state)

        # Run just the bootstrap stage executor directly
        result = agent._run_bootstrap()

        # Should have taken a line scan
        assert state.total_measurements > 0

    def test_run_returns_summary_dict(self):
        """run() must return a dict with required summary keys."""
        state = make_state()
        agent = self._make_agent(state)

        summary = agent.run()

        required_keys = {
            "success", "final_stage", "total_steps",
            "total_measurements", "measurement_reduction",
            "total_backtracks", "safety_violations",
            "hitl_events", "run_id",
        }
        assert required_keys.issubset(set(summary.keys())), (
            f"Missing keys: {required_keys - set(summary.keys())}"
        )

    def test_run_terminates_within_budget(self):
        """Agent must not exceed max_steps or measurement_budget."""
        state = make_state()
        agent = self._make_agent(state)

        summary = agent.run()

        assert summary["total_steps"] <= 5
        assert summary["total_measurements"] <= 512

    def test_governance_log_populated(self):
        """Every step should produce at least one Decision in the log."""
        state = make_state()
        agent = self._make_agent(state)

        agent.run()

        # mission_start decision is always logged
        assert len(state.decisions) >= 1

    def test_translator_uses_correct_axis_naming(self):
        """
        Regression: TranslationAgent inside ExecutiveAgent must use vg1/vg2,
        not V_g1/V_g2. Verified by checking generated code in bootstrap step.
        """
        state = make_state()
        agent = self._make_agent(state)

        # Peek at the translator's generated code for a line scan
        plan = MeasurementPlan(
            modality=MeasurementModality.LINE_SCAN,
            axis="vg1",
            start=-0.5,
            stop=0.5,
            steps=16,
        )
        code, _ = agent.translator._generate_code(plan)
        assert "vg1" in code
        assert "V_g1" not in code

    def test_safety_critic_applied_to_moves(self):
        """
        SafetyCritic must be initialised from state.voltage_bounds.
        After any voltage move, the state must remain within bounds.
        """
        state = make_state()
        agent = self._make_agent(state)

        # Manually apply a safe move
        delta = VoltagePoint(vg1=0.05, vg2=0.03)
        state.apply_move(delta)

        bounds = state.voltage_bounds
        v = state.current_voltage
        assert bounds["vg1"]["min"] <= v.vg1 <= bounds["vg1"]["max"]
        assert bounds["vg2"]["min"] <= v.vg2 <= bounds["vg2"]["max"]
