"""
tests/test_state.py
===================
Tests for ExperimentState — the single source of truth.
"""

import uuid
import pytest
import numpy as np

from qdot.core.state import ExperimentState, BeliefState
from qdot.core.types import (
    ChargeLabel,
    Classification,
    DQCResult,
    DQCQuality,
    Measurement,
    MeasurementModality,
    OODResult,
    TuningStage,
    VoltagePoint,
)


class TestBeliefState:
    def test_initialise_uniform(self):
        b = BeliefState()
        b.initialise_uniform()
        probs = list(b.charge_probs.values())
        assert abs(sum(probs) - 1.0) < 1e-9
        # All equal
        assert all(abs(p - probs[0]) < 1e-9 for p in probs)

    def test_entropy_is_high_for_uniform(self):
        b = BeliefState()
        b.initialise_uniform()
        assert b.entropy() > 2.0  # log2(9) ≈ 3.17 for 9 states

    def test_most_likely_state(self):
        b = BeliefState()
        b.charge_probs = {(0, 0): 0.1, (1, 1): 0.8, (2, 0): 0.1}
        assert b.most_likely_state() == (1, 1)


class TestExperimentStateFactory:
    def test_new_creates_valid_state(self):
        state = ExperimentState.new(device_id="test_device")
        assert state.device_id == "test_device"
        assert state.run_id != ""
        assert state.stage == TuningStage.BOOTSTRAPPING
        assert len(state.trajectory) == 1
        assert state.step == 0

    def test_initial_voltage_at_origin(self):
        state = ExperimentState.new(device_id="x")
        assert state.current_voltage.vg1 == 0.0
        assert state.current_voltage.vg2 == 0.0


class TestExperimentStateMutation:
    def setup_method(self):
        self.state = ExperimentState.new(device_id="test")

    def test_add_measurement_increments_count(self):
        m = Measurement(
            modality=MeasurementModality.COARSE_2D,
            resolution=32,
            device_id="test",
        )
        self.state.add_measurement(m)
        assert self.state.total_measurements == 32 * 32
        assert m.id in self.state.measurements

    def test_add_line_scan_measurement(self):
        m = Measurement(
            modality=MeasurementModality.LINE_SCAN,
            steps=128,
            device_id="test",
        )
        self.state.add_measurement(m)
        assert self.state.total_measurements == 128

    def test_add_classification_updates_last(self):
        mid = uuid.uuid4()
        m = Measurement(id=mid, modality=MeasurementModality.COARSE_2D, resolution=16)
        self.state.add_measurement(m)

        cls = Classification(
            measurement_id=mid,
            label=ChargeLabel.DOUBLE_DOT,
            confidence=0.88,
        )
        self.state.add_classification(cls)
        assert self.state.last_classification is cls
        assert self.state.last_confidence == 0.88
        assert self.state.last_label == ChargeLabel.DOUBLE_DOT
        # BO history should have one entry
        assert len(self.state.bo_history) == 1

    def test_add_ood_result(self):
        mid = uuid.uuid4()
        ood = OODResult(measurement_id=mid, score=30.0, threshold=24.0, flag=True)
        self.state.add_ood_result(ood)
        assert self.state.is_ood is True
        assert len(self.state.ood_history) == 1

    def test_apply_move(self):
        delta = VoltagePoint(0.05, -0.03)
        self.state.apply_move(delta)
        assert abs(self.state.current_voltage.vg1 - 0.05) < 1e-9
        assert abs(self.state.current_voltage.vg2 + 0.03) < 1e-9
        assert len(self.state.trajectory) == 2

    def test_advance_stage_resets_backtrack_counter(self):
        self.state.consecutive_backtracks = 3
        self.state.advance_stage(TuningStage.CHARGE_ID)
        assert self.state.stage == TuningStage.CHARGE_ID
        assert self.state.consecutive_backtracks == 0

    def test_step_count_equals_decisions(self):
        from qdot.core.types import Decision
        for i in range(3):
            self.state.add_decision(Decision(run_id=self.state.run_id, step=i))
        assert self.state.step == 3


class TestBeliefSummary:
    def test_summary_is_string(self):
        state = ExperimentState.new(device_id="test")
        summary = state.current_belief_summary()
        assert isinstance(summary, str)
        assert "Step" in summary
