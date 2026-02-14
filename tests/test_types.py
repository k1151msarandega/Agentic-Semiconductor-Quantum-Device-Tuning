"""
tests/test_types.py
===================
Unit tests for qdot/core/types.py.

These tests verify:
  - Every dataclass can be instantiated with default values
  - Enum values are stable (string values used in governance logs)
  - VoltagePoint arithmetic is correct
  - SafetyVerdict.all_passed logic is correct
"""

import pytest
from qdot.core.types import (
    ActionProposal,
    BacktrackEvent,
    BOPoint,
    ChargeLabel,
    Classification,
    Decision,
    DQCQuality,
    DQCResult,
    HITLEvent,
    HITLOutcome,
    Measurement,
    MeasurementModality,
    MeasurementPlan,
    OODResult,
    SafetyCheckResult,
    SafetyVerdict,
    TuningStage,
    VoltagePoint,
)
import uuid


# ---------------------------------------------------------------------------
# VoltagePoint
# ---------------------------------------------------------------------------

class TestVoltagePoint:
    def test_as_dict(self):
        vp = VoltagePoint(0.5, -0.3)
        assert vp.as_dict() == {"vg1": 0.5, "vg2": -0.3}

    def test_from_dict(self):
        vp = VoltagePoint.from_dict({"vg1": 0.1, "vg2": 0.2})
        assert vp.vg1 == 0.1
        assert vp.vg2 == 0.2

    def test_l1_norm(self):
        vp = VoltagePoint(0.05, -0.03)
        assert abs(vp.l1_norm - 0.08) < 1e-9

    def test_delta_to(self):
        a = VoltagePoint(0.0, 0.0)
        b = VoltagePoint(0.1, -0.05)
        delta = a.delta_to(b)
        assert abs(delta.vg1 - 0.1) < 1e-9
        assert abs(delta.vg2 + 0.05) < 1e-9

    def test_immutable(self):
        vp = VoltagePoint(0.5, 0.5)
        with pytest.raises(Exception):  # frozen dataclass raises FrozenInstanceError
            vp.vg1 = 1.0


# ---------------------------------------------------------------------------
# Enum stability (values used in stored governance logs)
# ---------------------------------------------------------------------------

class TestEnumStability:
    """If these fail after a rename, governance logs from previous runs break."""

    def test_dqc_quality_values(self):
        assert DQCQuality.HIGH.value == "high"
        assert DQCQuality.MODERATE.value == "moderate"
        assert DQCQuality.LOW.value == "low"

    def test_charge_label_values(self):
        assert ChargeLabel.SINGLE_DOT.value == "single-dot"
        assert ChargeLabel.DOUBLE_DOT.value == "double-dot"
        assert ChargeLabel.MISC.value == "misc"

    def test_tuning_stage_ordering(self):
        # Stage order must be stable for state machine logic
        assert TuningStage.BOOTSTRAPPING.value == 0
        assert TuningStage.COMPLETE.value == 5
        assert TuningStage.FAILED.value == -1

    def test_hitl_outcome_values(self):
        assert HITLOutcome.APPROVED.value == "approved"
        assert HITLOutcome.REJECTED.value == "rejected"
        assert HITLOutcome.MODIFIED.value == "modified"
        assert HITLOutcome.PENDING.value == "pending"


# ---------------------------------------------------------------------------
# SafetyVerdict
# ---------------------------------------------------------------------------

class TestSafetyVerdict:
    def _make_check(self, passed: bool, margin: float = 0.1) -> SafetyCheckResult:
        return SafetyCheckResult(
            check_name="test",
            passed=passed,
            margin=margin,
            per_gate={"vg1": margin, "vg2": margin},
        )

    def test_all_passed_when_all_checks_pass(self):
        v = SafetyVerdict(
            voltage_bounds=self._make_check(True, 0.5),
            slew_rate=self._make_check(True, 0.05),
            voltage_margin=self._make_check(True, 0.1),
        )
        assert v.all_passed is True

    def test_all_passed_false_if_any_check_fails(self):
        v = SafetyVerdict(
            voltage_bounds=self._make_check(True, 0.5),
            slew_rate=self._make_check(False, -0.02),
            voltage_margin=self._make_check(True, 0.1),
        )
        assert v.all_passed is False

    def test_min_margin(self):
        v = SafetyVerdict(
            voltage_bounds=self._make_check(True, 0.5),
            slew_rate=self._make_check(True, 0.03),
            voltage_margin=self._make_check(True, 0.1),
        )
        assert abs(v.min_margin - 0.03) < 1e-9


# ---------------------------------------------------------------------------
# Default instantiation (smoke tests)
# ---------------------------------------------------------------------------

class TestDefaultInstantiation:
    def test_measurement(self):
        m = Measurement()
        assert m.id is not None
        assert m.is_2d is True  # default modality is COARSE_2D

    def test_classification(self):
        mid = uuid.uuid4()
        c = Classification(
            measurement_id=mid,
            label=ChargeLabel.DOUBLE_DOT,
            confidence=0.87,
        )
        assert c.label == ChargeLabel.DOUBLE_DOT

    def test_ood_result_margin(self):
        ood = OODResult(
            measurement_id=uuid.uuid4(),
            score=18.0,
            threshold=24.0,
            flag=False,
        )
        assert abs(ood.margin - 6.0) < 1e-9
        assert not ood.flag

    def test_decision_defaults(self):
        d = Decision()
        assert d.run_id == ""
        assert d.llm_tokens_used == 0

    def test_measurement_plan(self):
        plan = MeasurementPlan(modality=MeasurementModality.LINE_SCAN)
        assert plan.steps == 128
