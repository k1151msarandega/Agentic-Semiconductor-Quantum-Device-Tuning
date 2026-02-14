"""
qdot/core/state.py
==================
ExperimentState — single source of truth for the entire agent run.

The old system scattered state across _MEAS dicts, separate BO history lists,
governance JSONL files, and Granite prompt logs. Everything lives here now.

All modules read from and write to this object. No module holds private state
that other modules need.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from uuid import UUID

import numpy as np

from qdot.core.types import (
    ActionProposal,
    BacktrackEvent,
    BOPoint,
    ChargeLabel,
    Classification,
    Decision,
    DQCResult,
    HITLEvent,
    Measurement,
    MeasurementModality,
    OODResult,
    SafetyVerdict,
    TuningStage,
    VoltagePoint,
)


@dataclass
class BeliefState:
    """
    POMDP belief state: P(charge configuration | observations, CIM prior).

    Phase 0 stub — particle filter and CIM observation model are Phase 2.
    For now this holds the data structures that Phase 2 will populate.
    """
    # Probability distribution over charge states (n1, n2)
    # Keys are (n1, n2) tuples; values are probabilities summing to 1.0
    charge_probs: Dict[tuple, float] = field(default_factory=dict)

    # Uncertainty map over voltage space: shape (H, W), values ∈ [0, 1]
    uncertainty_map: Optional[Any] = None  # np.ndarray once fitted

    # CIM physics parameters for this device (updated by DisorderLearner)
    device_params: Dict[str, float] = field(default_factory=lambda: {
        "E_c1": 2.3,
        "E_c2": 2.5,
        "t_c": 0.15,
        "T": 0.1,
        "lever_arm": 0.5,
        "noise_level": 0.02,
    })

    # Disorder posterior (None until DisorderLearner has run)
    disorder_estimate: Optional[Dict[str, Any]] = None

    def entropy(self) -> float:
        """Shannon entropy over charge state distribution."""
        if not self.charge_probs:
            return float("inf")
        probs = np.array(list(self.charge_probs.values()), dtype=float)
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))

    def most_likely_state(self) -> Optional[tuple]:
        if not self.charge_probs:
            return None
        return max(self.charge_probs, key=lambda k: self.charge_probs[k])

    def initialise_uniform(self, charge_states: Optional[List[tuple]] = None) -> None:
        """Initialise with uniform prior over given charge states."""
        if charge_states is None:
            charge_states = [(n1, n2) for n1 in range(3) for n2 in range(3)]
        p = 1.0 / len(charge_states)
        self.charge_probs = {s: p for s in charge_states}


@dataclass
class ExperimentState:
    """
    Centralised state object. All modules read from and write to this.

    Usage:
        state = ExperimentState.new(device_id="sim_device_001")
        state.add_measurement(measurement)
        state.add_classification(classification)
        state.add_decision(decision)
    """

    # -----------------------------------------------------------------------
    # Identity
    # -----------------------------------------------------------------------
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    device_id: str = ""
    started_at: float = field(default_factory=time.time)
    target_label: ChargeLabel = ChargeLabel.DOUBLE_DOT

    # -----------------------------------------------------------------------
    # Spatial / voltage
    # -----------------------------------------------------------------------
    current_voltage: VoltagePoint = field(default_factory=lambda: VoltagePoint(0.0, 0.0))
    trajectory: List[VoltagePoint] = field(default_factory=list)
    voltage_bounds: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "vg1": {"min": -1.0, "max": 1.0},
        "vg2": {"min": -1.0, "max": 1.0},
    })
    step_caps: Dict[str, float] = field(default_factory=lambda: {
        "l1_max": 0.10,
    })

    # -----------------------------------------------------------------------
    # Belief
    # -----------------------------------------------------------------------
    belief: BeliefState = field(default_factory=BeliefState)

    # -----------------------------------------------------------------------
    # Measurements and results
    # -----------------------------------------------------------------------
    measurements: Dict[UUID, Measurement] = field(default_factory=dict)
    dqc_results: Dict[UUID, DQCResult] = field(default_factory=dict)
    classifications: Dict[UUID, Classification] = field(default_factory=dict)
    ood_history: List[OODResult] = field(default_factory=list)

    # Most recent results (convenience accessors — updated on every add)
    last_classification: Optional[Classification] = None
    last_ood: Optional[OODResult] = None
    last_dqc: Optional[DQCResult] = None

    # -----------------------------------------------------------------------
    # Optimisation
    # -----------------------------------------------------------------------
    bo_history: List[BOPoint] = field(default_factory=list)

    # -----------------------------------------------------------------------
    # Governance / audit trail
    # -----------------------------------------------------------------------
    decisions: List[Decision] = field(default_factory=list)
    hitl_events: List[HITLEvent] = field(default_factory=list)
    backtrack_log: List[BacktrackEvent] = field(default_factory=list)

    # -----------------------------------------------------------------------
    # State machine
    # -----------------------------------------------------------------------
    stage: TuningStage = TuningStage.BOOTSTRAPPING
    consecutive_backtracks: int = 0   # at current stage
    total_backtracks: int = 0

    # -----------------------------------------------------------------------
    # Metrics (running totals)
    # -----------------------------------------------------------------------
    total_measurements: int = 0       # individual pixel/point count
    safety_violations: int = 0
    llm_tokens_total: int = 0

    # -----------------------------------------------------------------------
    # Configuration snapshot (frozen at run start)
    # -----------------------------------------------------------------------
    config: Dict[str, Any] = field(default_factory=dict)

    # -----------------------------------------------------------------------
    # Factory
    # -----------------------------------------------------------------------

    @classmethod
    def new(
        cls,
        device_id: str,
        target_label: ChargeLabel = ChargeLabel.DOUBLE_DOT,
        voltage_bounds: Optional[Dict] = None,
        config: Optional[Dict] = None,
    ) -> "ExperimentState":
        """Create a fresh state for a new run."""
        state = cls(
            device_id=device_id,
            target_label=target_label,
        )
        if voltage_bounds:
            state.voltage_bounds = voltage_bounds
        if config:
            state.config = config
        state.belief.initialise_uniform()
        state.trajectory.append(state.current_voltage)
        return state

    # -----------------------------------------------------------------------
    # Mutation helpers — use these instead of direct dict access
    # -----------------------------------------------------------------------

    def add_measurement(self, m: Measurement) -> None:
        self.measurements[m.id] = m
        self._update_measurement_count(m)

    def add_dqc_result(self, result: DQCResult) -> None:
        self.dqc_results[result.measurement_id] = result
        self.last_dqc = result

    def add_classification(self, cls: Classification) -> None:
        self.classifications[cls.measurement_id] = cls
        self.last_classification = cls
        # Update BO history with the new observation
        m = self.measurements.get(cls.measurement_id)
        score = cls.confidence if cls.label == self.target_label else 0.0
        self.bo_history.append(BOPoint(
            voltage=self.current_voltage,
            score=score,
            label=cls.label,
            confidence=cls.confidence,
            step=len(self.decisions),
        ))

    def add_ood_result(self, result: OODResult) -> None:
        self.ood_history.append(result)
        self.last_ood = result

    def add_decision(self, d: Decision) -> None:
        self.decisions.append(d)
        self.llm_tokens_total += d.llm_tokens_used

    def add_hitl_event(self, event: HITLEvent) -> None:
        self.hitl_events.append(event)

    def apply_move(self, safe_delta: VoltagePoint) -> None:
        """Apply a safety-approved voltage move and append to trajectory."""
        self.current_voltage = VoltagePoint(
            vg1=self.current_voltage.vg1 + safe_delta.vg1,
            vg2=self.current_voltage.vg2 + safe_delta.vg2,
        )
        self.trajectory.append(self.current_voltage)

    def record_backtrack(self, event: BacktrackEvent) -> None:
        self.backtrack_log.append(event)
        self.consecutive_backtracks += 1
        self.total_backtracks += 1

    def advance_stage(self, new_stage: TuningStage) -> None:
        """Move to a new stage, resetting the consecutive backtrack counter."""
        self.stage = new_stage
        self.consecutive_backtracks = 0

    def record_safety_violation(self) -> None:
        self.safety_violations += 1

    # -----------------------------------------------------------------------
    # Convenience queries
    # -----------------------------------------------------------------------

    @property
    def step(self) -> int:
        return len(self.decisions)

    @property
    def last_confidence(self) -> float:
        if self.last_classification is None:
            return 0.0
        return self.last_classification.confidence

    @property
    def last_label(self) -> ChargeLabel:
        if self.last_classification is None:
            return ChargeLabel.UNKNOWN
        return self.last_classification.label

    @property
    def is_ood(self) -> bool:
        if self.last_ood is None:
            return False
        return self.last_ood.flag

    @property
    def target_achieved(self) -> bool:
        return (
            self.last_label == self.target_label
            and self.last_confidence >= self.config.get("Ct_high", 0.85)
        )

    @property
    def elapsed_s(self) -> float:
        return time.time() - self.started_at

    def current_belief_summary(self) -> str:
        """LLM-readable string summary of current belief state."""
        most_likely = self.belief.most_likely_state()
        entropy = self.belief.entropy()
        last_cls = self.last_classification
        return (
            f"Step {self.step} | Stage: {self.stage.name} | "
            f"Voltage: ({self.current_voltage.vg1:.3f}, {self.current_voltage.vg2:.3f}) | "
            f"Most likely charge state: {most_likely} | "
            f"Belief entropy: {entropy:.2f} | "
            f"Last label: {last_cls.label.value if last_cls else 'none'} @ "
            f"{self.last_confidence:.1%} confidence | "
            f"OOD: {self.is_ood}"
        )

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _update_measurement_count(self, m: Measurement) -> None:
        if m.modality == MeasurementModality.LINE_SCAN:
            self.total_measurements += m.steps or 128
        elif m.is_2d:
            res = m.resolution or 32
            self.total_measurements += res * res
