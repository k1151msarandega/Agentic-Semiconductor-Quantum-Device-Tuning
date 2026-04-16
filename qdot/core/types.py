"""
qdot/core/types.py
==================
Central data types for the QDot Agentic Tuning system.

This file defines every data contract between modules.
Rule: if a module produces it or consumes it, the type lives here.
No module should define its own ad-hoc dicts for inter-module communication.

Build order: this file is Phase 0 day one — everything else imports from here.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any
from uuid import UUID


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class DQCQuality(Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"


class ChargeLabel(Enum):
    SINGLE_DOT = "single-dot"
    DOUBLE_DOT = "double-dot"
    MISC = "misc"
    UNKNOWN = "unknown"


class TuningStage(Enum):
    """
    Stages of the backtracking state machine (blueprint §3.2).

    HYPERSURFACE_SEARCH (added Phase 2.1) sits between COARSE_SURVEY and
    CHARGE_ID.  Its job is to navigate from "found a signal region" to
    "parked near the first charge boundary", so CHARGE_ID always operates
    in the correct voltage neighbourhood.  This matches the Ares-group
    methodology (Moon et al. 2020, Schuff et al. 2025) where an explicit
    pinch-off-boundary walk precedes any classification step.

    Stage ordering:
        BOOTSTRAPPING       → verify device responds
        COARSE_SURVEY       → locate any Coulomb signal in voltage space
        HYPERSURFACE_SEARCH → navigate to the charge boundary (NEW)
        CHARGE_ID           → classify the charge regime
        NAVIGATION          → move toward (1,1) state
        VERIFICATION        → confirm (1,1) is stable
        COMPLETE            → mission achieved
        FAILED              → unrecoverable
    """
    BOOTSTRAPPING       = 0
    COARSE_SURVEY       = 1
    HYPERSURFACE_SEARCH = 2   # NEW — navigate to charge boundary before classification
    CHARGE_ID           = 3   # was 2
    NAVIGATION          = 4   # was 3
    VERIFICATION        = 5   # was 4
    COMPLETE            = 6   # was 5
    FAILED              = -1


class MeasurementModality(Enum):
    LINE_SCAN = "line_scan"
    COARSE_2D = "coarse_2d"
    LOCAL_PATCH = "local_patch"
    FINE_2D = "fine_2d"
    NONE = "none"


class HITLOutcome(Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"
    PENDING = "pending"


class ActionType(Enum):
    MEASURE = "measure"
    MOVE = "move"
    SKIP = "skip"
    BACKTRACK = "backtrack"
    REQUEST_HITL = "request_hitl"


# ---------------------------------------------------------------------------
# Primitive value objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VoltagePoint:
    vg1: float
    vg2: float

    def as_dict(self) -> Dict[str, float]:
        return {"vg1": self.vg1, "vg2": self.vg2}

    def delta_to(self, other: "VoltagePoint") -> "VoltagePoint":
        return VoltagePoint(vg1=other.vg1 - self.vg1, vg2=other.vg2 - self.vg2)

    @property
    def l1_norm(self) -> float:
        return abs(self.vg1) + abs(self.vg2)

    @staticmethod
    def from_dict(d: Dict[str, float]) -> "VoltagePoint":
        return VoltagePoint(vg1=d["vg1"], vg2=d["vg2"])


# ---------------------------------------------------------------------------
# Measurement types
# ---------------------------------------------------------------------------

@dataclass
class Measurement:
    id: UUID = field(default_factory=uuid.uuid4)
    array: Any = None
    modality: MeasurementModality = MeasurementModality.COARSE_2D
    voltage_centre: Optional[VoltagePoint] = None
    v1_range: Optional[Tuple[float, float]] = None
    v2_range: Optional[Tuple[float, float]] = None
    axis: Optional[str] = None
    resolution: Optional[int] = None
    steps: Optional[int] = None
    device_id: str = ""
    timestamp: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_2d(self) -> bool:
        return self.modality in (
            MeasurementModality.COARSE_2D,
            MeasurementModality.LOCAL_PATCH,
            MeasurementModality.FINE_2D,
        )


@dataclass
class DQCResult:
    measurement_id: UUID
    quality: DQCQuality
    snr_db: float
    dynamic_range: float
    flatness_score: float
    physically_plausible: bool
    notes: str = ""


# ---------------------------------------------------------------------------
# Classification & OOD
# ---------------------------------------------------------------------------

@dataclass
class Classification:
    measurement_id: UUID
    label: ChargeLabel
    confidence: float
    ensemble_disagreement: float = 0.0
    features: Dict[str, float] = field(default_factory=dict)
    physics_override: bool = False
    nl_summary: str = ""


@dataclass
class OODResult:
    measurement_id: UUID
    score: float
    threshold: float
    flag: bool

    @property
    def margin(self) -> float:
        return self.threshold - self.score


# ---------------------------------------------------------------------------
# Planning & optimisation
# ---------------------------------------------------------------------------

@dataclass
class MeasurementPlan:
    modality: MeasurementModality
    v1_range: Optional[Tuple[float, float]] = None
    v2_range: Optional[Tuple[float, float]] = None
    axis: Optional[str] = None
    start: Optional[float] = None
    stop: Optional[float] = None
    steps: int = 128
    resolution: int = 32
    rationale: str = ""
    info_gain_per_cost: float = 0.0


@dataclass
class BOPoint:
    voltage: VoltagePoint
    score: float
    label: ChargeLabel = ChargeLabel.UNKNOWN
    confidence: float = 0.0
    step: int = 0


@dataclass
class ActionProposal:
    delta_v: VoltagePoint
    safe_delta_v: Optional[VoltagePoint] = None
    expected_new_voltage: Optional[VoltagePoint] = None
    info_gain: float = 0.0
    clipped: bool = False
    clip_warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------

@dataclass
class SafetyCheckResult:
    check_name: str
    passed: bool
    margin: float
    per_gate: Dict[str, float] = field(default_factory=dict)
    notes: str = ""


@dataclass
class SafetyVerdict:
    voltage_bounds: SafetyCheckResult
    slew_rate: SafetyCheckResult
    voltage_margin: SafetyCheckResult
    all_passed: bool = False

    def __post_init__(self) -> None:
        self.all_passed = (
            self.voltage_bounds.passed
            and self.slew_rate.passed
            and self.voltage_margin.passed
        )

    @property
    def min_margin(self) -> float:
        return min(
            self.voltage_bounds.margin,
            self.slew_rate.margin,
            self.voltage_margin.margin,
        )


# ---------------------------------------------------------------------------
# HITL
# ---------------------------------------------------------------------------

@dataclass
class HITLEvent:
    id: UUID = field(default_factory=uuid.uuid4)
    run_id: str = ""
    step: int = 0
    trigger_reason: str = ""
    risk_score: float = 0.0
    proposal: Optional[ActionProposal] = None
    safety_verdict: Optional[SafetyVerdict] = None
    outcome: HITLOutcome = HITLOutcome.PENDING
    modified_delta_v: Optional[VoltagePoint] = None
    queued_at: float = 0.0
    decided_at: Optional[float] = None
    deciding_human: str = ""


# ---------------------------------------------------------------------------
# Governance / audit trail
# ---------------------------------------------------------------------------

@dataclass
class Decision:
    id: UUID = field(default_factory=uuid.uuid4)
    run_id: str = ""
    step: int = 0
    timestamp: float = 0.0
    intent: str = ""
    stage: TuningStage = TuningStage.BOOTSTRAPPING
    observation_summary: Dict[str, Any] = field(default_factory=dict)
    action_summary: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    llm_tokens_used: int = 0
    llm_call_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Backtracking
# ---------------------------------------------------------------------------

@dataclass
class BacktrackEvent:
    id: UUID = field(default_factory=uuid.uuid4)
    run_id: str = ""
    step: int = 0
    timestamp: float = 0.0
    from_stage: TuningStage = TuningStage.COARSE_SURVEY
    to_stage: TuningStage = TuningStage.BOOTSTRAPPING
    reason: str = ""
    consecutive_backtracks_at_level: int = 0
    hitl_triggered: bool = False
