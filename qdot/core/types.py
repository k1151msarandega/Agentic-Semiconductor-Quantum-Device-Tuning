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
    """Data quality classification from the DQC Gatekeeper."""
    HIGH = "high"         # SNR > 20 dB, plausible, dynamic range > 0.3
    MODERATE = "moderate" # SNR 10–20 dB or borderline plausibility
    LOW = "low"           # SNR < 10 dB or physically implausible → STOP


class ChargeLabel(Enum):
    """Charge stability diagram classification labels."""
    SINGLE_DOT = "single-dot"
    DOUBLE_DOT = "double-dot"
    MISC = "misc"
    UNKNOWN = "unknown"


class TuningStage(Enum):
    """Stages of the backtracking state machine (Section 3.2 of blueprint)."""
    BOOTSTRAPPING = 0      # Device response confirmed
    COARSE_SURVEY = 1      # Locate any Coulomb peak
    CHARGE_ID = 2          # Classify current charge region
    NAVIGATION = 3         # Move toward (1,1) state
    VERIFICATION = 4       # Confirm (1,1) is stable
    COMPLETE = 5           # Mission achieved
    FAILED = -1            # Unrecoverable failure


class MeasurementModality(Enum):
    """Measurement type selected by the Active Sensing Policy."""
    LINE_SCAN = "line_scan"
    COARSE_2D = "coarse_2d"
    LOCAL_PATCH = "local_patch"
    FINE_2D = "fine_2d"
    NONE = "none"            # Skip: belief already peaked


class HITLOutcome(Enum):
    """Possible outcomes of a HITL approval request."""
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"
    PENDING = "pending"


class ActionType(Enum):
    """High-level action types the Executive Agent can take."""
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
    """A point in gate-voltage space. Immutable."""
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
    """
    A raw conductance measurement returned by the Device Adapter.
    The array is always normalised to [0, 1] by the adapter before this point.
    """
    id: UUID = field(default_factory=uuid.uuid4)
    array: Any = None                 # np.ndarray — avoid numpy import at type level
    modality: MeasurementModality = MeasurementModality.COARSE_2D
    voltage_centre: Optional[VoltagePoint] = None
    v1_range: Optional[Tuple[float, float]] = None
    v2_range: Optional[Tuple[float, float]] = None
    axis: Optional[str] = None        # for line scans: "vg1" or "vg2"
    resolution: Optional[int] = None  # for 2D: number of pixels per side
    steps: Optional[int] = None       # for line scans: number of points
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
    """
    Output of the DQC Gatekeeper.
    Attached to every Measurement before it reaches the Inspection Agent.
    """
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
    """Output of the Inspection Agent for a single 2D measurement."""
    measurement_id: UUID
    label: ChargeLabel
    confidence: float                 # ∈ [0, 1], from primary classifier
    ensemble_disagreement: float = 0.0  # max pairwise disagreement across ensemble ∈ [0, 1]
    features: Dict[str, float] = field(default_factory=dict)
    physics_override: bool = False    # True if heuristic validator overrode CNN
    nl_summary: str = ""              # LLM-generated summary for Executive Agent


@dataclass
class OODResult:
    """Out-of-distribution detection result."""
    measurement_id: UUID
    score: float
    threshold: float
    flag: bool                        # True → trigger DisorderLearner

    @property
    def margin(self) -> float:
        """Positive = in-distribution, negative = OOD."""
        return self.threshold - self.score


# ---------------------------------------------------------------------------
# Planning & optimisation
# ---------------------------------------------------------------------------

@dataclass
class MeasurementPlan:
    """
    Output of the Active Sensing Policy.
    Instructs the Translation Agent which measurement to take next.
    """
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
    """A single observation in the Bayesian Optimisation history."""
    voltage: VoltagePoint
    score: float                  # ∈ [0, 1], 1 = target achieved
    label: ChargeLabel = ChargeLabel.UNKNOWN
    confidence: float = 0.0
    step: int = 0


@dataclass
class ActionProposal:
    """
    A proposed voltage move from the Multi-fidelity BO.
    Goes through the Safety Critic before any action is taken.
    """
    delta_v: VoltagePoint             # Proposed ΔV (before safety clipping)
    safe_delta_v: Optional[VoltagePoint] = None  # After Safety Critic clips
    expected_new_voltage: Optional[VoltagePoint] = None
    info_gain: float = 0.0
    clipped: bool = False
    clip_warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------

@dataclass
class SafetyCheckResult:
    """Result of a single safety check from the Safety Critic."""
    check_name: str                   # "voltage_bounds" | "slew_rate" | "voltage_margin"
    passed: bool
    margin: float                     # Positive = safe, negative = violation
    per_gate: Dict[str, float] = field(default_factory=dict)
    notes: str = ""


@dataclass
class SafetyVerdict:
    """
    Aggregated safety verdict from the Safety Critic.
    All checks must pass before a move is applied.
    """
    voltage_bounds: SafetyCheckResult
    slew_rate: SafetyCheckResult
    voltage_margin: SafetyCheckResult  # Feeds into Risk Score
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
    """A single HITL approval request and its outcome."""
    id: UUID = field(default_factory=uuid.uuid4)
    run_id: str = ""
    step: int = 0
    trigger_reason: str = ""          # Human-readable trigger description
    risk_score: float = 0.0
    proposal: Optional[ActionProposal] = None
    safety_verdict: Optional[SafetyVerdict] = None
    outcome: HITLOutcome = HITLOutcome.PENDING
    modified_delta_v: Optional[VoltagePoint] = None  # If human modified the proposal
    queued_at: float = 0.0
    decided_at: Optional[float] = None
    deciding_human: str = ""


# ---------------------------------------------------------------------------
# Governance / audit trail
# ---------------------------------------------------------------------------

@dataclass
class Decision:
    """
    A single entry in the governance log. Immutable after creation.
    Every agent action, observation, and plan transition is logged as a Decision.
    """
    id: UUID = field(default_factory=uuid.uuid4)
    run_id: str = ""
    step: int = 0
    timestamp: float = 0.0
    intent: str = ""                  # e.g. "observe", "plan_move", "start", "backtrack"
    stage: TuningStage = TuningStage.BOOTSTRAPPING

    # What did the agent see?
    observation_summary: Dict[str, Any] = field(default_factory=dict)
    # What did the agent decide?
    action_summary: Dict[str, Any] = field(default_factory=dict)
    # Why?
    rationale: str = ""
    # LLM metadata
    llm_tokens_used: int = 0
    llm_call_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Backtracking
# ---------------------------------------------------------------------------

@dataclass
class BacktrackEvent:
    """Logged whenever the state machine backtracks to a previous stage."""
    id: UUID = field(default_factory=uuid.uuid4)
    run_id: str = ""
    step: int = 0
    timestamp: float = 0.0
    from_stage: TuningStage = TuningStage.COARSE_SURVEY
    to_stage: TuningStage = TuningStage.BOOTSTRAPPING
    reason: str = ""
    consecutive_backtracks_at_level: int = 0
    hitl_triggered: bool = False
