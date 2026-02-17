"""
qdot/planning/state_machine.py
================================
Backtracking State Machine — 5-stage autonomous tuning orchestrator.

Five sequential stages from blueprint §3.2:
    BOOTSTRAPPING → COARSE_SURVEY → CHARGE_ID → NAVIGATION → VERIFICATION → COMPLETE

Each stage has:
    - Success threshold to advance
    - Max retries before backtracking
    - Max backtracks before HITL

Key types imported from Phase 0 (NEVER redefined here):
    TuningStage    — enum from qdot.core.types
    BacktrackEvent — dataclass from qdot.core.types

Integration with ExperimentState:
    state.stage                   — updated on advance/backtrack
    state.consecutive_backtracks  — updated on record_backtrack()
    state.backtrack_log           — BacktrackEvent appended via state.record_backtrack()
    state.advance_stage()         — resets consecutive_backtracks

Blueprint reference: §3.2 (Backtracking State Machine), Fig. 3
"""

from __future__ import annotations

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Phase 0 types — imported, never redefined
from qdot.core.types import BacktrackEvent, TuningStage
from qdot.core.state import ExperimentState


# ---------------------------------------------------------------------------
# Stage result (Phase 2 internal — not in types.py)
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    """
    Outcome of one attempt at a tuning stage.
    Internal to the state machine — not shared with governance log directly.
    The Executive Agent translates this into a Decision for governance.
    """
    success: bool
    confidence: float          # ∈ [0, 1]
    reason: str
    measurements_taken: int = 0
    data: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Stage configurations
# ---------------------------------------------------------------------------

@dataclass
class StageConfig:
    """Configuration for a single tuning stage."""
    stage: TuningStage
    success_threshold: float    # Minimum confidence to advance
    max_retries: int            # Max attempts before backtracking
    max_backtracks: int         # Max backtracks at this stage before HITL
    description: str = ""


# Stage ordering (must match TuningStage enum values)
STAGE_ORDER: List[TuningStage] = [
    TuningStage.BOOTSTRAPPING,
    TuningStage.COARSE_SURVEY,
    TuningStage.CHARGE_ID,
    TuningStage.NAVIGATION,
    TuningStage.VERIFICATION,
    TuningStage.COMPLETE,
]

DEFAULT_STAGE_CONFIGS: Dict[TuningStage, StageConfig] = {
    TuningStage.BOOTSTRAPPING: StageConfig(
        stage=TuningStage.BOOTSTRAPPING,
        success_threshold=0.5,
        max_retries=3,
        max_backtracks=0,    # Cannot backtrack from first stage
        description="Verify device responds to gates and charge sensor is functional",
    ),
    TuningStage.COARSE_SURVEY: StageConfig(
        stage=TuningStage.COARSE_SURVEY,
        success_threshold=0.3,
        max_retries=3,
        max_backtracks=2,
        description="Locate any Coulomb peak boundary",
    ),
    TuningStage.CHARGE_ID: StageConfig(
        stage=TuningStage.CHARGE_ID,
        success_threshold=0.7,
        max_retries=2,
        max_backtracks=2,
        description="Classify current charge region via InspectionAgent",
    ),
    TuningStage.NAVIGATION: StageConfig(
        stage=TuningStage.NAVIGATION,
        success_threshold=0.8,
        max_retries=3,
        max_backtracks=2,
        description="Navigate to target (1,1) charge state via BO",
    ),
    TuningStage.VERIFICATION: StageConfig(
        stage=TuningStage.VERIFICATION,
        success_threshold=0.9,
        max_retries=2,
        max_backtracks=1,
        description="Confirm (1,1) is stable across repeated measurements",
    ),
}


# ---------------------------------------------------------------------------
# State Machine
# ---------------------------------------------------------------------------

class StateMachine:
    """
    Five-stage backtracking state machine.

    Operates on ExperimentState — reads and writes stage, consecutive_backtracks,
    and backtrack_log via the state's mutation helpers.

    Decides:
        advance()   — stage succeeded, move forward
        retry()     — stage failed, retry with new candidate
        backtrack() — retries exhausted, go back one stage
        hitl()      — backtracks exhausted, request human help
    """

    def __init__(
        self,
        state: ExperimentState,
        configs: Optional[Dict[TuningStage, StageConfig]] = None,
    ):
        """
        Args:
            state: The central ExperimentState (Phase 0).
            configs: Stage configs (uses defaults if None).
        """
        self.state = state
        self.configs = configs if configs is not None else DEFAULT_STAGE_CONFIGS

        # Per-stage retry counters (reset on advance or backtrack)
        self._retries: Dict[TuningStage, int] = {s: 0 for s in TuningStage}
        # Per-stage backtrack counters (accumulate)
        self._backtracks_at_stage: Dict[TuningStage, int] = {s: 0 for s in TuningStage}

    # ------------------------------------------------------------------
    # Main decision interface
    # ------------------------------------------------------------------

    def process_result(self, result: StageResult) -> Tuple[TuningStage, str, bool]:
        """
        Evaluate a stage result and decide the next action.

        Args:
            result: Outcome of the current stage attempt.

        Returns:
            (new_stage, rationale, hitl_triggered)
        """
        stage = self.state.stage
        config = self.configs.get(stage)

        if config is None:
            return stage, f"No config for stage {stage.name}", False

        # Check HITL triggers first (blueprint §4, conditions 8 and 9)
        hitl, hitl_reason = self._check_hitl(stage, config)
        if hitl:
            return stage, hitl_reason, True

        # Can we advance?
        if result.success and result.confidence >= config.success_threshold:
            new_stage, rationale = self._advance(stage, result)
            return new_stage, rationale, False

        # Should we backtrack?
        if self._retries[stage] >= config.max_retries:
            # Retries exhausted
            if stage == TuningStage.BOOTSTRAPPING or config.max_backtracks == 0:
                # Can't backtrack — trigger HITL
                return stage, f"Retries exhausted at {stage.name} with no backtrack available", True
            new_stage, rationale = self._backtrack(stage, result)
            # Re-check HITL after backtrack
            hitl, hitl_reason = self._check_hitl(new_stage, self.configs.get(new_stage, config))
            return new_stage, rationale, hitl

        # Retry
        self._retries[stage] += 1
        config_max = config.max_retries
        rationale = (
            f"Stage {stage.name} attempt {self._retries[stage]}/{config_max} failed "
            f"(confidence={result.confidence:.2f} < threshold={config.success_threshold}). "
            f"Reason: {result.reason}"
        )
        return stage, rationale, False

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def _advance(self, stage: TuningStage, result: StageResult) -> Tuple[TuningStage, str]:
        """Advance to next stage."""
        idx = STAGE_ORDER.index(stage)
        new_stage = STAGE_ORDER[min(idx + 1, len(STAGE_ORDER) - 1)]

        # Reset retry counter for current stage
        self._retries[stage] = 0

        # Update ExperimentState via its mutation helper
        self.state.advance_stage(new_stage)

        rationale = (
            f"Stage {stage.name} succeeded (confidence={result.confidence:.2f} "
            f">= threshold={self.configs[stage].success_threshold}). "
            f"Advancing to {new_stage.name}."
        )
        return new_stage, rationale

    def _backtrack(self, stage: TuningStage, result: StageResult) -> Tuple[TuningStage, str]:
        """Backtrack to the previous stage."""
        idx = STAGE_ORDER.index(stage)
        prev_stage = STAGE_ORDER[max(idx - 1, 0)]

        # Record backtrack in ExperimentState
        event = BacktrackEvent(
            run_id=self.state.run_id,
            step=self.state.step,
            timestamp=time.time(),
            from_stage=stage,
            to_stage=prev_stage,
            reason=result.reason,
            consecutive_backtracks_at_level=self.state.consecutive_backtracks + 1,
            hitl_triggered=False,
        )
        self.state.record_backtrack(event)

        # Update per-stage counters
        self._backtracks_at_stage[stage] += 1
        self._retries[stage] = 0
        self._retries[prev_stage] = 0

        # Update ExperimentState stage (but DON'T reset consecutive_backtracks —
        # that's done only by advance_stage)
        self.state.stage = prev_stage

        rationale = (
            f"Backtracking from {stage.name} to {prev_stage.name} "
            f"after {self.configs[stage].max_retries} retries. "
            f"Reason: {result.reason}. "
            f"Consecutive backtracks: {self.state.consecutive_backtracks}."
        )
        return prev_stage, rationale

    # ------------------------------------------------------------------
    # HITL trigger checks
    # ------------------------------------------------------------------

    def _check_hitl(self, stage: TuningStage, config: StageConfig) -> Tuple[bool, str]:
        """
        Check conditions 8 (consecutive backtracks ≥ 2) from blueprint §4.
        Conditions 1-7 and 9-12 are checked by HITLManager.compute_risk_score().
        """
        # Condition 8: consecutive backtracks ≥ 2 at same stage
        if self.state.consecutive_backtracks >= 2:
            return True, (
                f"Consecutive backtracks >= 2 at stage {stage.name} "
                f"(count={self.state.consecutive_backtracks}). HITL required."
            )

        # Per-stage backtrack limit
        n_bt = self._backtracks_at_stage.get(stage, 0)
        if n_bt >= config.max_backtracks and config.max_backtracks > 0:
            return True, (
                f"Stage {stage.name} backtrack limit reached "
                f"({n_bt}/{config.max_backtracks}). HITL required."
            )

        # Loop detection: same stage visited more than 5 times
        stage_count = sum(1 for s in self.state.backtrack_log
                         if s.from_stage == stage or s.to_stage == stage)
        if stage_count > 5:
            return True, f"Loop detected: stage {stage.name} appeared {stage_count} times."

        return False, ""


# ---------------------------------------------------------------------------
# Stage result factory functions (called by ExecutiveAgent)
# ---------------------------------------------------------------------------

def bootstrap_result(device_responds: bool, signal_detected: bool) -> StageResult:
    """Create StageResult for BOOTSTRAPPING stage."""
    success = device_responds and signal_detected
    reasons = []
    if not device_responds:
        reasons.append("gates do not modulate current")
    if not signal_detected:
        reasons.append("no charge sensor signal")
    return StageResult(
        success=success,
        confidence=1.0 if success else 0.0,
        reason="Device OK" if success else "; ".join(reasons),
        data={"device_responds": device_responds, "signal_detected": signal_detected},
    )


def survey_result(peak_found: bool, peak_quality: float) -> StageResult:
    """Create StageResult for COARSE_SURVEY stage."""
    return StageResult(
        success=peak_found,
        confidence=float(np.clip(peak_quality, 0.0, 1.0)),
        reason="Coulomb peak found" if peak_found else "No clear Coulomb peak",
        data={"peak_quality": peak_quality},
    )


def charge_id_result(
    label: str,
    confidence: float,
    physics_override: bool = False,
) -> StageResult:
    """Create StageResult for CHARGE_ID stage."""
    # physics_override → reduce effective confidence (blueprint §5.1)
    effective = min(0.65, confidence) if physics_override else confidence
    success = label in ("single-dot", "double-dot") and effective > 0.5
    reason = f"Classified as {label}"
    if physics_override:
        reason += " (physics override: confidence capped at 0.65)"
    return StageResult(
        success=success,
        confidence=effective,
        reason=reason,
        data={"label": label, "raw_confidence": confidence, "physics_override": physics_override},
    )


def navigation_result(target_reached: bool, belief_confidence: float) -> StageResult:
    """Create StageResult for NAVIGATION stage."""
    return StageResult(
        success=target_reached and belief_confidence >= 0.7,
        confidence=belief_confidence,
        reason="(1,1) state reached" if target_reached else "Target not yet reached",
        data={"target_reached": target_reached, "belief_confidence": belief_confidence},
    )


def verification_result(
    stable: bool,
    reproducibility: float,
    charge_noise: float,
) -> StageResult:
    """Create StageResult for VERIFICATION stage."""
    success = stable and reproducibility > 0.8 and charge_noise < 0.1
    confidence = float(reproducibility * (1.0 - charge_noise))
    return StageResult(
        success=success,
        confidence=confidence,
        reason=f"Reproducibility={reproducibility:.2f}, charge_noise={charge_noise:.3f}",
        data={"stable": stable, "reproducibility": reproducibility, "charge_noise": charge_noise},
    )
