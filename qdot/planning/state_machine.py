"""
qdot/planning/state_machine.py
================================
Backtracking State Machine — 6-stage autonomous tuning orchestrator.

Stage ordering (blueprint §3.2 + Phase 2.1 addition):
    BOOTSTRAPPING → COARSE_SURVEY → HYPERSURFACE_SEARCH → CHARGE_ID
                 → NAVIGATION → VERIFICATION → COMPLETE

HYPERSURFACE_SEARCH (new): navigate to the charge boundary found by
COARSE_SURVEY before handing off to CHARGE_ID.  Without this stage the
classification step runs at a voltage that may be far from any charge
feature, producing MISC on every attempt.  This matches Schuff et al.
(2025) and Moon et al. (2020) where boundary-walking precedes classification.
"""

from __future__ import annotations

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from qdot.core.types import BacktrackEvent, TuningStage
from qdot.core.state import ExperimentState


# ---------------------------------------------------------------------------
# Stage result
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    success: bool
    confidence: float
    reason: str
    measurements_taken: int = 0
    data: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Stage configurations
# ---------------------------------------------------------------------------

@dataclass
class StageConfig:
    stage: TuningStage
    success_threshold: float
    max_retries: int
    max_backtracks: int
    description: str = ""


STAGE_ORDER: List[TuningStage] = [
    TuningStage.BOOTSTRAPPING,
    TuningStage.COARSE_SURVEY,
    TuningStage.HYPERSURFACE_SEARCH,   # NEW
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
        max_backtracks=0,
        description="Verify device responds to gates and charge sensor is functional",
    ),
    TuningStage.COARSE_SURVEY: StageConfig(
        stage=TuningStage.COARSE_SURVEY,
        success_threshold=0.2,
        max_retries=3,
        max_backtracks=2,
        description="Locate any Coulomb signal in voltage space",
    ),
    TuningStage.HYPERSURFACE_SEARCH: StageConfig(
        stage=TuningStage.HYPERSURFACE_SEARCH,
        success_threshold=0.4,   # boundary_proximity confidence from hypersurface_result()
        max_retries=3,
        max_backtracks=2,
        description=(
            "Navigate to the charge boundary found by COARSE_SURVEY. "
            "Agent walks along conductance gradient until a charge feature "
            "is visible within the scan window."
        ),
    ),
    TuningStage.CHARGE_ID: StageConfig(
        stage=TuningStage.CHARGE_ID,
        success_threshold=0.35,   # was 0.5; ensemble rarely exceeds 0.5 on 3-class boundary cases
        max_retries=2,
        max_backtracks=2,
        description="Classify current charge region via InspectionAgent",
    ),
    TuningStage.NAVIGATION: StageConfig(
        stage=TuningStage.NAVIGATION,
        success_threshold=0.15,   # was 0.7; MAP-based success — belief just needs to concentrate on (1,1)
        max_retries=20,
        max_backtracks=2,
        description="Navigate to target (1,1) charge state via BO",
    ),
    TuningStage.VERIFICATION: StageConfig(
        stage=TuningStage.VERIFICATION,
        success_threshold=0.7,
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
    Six-stage backtracking state machine.

    Operates on ExperimentState — reads and writes stage,
    consecutive_backtracks, and backtrack_log via the state's helpers.
    """

    def __init__(
        self,
        state: ExperimentState,
        configs: Optional[Dict[TuningStage, StageConfig]] = None,
    ):
        self.state = state
        self.configs = configs if configs is not None else DEFAULT_STAGE_CONFIGS
        self._retries: Dict[TuningStage, int] = {s: 0 for s in TuningStage}
        self._backtracks_at_stage: Dict[TuningStage, int] = {s: 0 for s in TuningStage}

    def process_result(self, result: StageResult) -> Tuple[TuningStage, str, bool]:
        stage = self.state.stage
        config = self.configs.get(stage)
        if config is None:
            return stage, f"No config for stage {stage.name}", False

        # SUCCESS CHECK before HITL — advance immediately when good enough.
        if result.success and result.confidence >= config.success_threshold:
            new_stage, rationale = self._advance(stage, result)
            return new_stage, rationale, False

        hitl, hitl_reason = self._check_hitl(stage, config)
        if hitl:
            return stage, hitl_reason, True

        if self._retries[stage] >= config.max_retries:
            if stage == TuningStage.BOOTSTRAPPING or config.max_backtracks == 0:
                return stage, f"Retries exhausted at {stage.name} with no backtrack available", True
            new_stage, rationale = self._backtrack(stage, result)
            hitl, hitl_reason = self._check_hitl(new_stage, self.configs.get(new_stage, config))
            return new_stage, rationale, hitl

        self._retries[stage] += 1
        rationale = (
            f"Stage {stage.name} attempt {self._retries[stage]}/{config.max_retries} failed "
            f"(confidence={result.confidence:.2f} < threshold={config.success_threshold}). "
            f"Reason: {result.reason}"
        )
        return stage, rationale, False

    def _advance(self, stage: TuningStage, result: StageResult) -> Tuple[TuningStage, str]:
        idx = STAGE_ORDER.index(stage)
        new_stage = STAGE_ORDER[min(idx + 1, len(STAGE_ORDER) - 1)]
        self._retries[stage] = 0
        self.state.advance_stage(new_stage)
        rationale = (
            f"Stage {stage.name} succeeded (confidence={result.confidence:.2f} "
            f">= threshold={self.configs[stage].success_threshold}). "
            f"Advancing to {new_stage.name}."
        )
        return new_stage, rationale

    def _backtrack(self, stage: TuningStage, result: StageResult) -> Tuple[TuningStage, str]:
        idx = STAGE_ORDER.index(stage)
        prev_stage = STAGE_ORDER[max(idx - 1, 0)]
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
        self._backtracks_at_stage[stage] += 1
        self._retries[stage] = 0
        self._retries[prev_stage] = 0
        self.state.stage = prev_stage
        rationale = (
            f"Backtracking from {stage.name} to {prev_stage.name} "
            f"after {self.configs[stage].max_retries} retries. "
            f"Reason: {result.reason}. "
            f"Consecutive backtracks: {self.state.consecutive_backtracks}."
        )
        return prev_stage, rationale

    def _check_hitl(self, stage: TuningStage, config: StageConfig) -> Tuple[bool, str]:
        if self.state.consecutive_backtracks >= 2:
            return True, (
                f"Consecutive backtracks >= 2 at stage {stage.name} "
                f"(count={self.state.consecutive_backtracks}). HITL required."
            )
        n_bt = self._backtracks_at_stage.get(stage, 0)
        if n_bt >= config.max_backtracks and config.max_backtracks > 0:
            return True, (
                f"Stage {stage.name} backtrack limit reached "
                f"({n_bt}/{config.max_backtracks}). HITL required."
            )
        stage_count = sum(1 for s in self.state.backtrack_log
                         if s.from_stage == stage or s.to_stage == stage)
        if stage_count > 5:
            return True, f"Loop detected: stage {stage.name} appeared {stage_count} times."
        return False, ""


# ---------------------------------------------------------------------------
# Stage result factory functions
# ---------------------------------------------------------------------------

def bootstrap_result(device_responds: bool, signal_detected: bool) -> StageResult:
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
    return StageResult(
        success=peak_found,
        confidence=float(np.clip(peak_quality, 0.0, 1.0)),
        reason="Coulomb peak found" if peak_found else "No clear Coulomb peak",
        data={"peak_quality": peak_quality},
    )


def hypersurface_result(boundary_found: bool, proximity_confidence: float) -> StageResult:
    """
    Create StageResult for HYPERSURFACE_SEARCH stage.

    Args:
        boundary_found:        True if a charge boundary is visible in the
                               current scan window (SNR check passed).
        proximity_confidence:  Continuous estimate of how close the agent is
                               to the charge boundary, ∈ [0, 1].  Derived from
                               the conductance peak quality at the new voltage.
    """
    return StageResult(
        success=boundary_found,
        confidence=float(np.clip(proximity_confidence, 0.0, 1.0)),
        reason=(
            "Charge boundary located in scan window"
            if boundary_found
            else "Charge boundary not yet visible; continuing gradient walk"
        ),
        data={
            "boundary_found": boundary_found,
            "proximity_confidence": proximity_confidence,
        },
    )


def charge_id_result(
    label: str,
    confidence: float,
    physics_override: bool = False,
) -> StageResult:
    effective = min(0.65, confidence) if physics_override else confidence
    success = label in ("single-dot", "double-dot") and effective > 0.35  # was 0.5
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
    return StageResult(
        success=target_reached,   # was: target_reached and belief_confidence >= 0.7
        confidence=belief_confidence,
        reason="(1,1) state reached" if target_reached else "Target not yet reached",
        data={"target_reached": target_reached, "belief_confidence": belief_confidence},
    )


def verification_result(
    stable: bool, reproducibility: float, charge_noise: float,
) -> StageResult:
    success = stable and reproducibility > 0.8 and charge_noise < 0.1
    confidence = float(reproducibility * (1.0 - charge_noise))
    return StageResult(
        success=success,
        confidence=confidence,
        reason=f"Reproducibility={reproducibility:.2f}, charge_noise={charge_noise:.3f}",
        data={"stable": stable, "reproducibility": reproducibility, "charge_noise": charge_noise},
    )
