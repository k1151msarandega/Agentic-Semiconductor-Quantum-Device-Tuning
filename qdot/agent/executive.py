"""
qdot/agent/executive.py
========================
Executive Agent — main agent loop orchestrator.

Implements Blueprint Fig. 2 (Main Agent Loop):
    1.  ActiveSensingPolicy  → MeasurementPlan
    2.  TranslationAgent     → Measurement (via DeviceAdapter)
    3.  DQCGatekeeper        → DQCResult
    4.  InspectionAgent      → Classification, OODResult  (2D only)
    5.  BeliefUpdater        → updates ExperimentState.belief
    6.  MultiResBO.propose() → ActionProposal
    7.  SafetyCritic.clip()  → clipped ActionProposal
    8.  HITLManager.compute_risk_score() → risk ∈ [0,1]
    9.  HITLManager.queue_request() + await_decision()  (if risk ≥ 0.70)
    10. adapter.set_voltages() + state.apply_move()
    11. GovernanceLogger.log(Decision)
    12. StateMachine.process_result()
    Repeat until COMPLETE or budget exhausted.

LLM call budget (blueprint §5.1):
    - ONE call per stage transition (rationale generation)
    - ONE call per HITL trigger (justification)
    - NOT per measurement, NOT per voltage move
    Phase 2: template-based rationale (no LLM). Phase 3 adds Granite 3-8B.

Key design decisions honoured:
    - Line scans → BeliefUpdater.update_from_1d() (bypass InspectionAgent)
    - 2D patches → DQC → InspectionAgent → BeliefUpdater.update_from_2d()
    - physics_override → BeliefUpdater handles uncertainty inflation
    - HITL is BLOCKING — no auto-approval (blueprint §0, principle #5)
    - DisorderLearner is Phase 3 — OOD flags are logged only
"""

from __future__ import annotations

import time
import uuid
from typing import Optional

# Phase 0 types — ALL imported from canonical locations
from qdot.core.types import (
    ActionProposal,
    ChargeLabel,
    Decision,
    DQCQuality,
    MeasurementModality,
    TuningStage,
    VoltagePoint,
)
from qdot.core.state import ExperimentState
from qdot.core.governance import GovernanceLogger
from qdot.core.hitl import HITLManager

# Phase 0 hardware
from qdot.hardware.adapter import DeviceAdapter
from qdot.hardware.safety import SafetyCritic

# Phase 1 perception
from qdot.perception.dqc import DQCGatekeeper
from qdot.perception.inspector import InspectionAgent

# Phase 2 planning
from qdot.planning.belief import BeliefUpdater, CIMObservationModel
from qdot.planning.sensing import ActiveSensingPolicy
from qdot.planning.bayesian_opt import MultiResBO
from qdot.planning.state_machine import (
    StateMachine, StageResult,
    bootstrap_result, survey_result,
    charge_id_result, navigation_result, verification_result,
)

# Phase 2 agent
from qdot.agent.translator import TranslationAgent


class ExecutiveAgent:
    """
    Main agent loop for autonomous quantum dot tuning.

    One instance per tuning run. Uses ExperimentState as single source of truth.

    Usage:
        agent = ExecutiveAgent(
            state=state,
            adapter=adapter,
            inspection_agent=inspector,
        )
        agent.run()
    """

    def __init__(
        self,
        state: ExperimentState,
        adapter: DeviceAdapter,
        inspection_agent: Optional[InspectionAgent] = None,
        dqc: Optional[DQCGatekeeper] = None,
        safety_critic: Optional[SafetyCritic] = None,
        hitl_manager: Optional[HITLManager] = None,
        governance_logger: Optional[GovernanceLogger] = None,
        max_steps: int = 100,
        measurement_budget: int = 2048,
    ):
        """
        Args:
            state: Central ExperimentState from Phase 0.
            adapter: DeviceAdapter (CIMSimulatorAdapter or hardware).
            inspection_agent: Phase 1 InspectionAgent (required for 2D classification).
            dqc: DQC Gatekeeper (created with defaults if None).
            safety_critic: SafetyCritic (created from state.voltage_bounds if None).
            hitl_manager: HITLManager (created with test_mode off if None).
            governance_logger: GovernanceLogger (created from state.run_id if None).
            max_steps: Hard step limit.
            measurement_budget: Max measurement points (≤2048 for ≥50% reduction target).
        """
        self.state = state
        self.adapter = adapter
        self.inspection_agent = inspection_agent
        self.max_steps = max_steps
        self.measurement_budget = measurement_budget

        # Phase 0 components (create with sensible defaults if not injected)
        self.dqc = dqc or DQCGatekeeper()
        self.safety_critic = safety_critic or SafetyCritic(
            voltage_bounds=state.voltage_bounds,
            l1_max=state.step_caps.get("l1_max", 0.10),
        )
        self.hitl_manager = hitl_manager or HITLManager()
        self.governance_logger = governance_logger or GovernanceLogger(
            run_id=state.run_id,
            log_dir=f"data/governance/{state.run_id}",
        )

        # Phase 2 planning components
        self.belief_updater = BeliefUpdater(
            belief=state.belief,
            obs_model=CIMObservationModel(device_params=state.belief.device_params),
        )
        self.sensing_policy = ActiveSensingPolicy()
        self.bo = MultiResBO(belief=state.belief, voltage_bounds=state.voltage_bounds)
        self.state_machine = StateMachine(state=state)
        self.translator = TranslationAgent(adapter=adapter)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """
        Execute the full autonomous tuning mission.

        Returns:
            Summary dict with success, metrics, final stage, etc.
        """
        self._log_decision(
            intent="mission_start",
            obs={},
            action={"step_budget": self.max_steps, "meas_budget": self.measurement_budget},
            rationale="Mission start — first voltage always triggers HITL (risk=1.0)",
        )

        while not self._should_terminate():
            self._step()

        return self._mission_summary()

    def _step(self) -> None:
        """Execute one iteration of the main agent loop."""
        stage = self.state.stage

        # Execute stage-specific logic
        if stage == TuningStage.BOOTSTRAPPING:
            result = self._run_bootstrap()
        elif stage == TuningStage.COARSE_SURVEY:
            result = self._run_survey()
        elif stage == TuningStage.CHARGE_ID:
            result = self._run_charge_id()
        elif stage == TuningStage.NAVIGATION:
            result = self._run_navigation()
        elif stage == TuningStage.VERIFICATION:
            result = self._run_verification()
        else:
            return  # COMPLETE or FAILED

        # Process result through state machine
        new_stage, rationale, hitl_triggered = self.state_machine.process_result(result)

        # Log stage transition (with one LLM call budget slot — template here)
        if new_stage != stage:
            self._log_decision(
                intent="stage_transition",
                obs={"from_stage": stage.name, "result_confidence": result.confidence},
                action={"to_stage": new_stage.name},
                rationale=rationale,
            )

        # Handle HITL if triggered (BLOCKING — no timeout)
        if hitl_triggered:
            self._handle_hitl(rationale)

    # ------------------------------------------------------------------
    # Stage executors (called by _step)
    # ------------------------------------------------------------------

    def _run_bootstrap(self) -> StageResult:
        """
        BOOTSTRAPPING: verify device responds and charge sensor is functional.
        """
        # Take a quick line scan to check for electrical response
        from qdot.core.types import MeasurementPlan, MeasurementModality
        plan = MeasurementPlan(
            modality=MeasurementModality.LINE_SCAN,
            axis="vg1",
            start=self.state.voltage_bounds["vg1"]["min"] * 0.5,
            stop=self.state.voltage_bounds["vg1"]["max"] * 0.5,
            steps=32,
            rationale="Bootstrap: electrical integrity check",
        )
        tr = self.translator.execute(plan)
        if tr.measurement is None:
            return bootstrap_result(device_responds=False, signal_detected=False)

        m = tr.measurement
        self.state.add_measurement(m)

        arr = m.array
        import numpy as np
        signal_detected = float(arr.max() - arr.min()) > 0.1
        device_responds = float(arr.var()) > 1e-6
        return bootstrap_result(device_responds, signal_detected)

    def _run_survey(self) -> StageResult:
        """
        COARSE_SURVEY: locate any Coulomb peak with a coarse 2D scan.
        """
        v1_range = (
            self.state.voltage_bounds["vg1"]["min"],
            self.state.voltage_bounds["vg1"]["max"],
        )
        v2_range = (
            self.state.voltage_bounds["vg2"]["min"],
            self.state.voltage_bounds["vg2"]["max"],
        )

        plan = self.sensing_policy.select(self.state.belief, v1_range, v2_range)
        tr = self.translator.execute(plan)

        if tr.measurement is None:
            return survey_result(peak_found=False, peak_quality=0.0)

        m = tr.measurement
        self.state.add_measurement(m)

        # DQC check
        dqc = self.dqc.assess(m)
        self.state.add_dqc_result(dqc)

        if dqc.quality == DQCQuality.LOW:
            # DQC LOW → stop, don't pass to InspectionAgent
            return survey_result(peak_found=False, peak_quality=0.0)

        import numpy as np
        arr = m.array if m.array is not None else []
        arr = np.asarray(arr)
        peak_quality = float((arr.max() - arr.mean()) / (arr.max() + 1e-8))

        # Update belief with this measurement
        if m.is_2d:
            self.belief_updater.update_from_2d(m)

        return survey_result(peak_found=peak_quality > 0.2, peak_quality=peak_quality)

    def _run_charge_id(self) -> StageResult:
        """
        CHARGE_ID: classify current region using InspectionAgent.
        Requires inspection_agent to be injected.
        """
        v1_range = (
            self.state.current_voltage.vg1 - 0.1,
            self.state.current_voltage.vg1 + 0.1,
        )
        v2_range = (
            self.state.current_voltage.vg2 - 0.1,
            self.state.current_voltage.vg2 + 0.1,
        )

        plan = self.sensing_policy.select(self.state.belief, v1_range, v2_range)
        tr = self.translator.execute(plan)

        if tr.measurement is None:
            return charge_id_result("unknown", 0.0)

        m = tr.measurement
        self.state.add_measurement(m)
        dqc = self.dqc.assess(m)
        self.state.add_dqc_result(dqc)

        if dqc.quality == DQCQuality.LOW:
            return charge_id_result("unknown", 0.0)

        if not m.is_2d or self.inspection_agent is None:
            return charge_id_result("unknown", 0.3)

        # InspectionAgent: only for 2D measurements (blueprint §5.3)
        classification, ood_result = self.inspection_agent.inspect(m, dqc)
        self.state.add_classification(classification)
        self.state.add_ood_result(ood_result)

        # Update belief (handles physics_override internally)
        self.belief_updater.update_from_2d(m, classification)

        return charge_id_result(
            label=classification.label.value,
            confidence=classification.confidence,
            physics_override=classification.physics_override,
        )

    def _run_navigation(self) -> StageResult:
        """
        NAVIGATION: use BO to move toward target (1,1) charge state.
        """
        # Update BO with latest history
        self.bo.update(self.state.bo_history)

        # Propose move
        proposal = self.bo.propose(
            current=self.state.current_voltage,
            l1_max=self.state.step_caps.get("l1_max", 0.10),
        )

        # Safety check
        proposal = self.safety_critic.clip(proposal, self.state.current_voltage)
        safety_verdict = self.safety_critic.verify(self.state.current_voltage, proposal)

        if not safety_verdict.all_passed:
            self.state.record_safety_violation()
            return navigation_result(target_reached=False, belief_confidence=0.0)

        # Risk score (HITLManager handles conditions 1-7, 9-12)
        dqc_flag = self.state.last_dqc.quality.value if self.state.last_dqc else "high"
        ood_score = self.state.last_ood.score if self.state.last_ood else 0.0
        disagreement = (
            self.state.last_classification.ensemble_disagreement
            if self.state.last_classification else 0.0
        )
        risk = self.hitl_manager.compute_risk_score(
            proposal=proposal,
            safety_verdict=safety_verdict,
            dqc_flag=dqc_flag,
            ood_score=ood_score,
            ensemble_disagreement=disagreement,
            consecutive_backtracks=self.state.consecutive_backtracks,
            step=self.state.step + 1,
        )

        # HITL gate (BLOCKING — no timeout)
        if risk >= HITLManager.HITL_THRESHOLD:
            event = self.hitl_manager.queue_request(
                run_id=self.state.run_id,
                step=self.state.step,
                stage=self.state.stage,
                trigger_reason=f"Risk score={risk:.2f} ≥ threshold=0.70",
                risk_score=risk,
                proposal=proposal,
                safety_verdict=safety_verdict,
            )
            event = self.hitl_manager.await_decision(event)
            self.state.add_hitl_event(event)

            from qdot.core.types import HITLOutcome
            if event.outcome == HITLOutcome.REJECTED:
                return navigation_result(target_reached=False, belief_confidence=0.0)
            if event.outcome == HITLOutcome.MODIFIED and event.modified_delta_v:
                safe_delta = event.modified_delta_v
                proposal = ActionProposal(
                    delta_v=proposal.delta_v,
                    safe_delta_v=safe_delta,
                    expected_new_voltage=VoltagePoint(
                        vg1=self.state.current_voltage.vg1 + safe_delta.vg1,
                        vg2=self.state.current_voltage.vg2 + safe_delta.vg2,
                    ),
                    info_gain=proposal.info_gain,
                )

        # Apply the move
        safe_dv = proposal.safe_delta_v or proposal.delta_v
        self.translator.execute_voltage_move(
            vg1=self.state.current_voltage.vg1 + safe_dv.vg1,
            vg2=self.state.current_voltage.vg2 + safe_dv.vg2,
        )
        self.state.apply_move(safe_dv)

        # Check if target reached
        most_likely = self.state.belief.most_likely_state()
        target_reached = (most_likely == (1, 1))
        belief_confidence = (
            self.state.belief.charge_probs.get((1, 1), 0.0)
        )

        self._log_decision(
            intent="voltage_move",
            obs={
                "risk_score": risk,
                "dqc_flag": dqc_flag,
                "belief_mode": str(most_likely),
                "belief_confidence": belief_confidence,
            },
            action={
                "delta_vg1": safe_dv.vg1,
                "delta_vg2": safe_dv.vg2,
                "clipped": proposal.clipped,
            },
            rationale=f"BO proposal: info_gain={proposal.info_gain:.4f}",
        )

        return navigation_result(target_reached, belief_confidence)

    def _run_verification(self) -> StageResult:
        """
        VERIFICATION: confirm (1,1) is stable over repeated measurements.
        """
        import numpy as np
        confirmations = 0
        n_checks = 3

        for _ in range(n_checks):
            plan = self.sensing_policy.select(
                self.state.belief,
                v1_range=(self.state.current_voltage.vg1 - 0.05, self.state.current_voltage.vg1 + 0.05),
                v2_range=(self.state.current_voltage.vg2 - 0.05, self.state.current_voltage.vg2 + 0.05),
            )
            tr = self.translator.execute(plan)
            if tr.measurement is None:
                continue

            m = tr.measurement
            self.state.add_measurement(m)
            dqc = self.dqc.assess(m)
            self.state.add_dqc_result(dqc)

            if dqc.quality == DQCQuality.LOW:
                continue

            if m.is_2d and self.inspection_agent:
                classification, ood_result = self.inspection_agent.inspect(m, dqc)
                self.state.add_classification(classification)
                self.state.add_ood_result(ood_result)
                self.belief_updater.update_from_2d(m, classification)

                if classification.label == ChargeLabel.DOUBLE_DOT:
                    confirmations += 1

        reproducibility = confirmations / n_checks
        charge_noise = 1.0 - reproducibility  # simplified estimate
        return verification_result(
            stable=(confirmations >= 2),
            reproducibility=reproducibility,
            charge_noise=charge_noise,
        )

    # ------------------------------------------------------------------
    # HITL and governance
    # ------------------------------------------------------------------

    def _handle_hitl(self, reason: str) -> None:
        """
        Queue a HITL event and BLOCK until resolved.

        Blueprint §0 principle #5: no auto-approval on timeout.
        """
        dummy_proposal = ActionProposal(
            delta_v=VoltagePoint(vg1=0.0, vg2=0.0),
        )
        dummy_verdict = self.safety_critic.verify(
            self.state.current_voltage, dummy_proposal
        )
        risk = self.hitl_manager.compute_risk_score(
            proposal=dummy_proposal,
            safety_verdict=dummy_verdict,
            consecutive_backtracks=self.state.consecutive_backtracks,
            step=self.state.step + 1,
        )
        event = self.hitl_manager.queue_request(
            run_id=self.state.run_id,
            step=self.state.step,
            stage=self.state.stage,
            trigger_reason=reason,
            risk_score=max(risk, 0.70),
            proposal=dummy_proposal,
            safety_verdict=dummy_verdict,
        )
        event = self.hitl_manager.await_decision(event)  # BLOCKS
        self.state.add_hitl_event(event)

        self._log_decision(
            intent="hitl_trigger",
            obs={"reason": reason, "consecutive_backtracks": self.state.consecutive_backtracks},
            action={"outcome": event.outcome.value},
            rationale=reason,
        )

    def _log_decision(
        self,
        intent: str,
        obs: dict,
        action: dict,
        rationale: str,
    ) -> None:
        """Log a decision to both ExperimentState and GovernanceLogger."""
        d = Decision(
            run_id=self.state.run_id,
            step=self.state.step,
            timestamp=time.time(),
            intent=intent,
            stage=self.state.stage,
            observation_summary=obs,
            action_summary=action,
            rationale=rationale,
            llm_tokens_used=0,  # Phase 2: no LLM calls
        )
        self.state.add_decision(d)
        self.governance_logger.log(d)

    # ------------------------------------------------------------------
    # Termination and summary
    # ------------------------------------------------------------------

    def _should_terminate(self) -> bool:
        return (
            self.state.step >= self.max_steps
            or self.state.total_measurements >= self.measurement_budget
            or self.state.stage in (TuningStage.COMPLETE, TuningStage.FAILED)
        )

    def _mission_summary(self) -> dict:
        dense_baseline = 64 * 64  # 4096 points
        reduction = 1.0 - (self.state.total_measurements / dense_baseline)
        return {
            "success": self.state.stage == TuningStage.COMPLETE,
            "final_stage": self.state.stage.name,
            "total_steps": self.state.step,
            "total_measurements": self.state.total_measurements,
            "measurement_reduction": reduction,
            "total_backtracks": self.state.total_backtracks,
            "safety_violations": self.state.safety_violations,
            "hitl_events": len(self.state.hitl_events),
            "run_id": self.state.run_id,
        }
