"""
qdot/agent/executive.py
========================
Executive Agent — main agent loop orchestrator.
(Same as original — only _measurement_fits() helper added and budget guards
inserted before each measurement acquisition.)
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from qdot.core.types import (
    ActionProposal,
    ChargeLabel,
    Decision,
    DQCQuality,
    HITLOutcome,
    MeasurementModality,
    MeasurementPlan,
    TuningStage,
    VoltagePoint,
)
from qdot.core.state import ExperimentState
from qdot.core.governance import GovernanceLogger
from qdot.core.hitl import HITLManager

from qdot.hardware.adapter import DeviceAdapter
from qdot.hardware.safety import SafetyCritic

from qdot.perception.dqc import DQCGatekeeper
from qdot.perception.inspector import InspectionAgent

from qdot.planning.belief import BeliefUpdater, CIMObservationModel
from qdot.planning.sensing import ActiveSensingPolicy
from qdot.planning.bayesian_opt import MultiResBO
from qdot.planning.state_machine import (
    StateMachine, StageResult,
    bootstrap_result, survey_result,
    charge_id_result, navigation_result, verification_result,
)

from qdot.agent.translator import TranslationAgent


class ExecutiveAgent:
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
        self.state = state
        self.adapter = adapter
        self.inspection_agent = inspection_agent
        self.max_steps = max_steps
        self.measurement_budget = measurement_budget

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

        self.belief_updater = BeliefUpdater(
            belief=state.belief,
            obs_model=CIMObservationModel(device_params=state.belief.device_params),
        )
        self.sensing_policy = ActiveSensingPolicy()
        self.bo = MultiResBO(belief=state.belief, voltage_bounds=state.voltage_bounds)
        self.state_machine = StateMachine(state=state)
        self.translator = TranslationAgent(adapter=adapter)

    # ------------------------------------------------------------------
    # Budget guard
    # ------------------------------------------------------------------

    def _measurement_fits(self, plan: MeasurementPlan) -> bool:
        """
        Return True if executing this plan would keep total_measurements
        within budget.

        The termination check in _should_terminate() runs between steps,
        not inside them.  A single step can call the adapter multiple times
        (e.g. _run_verification loops 3×) or request a high-resolution scan
        (FINE_2D = 4096 points).  Without this guard, any step that starts
        with total_measurements < budget but takes a large scan will overshoot.

        Cost is conservative: we use the modality cost table rather than the
        actual points taken, which is always equal or smaller.
        """
        from qdot.planning.sensing import MODALITY_COST
        cost = MODALITY_COST.get(plan.modality, 0)
        return (self.state.total_measurements + cost) <= self.measurement_budget

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> dict:
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
        stage = self.state.stage

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
            return

        new_stage, rationale, hitl_triggered = self.state_machine.process_result(result)

        if new_stage != stage:
            self._log_decision(
                intent="stage_transition",
                obs={"from_stage": stage.name, "result_confidence": result.confidence},
                action={"to_stage": new_stage.name},
                rationale=rationale,
            )

        if hitl_triggered:
            self._handle_hitl(rationale)

    # ------------------------------------------------------------------
    # Stage executors
    # ------------------------------------------------------------------

    def _run_bootstrap(self) -> StageResult:
        plan = MeasurementPlan(
            modality=MeasurementModality.LINE_SCAN,
            axis="vg1",
            start=self.state.voltage_bounds["vg1"]["min"] * 0.5,
            stop=self.state.voltage_bounds["vg1"]["max"] * 0.5,
            steps=32,
            rationale="Bootstrap: electrical integrity check",
        )
        # Budget guard: bootstrap scan is 32 points — refuse if it would overshoot
        if not self._measurement_fits(plan):
            return bootstrap_result(device_responds=False, signal_detected=False)

        tr = self.translator.execute(plan)
        if tr.measurement is None:
            return bootstrap_result(device_responds=False, signal_detected=False)

        m = tr.measurement
        self.state.add_measurement(m)

        arr = m.array
        signal_detected = float(arr.max() - arr.min()) > 0.1
        device_responds = float(arr.var()) > 1e-6
        return bootstrap_result(device_responds, signal_detected)

    def _run_survey(self) -> StageResult:
        v1_range = (
            self.state.voltage_bounds["vg1"]["min"],
            self.state.voltage_bounds["vg1"]["max"],
        )
        v2_range = (
            self.state.voltage_bounds["vg2"]["min"],
            self.state.voltage_bounds["vg2"]["max"],
        )

        plan = self.sensing_policy.select(self.state.belief, v1_range, v2_range)

        # Budget guard: check before executing — a FINE_2D scan is 4096 points
        if not self._measurement_fits(plan):
            return survey_result(peak_found=False, peak_quality=0.0)

        tr = self.translator.execute(plan)

        if tr.measurement is None:
            return survey_result(peak_found=False, peak_quality=0.0)

        m = tr.measurement
        self.state.add_measurement(m)

        dqc = self.dqc.assess(m)
        self.state.add_dqc_result(dqc)

        if dqc.quality == DQCQuality.LOW:
            return survey_result(peak_found=False, peak_quality=0.0)

        arr = m.array if m.array is not None else []
        arr = np.asarray(arr)
        peak_quality = float((arr.max() - arr.mean()) / (arr.max() + 1e-8))

        if m.is_2d:
            self.belief_updater.update_from_2d(m)

        return survey_result(peak_found=peak_quality > 0.2, peak_quality=peak_quality)

    def _run_charge_id(self) -> StageResult:
        v1_range = (
            self.state.current_voltage.vg1 - 0.1,
            self.state.current_voltage.vg1 + 0.1,
        )
        v2_range = (
            self.state.current_voltage.vg2 - 0.1,
            self.state.current_voltage.vg2 + 0.1,
        )

        plan = self.sensing_policy.select(self.state.belief, v1_range, v2_range)

        # Budget guard
        if not self._measurement_fits(plan):
            return charge_id_result("unknown", 0.0)

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

        classification, ood_result = self.inspection_agent.inspect(m, dqc)
        self.state.add_classification(classification)
        self.state.add_ood_result(ood_result)

        self.belief_updater.update_from_2d(m, classification)

        return charge_id_result(
            label=classification.label.value,
            confidence=classification.confidence,
            physics_override=classification.physics_override,
        )

    def _run_navigation(self) -> StageResult:
        self.bo.update(self.state.bo_history)

        proposal = self.bo.propose(
            current=self.state.current_voltage,
            l1_max=self.state.step_caps.get("l1_max", 0.10),
        )

        proposal = self.safety_critic.clip(proposal, self.state.current_voltage)
        safety_verdict = self.safety_critic.verify(self.state.current_voltage, proposal)

        if not safety_verdict.all_passed:
            self.state.record_safety_violation()
            return navigation_result(target_reached=False, belief_confidence=0.0)

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

        safe_dv = proposal.safe_delta_v or proposal.delta_v
        self.translator.execute_voltage_move(
            vg1=self.state.current_voltage.vg1 + safe_dv.vg1,
            vg2=self.state.current_voltage.vg2 + safe_dv.vg2,
        )
        self.state.apply_move(safe_dv)

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
        confirmations = 0
        n_checks = 3

        for _ in range(n_checks):
            plan = self.sensing_policy.select(
                self.state.belief,
                v1_range=(self.state.current_voltage.vg1 - 0.05, self.state.current_voltage.vg1 + 0.05),
                v2_range=(self.state.current_voltage.vg2 - 0.05, self.state.current_voltage.vg2 + 0.05),
            )

            # Budget guard inside the loop — each iteration can take a new scan
            if not self._measurement_fits(plan):
                break

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
        charge_noise = 1.0 - reproducibility
        return verification_result(
            stable=(confirmations >= 2),
            reproducibility=reproducibility,
            charge_noise=charge_noise,
        )

    # ------------------------------------------------------------------
    # HITL and governance
    # ------------------------------------------------------------------

    def _handle_hitl(self, reason: str) -> None:
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
        event = self.hitl_manager.await_decision(event)
        self.state.add_hitl_event(event)

        self._log_decision(
            intent="hitl_trigger",
            obs={"reason": reason, "consecutive_backtracks": self.state.consecutive_backtracks},
            action={"outcome": event.outcome.value},
            rationale=reason,
        )

    def _log_decision(self, intent: str, obs: dict, action: dict, rationale: str) -> None:
        d = Decision(
            run_id=self.state.run_id,
            step=self.state.step,
            timestamp=time.time(),
            intent=intent,
            stage=self.state.stage,
            observation_summary=obs,
            action_summary=action,
            rationale=rationale,
            llm_tokens_used=0,
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
        dense_baseline = 64 * 64
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
