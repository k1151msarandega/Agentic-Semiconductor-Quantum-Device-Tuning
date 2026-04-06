"""
qdot/agent/executive.py
========================
Executive Agent — main agent loop orchestrator.
"""

from __future__ import annotations

import math
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
        self.control_steps = 0

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

    def _step(self) -> bool:
        """
        Execute one iteration of the main agent loop.

        Returns True if the loop should continue, False if it should stop.
        Used by diagnose_trial.py for step-by-step monitoring.
        """
        self.control_steps += 1
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
            return False

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

        return not self._should_terminate()

    # ------------------------------------------------------------------
    # Stage executors
    # ------------------------------------------------------------------

    def _run_bootstrap(self) -> StageResult:
        plan = MeasurementPlan(
            modality=MeasurementModality.LINE_SCAN,
            axis="vg1",
            start=self.state.voltage_bounds["vg1"]["min"],
            stop=self.state.voltage_bounds["vg1"]["max"],
            steps=64,
            rationale="Bootstrap: electrical integrity check across full voltage range",
        )
        plan = self._fit_plan_to_remaining_budget(plan)

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
        plan = self._fit_plan_to_remaining_budget(plan)

        tr = self.translator.execute(plan)

        if tr.measurement is None:
            return survey_result(peak_found=False, peak_quality=0.0)

        m = tr.measurement
        self.state.add_measurement(m)

        dqc = self.dqc.assess(m)
        self.state.add_dqc_result(dqc)

        if dqc.quality == DQCQuality.LOW:
            return survey_result(peak_found=False, peak_quality=0.0)

        arr = np.asarray(m.array) if m.array is not None else np.array([])
        peak_quality = float((arr.max() - arr.mean()) / (arr.max() + 1e-8))

        if m.is_2d:
            self.belief_updater.update_from_2d(m)

            # Store the peak voltage in state.config so CHARGE_ID scans the
            # right location. We do NOT physically move here — the slew rate
            # cap (0.1V L1) would require 5+ survey iterations to reach a peak
            # at -0.5V, and the state machine advances after the first success.
            # Storing the location and passing it to CHARGE_ID is the correct
            # architectural approach: no safety constraint is worked around,
            # and CHARGE_ID always scans the known feature location.
            if peak_quality > 0.2 and arr.size > 0 and arr.ndim >= 2:
                row, col = np.unravel_index(np.argmax(arr), arr.shape)
                v1_lo, v1_hi = m.v1_range or (-1.0, 1.0)
                v2_lo, v2_hi = m.v2_range or (-1.0, 1.0)
                res = arr.shape[0]
                if res > 1:
                    peak_vg1 = v1_lo + (col / (res - 1)) * (v1_hi - v1_lo)
                    peak_vg2 = v2_lo + (row / (res - 1)) * (v2_hi - v2_lo)
                    self.state.config["survey_peak_vg1"] = peak_vg1
                    self.state.config["survey_peak_vg2"] = peak_vg2

        return survey_result(peak_found=peak_quality > 0.2, peak_quality=peak_quality)

    def _run_charge_id(self) -> StageResult:
        # Use the peak location found during survey if available.
        # This is the core fix: CHARGE_ID scans where we know features are,
        # not where current_voltage happens to be (which is still near origin
        # since the slew-limited bootstrap/survey don't move far).
        centre_vg1 = self.state.config.get(
            "survey_peak_vg1", self.state.current_voltage.vg1
        )
        centre_vg2 = self.state.config.get(
            "survey_peak_vg2", self.state.current_voltage.vg2
        )

        # ±0.2V window around the known feature location (wider than ±0.1
        # to account for argmax pixel-quantisation error in the survey scan).
        v1_range = (centre_vg1 - 0.2, centre_vg1 + 0.2)
        v2_range = (centre_vg2 - 0.2, centre_vg2 + 0.2)

        # CHARGE_ID always needs a 2D scan — the InspectionAgent requires is_2d=True.
        # The sensing policy systematically selects LINE_SCAN here because its
        # IG/cost ratio beats all 2D options' theoretical maxima (LINE_SCAN cost is
        # 8x lower). Bypassing it for this stage is architecturally correct:
        # CHARGE_ID is a classification task, not an exploration task.
        plan = MeasurementPlan(
            modality=MeasurementModality.COARSE_2D,
            v1_range=v1_range,
            v2_range=v2_range,
            resolution=32,
            rationale="CHARGE_ID: 2D scan required for InspectionAgent classification",
        )
        plan = self._fit_plan_to_remaining_budget(plan)

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
        belief_confidence = self.state.belief.charge_probs.get((1, 1), 0.0)

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
            # VERIFICATION always needs 2D scans — the InspectionAgent requires is_2d=True.
            # Like CHARGE_ID, this is a classification task (confirming DOUBLE_DOT state).
            # The sensing policy would select LINE_SCAN for the same reason (8x cost advantage).
            # Bypassing it for this stage is architecturally correct.

            # Clip the scan window to stay within voltage bounds.
            v1_lo = max(
                self.state.voltage_bounds["vg1"]["min"],
                self.state.current_voltage.vg1 - 0.05
            )
            v1_hi = min(
                self.state.voltage_bounds["vg1"]["max"],
                self.state.current_voltage.vg1 + 0.05
            )
            v2_lo = max(
                self.state.voltage_bounds["vg2"]["min"],
                self.state.current_voltage.vg2 - 0.05
            )
            v2_hi = min(
                self.state.voltage_bounds["vg2"]["max"],
                self.state.current_voltage.vg2 + 0.05
            )

            plan = MeasurementPlan(
                modality=MeasurementModality.COARSE_2D,
                v1_range=(v1_lo, v1_hi),
                v2_range=(v2_lo, v2_hi),
                resolution=16,
                rationale="VERIFICATION: 2D scan required for InspectionAgent classification",
            )
            plan = self._fit_plan_to_remaining_budget(plan)

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
    # Budget guard
    # ------------------------------------------------------------------

    def _estimate_plan_cost(self, plan: MeasurementPlan) -> int:
        """Estimate measurement points consumed by a plan."""
        if plan.modality == MeasurementModality.NONE:
            return 0
        if plan.modality == MeasurementModality.LINE_SCAN:
            return int(plan.steps or 128)
        if plan.resolution is None:
            return 0
        return int(plan.resolution * plan.resolution)

    def _fit_plan_to_remaining_budget(self, plan: MeasurementPlan) -> MeasurementPlan:
        """
        Clamp or downgrade a plan so it cannot exceed remaining budget.

        - Line scan: clamp steps to remaining points (min 2).
        - 2D scan:   reduce resolution to largest res where res² fits;
                     fall back to line scan only if even 8×8 doesn't fit.
        - Budget exhausted: return NONE.
        """
        remaining = self.measurement_budget - self.state.total_measurements
        if remaining <= 0:
            return MeasurementPlan(
                modality=MeasurementModality.NONE,
                rationale="Budget exhausted",
            )

        cost = self._estimate_plan_cost(plan)
        if cost <= remaining:
            return plan

        if plan.modality == MeasurementModality.LINE_SCAN:
            steps = max(2, min(int(plan.steps or 128), remaining))
            return MeasurementPlan(
                modality=MeasurementModality.LINE_SCAN,
                axis=plan.axis or "vg1",
                start=plan.start,
                stop=plan.stop,
                steps=steps,
                rationale=f"{plan.rationale} (budget-clamped to {steps} steps)",
                info_gain_per_cost=plan.info_gain_per_cost,
            )

        # 2D scan doesn't fit at requested resolution — try smaller 2D first.
        v1_range = plan.v1_range or (
            self.state.voltage_bounds["vg1"]["min"],
            self.state.voltage_bounds["vg1"]["max"],
        )
        v2_range = plan.v2_range or (
            self.state.voltage_bounds["vg2"]["min"],
            self.state.voltage_bounds["vg2"]["max"],
        )

        requested_res = int(plan.resolution or math.isqrt(max(cost, 1)))
        max_fit_res = int(math.isqrt(remaining))
        res = min(requested_res, max_fit_res)

        if res >= 8:
            return MeasurementPlan(
                modality=plan.modality,
                v1_range=v1_range,
                v2_range=v2_range,
                resolution=res,
                rationale=(
                    f"{plan.rationale} "
                    f"(resolution reduced {requested_res}→{res} to fit budget)"
                ),
                info_gain_per_cost=plan.info_gain_per_cost,
            )

        # Even 8×8 doesn't fit — fall back to line scan.
        steps = max(2, min(128, remaining))
        return MeasurementPlan(
            modality=MeasurementModality.LINE_SCAN,
            axis="vg1",
            start=v1_range[0],
            stop=v1_range[1],
            steps=steps,
            rationale=f"{plan.rationale} (downgraded to {steps}-pt line scan; budget remaining={remaining})",
            info_gain_per_cost=plan.info_gain_per_cost,
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
            self.control_steps >= self.max_steps
            or self.state.total_measurements >= self.measurement_budget
            or self.state.stage in (TuningStage.COMPLETE, TuningStage.FAILED)
        )

    def _mission_summary(self) -> dict:
        dense_baseline = 64 * 64
        reduction = 1.0 - (self.state.total_measurements / dense_baseline)
        return {
            "success": self.state.stage == TuningStage.COMPLETE,
            "final_stage": self.state.stage.name,
            "total_steps": self.control_steps,
            "total_measurements": self.state.total_measurements,
            "measurement_reduction": reduction,
            "total_backtracks": self.state.total_backtracks,
            "safety_violations": self.state.safety_violations,
            "hitl_events": len(self.state.hitl_events),
            "run_id": self.state.run_id,
        }
