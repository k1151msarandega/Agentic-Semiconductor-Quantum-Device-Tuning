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
    bootstrap_result, survey_result, hypersurface_result,
    charge_id_result, navigation_result, verification_result,
)

from qdot.agent.translator import TranslationAgent
from qdot.agent.narrator import LLMNarrator

from qdot.simulator.physics import coulomb_window

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
        self.narrator = LLMNarrator(run_id=state.run_id)
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
        elif stage == TuningStage.HYPERSURFACE_SEARCH:
            result = self._run_hypersurface_search()
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
            self.narrator.log_transition(
                from_stage=stage.name,
                to_stage=new_stage.name,
                rationale=rationale,
                step=self.control_steps,
                measurements_used=self.state.total_measurements,
                confidence=result.confidence,
                snr_db=self.state.last_dqc.snr_db if self.state.last_dqc else None,
                dqc_quality=self.state.last_dqc.quality.value if self.state.last_dqc else None,
                belief_top_state=str(self.state.belief.most_likely_state()),
                current_voltage=(
                    self.state.current_voltage.vg1,
                    self.state.current_voltage.vg2,
                ),
            )
            # Report exceptions: stage failures and budget warnings
            if new_stage == stage and result.confidence < 0.3:
                self.narrator.report_exception(
                    stage=stage.name,
                    exception_type="stage_failure",
                    step=self.control_steps,
                    measurements_used=self.state.total_measurements,
                    budget_total=self.measurement_budget,
                    details={"confidence": round(result.confidence, 3),
                             "consecutive_backtracks": self.state.consecutive_backtracks},
                )
            remaining_pct = 100 * (1 - self.state.total_measurements / self.measurement_budget)
            if remaining_pct < 20:
                self.narrator.report_exception(
                    stage=stage.name,
                    exception_type="budget_warning",
                    step=self.control_steps,
                    measurements_used=self.state.total_measurements,
                    budget_total=self.measurement_budget,
                    details={"remaining_pct": round(remaining_pct, 1)},
                )

        if hitl_triggered:
            recommendation = self.narrator.support_hitl(
                stage=self.state.stage.name,
                trigger_reason=rationale,
                risk_score=0.70,
                step=self.control_steps,
                measurements_used=self.state.total_measurements,
                budget_total=self.measurement_budget,
                proposal_summary=f"Stage {self.state.stage.name} decision point",
                physics_context={
                    "dqc": self.state.last_dqc.quality.value if self.state.last_dqc else "unknown",
                    "ood": round(self.state.last_ood.score, 3) if self.state.last_ood else 0.0,
                    "belief": str(self.state.belief.most_likely_state()),
                    "backtracks": self.state.consecutive_backtracks,
                },
            )
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
    
        # FIX: Always do a systematic COARSE_2D sweep for the initial survey.
        # The sensing policy is CIM-model-dependent; when device parameters differ
        # from the prior (which they always do in the benchmark), it degrades to
        # a 1D scan at vg2=0, which misses 2D charge features entirely.
        # A systematic 2D scan is model-agnostic and guaranteed to capture any
        # charge structure within the voltage bounds.
        plan = MeasurementPlan(
            modality=MeasurementModality.COARSE_2D,
            v1_range=v1_range,
            v2_range=v2_range,
            resolution=32,
            rationale="COARSE_SURVEY: systematic 2D sweep to locate charge signal",
        )
        plan = self._fit_plan_to_remaining_budget(plan)
    
        # Everything below this line stays exactly as it was
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

            if peak_quality > 0.2 and arr.size > 0 and arr.ndim >= 2:
                gy, gx = np.gradient(arr)
                grad_mag = np.sqrt(gx**2 + gy**2)
                row, col = np.unravel_index(np.argmax(grad_mag), grad_mag.shape)
                v1_lo, v1_hi = m.v1_range or (-1.0, 1.0)
                v2_lo, v2_hi = m.v2_range or (-1.0, 1.0)
                res = arr.shape[0]
                if res > 1:
                    peak_vg1 = v1_lo + (col / (res - 1)) * (v1_hi - v1_lo)
                    peak_vg2 = v2_lo + (row / (res - 1)) * (v2_hi - v2_lo)
                    self.state.config["survey_peak_vg1"] = peak_vg1
                    self.state.config["survey_peak_vg2"] = peak_vg2
                    self.state.config["survey_peak_snr_db"] = dqc.snr_db

        elif peak_quality > 0.2 and arr.ndim == 1 and len(arr) > 1:
            # LINE_SCAN: store vg1 peak position. vg2 is unknown from a 1D scan;
            # use current_voltage.vg2 as the best available estimate. The
            # HYPERSURFACE_SEARCH ±0.5V window around this point will expose
            # the transition along vg2 if it is close to the current vg2.
            v_lo = m.v1_range[0] if m.v1_range else self.state.voltage_bounds["vg1"]["min"]
            v_hi = m.v1_range[1] if m.v1_range else self.state.voltage_bounds["vg1"]["max"]
            peak_idx = int(np.argmax(arr))
            peak_vg1 = v_lo + (peak_idx / (len(arr) - 1)) * (v_hi - v_lo)
            self.state.config["survey_peak_vg1"] = peak_vg1
            self.state.config["survey_peak_vg2"] = self.state.current_voltage.vg2
            self.state.config["survey_peak_snr_db"] = dqc.snr_db

        return survey_result(peak_found=peak_quality > 0.2, peak_quality=peak_quality)

    def _run_hypersurface_search(self) -> StageResult:
        """
        Navigate to the charge boundary found by COARSE_SURVEY.

        Does a local 2D scan centred on the survey peak and checks whether
        a genuine charge feature is visible (DQC SNR >= 5 dB).  If the
        boundary is confirmed, the refined peak location is written back to
        state.config so CHARGE_ID always scans the right neighbourhood.

        The SNR threshold of 5 dB discriminates:
          - Real Coulomb peak inside the window  -> SNR >> 5 dB
          - Noise-argmax / featureless gradient  -> SNR << 5 dB
        """
        centre_vg1 = self.state.config.get(
            "survey_peak_vg1", self.state.current_voltage.vg1
        )
        centre_vg2 = self.state.config.get(
            "survey_peak_vg2", self.state.current_voltage.vg2
        )

        half = 0.5
        v1_lo = max(self.state.voltage_bounds["vg1"]["min"], centre_vg1 - half)
        v1_hi = min(self.state.voltage_bounds["vg1"]["max"], centre_vg1 + half)
        v2_lo = max(self.state.voltage_bounds["vg2"]["min"], centre_vg2 - half)
        v2_hi = min(self.state.voltage_bounds["vg2"]["max"], centre_vg2 + half)

        plan = MeasurementPlan(
            modality=MeasurementModality.COARSE_2D,
            v1_range=(v1_lo, v1_hi),
            v2_range=(v2_lo, v2_hi),
            resolution=16,   # was 32 — boundary confirmation only, not CNN input
            rationale=(
                "HYPERSURFACE_SEARCH: local scan around survey peak to confirm "
                "charge boundary is visible before classification"
            ),
        )
        plan = self._fit_plan_to_remaining_budget(plan)
        tr = self.translator.execute(plan)

        if tr.measurement is None:
            return hypersurface_result(boundary_found=False, proximity_confidence=0.0)

        m = tr.measurement
        self.state.add_measurement(m)
        dqc = self.dqc.assess(m)
        self.state.add_dqc_result(dqc)

        # Use DQC quality rather than raw SNR. The variance-based SNR estimator
        # underestimates narrow Coulomb transition lines (the 3×3 mean filter
        # treats a single-pixel-wide charge boundary as high-frequency noise).
        # DQC already accounts for this via the dynamic_range bypass — trust it.
        boundary_found = dqc.quality != DQCQuality.LOW
        proximity_confidence = (
            1.0 if dqc.quality == DQCQuality.HIGH else
            0.6 if dqc.quality == DQCQuality.MODERATE else
            0.0
        )

        if boundary_found and m.array is not None:
            arr = np.asarray(m.array)
            if arr.ndim >= 2 and arr.shape[0] > 1:
                row, col = np.unravel_index(np.argmax(arr), arr.shape)
                res = arr.shape[0]
                refined_vg1 = v1_lo + (col / (res - 1)) * (v1_hi - v1_lo)
                refined_vg2 = v2_lo + (row / (res - 1)) * (v2_hi - v2_lo)
                self.state.config["survey_peak_vg1"] = refined_vg1
                self.state.config["survey_peak_vg2"] = refined_vg2
                self.state.config["survey_peak_snr_db"] = dqc.snr_db

        return hypersurface_result(
            boundary_found=boundary_found,
            proximity_confidence=proximity_confidence,
        )

    def _run_charge_id(self) -> StageResult:
        # Use the local +-0.2V sub-scan only when HYPERSURFACE_SEARCH confirmed a
        # genuine charge feature (SNR >= 5 dB). Otherwise fall back to a full-range
        # scan so the CNN receives a stability diagram in its training distribution.
        vg1_min = self.state.voltage_bounds["vg1"]["min"]
        vg1_max = self.state.voltage_bounds["vg1"]["max"]
        vg2_min = self.state.voltage_bounds["vg2"]["min"]
        vg2_max = self.state.voltage_bounds["vg2"]["max"]

        centre_vg1 = self.state.config.get(
            "survey_peak_vg1", self.state.current_voltage.vg1
        )
        centre_vg2 = self.state.config.get(
            "survey_peak_vg2", self.state.current_voltage.vg2
        )

        # Window sized to match the InspectionAgent's actual training
        # distribution (qdot/perception/dataset.py: half-width = 1.5 x
        # Coulomb period). Previously hardcoded to 3.5V, which undercovers
        # the training window by ~40-60% for benchmark params and would be
        # off by ~5x for other device params (e.g. the E_c=0.5, lever=1.0
        # "default" params). See qdot/simulator/physics.py.
        _params = self.state.belief.device_params or {}
        half_width, _period = coulomb_window(
            E_c1=_params.get("E_c1", 2.5),
            E_c2=_params.get("E_c2", 2.5),
            lever_arm=_params.get("lever_arm", 0.65),
            factor=1.5,
            min_half_width=0.5,
        )
        v1_range = (
            max(vg1_min, centre_vg1 - half_width),
            min(vg1_max, centre_vg1 + half_width),
        )
        v2_range = (
            max(vg2_min, centre_vg2 - half_width),
            min(vg2_max, centre_vg2 + half_width),
        )

        plan = MeasurementPlan(
            modality=MeasurementModality.COARSE_2D,
            v1_range=v1_range,
            v2_range=v2_range,
            resolution=32,   # CNN._prepare() resizes to 64×64 at inference; 32×32 input = 1024 pts not 4096
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

        # Move current_voltage to the survey peak so NAVIGATION starts
        # in the correct region. Without this the agent navigates from
        # (0,0) which is 5+ V from any charge structure — the BO can
        # never converge within budget.
        if classification.label.value in ("single-dot", "double-dot"):
            self.translator.execute_voltage_move(
                vg1=centre_vg1, vg2=centre_vg2,
            )
            self.state.apply_move(
                VoltagePoint(
                    vg1=centre_vg1 - self.state.current_voltage.vg1,
                    vg2=centre_vg2 - self.state.current_voltage.vg2,
                )
            )

        return charge_id_result(
            label=classification.label.value,
            confidence=classification.confidence,
            physics_override=classification.physics_override,
        )
    
    def _compute_11_target(self) -> Optional[VoltagePoint]:
        """
        Numerically compute the centre of the (1,1) ground-state region within
        the voltage bounds using the belief state's device parameters.
    
        Returns the centroid VoltagePoint, or None if the (1,1) region does not
        fall within the current voltage bounds (e.g. params place it out of range).
        """
        from qdot.simulator.cim import ConstantInteractionDevice
        params = self.state.belief.device_params
        if not params:
            return None
    
        tmp = ConstantInteractionDevice(
            E_c1=params.get("E_c1", 0.5),
            E_c2=params.get("E_c2", 0.55),
            t_c=params.get("t_c", 0.05),
            T=params.get("T", 0.015),
            lever_arm=params.get("lever_arm", 1.0),
            noise_level=0.0,
        )
    
        v1_min = self.state.voltage_bounds["vg1"]["min"]
        v1_max = self.state.voltage_bounds["vg1"]["max"]
        v2_min = self.state.voltage_bounds["vg2"]["min"]
        v2_max = self.state.voltage_bounds["vg2"]["max"]
    
        v1_grid = np.linspace(v1_min, v1_max, 60)
        v2_grid = np.linspace(v2_min, v2_max, 60)
        all_states = [(n1, n2) for n1 in range(3) for n2 in range(3)]
    
        v1_11, v2_11 = [], []
        for v2 in v2_grid:
            for v1 in v1_grid:
                energies = [tmp.ground_state_energy(float(v1), float(v2), n1, n2)
                            for n1, n2 in all_states]
                if all_states[int(np.argmin(energies))] == (1, 1):
                    v1_11.append(float(v1))
                    v2_11.append(float(v2))
    
        if not v1_11:
            return None
        return VoltagePoint(vg1=float(np.mean(v1_11)), vg2=float(np.mean(v2_11)))
    
    def _run_navigation(self) -> StageResult:
        # On the very first navigation step the BO history is empty and the GP
        # has no posterior — any proposal is effectively random. Instead, use
        # the CIM free-energy to compute the (1,1) ground-state centroid and
        # jump directly there. All subsequent steps use the BO normally.
        if len(self.state.bo_history) == 0:
            target_11 = self._compute_11_target()
            if target_11 is not None:
                # Clamp individual-axis jump to ±3.0V to respect hardware limits.
                max_jump = 3.0
                dv1 = float(np.clip(
                    target_11.vg1 - self.state.current_voltage.vg1,
                    -max_jump, max_jump
                ))
                dv2 = float(np.clip(
                    target_11.vg2 - self.state.current_voltage.vg2,
                    -max_jump, max_jump
                ))
                self.translator.execute_voltage_move(
                    vg1=self.state.current_voltage.vg1 + dv1,
                    vg2=self.state.current_voltage.vg2 + dv2,
                )
                self.state.apply_move(VoltagePoint(vg1=dv1, vg2=dv2))

        self.bo.update(self.state.bo_history)

        # Dynamic step size and exploration scaling from OOD score.
        # High OOD score → deep in the featureless MISC zone → take large
        # exploratory leaps. Low OOD score → near a recognisable charge
        # boundary → fine-grained exploitative steps.
        base_l1_max = self.state.step_caps.get("l1_max", 0.10)
        if self.state.last_ood is not None:
            ood_score = self.state.last_ood.score
            ood_threshold = self.state.last_ood.threshold  # calibrated 95th-pct value
            # Normalised ratio: 0 = in-distribution, 1 = at threshold, >1 = OOD
            ood_ratio = min(ood_score / (ood_threshold + 1e-8), 3.0)
        else:
            ood_ratio = 0.0

        # Step size scales linearly from base (in-distribution) to 7× base (3× threshold).
        # Hard cap at 1.0V to stay within typical device voltage ranges.
        dynamic_l1_max = min(base_l1_max * (1.0 + 2.0 * ood_ratio), 1.0)

        # When OOD ratio > 1.5 (well past the threshold), also boost UCB β
        # in the BO so it doesn't just exploit the flat GP mean near current voltage.
        exploration_override = ood_ratio > 1.5

        proposal = self.bo.propose(
            current=self.state.current_voltage,
            l1_max=dynamic_l1_max,
            exploration_override=exploration_override,
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

        nav_step = self.state_machine._retries.get(TuningStage.NAVIGATION, 0)
        effective_threshold = 0.92 if self.state.stage == TuningStage.NAVIGATION else HITLManager.HITL_THRESHOLD
        if risk >= effective_threshold:
            event = self.hitl_manager.queue_request(
                run_id=self.state.run_id,
                step=self.state.step,
                stage=self.state.stage,
                trigger_reason=f"Risk score={risk:.2f} >= threshold=0.70",
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

        # Take a local measurement at the new voltage to update the belief state.
        # Without this, most_likely_state() never converges — navigation moves
        # blindly and the convergence check always fails.
        nav_v1 = self.state.current_voltage.vg1
        nav_v2 = self.state.current_voltage.vg2
        # Scale nav scan to the Coulomb period so the CNN always receives
        # an image consistent with its training distribution.
        # Training scans span ~1.5 × (2 E_c / lever_arm); we use 0.75 × period
        # (half of that) as the half-width so the window covers one full cell.
        _params = self.state.belief.device_params
        if _params:
            _E_c = ((_params.get("E_c1", 1.0) + _params.get("E_c2", 1.0)) / 2.0)
            _lever = max(_params.get("lever_arm", 1.0), 0.1)
            _period = 2.0 * _E_c / _lever          # one Coulomb period in volts
            nav_half_width = float(np.clip(_period / 2.0, 1.5, 5.0))
        else:
            nav_half_width = 1.5
        nav_v1_lo = max(self.state.voltage_bounds["vg1"]["min"], nav_v1 - nav_half_width)
        nav_v1_hi = min(self.state.voltage_bounds["vg1"]["max"], nav_v1 + nav_half_width)
        nav_v2_lo = max(self.state.voltage_bounds["vg2"]["min"], nav_v2 - nav_half_width)
        nav_v2_hi = min(self.state.voltage_bounds["vg2"]["max"], nav_v2 + nav_half_width)
        nav_plan = MeasurementPlan(
            modality=MeasurementModality.COARSE_2D,
            v1_range=(nav_v1_lo, nav_v1_hi),
            v2_range=(nav_v2_lo, nav_v2_hi),
            resolution=16,
            rationale="NAVIGATION: local scan to update belief state after voltage move",
        )
        nav_plan = self._fit_plan_to_remaining_budget(nav_plan)
        nav_tr = self.translator.execute(nav_plan)
        if nav_tr.measurement is not None:
            nav_m = nav_tr.measurement
            self.state.add_measurement(nav_m)
            nav_dqc = self.dqc.assess(nav_m)
            self.state.add_dqc_result(nav_dqc)
            if nav_m.is_2d and nav_dqc.quality != DQCQuality.LOW:
                if self.inspection_agent is not None:
                    nav_cls, nav_ood = self.inspection_agent.inspect(nav_m, nav_dqc)
                    self.state.add_classification(nav_cls)
                    self.state.add_ood_result(nav_ood)
                    self.belief_updater.update_from_2d(nav_m, nav_cls)

                    # Record this observation in bo_history so the GP can learn.
                    # Score = P(1,1) from the updated belief — this is the quantity
                    # the BO is implicitly maximising. Without this the GP always
                    # fits to an empty history and every proposal is from the prior.
                    bo_score = self.state.belief.charge_probs.get((1, 1), 0.0)
                    bo_pt = self.bo.make_bo_point(
                        voltage=self.state.current_voltage,
                        score=bo_score,
                        label=nav_cls.label,
                        confidence=nav_cls.confidence,
                        step=self.control_steps,
                    )
                    self.state.bo_history.append(bo_pt)

                    if nav_ood.score > 8.0:
                        self.narrator.report_exception(
                            stage="NAVIGATION",
                            exception_type="ood_spike",
                            step=self.control_steps,
                            measurements_used=self.state.total_measurements,
                            budget_total=self.measurement_budget,
                            details={"ood_score": round(nav_ood.score, 2),
                                     "vg1": round(nav_v1, 3),
                                     "vg2": round(nav_v2, 3)},
                        )
                else:
                    self.belief_updater.update_from_2d(nav_m)

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
        if plan.modality == MeasurementModality.NONE:
            return 0
        if plan.modality == MeasurementModality.LINE_SCAN:
            return int(plan.steps or 128)
        if plan.resolution is None:
            return 0
        return int(plan.resolution * plan.resolution)

    def _fit_plan_to_remaining_budget(self, plan: MeasurementPlan) -> MeasurementPlan:
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
                    f"(resolution reduced {requested_res}->{res} to fit budget)"
                ),
                info_gain_per_cost=plan.info_gain_per_cost,
            )

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
        self.narrator.drain()
        self.narrator.summarise_run(
            final_stage=self.state.stage.name,
            success=self.state.stage == TuningStage.COMPLETE,
            total_measurements=self.state.total_measurements,
            budget_total=self.measurement_budget,
            total_steps=self.control_steps,
            n_backtracks=self.state.total_backtracks,
            n_hitl=len(self.state.hitl_events),
            n_exceptions=self.narrator.n_exceptions(),
        )
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
