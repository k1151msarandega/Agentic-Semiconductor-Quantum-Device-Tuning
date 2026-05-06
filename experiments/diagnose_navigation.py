"""
experiments/diagnose_navigation.py
====================================
Targeted diagnostic for NAVIGATION stage failures.

Runs a single trial through to NAVIGATION and then traces every BO step:
voltage position, BO proposal, risk score breakdown, belief state,
CNN classification, and whether the agent is actually converging.

Usage:
    python experiments/diagnose_navigation.py
    python experiments/diagnose_navigation.py --seed 42 --budget 8192
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from qdot.core.types import (
    ChargeLabel, HITLOutcome, TuningStage,
    ActionProposal, MeasurementPlan, MeasurementModality,
)
from qdot.core.state import ExperimentState
from qdot.core.hitl import HITLManager
from qdot.core.governance import GovernanceLogger
from qdot.simulator.cim import CIMSimulatorAdapter
from qdot.perception.dqc import DQCGatekeeper
from qdot.perception.inspector import InspectionAgent
from qdot.perception.classifier import EnsembleCNN
from qdot.perception.ood import MahalanobisOOD
from qdot.agent.executive import ExecutiveAgent

DIVIDER   = "─" * 68
THICK_DIV = "═" * 68


class NavigationDiagnosticAgent(ExecutiveAgent):
    """Subclass that dumps a full diagnostic block on every navigation step."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._nav_step = 0
        # We'll compute the true (1,1) location from device params
        self._true_v11_vg1 = None
        self._true_v11_vg2 = None

    def _set_true_target(self, E_c, lever):
        """Store the analytically known (1,1) location for reference."""
        # (1,1) triple point is where both transitions coincide:
        # vg1 ≈ -(E_c1 + E_c2/2) / lever  (rough estimate for symmetric device)
        self._true_v11_vg1 = -(E_c * 1.5) / lever
        self._true_v11_vg2 = -(E_c * 1.5) / lever

    def _run_navigation(self) -> object:
        self._nav_step += 1

        # --- Capture pre-step state ---
        v_before = self.state.current_voltage
        belief_before = dict(self.state.belief.charge_probs)
        most_likely_before = self.state.belief.most_likely_state()

        # Run the actual navigation step
        result = super()._run_navigation()

        # --- Capture post-step state ---
        v_after = self.state.current_voltage
        most_likely_after = self.state.belief.most_likely_state()
        conf_11 = self.state.belief.charge_probs.get((1, 1), 0.0)
        last_cls = self.state.last_classification
        last_dqc = self.state.last_dqc
        last_risk = self._last_risk if hasattr(self, '_last_risk') else None

        # Distance to true (1,1) if known
        dist_before = dist_after = None
        if self._true_v11_vg1 is not None:
            dist_before = np.sqrt(
                (v_before.vg1 - self._true_v11_vg1)**2 +
                (v_before.vg2 - self._true_v11_vg2)**2
            )
            dist_after = np.sqrt(
                (v_after.vg1 - self._true_v11_vg1)**2 +
                (v_after.vg2 - self._true_v11_vg2)**2
            )
            moving = "→ CLOSER" if dist_after < dist_before else "→ FARTHER"
        else:
            moving = ""

        print(f"\n{THICK_DIV}")
        print(f"  NAV STEP {self._nav_step:3d}  |  "
              f"step={self.state.step}  |  "
              f"meas={self.state.total_measurements}")
        print(THICK_DIV)

        # Voltage movement
        dv1 = v_after.vg1 - v_before.vg1
        dv2 = v_after.vg2 - v_before.vg2
        print(f"\n  [VOLTAGE]")
        print(f"    Before:  vg1={v_before.vg1:+.4f}V  vg2={v_before.vg2:+.4f}V")
        print(f"    After:   vg1={v_after.vg1:+.4f}V  vg2={v_after.vg2:+.4f}V")
        print(f"    Move:    Δvg1={dv1:+.4f}  Δvg2={dv2:+.4f}  "
              f"L1={abs(dv1)+abs(dv2):.4f}")
        if self._true_v11_vg1 is not None:
            print(f"    True (1,1) target: vg1≈{self._true_v11_vg1:.3f}V  "
                  f"vg2≈{self._true_v11_vg2:.3f}V")
            print(f"    Distance: {dist_before:.4f}V → {dist_after:.4f}V  {moving}")

        # Risk score
        print(f"\n  [RISK]")
        # Re-compute risk components for display
        sv = self.safety_critic.verify(v_before,
            type('P', (), {
                'delta_v': type('V', (), {'vg1': dv1, 'vg2': dv2, 'l1_norm': abs(dv1)+abs(dv2)})(),
                'safe_delta_v': None, 'clipped': False, 'clip_warnings': [],
                'info_gain': 0.0,
            })()
        ) if abs(dv1) + abs(dv2) > 0 else None

        dqc_flag = last_dqc.quality.value if last_dqc else "high"
        disagreement = last_cls.ensemble_disagreement if last_cls else 0.0
        ood_score = self.state.last_ood.score if self.state.last_ood else 0.0

        print(f"    DQC flag:       {dqc_flag}")
        print(f"    OOD score:      {ood_score:.3f}")
        print(f"    Disagreement:   {disagreement:.4f}  "
              f"{'!! >0.30 → +0.35 risk' if disagreement > 0.30 else '✓'}")
        print(f"    Backtracks:     {self.state.consecutive_backtracks}  "
              f"{'!! ≥2 → +0.45 risk' if self.state.consecutive_backtracks >= 2 else '✓'}")
        print(f"    HITL triggers so far: {len(self.state.hitl_events)}")

        # Belief state
        print(f"\n  [BELIEF STATE]")
        top_states = sorted(
            self.state.belief.charge_probs.items(),
            key=lambda x: x[1], reverse=True
        )[:5]
        for state_key, prob in top_states:
            bar = "█" * int(prob * 30)
            winner = "  ← most likely" if state_key == most_likely_after else ""
            target = "  ← TARGET" if state_key == (1, 1) else ""
            print(f"    {str(state_key):<10}  {prob:.4f}  {bar}{winner}{target}")
        print(f"    Entropy: {self.state.belief.entropy():.4f}")
        converging = conf_11 > 0.3
        print(f"    P(1,1)={conf_11:.4f}  "
              f"{'✓ converging' if converging else '✗ not converging toward (1,1)'}")

        # CNN
        if last_cls is not None:
            print(f"\n  [CNN @ NEW POSITION]")
            print(f"    Label:        {last_cls.label.value}")
            print(f"    Confidence:   {last_cls.confidence:.4f}")
            print(f"    Disagreement: {last_cls.ensemble_disagreement:.4f}")

        # Verdict
        print(f"\n  [STEP RESULT]")
        print(f"    success={result.success}  "
              f"confidence={result.confidence:.4f}")
        if result.success:
            print(f"    ✓ NAVIGATION COMPLETE")
        else:
            remaining = self.measurement_budget - self.state.total_measurements
            print(f"    Budget remaining: {remaining} pts")
        print(DIVIDER)

        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--budget", type=int, default=8192)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--out", type=str, default="results/diagnose_navigation")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    E_c = 2.5
    lever = 0.65
    t_c = 0.3

    adapter = CIMSimulatorAdapter(
        device_id="nav_diag",
        params={
            "E_c1": E_c, "E_c2": E_c + 0.2, "t_c": t_c,
            "T": 0.08, "lever_arm": lever, "noise_level": 0.015,
        },
        seed=args.seed,
    )

    V_t = -E_c / lever
    print(f"\n{THICK_DIV}")
    print("  NAVIGATION DIAGNOSTIC RUN")
    print(THICK_DIV)
    print(f"  CIM: E_c1={E_c}, lever_arm={lever}")
    print(f"  Charge transition at V_t ≈ {V_t:.3f}V")
    print(f"  Approximate (1,1) triple point: vg1≈{V_t*1.5:.3f}V")

    ckpt = Path("experiments/checkpoints/phase1")
    try:
        ensemble = EnsembleCNN.load(str(ckpt))
        ood = MahalanobisOOD.load(str(ckpt / "ood_detector.pkl"))
        inspector = InspectionAgent(ensemble=ensemble, ood_detector=ood)
        print(f"  ✓ Loaded InspectionAgent from {ckpt}")
    except Exception as exc:
        print(f"  ⚠ Checkpoint load failed ({exc}); using untrained CNN")
        inspector = InspectionAgent()

    state = ExperimentState.new(device_id="nav_diag", target_label=ChargeLabel.DOUBLE_DOT)
    hitl = HITLManager(enabled=True)
    hitl.set_test_mode(auto_outcome=HITLOutcome.APPROVED)
    gov = GovernanceLogger(run_id=state.run_id,
                           log_dir=str(out_dir / "governance"))

    agent = NavigationDiagnosticAgent(
        state=state, adapter=adapter, inspection_agent=inspector,
        hitl_manager=hitl, governance_logger=gov,
        max_steps=args.max_steps, measurement_budget=args.budget,
    )
    agent._set_true_target(E_c, lever)

    print(f"\n  Running to NAVIGATION stage...")
    print(DIVIDER)

    for _ in range(args.max_steps):
        if not agent._step():
            break
        if state.stage == TuningStage.NAVIGATION:
            break

    if state.stage != TuningStage.NAVIGATION:
        print(f"\n  ✗ Never reached NAVIGATION. Final stage: {state.stage.name}")
        print(f"    Measurements used: {state.total_measurements}")
        return

    print(f"\n  ✓ Reached NAVIGATION at step {state.step}, "
          f"meas={state.total_measurements}")
    print(f"  Starting voltage: vg1={state.current_voltage.vg1:.4f}  "
          f"vg2={state.current_voltage.vg2:.4f}")
    print(DIVIDER)

    # Now run navigation steps explicitly
    for _ in range(args.max_steps):
        if not agent._step():
            break
        if state.stage != TuningStage.NAVIGATION:
            break

    print(f"\n{THICK_DIV}")
    print("  NAVIGATION SUMMARY")
    print(THICK_DIV)
    print(f"  Nav steps taken:    {agent._nav_step}")
    print(f"  Final stage:        {state.stage.name}")
    print(f"  Final voltage:      vg1={state.current_voltage.vg1:.4f}  "
          f"vg2={state.current_voltage.vg2:.4f}")
    print(f"  Total measurements: {state.total_measurements}")
    print(f"  HITL events:        {len(state.hitl_events)}")
    print(f"  Final P(1,1):       "
          f"{state.belief.charge_probs.get((1,1), 0.0):.4f}")
    print(f"  Most likely state:  {state.belief.most_likely_state()}")
    print(DIVIDER)


if __name__ == "__main__":
    main()
