"""
experiments/diagnose_charge_id.py
===================================
Targeted diagnostic for CHARGE_ID stage failures.

Runs a single trial and, for every CHARGE_ID attempt, dumps a full
diagnostic block: scan range, array statistics, DQC result, CNN class
probabilities, physics features, and a pass/fail verdict.

Usage:
    python experiments/diagnose_charge_id.py
    python experiments/diagnose_charge_id.py --seed 1000 --out results/diag/
    python experiments/diagnose_charge_id.py --skip-checkpoints   # stub CNN
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from qdot.core.types import ChargeLabel, HITLOutcome, TuningStage
from qdot.core.state import ExperimentState
from qdot.core.hitl import HITLManager
from qdot.core.governance import GovernanceLogger
from qdot.simulator.cim import CIMSimulatorAdapter
from qdot.perception.dqc import DQCGatekeeper
from qdot.perception.features import physics_features, log_preprocess
from qdot.perception.inspector import InspectionAgent
from qdot.perception.classifier import EnsembleCNN
from qdot.agent.executive import ExecutiveAgent

DIVIDER   = "─" * 68
THICK_DIV = "═" * 68


def _boundary_check(val, lo, hi, tol=0.05):
    return val <= lo + tol or val >= hi - tol


class DiagnosticAgent(ExecutiveAgent):
    """Subclass that injects a diagnostic dump into every _run_charge_id call."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._charge_id_attempt = 0

    def _run_charge_id(self):
        self._charge_id_attempt += 1

        # Reproduce the scan-range logic so we can report it BEFORE the stage runs.
        vg1_min = self.state.voltage_bounds["vg1"]["min"]
        vg1_max = self.state.voltage_bounds["vg1"]["max"]
        vg2_min = self.state.voltage_bounds["vg2"]["min"]
        vg2_max = self.state.voltage_bounds["vg2"]["max"]
        centre_vg1 = self.state.config.get("survey_peak_vg1", self.state.current_voltage.vg1)
        centre_vg2 = self.state.config.get("survey_peak_vg2", self.state.current_voltage.vg2)
        at_v1_bnd = _boundary_check(centre_vg1, vg1_min, vg1_max)
        at_v2_bnd = _boundary_check(centre_vg2, vg2_min, vg2_max)

        # Run the actual stage (parent logic unchanged).
        stage_result = super()._run_charge_id()

        # Now gather evidence from state.
        last_cls = self.state.last_classification
        last_dqc = self.state.last_dqc
        last_2d_arr = None
        for m in reversed(list(self.state.measurements.values())):
            if m.is_2d and m.array is not None:
                last_2d_arr = np.asarray(m.array, dtype=np.float64)
                break

        # ── Print diagnostic ────────────────────────────────────────────────
        print(f"\n{THICK_DIV}")
        print(f"  CHARGE_ID DIAGNOSTIC  attempt={self._charge_id_attempt}  "
              f"step={self.state.step}  meas_used={self.state.total_measurements}")
        print(THICK_DIV)

        # --- Scan range ---
        print("\n  [SCAN RANGE]")
        print(f"    Survey peak:  vg1={centre_vg1:+.4f}V  vg2={centre_vg2:+.4f}V")
        print(f"    At vg1 boundary: {at_v1_bnd}   At vg2 boundary: {at_v2_bnd}")
        if at_v1_bnd or at_v2_bnd:
            print()
            print("    !! BOUNDARY ARTEFACT: survey argmax is at the scan window edge.")
            print("       The charge transition is likely OUTSIDE the ±1V window.")
            print("       The ±0.2V CHARGE_ID sub-scan contains noise only — no features.")
            print("       The CNN will (correctly) classify this as MISC every time.")
        else:
            print("    ✓  Interior peak — ±0.2V local scan is appropriate.")

        # --- Array stats ---
        if last_2d_arr is not None:
            print("\n  [ARRAY STATISTICS]  (normalised [0,1] from CIM)")
            print(f"    Shape:  {last_2d_arr.shape}")
            print(f"    Min:    {last_2d_arr.min():.6f}")
            print(f"    Max:    {last_2d_arr.max():.6f}")
            print(f"    Mean:   {last_2d_arr.mean():.6f}")
            print(f"    Std:    {last_2d_arr.std():.6f}")
            lp = log_preprocess(last_2d_arr)
            print(f"    Log-preprocessed  mean={lp.mean():.4f}  std={lp.std():.4f}")

        # --- DQC ---
        if last_dqc is not None:
            icon = {"high": "✓", "moderate": "△", "low": "✗"}.get(last_dqc.quality.value, "?")
            print("\n  [DQC RESULT]")
            print(f"    Quality:       {icon} {last_dqc.quality.value.upper()}")
            print(f"    SNR:           {last_dqc.snr_db:.2f} dB")
            print(f"    Dynamic range: {last_dqc.dynamic_range:.4f}")
            print(f"    Flatness:      {last_dqc.flatness_score:.6f}")
            print(f"    Plausible:     {last_dqc.physically_plausible}")
            if last_dqc.notes:
                print(f"    Notes: {last_dqc.notes}")
        else:
            print("\n  [DQC RESULT]  — not available (measurement skipped or budget hit)")

        # --- CNN ---
        if last_cls is not None:
            print("\n  [CNN CLASSIFICATION]")
            print(f"    Final label:      {last_cls.label.value}")
            print(f"    Confidence:       {last_cls.confidence:.4f}  (need > 0.50 to pass)")
            print(f"    Physics override: {last_cls.physics_override}")
            print(f"    Disagreement:     {last_cls.ensemble_disagreement:.4f}  (HITL > 0.30)")
            if last_2d_arr is not None and self.inspection_agent is not None:
                try:
                    probs = self.inspection_agent.ensemble.predict_proba(
                        last_2d_arr.astype(np.float32)
                    )
                    print("\n    Class probabilities (mean ensemble):")
                    for lbl, p in zip(["double-dot", "single-dot", "misc"], probs):
                        bar = "█" * int(p * 32)
                        winner = "  ← winner" if lbl == last_cls.label.value else ""
                        print(f"      {lbl:<12}  {p:.4f}  {bar}{winner}")
                except Exception as exc:
                    print(f"    (Could not re-run CNN: {exc})")
        else:
            print("\n  [CNN CLASSIFICATION]  — not available (DQC LOW or non-2D)")

        # --- Physics features ---
        if last_2d_arr is not None:
            feats = physics_features(last_2d_arr)
            print("\n  [PHYSICS FEATURES]")
            ds = feats["diagonal_strength"]
            ds_warn = "  !! LOW — no diagonal transitions visible" if ds < 0.25 else "  ✓"
            print(f"    diagonal_strength:  {ds:.4f}{ds_warn}")
            print(f"    fft_peak_ratio:     {feats['fft_peak_ratio']:.4f}")
            print(f"    mean_conductance:   {feats['mean_conductance']:.4f}")
            cs = feats["conductance_std"]
            cs_warn = "  !! Featureless?" if cs < 0.05 else ""
            print(f"    conductance_std:    {cs:.4f}{cs_warn}")

        # --- Verdict ---
        success_label = last_cls is not None and last_cls.label.value in ("single-dot", "double-dot")
        success_conf  = last_cls is not None and last_cls.confidence > 0.50
        print("\n  [VERDICT]")
        if success_label and success_conf:
            print("    ✓  PASS  — charge_id_result() will advance to NAVIGATION")
        else:
            print("    ✗  FAIL  — charge_id_result() will retry or backtrack")
            if not success_label:
                v = last_cls.label.value if last_cls else "none"
                print(f"       label='{v}' not in ('single-dot', 'double-dot')")
            if not success_conf:
                c = last_cls.confidence if last_cls else 0.0
                print(f"       confidence={c:.4f} ≤ 0.50")
        print(DIVIDER)

        return stage_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--budget", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--out", type=str, default="results/diagnose_charge_id")
    parser.add_argument("--skip-checkpoints", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use benchmark-representative CIM params (seed 1000 = benchmark trial 0)
    E_c = 1.5
    lever = 0.65
    adapter = CIMSimulatorAdapter(
        device_id="diag",
        params={"E_c1": E_c, "E_c2": E_c + 0.2, "t_c": 0.3,
                "T": 0.08, "lever_arm": lever, "noise_level": 0.015},
        seed=args.seed,
    )

    t1 = -E_c / lever
    t2 = -(E_c + 0.2) / lever
    print(f"\n{THICK_DIV}")
    print("  CHARGE_ID DIAGNOSTIC RUN")
    print(THICK_DIV)
    print(f"  CIM: E_c1={E_c}, lever_arm={lever}")
    print(f"  Transition vg1 ≈ {t1:.2f}V  ({'INSIDE' if abs(t1)<=1 else 'OUTSIDE'} ±1V)")
    print(f"  Transition vg2 ≈ {t2:.2f}V  ({'INSIDE' if abs(t2)<=1 else 'OUTSIDE'} ±1V)")

    # Load checkpoint if available
    ckpt = Path("experiments/checkpoints/phase1")
    if not args.skip_checkpoints and ckpt.exists():
        try:
            from qdot.perception.ood import MahalanobisOOD
            ensemble = EnsembleCNN.load(str(ckpt))
            ood = MahalanobisOOD.load(str(ckpt / "ood_detector.pkl"))
            inspector = InspectionAgent(ensemble=ensemble, ood_detector=ood)
            print(f"  ✓ Loaded trained InspectionAgent from {ckpt}")
        except Exception as exc:
            print(f"  ⚠  Checkpoint load failed ({exc}); using untrained CNN")
            inspector = InspectionAgent()
    else:
        print("  ⚠  Stub CNN (untrained)")
        inspector = InspectionAgent()

    state = ExperimentState.new(device_id="diag", target_label=ChargeLabel.DOUBLE_DOT)
    hitl = HITLManager(enabled=True)
    hitl.set_test_mode(auto_outcome=HITLOutcome.APPROVED)
    gov = GovernanceLogger(run_id=state.run_id, log_dir=str(out_dir / "governance"))

    agent = DiagnosticAgent(
        state=state, adapter=adapter, inspection_agent=inspector,
        hitl_manager=hitl, governance_logger=gov,
        max_steps=args.max_steps, measurement_budget=args.budget,
    )

    for _ in range(args.max_steps):
        if not agent._step():
            break

    # Root-cause summary
    print(f"\n{THICK_DIV}")
    print("  ROOT-CAUSE SUMMARY")
    print(THICK_DIV)
    in_window = abs(t1) <= 1.0 and abs(t2) <= 1.0
    print(f"  {'✓' if in_window else '✗'}  Transition inside ±1V: {in_window}")
    E_ok = 1.8 <= E_c <= 5.5
    l_ok = 0.35 <= lever <= 0.85
    print(f"  {'✓' if (E_ok and l_ok) else '✗'}  Params in training dist: E_c={E_ok}, lever={l_ok}")
    n_pass = sum(
        1 for c in state.classifications.values()
        if c.label.value in ("single-dot", "double-dot") and c.confidence > 0.5
    )
    print(f"  {'✓' if n_pass>0 else '✗'}  CHARGE_ID successes: {n_pass}/{agent._charge_id_attempt}")
    print(f"\n  Final stage: {state.stage.name}  |  Steps: {state.step}  |  "
          f"Measurements: {state.total_measurements}")
    print(DIVIDER)


if __name__ == "__main__":
    main()
