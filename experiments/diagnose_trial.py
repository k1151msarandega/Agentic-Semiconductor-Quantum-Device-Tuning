"""
experiments/diagnose_trial.py
==============================
Diagnostic tool for debugging failed Phase 2 trials.

Runs a single trial with verbose logging to understand why it's failing.

Usage:
    python experiments/diagnose_trial.py --verbose
    python experiments/diagnose_trial.py --seed 42 --out debug_trial/
"""

import argparse
import json
from pathlib import Path

import numpy as np

from qdot.core.types import ChargeLabel, HITLOutcome
from qdot.core.state import ExperimentState
from qdot.core.hitl import HITLManager
from qdot.core.governance import GovernanceLogger
from qdot.simulator.cim import CIMSimulatorAdapter
from qdot.perception.inspector import InspectionAgent
from qdot.agent.executive import ExecutiveAgent


def main():
    parser = argparse.ArgumentParser(description="Diagnose a single Phase 2 trial")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--budget", type=int, default=2048)
    parser.add_argument("--verbose", action="store_true", help="Print step-by-step progress")
    parser.add_argument("--out", type=str, default="results/diagnose")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("PHASE 2 DIAGNOSTIC TRIAL")
    print("="*70 + "\n")

    # Setup
    state = ExperimentState.new(device_id="diagnostic_trial", target_label=ChargeLabel.DOUBLE_DOT)
    adapter = CIMSimulatorAdapter(device_id="diagnostic_trial", seed=args.seed)
    inspector = InspectionAgent(ensemble=None, ood_detector=None)

    hitl = HITLManager(enabled=True)
    hitl.set_test_mode(auto_outcome=HITLOutcome.APPROVED)

    governance = GovernanceLogger(run_id=state.run_id, log_dir=str(out_dir / "governance"))

    # Create agent
    agent = ExecutiveAgent(
        state=state,
        adapter=adapter,
        inspection_agent=inspector,
        hitl_manager=hitl,
        governance_logger=governance,
        max_steps=args.max_steps,
        measurement_budget=args.budget,
    )

    # Run with step-by-step monitoring
    print("Running trial...")
    print(f"Max steps: {args.max_steps}")
    print(f"Measurement budget: {args.budget}\n")

    step_log = []
    last_stage = state.stage

    for step_num in range(args.max_steps):
        # Take one step
        should_continue = agent._step()

        # Log progress
        step_info = {
            "step": state.step,
            "stage": str(state.stage),
            "measurements": state.total_measurements,
            "backtracks": state.consecutive_backtracks,
            "voltage": {"vg1": state.current_voltage.vg1, "vg2": state.current_voltage.vg2},
        }
        step_log.append(step_info)

        # Verbose output
        if args.verbose:
            if state.stage != last_stage:
                print(f"\n{'='*60}")
                print(f"STAGE TRANSITION: {last_stage} → {state.stage}")
                print(f"{'='*60}")
                last_stage = state.stage

            print(f"[{state.step:3d}] {state.stage.name:<16} | "
                  f"meas={state.total_measurements:4d} | "
                  f"backtracks={state.consecutive_backtracks} | "
                  f"V=({state.current_voltage.vg1:+.3f}, {state.current_voltage.vg2:+.3f})")

        if not should_continue:
            break

    # Analyze results
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70 + "\n")

    print(f"Final stage:        {state.stage}")
    print(f"Success:            {state.stage.name == 'COMPLETE'}")
    print(f"Total steps:        {state.step}")
    print(f"Total measurements: {state.total_measurements}")
    print(f"Total backtracks:   {state.total_backtracks}")
    print(f"Safety violations:  {state.safety_violations}")
    print(f"HITL events:        {len(state.hitl_events)}")

    # Stage distribution
    from collections import Counter
    stage_counts = Counter(s["stage"] for s in step_log)
    print(f"\nSteps per stage:")
    for stage, count in stage_counts.most_common():
        print(f"  {stage:<20} {count:3d} steps ({100*count/len(step_log):.1f}%)")

    # Identify problems
    print(f"\nPOTENTIAL ISSUES:")
    if state.stage.name == 'COMPLETE':
        print("  ✓ Trial completed successfully")
    else:
        if state.step >= args.max_steps:
            print(f"  ✗ Hit max step limit ({args.max_steps}) before completing")
            print(f"    → Consider increasing max_steps or investigating stage inefficiency")

        if state.total_measurements >= args.budget:
            print(f"  ✗ Exhausted measurement budget ({args.budget})")
            print(f"    → Agent is taking too many measurements")

        stuck_stage = max(stage_counts, key=stage_counts.get)
        if stage_counts[stuck_stage] > 10:
            print(f"  ✗ Agent spent {stage_counts[stuck_stage]} steps in {stuck_stage}")
            print(f"    → Check stage success criteria and retry logic")

        if state.consecutive_backtracks > 5:
            print(f"  ✗ High backtrack count ({state.consecutive_backtracks})")
            print(f"    → State machine may be stuck in a loop")

    # Save detailed log
    log_path = out_dir / "step_log.json"
    with open(log_path, "w") as f:
        json.dump({
            "summary": {
                "final_stage": str(state.stage),
                "success": state.stage.name == 'COMPLETE',
                "total_steps": state.step,
                "total_measurements": state.total_measurements,
                "total_backtracks": state.total_backtracks,
            },
            "steps": step_log,
        }, f, indent=2)

    print(f"\nDetailed log saved to: {log_path}")
    print(f"Governance logs saved to: {out_dir}/governance/")


if __name__ == "__main__":
    main()
