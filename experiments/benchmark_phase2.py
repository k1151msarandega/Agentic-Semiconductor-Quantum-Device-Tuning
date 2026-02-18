"""
experiments/benchmark_phase2.py
================================
Phase 2 benchmark: 100 CIM simulation trials with full agent loop.

Validates:
    ≥50% measurement reduction vs 64×64 dense baseline (4096 points)
    ≥90% success rate (reaching target (1,1) charge state)

Usage:
    python experiments/benchmark_phase2.py --n-trials 100 --budget 2048
    python experiments/benchmark_phase2.py --fast  # 10 trials for quick testing
    python experiments/benchmark_phase2.py --skip-missing-checkpoints  # run without trained models

Outputs:
    - Summary report printed to stdout
    - Detailed per-trial logs saved to --out directory
    - CSV with metrics for each trial
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

# Phase 0 types and state
from qdot.core.types import ChargeLabel, TuningStage
from qdot.core.state import ExperimentState
from qdot.core.governance import GovernanceLogger
from qdot.core.hitl import HITLManager, HITLOutcome

# Phase 0 hardware
from qdot.simulator.cim import CIMSimulatorAdapter
from qdot.hardware.safety import SafetyCritic

# Phase 1 perception
from qdot.perception.dqc import DQCGatekeeper
from qdot.perception.inspector import InspectionAgent
from qdot.perception.classifier import EnsembleCNN
from qdot.perception.ood import MahalanobisOOD

# Phase 2 agent
from qdot.agent.executive import ExecutiveAgent


def main():
    parser = argparse.ArgumentParser(description="Phase 2 benchmark")
    parser.add_argument("--n-trials", type=int, default=100,
                        help="Number of simulation trials (default: 100)")
    parser.add_argument("--budget", type=int, default=2048,
                        help="Measurement budget per trial (default: 2048)")
    parser.add_argument("--max-steps", type=int, default=100,
                        help="Max control steps per trial (default: 100)")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: 10 trials, reduced budgets for CI")
    parser.add_argument("--profile", action="store_true",
                        help="Enable profiling to identify bottlenecks")
    parser.add_argument("--skip-missing-checkpoints", action="store_true",
                        help="Run without trained InspectionAgent (for CI)")
    parser.add_argument("--out", type=str, default="results/benchmark_phase2",
                        help="Output directory for detailed logs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.fast:
        args.n_trials = 10
        args.budget = 512
        args.max_steps = 50  # Needs to be high enough for full stage sequence
        print("FAST MODE: 10 trials, 512 budget, 50 max steps")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("PHASE 2 BENCHMARK — Agentic Tuning on CIM Simulator")
    print(f"{'='*70}\n")
    print(f"Trials:            {args.n_trials}")
    print(f"Measurement budget: {args.budget} points")
    print(f"Max steps:         {args.max_steps}")
    print(f"Target:            (1,1) charge state")
    print(f"Dense baseline:    64×64 = 4096 points")
    print(f"Reduction target:  ≥50% (≤2048 measurements)")
    print(f"Success target:    ≥90% of trials\n")

    # Load Phase 1 components
    inspector = load_inspector(args.skip_missing_checkpoints)

    # Start profiling if requested
    profiler = None
    if args.profile:
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()
        print("⚙️  Profiling enabled\n")

    # Run trials
    np.random.seed(args.seed)
    results = []
    for trial_idx in range(args.n_trials):
        print(f"[{trial_idx+1}/{args.n_trials}] ", end="", flush=True)
        result = run_trial(
            trial_idx=trial_idx,
            inspector=inspector,
            measurement_budget=args.budget,
            max_steps=args.max_steps,
            out_dir=out_dir,
        )
        results.append(result)
        status = "✓" if result["success"] else "✗"
        print(f"{status} {result['final_stage']} | {result['total_measurements']} meas | {result['total_steps']} steps")

    # Aggregate metrics
    summary = compute_summary(results, args)

    # Print report
    print_report(summary)

    # Save detailed results
    save_results(summary, results, out_dir)

    # Stop profiling and print results
    if profiler is not None:
        profiler.disable()
        import pstats
        print("\n" + "="*70)
        print("PROFILING RESULTS — Top 20 Time Sinks")
        print("="*70)
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(20)

    # Exit with appropriate code
    if summary["success_rate"] < 0.90:
        print(f"\n❌ BENCHMARK FAILED: success rate {summary['success_rate']:.1%} < 90%")
        return 1
    if summary["mean_reduction"] < 0.50:
        print(f"\n❌ BENCHMARK FAILED: mean reduction {summary['mean_reduction']:.1%} < 50%")
        return 1

    print("\n✅ PHASE 2 BENCHMARK PASSED")
    return 0


def load_inspector(skip_missing: bool) -> InspectionAgent:
    """Load trained InspectionAgent or create untrained stub if skipping."""
    checkpoint_dir = Path("experiments/checkpoints/phase1")

    if skip_missing or not checkpoint_dir.exists():
        print("⚠️  Running without trained checkpoints (InspectionAgent in stub mode)\n")
        return InspectionAgent(
            ensemble=None,      # Untrained — predictions will be random
            ood_detector=None,  # OOD detection skipped
        )

    # Load trained ensemble and OOD detector
    try:
        ensemble = EnsembleCNN.load(str(checkpoint_dir))
        ood = MahalanobisOOD.load(str(checkpoint_dir / "ood_detector.pkl"))
        print(f"✓ Loaded trained InspectionAgent from {checkpoint_dir}\n")
        return InspectionAgent(ensemble=ensemble, ood_detector=ood)
    except Exception as e:
        print(f"⚠️  Failed to load checkpoints: {e}")
        print("   Continuing with untrained InspectionAgent\n")
        return InspectionAgent(ensemble=None, ood_detector=None)


def run_trial(
    trial_idx: int,
    inspector: InspectionAgent,
    measurement_budget: int,
    max_steps: int,
    out_dir: Path,
) -> Dict[str, Any]:
    """Run a single tuning trial."""
    # Create fresh state
    device_id = f"cim_trial_{trial_idx:03d}"
    state = ExperimentState.new(
        device_id=device_id,
        target_label=ChargeLabel.DOUBLE_DOT,
    )
    state.config = {
        "measurement_budget": measurement_budget,
        "max_steps": max_steps,
        "trial_idx": trial_idx,
    }

    # Create device adapter with varying CIM parameters (test generalization)
    E_c_base = 3.0 + np.random.uniform(-0.5, 0.5)
    t_c_base = 0.3 + np.random.uniform(-0.1, 0.1)
    adapter = CIMSimulatorAdapter(
        device_id=device_id,
        params={
            "E_c1": E_c_base,
            "E_c2": E_c_base + 0.2,
            "t_c": t_c_base,
            "T": 0.08,
            "lever_arm": 0.55,
            "noise_level": 0.015,
        },
        seed=trial_idx + 1000,
    )

    # HITL in auto-approve test mode (no blocking)
    hitl = HITLManager(enabled=True)
    hitl.set_test_mode(auto_outcome=HITLOutcome.APPROVED)

    # Governance logger
    gov_log_dir = out_dir / "governance" / f"trial_{trial_idx:03d}"
    governance = GovernanceLogger(run_id=state.run_id, log_dir=str(gov_log_dir))

    # Create agent
    agent = ExecutiveAgent(
        state=state,
        adapter=adapter,
        inspection_agent=inspector,
        hitl_manager=hitl,
        governance_logger=governance,
        max_steps=max_steps,
        measurement_budget=measurement_budget,
    )

    # Run
    t_start = time.time()
    summary = agent.run()
    duration = time.time() - t_start

    # Add trial-specific info
    summary["trial_idx"] = trial_idx
    summary["duration_s"] = duration
    summary["device_params"] = {
        "E_c1": adapter.device.E_c1,
        "E_c2": adapter.device.E_c2,
        "t_c": adapter.device.t_c,
    }

    return summary


def compute_summary(results: List[Dict], args) -> Dict[str, Any]:
    """Aggregate trial results into summary statistics."""
    n = len(results)
    successes = sum(1 for r in results if r["success"])
    measurements = [r["total_measurements"] for r in results]
    steps = [r["total_steps"] for r in results]
    reductions = [r["measurement_reduction"] for r in results]
    backtracks = [r["total_backtracks"] for r in results]
    hitl_counts = [r["hitl_events"] for r in results]

    dense_baseline = 64 * 64  # 4096 points

    return {
        "n_trials": n,
        "success_rate": successes / n if n > 0 else 0.0,
        "mean_measurements": float(np.mean(measurements)),
        "std_measurements": float(np.std(measurements)),
        "mean_steps": float(np.mean(steps)),
        "mean_reduction": float(np.mean(reductions)),
        "median_reduction": float(np.median(reductions)),
        "min_reduction": float(np.min(reductions)),
        "max_reduction": float(np.max(reductions)),
        "mean_backtracks": float(np.mean(backtracks)),
        "mean_hitl": float(np.mean(hitl_counts)),
        "dense_baseline": dense_baseline,
        "measurement_budget": args.budget,
        "max_steps": args.max_steps,
        "targets": {
            "success_rate_min": 0.90,
            "reduction_min": 0.50,
        },
    }


def print_report(summary: Dict[str, Any]):
    """Print formatted summary report."""
    print(f"\n{'='*70}")
    print("BENCHMARK RESULTS")
    print(f"{'='*70}\n")

    success_pass = "✓" if summary["success_rate"] >= 0.90 else "✗"
    reduction_pass = "✓" if summary["mean_reduction"] >= 0.50 else "✗"

    print(f"Success rate:        {summary['success_rate']:>6.1%}  {success_pass}  (target ≥90%)")
    print(f"Mean reduction:      {summary['mean_reduction']:>6.1%}  {reduction_pass}  (target ≥50%)")
    print(f"Median reduction:    {summary['median_reduction']:>6.1%}")
    print(f"Reduction range:     [{summary['min_reduction']:.1%}, {summary['max_reduction']:.1%}]")
    print()
    print(f"Mean measurements:   {summary['mean_measurements']:>6.0f} ± {summary['std_measurements']:.0f}")
    print(f"Mean steps:          {summary['mean_steps']:>6.1f}")
    print(f"Mean backtracks:     {summary['mean_backtracks']:>6.1f}")
    print(f"Mean HITL triggers:  {summary['mean_hitl']:>6.1f}")
    print(f"\n{'='*70}")


def save_results(summary: Dict, results: List[Dict], out_dir: Path):
    """Save detailed results to disk."""
    # Summary JSON
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Per-trial CSV
    import csv
    with open(out_dir / "trials.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "trial_idx", "success", "final_stage", "total_measurements",
            "total_steps", "measurement_reduction", "total_backtracks",
            "hitl_events", "duration_s",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "trial_idx": r["trial_idx"],
                "success": r["success"],
                "final_stage": r["final_stage"],
                "total_measurements": r["total_measurements"],
                "total_steps": r["total_steps"],
                "measurement_reduction": r["measurement_reduction"],
                "total_backtracks": r["total_backtracks"],
                "hitl_events": r["hitl_events"],
                "duration_s": r.get("duration_s", 0.0),
            })

    print(f"\nDetailed results saved to: {out_dir}/")
    print(f"  - summary.json")
    print(f"  - trials.csv")
    print(f"  - governance/trial_XXX/*.jsonl")


if __name__ == "__main__":
    import sys
    sys.exit(main())
