"""
experiments/ablation_phase2.py
===============================
Ablation study for Phase 2 computational budgets.

Validates whether reducing n_particles and n_mc_samples affects:
- Success rate (reaching target state)
- Measurement efficiency (reduction vs dense baseline)
- Execution time

Usage:
    python experiments/ablation_phase2.py --n-trials 20 --out results/ablation

Runs experiments with different parameter combinations and compares results.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

# Phase 0
from qdot.core.types import ChargeLabel
from qdot.core.state import ExperimentState
from qdot.core.hitl import HITLManager, HITLOutcome
from qdot.core.governance import GovernanceLogger

# Hardware
from qdot.simulator.cim import CIMSimulatorAdapter

# Phase 1
from qdot.perception.inspector import InspectionAgent

# Phase 2
from qdot.agent.executive import ExecutiveAgent
from qdot.planning.belief import BeliefUpdater
from qdot.planning.sensing import ActiveSensingPolicy


def main():
    parser = argparse.ArgumentParser(description="Phase 2 ablation study")
    parser.add_argument("--n-trials", type=int, default=20,
                        help="Trials per configuration (default: 20)")
    parser.add_argument("--budget", type=int, default=1024,
                        help="Measurement budget per trial")
    parser.add_argument("--max-steps", type=int, default=50,
                        help="Max steps per trial")
    parser.add_argument("--out", type=str, default="results/ablation",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("PHASE 2 ABLATION STUDY — Computational Budget Validation")
    print("="*70 + "\n")
    print(f"Trials per config: {args.n_trials}")
    print(f"Budget: {args.budget} points")
    print(f"Max steps: {args.max_steps}\n")

    # Configurations to test
    configs = [
        {"name": "baseline", "n_particles": 1000, "n_mc": 8},
        {"name": "reduced_particles", "n_particles": 500, "n_mc": 8},
        {"name": "reduced_mc", "n_particles": 1000, "n_mc": 4},
        {"name": "both_reduced", "n_particles": 500, "n_mc": 4},
    ]

    results_by_config = {}
    np.random.seed(args.seed)

    for config in configs:
        print(f"\n{'='*70}")
        print(f"CONFIG: {config['name']} (particles={config['n_particles']}, mc={config['n_mc']})")
        print(f"{'='*70}\n")

        config_results = []
        for trial_idx in range(args.n_trials):
            print(f"[{trial_idx+1}/{args.n_trials}] ", end="", flush=True)
            
            result = run_trial(
                trial_idx=trial_idx,
                n_particles=config["n_particles"],
                n_mc_samples=config["n_mc"],
                budget=args.budget,
                max_steps=args.max_steps,
                out_dir=out_dir / config["name"],
            )
            config_results.append(result)
            
            status = "✓" if result["success"] else "✗"
            print(f"{status} {result['final_stage']} | {result['total_measurements']} meas | {result['duration_s']:.1f}s")

        results_by_config[config["name"]] = config_results

    # Analyze and compare
    summary = analyze_results(results_by_config, args)
    
    # Print comparison
    print_comparison(summary)
    
    # Save results
    with open(out_dir / "ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {out_dir}/ablation_summary.json")


def run_trial(
    trial_idx: int,
    n_particles: int,
    n_mc_samples: int,
    budget: int,
    max_steps: int,
    out_dir: Path,
) -> Dict[str, Any]:
    """Run a single trial with specified parameters."""
    device_id = f"ablation_trial_{trial_idx:03d}"
    state = ExperimentState.new(device_id=device_id, target_label=ChargeLabel.DOUBLE_DOT)
    
    adapter = CIMSimulatorAdapter(
        device_id=device_id,
        params={
            "E_c1": 3.0 + np.random.uniform(-0.3, 0.3),
            "E_c2": 3.2 + np.random.uniform(-0.3, 0.3),
            "t_c": 0.3 + np.random.uniform(-0.05, 0.05),
        },
        seed=trial_idx + 1000,
    )
    
    # Untrained inspector (for ablation, we just test planning/agent)
    inspector = InspectionAgent(ensemble=None, ood_detector=None)
    
    hitl = HITLManager(enabled=True)
    hitl.set_test_mode(auto_outcome=HITLOutcome.APPROVED)
    
    gov_log_dir = out_dir / "governance" / f"trial_{trial_idx:03d}"
    governance = GovernanceLogger(run_id=state.run_id, log_dir=str(gov_log_dir))
    
    # Create agent with custom budgets
    agent = ExecutiveAgent(
        state=state,
        adapter=adapter,
        inspection_agent=inspector,
        hitl_manager=hitl,
        governance_logger=governance,
        max_steps=max_steps,
        measurement_budget=budget,
    )
    
    # Override planning component budgets
    agent.belief_updater = BeliefUpdater(
        belief=state.belief,
        n_particles=n_particles,
    )
    agent.sensing_policy = ActiveSensingPolicy(
        n_mc_samples=n_mc_samples,
    )
    
    t_start = time.time()
    summary = agent.run()
    duration = time.time() - t_start
    
    summary["trial_idx"] = trial_idx
    summary["duration_s"] = duration
    summary["n_particles"] = n_particles
    summary["n_mc_samples"] = n_mc_samples
    
    return summary


def analyze_results(results_by_config: Dict[str, List[Dict]], args) -> Dict:
    """Aggregate and compare results across configurations."""
    summary = {}
    
    for config_name, results in results_by_config.items():
        n = len(results)
        successes = sum(1 for r in results if r["success"])
        measurements = [r["total_measurements"] for r in results]
        reductions = [r["measurement_reduction"] for r in results]
        durations = [r["duration_s"] for r in results]
        
        summary[config_name] = {
            "success_rate": successes / n if n > 0 else 0.0,
            "mean_measurements": float(np.mean(measurements)),
            "std_measurements": float(np.std(measurements)),
            "mean_reduction": float(np.mean(reductions)),
            "std_reduction": float(np.std(reductions)),
            "mean_duration": float(np.mean(durations)),
            "std_duration": float(np.std(durations)),
            "speedup_vs_baseline": None,  # Computed below
        }
    
    # Compute speedups
    baseline_duration = summary["baseline"]["mean_duration"]
    for config_name in summary:
        if config_name != "baseline":
            speedup = baseline_duration / summary[config_name]["mean_duration"]
            summary[config_name]["speedup_vs_baseline"] = speedup
    
    return summary


def print_comparison(summary: Dict):
    """Print formatted comparison table."""
    print("\n" + "="*70)
    print("ABLATION RESULTS")
    print("="*70 + "\n")
    
    configs = ["baseline", "reduced_particles", "reduced_mc", "both_reduced"]
    
    print(f"{'Config':<20} {'Success%':<10} {'Reduction%':<12} {'Duration(s)':<12} {'Speedup':<10}")
    print("-" * 70)
    
    for config in configs:
        if config not in summary:
            continue
        s = summary[config]
        speedup = f"{s['speedup_vs_baseline']:.2f}x" if s['speedup_vs_baseline'] else "-"
        print(f"{config:<20} {s['success_rate']:>7.1%}   {s['mean_reduction']:>7.1%} ± {s['std_reduction']:.1%}  "
              f"{s['mean_duration']:>6.1f} ± {s['std_duration']:.1f}  {speedup:>8}")
    
    print("\n" + "="*70)
    
    # Statistical significance tests
    print("\nKEY FINDINGS:\n")
    
    baseline = summary["baseline"]
    
    for config in ["reduced_particles", "reduced_mc", "both_reduced"]:
        if config not in summary:
            continue
        s = summary[config]
        
        # Success rate difference
        success_diff = abs(s["success_rate"] - baseline["success_rate"])
        # Reduction difference
        reduction_diff = abs(s["mean_reduction"] - baseline["mean_reduction"])
        
        print(f"{config}:")
        if success_diff < 0.05 and reduction_diff < 0.05:
            print(f"  ✓ Performance equivalent to baseline (Δsuccess={success_diff:.1%}, Δreduction={reduction_diff:.1%})")
            print(f"    → Safe to use for {s['speedup_vs_baseline']:.1f}× speedup")
        else:
            print(f"  ✗ Performance differs from baseline (Δsuccess={success_diff:.1%}, Δreduction={reduction_diff:.1%})")
            print(f"    → Not recommended despite {s['speedup_vs_baseline']:.1f}× speedup")
        print()
    
    print("="*70)


if __name__ == "__main__":
    main()
