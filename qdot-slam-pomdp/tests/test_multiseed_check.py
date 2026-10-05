import time

import numpy as np

from qarray_env import DeviceParams, QArrayEnv
from particle_init import SharedPriorSpec, CouplingStratum, init_particle_filter
from noise_model import (
    GaussianReadoutNoise,
    calibrate_red_threshold,
    calibrate_threshold_low,
    derive_phase_thresholds,
)
from convergence import ConvergenceMonitor
from phase_switch import Phase, PhaseSwitchController
from run_active_slam import run_active_slam

T = 100.0

E_cm_intra1 = 0.3
E_cm_intra2 = 0.3
ground_truth_params = DeviceParams(
    E_c1=2.1, E_c2=2.1, alpha1=0.06, alpha2=0.06,
    E_cm_intra1=E_cm_intra1, E_cm_intra2=E_cm_intra2,
    cross_capacitance=0.05,
)
shared_prior = SharedPriorSpec(
    mean={"E_c1": 2.0, "E_c2": 2.0, "alpha1": 0.06, "alpha2": 0.06},
    std={"E_c1": 0.3, "E_c2": 0.3, "alpha1": 0.015, "alpha2": 0.015},
)
coupling_strata = [
    CouplingStratum(mean=0.0, std=0.01, weight=0.34),
    CouplingStratum(mean=0.05, std=0.02, weight=0.33),
    CouplingStratum(mean=0.12, std=0.03, weight=0.33),
]
prior_mu = np.array([2.0, 2.0, 0.06, 0.06, 0.05])
prior_sigma_diag = np.array([0.09, 0.09, 0.000225, 0.000225, 0.0009])

SEEDS = [0, 1, 2, 3, 4]
results = []

for seed in SEEDS:
    t_start = time.time()
    rng = np.random.default_rng(seed)

    ground_truth = QArrayEnv(ground_truth_params)
    pf = init_particle_filter(
        n_particles=15, shared_prior=shared_prior, coupling_strata=coupling_strata,
        prior_sigma_diag=prior_sigma_diag, E_cm_intra1=E_cm_intra1, E_cm_intra2=E_cm_intra2,
        rng=rng,
    )
    noise = GaussianReadoutNoise(sigma=0.05)

    blue_low = calibrate_threshold_low(
        prior_mu=prior_mu, prior_Sigma=np.diag(prior_sigma_diag),
        ground_truth_params=ground_truth_params,
        n_measurements=40, n_candidates_per_step=10, R_sigma=0.05, T=T, rng=rng,
    )
    blue_threshold_low, blue_threshold_high = derive_phase_thresholds(blue_low)

    red_low = calibrate_red_threshold(
        shared_prior=shared_prior, coupling_strata=coupling_strata,
        prior_sigma_diag=prior_sigma_diag, ground_truth_params=ground_truth_params,
        n_particles=pf.n_particles, n_measurements=40, n_candidates_per_step=10,
        R_sigma=0.05, T=T, rng=rng,
    )
    red_threshold_low, red_threshold_high = derive_phase_thresholds(red_low)

    phase_controller = PhaseSwitchController(
        blue_threshold_low=blue_threshold_low, blue_threshold_high=blue_threshold_high,
        red_threshold_low=red_threshold_low, red_threshold_high=red_threshold_high,
        mass_fraction_required=0.9,
    )
    monitor = ConvergenceMonitor(
        p_conf=0.9, epsilon=0.05, N=5,
        target_occupation=np.array([1.0, 1.0, 1.0, 1.0]),
        require_actuation_quiescence=True, T=T,
    )

    result = run_active_slam(
        pf, ground_truth, noise, monitor, phase_controller,
        w_discrete=1.0, w_continuous=1.0,
        vg_range=(0.0, 15.0), n_candidates=10, max_steps=60, T=T,
        forced_resample_period=10,
        run_phase0=True, phase0_n_points_per_line=25,
        rng=rng, verbose=False,
    )

    phase_2_steps = [h for h in result.history if h.phase is Phase.PHASE_2_FEATURE_BASED]
    final_red = result.history[-1].between_particle_logdet if result.history else None
    elapsed = time.time() - t_start

    row = {
        "seed": seed,
        "red_threshold_low": red_threshold_low,
        "phase2_reached": len(phase_2_steps) > 0,
        "phase2_first_step": phase_2_steps[0].step if phase_2_steps else None,
        "n_steps_in_phase2": len(phase_2_steps),
        "final_red": final_red,
        "converged": result.converged,
        "elapsed_s": elapsed,
    }
    results.append(row)
    print(f"seed={seed}  phase2_reached={row['phase2_reached']}  "
          f"first_step={row['phase2_first_step']}  "
          f"n_in_phase2={row['n_steps_in_phase2']:2d}/60  "
          f"final_red={final_red:7.3f}  red_thresh_low={red_threshold_low:7.3f}  "
          f"converged={row['converged']}  ({elapsed:.1f}s)")

n_reached = sum(1 for r in results if r["phase2_reached"])
print(f"\n{n_reached}/{len(SEEDS)} seeds reached Phase 2.")
if n_reached < len(SEEDS):
    failed = [r["seed"] for r in results if not r["phase2_reached"]]
    print(f"Did NOT reach Phase 2: seeds {failed} -- not uniform, worth looking at these specifically.")
