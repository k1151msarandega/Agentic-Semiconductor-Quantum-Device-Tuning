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

# T=100 throughout, passed explicitly everywhere below rather than relying
# on any function's default. Most signatures in this codebase still default
# to T=0.05, a leftover from before the primer's temperature calibration
# fix -- ~2000x too cold for a real device. This run exists specifically to
# exercise the corrected, calibrated path end to end, so nothing here is
# allowed to silently fall back to the old default.
T = 100.0

rng = np.random.default_rng(0)

# Ground truth -- Primer Section 4b's resolved feasible values.
E_cm_intra1 = 0.3
E_cm_intra2 = 0.3
ground_truth_params = DeviceParams(
    E_c1=2.1, E_c2=2.1, alpha1=0.06, alpha2=0.06,
    E_cm_intra1=E_cm_intra1, E_cm_intra2=E_cm_intra2,
    cross_capacitance=0.05,
)
ground_truth = QArrayEnv(ground_truth_params)
print("Ground truth constructed OK.")

# Ground-zero prior: diffuse, centered away from truth (we are NOT supposed
# to know E_c/alpha/coupling in advance -- ground-zero claim).
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

t0 = time.time()
pf = init_particle_filter(
    n_particles=15,
    shared_prior=shared_prior,
    coupling_strata=coupling_strata,
    prior_sigma_diag=prior_sigma_diag,
    E_cm_intra1=E_cm_intra1, E_cm_intra2=E_cm_intra2,
    rng=rng,
)
print(f"Particle filter init OK: {pf.n_particles} particles in {time.time()-t0:.2f}s")

noise = GaussianReadoutNoise(sigma=0.05)

# --- Calibrate the Phase-1/Phase-2 hysteresis thresholds ---
# calibrate_threshold_low defaults to n_measurements=300, n_candidates_
# per_step=30 (9000 Jacobian evals) -- trimmed here to the 40x10 scale
# already validated elsewhere in this project, because this is a SMOKE
# run, meant to confirm the Phase-2 gate fires and the goal-directed
# branch runs at all, not to produce a threshold worth treating as
# evidence. A real calibration pass should use the untrimmed defaults (or
# larger) before any downstream result is reported as a claim. This also
# now runs through the fixed jax_observation_jacobian (see the convergence
# .py/noise_model.py patch) rather than the finite-difference Jacobian it
# used before that fix -- so a threshold computed before that patch is not
# comparable to one computed after it, which is part of why this is
# recalibrated fresh here rather than reusing any previously-logged value.
print("\nCalibrating phase-switch thresholds (blue and red SEPARATELY -- "
      "see phase_switch.py's module docstring for why sharing one "
      "threshold between them was a real, confirmed bug)...")
t0 = time.time()
blue_low = calibrate_threshold_low(
    prior_mu=prior_mu, prior_Sigma=np.diag(prior_sigma_diag),
    ground_truth_params=ground_truth_params,
    n_measurements=40, n_candidates_per_step=10,
    R_sigma=0.05, T=T, rng=rng,
)
blue_threshold_low, blue_threshold_high = derive_phase_thresholds(blue_low)
print(f"blue: threshold_low={blue_threshold_low:.4f}  "
      f"threshold_high={blue_threshold_high:.4f}  ({time.time()-t0:.2f}s)")

t0 = time.time()
red_low = calibrate_red_threshold(
    shared_prior=shared_prior, coupling_strata=coupling_strata,
    prior_sigma_diag=prior_sigma_diag, ground_truth_params=ground_truth_params,
    n_particles=pf.n_particles, n_measurements=40, n_candidates_per_step=10,
    R_sigma=0.05, T=T, rng=rng,
)
red_threshold_low, red_threshold_high = derive_phase_thresholds(red_low)
print(f"red:  threshold_low={red_threshold_low:.4f}  "
      f"threshold_high={red_threshold_high:.4f}  ({time.time()-t0:.2f}s)")

phase_controller = PhaseSwitchController(
    blue_threshold_low=blue_threshold_low, blue_threshold_high=blue_threshold_high,
    red_threshold_low=red_threshold_low, red_threshold_high=red_threshold_high,
    mass_fraction_required=0.9,
)

monitor = ConvergenceMonitor(
    p_conf=0.9, epsilon=0.05, N=5,
    target_occupation=np.array([1.0, 1.0, 1.0, 1.0]),
    require_actuation_quiescence=True,
    T=T,
)

t0 = time.time()
result = run_active_slam(
    pf, ground_truth, noise, monitor, phase_controller,
    w_discrete=1.0, w_continuous=1.0,
    vg_range=(0.0, 15.0), n_candidates=10, max_steps=60,
    T=T,
    forced_resample_period=10,  # 6 forced resamples in 60 steps -- see below
    run_phase0=True, phase0_n_points_per_line=25,
    rng=rng, verbose=True,
)
elapsed = time.time() - t0

if result.coarse_sweep is not None:
    cs = result.coarse_sweep
    print(f"\nPhase 0: {cs.n_measurements} measurements, "
          f"{cs.n_transitions_found} transitions found")
    if cs.beta_trace.size > 0:
        print(f"Phase 0 beta: mean={cs.beta_trace.mean():.4f}  "
              f"median={np.median(cs.beta_trace):.4f}  "
              f"frac>=0.99={(cs.beta_trace >= 0.99).mean():.3f}  "
              f"frac<0.01={(cs.beta_trace < 0.01).mean():.3f}")

# --- IG/beta correlation check (Chat IV's hypothesis test) ---
phase1_steps = [h for h in result.history if h.best_ig_bald is not None]
if len(phase1_steps) >= 2:
    ig_total = np.array([h.best_ig for h in phase1_steps])
    ig_bald = np.array([h.best_ig_bald for h in phase1_steps])
    beta = np.array([h.beta for h in phase1_steps])
    print(f"\nPhase 1 beta: mean={beta.mean():.4f}  median={np.median(beta):.4f}  "
          f"frac<0.01={(beta < 0.01).mean():.3f}")
    print(f"corr(IG_total, beta) = {np.corrcoef(ig_total, beta)[0,1]:.4f}")
    print(f"corr(IG_BALD,  beta) = {np.corrcoef(ig_bald, beta)[0,1]:.4f}")

print(f"\nDone in {elapsed:.1f}s ({elapsed/len(result.history):.3f}s/step)")
print(f"converged={result.converged}  n_measurements={result.n_measurements}")

# --- Did the forced resample actually move red_line? The one check that
# tells us this addresses the plateau rather than just adding noise.
#
# NOT measured as an immediate single-step before/after delta around each
# resample: mu_jitter_scale deliberately widens particle positions right
# at the resample instant, so the one-step delta is expected to be noisy
# or even transiently positive (less converged) regardless of whether the
# fix works -- that's resampling doing its job, not a sign of failure.
# The real signal is the TREND over the steps following the first forced
# resample, where ordinary per-particle Kalman updates (which run every
# step regardless of resampling) get to act on a freshly-diversified,
# no-longer-stuck population. ---
forced_steps = [h for h in result.history if h.resample_reason == "forced"]
if forced_steps:
    red_by_step = {h.step: h.between_particle_logdet for h in result.history}
    first_forced_step = forced_steps[0].step
    pre = red_by_step.get(first_forced_step - 1)
    post_window = [
        red_by_step[s] for s in range(first_forced_step, first_forced_step + 15)
        if s in red_by_step
    ]
    print(f"\n{len(forced_steps)} forced resamples fired, first at step "
          f"{first_forced_step}.")
    if pre is not None and post_window:
        best_post = min(post_window)  # most negative = most converged
        print(f"red_line just before first forced resample: {pre:.3f}")
        print(f"red_line over the next {len(post_window)} steps: "
              f"{post_window[0]:.3f} -> {post_window[-1]:.3f} "
              f"(most converged: {best_post:.3f})")
        moved = best_post < pre - 1.0
        print(f"{'MOVING -- fix addresses the plateau' if moved else 'NOT CLEARLY MOVING -- reconsider, per the handoff note'}")
else:
    print("\nNo forced resamples fired -- forced_resample_period/max_steps mismatch, check config.")

# --- Did Phase 2 actually trigger? This is the point of this run. ---
phase_2_steps = [h for h in result.history if h.phase is Phase.PHASE_2_FEATURE_BASED]
if phase_2_steps:
    first = phase_2_steps[0]
    champion_mu = (
        phase_controller.champion.kalman.mu
        if phase_controller.champion is not None else None
    )
    print(f"\nPhase 2 REACHED at step {first.step} "
          f"({len(phase_2_steps)}/{len(result.history)} steps spent in Phase 2).")
    print(f"Champion particle's mu at end of run: {champion_mu}")
    first_goal_step = next(h for h in phase_2_steps if h.best_p_target is not None)
    print(f"First goal-directed step's best_p_target: {first_goal_step.best_p_target:.4f}")
else:
    print("\nPhase 2 NEVER REACHED in this run -- either max_steps is too "
          "small for this prior/threshold combination, or something in the "
          "Phase 1 -> 2 gate is wrong. This run does not confirm the "
          "goal-directed path works; treat as a fail, not a shrug.")

print(f"\nfinal mu (weighted mean, true=[2.1,2.1,0.06,0.06,0.05]):")
print(pf.weighted_mean_mu())
