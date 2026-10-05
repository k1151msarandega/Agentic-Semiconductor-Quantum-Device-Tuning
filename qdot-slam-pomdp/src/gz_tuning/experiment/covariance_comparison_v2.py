"""
Active (two-dial: BALD+FIM) vs raster-scan, compared on belief convergence
directly (within-particle covariance vs measurement count) rather than a
target-occupation-reached criterion.

Why: target-occupation convergence is evaluated at whatever voltage was
just measured, but the active policy chases INFORMATION (near transition
boundaries), not a fixed target location -- the two objectives pull in
different directions by design (no "go park at target" term exists in the
acquisition function). Belief covariance sidesteps this: it directly
answers "how many measurements to characterize the device," which is the
efficiency question Claim 1 is actually about, without needing the policy
to have also navigated anywhere in particular.
"""
from __future__ import annotations

import time
import numpy as np

from qarray_env import DeviceParams, QArrayEnv, N_GATE
from particle_init import SharedPriorSpec, CouplingStratum, init_particle_filter
from noise_model import GaussianReadoutNoise
from candidate_search import select_next_measurement
from coarse_sweep import run_coarse_sweep
from raster_scan import build_raster_grid

GROUND_TRUTH_PARAMS = DeviceParams(
    E_c1=2.1, E_c2=2.1, alpha1=0.06, alpha2=0.06,
    E_cm_intra1=0.3, E_cm_intra2=0.3, cross_capacitance=0.05,
)
VG_RANGE = (0.0, 15.0)
N_PARTICLES = 10
N_SEEDS = 6
N_STEPS = 40          # measurements AFTER phase0, matched between methods
N_CANDIDATES = 20
T = 0.05


def build_pf(rng):
    shared_prior = SharedPriorSpec(
        mean={"E_c1": 2.0, "E_c2": 2.0, "alpha1": 0.06, "alpha2": 0.06},
        std={"E_c1": 0.3, "E_c2": 0.3, "alpha1": 0.015, "alpha2": 0.015},
    )
    coupling_strata = [
        CouplingStratum(mean=0.0, std=0.01, weight=0.34),
        CouplingStratum(mean=0.05, std=0.02, weight=0.33),
        CouplingStratum(mean=0.12, std=0.03, weight=0.33),
    ]
    prior_sigma_diag = np.array([0.09, 0.09, 0.000225, 0.000225, 0.0009])
    return init_particle_filter(
        n_particles=N_PARTICLES, shared_prior=shared_prior,
        coupling_strata=coupling_strata, prior_sigma_diag=prior_sigma_diag,
        E_cm_intra1=0.3, E_cm_intra2=0.3, rng=rng,
    )


def run_active_trace(seed):
    rng = np.random.default_rng(seed)
    pf = build_pf(rng)
    ground_truth = QArrayEnv(GROUND_TRUTH_PARAMS)
    noise = GaussianReadoutNoise(sigma=0.05)
    R = noise.R_matrix()

    coarse = run_coarse_sweep(pf, ground_truth, noise, vg_range=VG_RANGE,
                               n_points_per_line=10, background_fractions=(1/3, 2/3),
                               T=T, rng=rng, verbose=False)
    seed_points = coarse.seed_points if coarse.n_transitions_found > 0 else None

    n_meas = [coarse.n_measurements]
    within_cov = [pf.within_particle_covariance()]

    for step in range(N_STEPS):
        search = select_next_measurement(
            pf, R, w_discrete=1.0, w_continuous=1.0, vg_range=VG_RANGE,
            n_candidates=N_CANDIDATES, T=T, seed_points=seed_points,
            seed_fraction=0.5, seed_jitter_sigma=0.3, rng=rng,
        )
        v_next = search.best_vg
        true_reading = ground_truth.soft_prediction(v_next, T=T)
        measured = noise.sample(true_reading, rng)
        pf.update(measured, v_next, R, T=T, envs=search.envs, target_ess_frac=0.5)

        ess = pf.effective_sample_size()
        if ess < 0.5 * N_PARTICLES:
            pf.resample(rng, mu_jitter_scale=0.1)

        n_meas.append(n_meas[-1] + 1)
        within_cov.append(pf.within_particle_covariance())

    return np.array(n_meas), np.array(within_cov)


def run_raster_trace(seed):
    rng = np.random.default_rng(seed)
    pf = build_pf(rng)
    ground_truth = QArrayEnv(GROUND_TRUTH_PARAMS)
    noise = GaussianReadoutNoise(sigma=0.05)
    R = noise.R_matrix()

    # Same total measurement budget: coarse-sweep-equivalent + N_STEPS more,
    # taken from a fixed alternating grid (no belief-informed choice at all).
    coarse = run_coarse_sweep(pf, ground_truth, noise, vg_range=VG_RANGE,
                               n_points_per_line=10, background_fractions=(1/3, 2/3),
                               T=T, rng=rng, verbose=False)
    n_meas = [coarse.n_measurements]
    within_cov = [pf.within_particle_covariance()]

    grid = build_raster_grid(VG_RANGE, n_points_per_gate=20, mode="alternating", n_alternations=6)
    for i in range(N_STEPS):
        vg = grid[i % len(grid)]
        true_reading = ground_truth.soft_prediction(vg, T=T)
        measured = noise.sample(true_reading, rng)
        pf.update(measured, vg, R, T=T, target_ess_frac=0.5)
        n_meas.append(n_meas[-1] + 1)
        within_cov.append(pf.within_particle_covariance())

    return np.array(n_meas), np.array(within_cov)


if __name__ == "__main__":
    import json, sys, os

    seeds = [int(s) for s in sys.argv[1:]] if len(sys.argv) > 1 else list(range(N_SEEDS))
    out_path = "/home/claude/gz/stage1_T0.05_oldrange.json"
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
    else:
        results = {"active": {}, "raster": {}}  # keyed by str(seed) -- reruns overwrite, never duplicate

    for seed in seeds:
        t0 = time.time()
        n_a, cov_a = run_active_trace(seed)
        n_r, cov_r = run_raster_trace(seed)
        results["active"][str(seed)] = cov_a.tolist()
        results["raster"][str(seed)] = cov_r.tolist()
        print(f"seed {seed}: active final within_cov={cov_a[-1]:.2f}  "
              f"raster final within_cov={cov_r[-1]:.2f}  [{time.time()-t0:.1f}s]", flush=True)
        results["n_meas"] = n_a.tolist()
        with open(out_path, "w") as f:
            json.dump(results, f)
    print(f"Saved. seeds present: {sorted(int(k) for k in results['active'].keys())}")
