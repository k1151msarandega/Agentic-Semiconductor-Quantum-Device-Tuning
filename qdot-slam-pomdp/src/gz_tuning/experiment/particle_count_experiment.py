import time
import numpy as np

from qarray_env import DeviceParams, QArrayEnv
from particle_init import SharedPriorSpec, CouplingStratum, init_particle_filter
from noise_model import GaussianReadoutNoise
from convergence import ConvergenceMonitor
from run_active_slam import run_active_slam

TRUE_VEC = np.array([2.1, 2.1, 0.06, 0.06, 0.05])  # E_c1, E_c2, alpha1, alpha2, coupling

def build_ground_truth():
    return DeviceParams(
        E_c1=2.1, E_c2=2.1, alpha1=0.06, alpha2=0.06,
        E_cm_intra1=0.3, E_cm_intra2=0.3, cross_capacitance=0.05,
    )

def run_for_n_particles(n_particles, seed):
    rng = np.random.default_rng(seed)
    ground_truth_params = build_ground_truth()
    ground_truth = QArrayEnv(ground_truth_params)

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

    pf = init_particle_filter(
        n_particles=n_particles, shared_prior=shared_prior,
        coupling_strata=coupling_strata, prior_sigma_diag=prior_sigma_diag,
        E_cm_intra1=0.3, E_cm_intra2=0.3, rng=rng,
    )

    noise = GaussianReadoutNoise(sigma=0.05)
    monitor = ConvergenceMonitor(
        p_conf=0.9, epsilon=0.05, N=5,
        target_occupation=np.array([1.0, 1.0, 1.0, 1.0]),
        require_actuation_quiescence=True,
    )

    t0 = time.time()
    result = run_active_slam(
        pf, ground_truth, noise, monitor,
        w_discrete=1.0, w_continuous=1.0,
        vg_range=(0.0, 15.0), n_candidates=8, max_steps=15,
        run_phase0=True, phase0_n_points_per_line=12,
        phase0_background_fractions=(1/3, 2/3),
        rng=rng, verbose=False,
    )
    elapsed = time.time() - t0

    final_mu = pf.weighted_mean_mu()
    rel_err = np.abs((final_mu - TRUE_VEC) / TRUE_VEC)
    ess_trace = np.array([h.ess for h in result.history])
    ig_trace = np.array([h.best_ig for h in result.history])
    n_resamples = sum(1 for h in result.history if h.resampled)

    return {
        "n_particles": n_particles,
        "elapsed_s": elapsed,
        "n_phase0_measurements": result.coarse_sweep.n_measurements if result.coarse_sweep else 0,
        "n_transitions_found": result.coarse_sweep.n_transitions_found if result.coarse_sweep else 0,
        "final_mu": final_mu,
        "rel_err_pct": rel_err * 100,
        "mean_rel_err_pct": float(np.mean(rel_err) * 100),
        "ess_min": float(ess_trace.min()),
        "ess_mean": float(ess_trace.mean()),
        "n_resamples": n_resamples,
        "ig_mean": float(ig_trace.mean()),
        "ig_zero_frac": float(np.mean(ig_trace < 1e-6)),
        "converged": result.converged,
        "within_cov_final": pf.within_particle_covariance(),
        "between_cov_final": pf.between_particle_covariance(),
    }

if __name__ == "__main__":
    for n in [10, 20, 40]:
        r = run_for_n_particles(n, seed=7)
        print(f"\n=== n_particles={n} ===")
        for k, v in r.items():
            print(f"  {k}: {v}")
