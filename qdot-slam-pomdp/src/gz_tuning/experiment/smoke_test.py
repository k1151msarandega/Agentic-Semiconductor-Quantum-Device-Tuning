import time
import numpy as np

from qarray_env import DeviceParams, QArrayEnv
from particle_init import SharedPriorSpec, CouplingStratum, init_particle_filter
from noise_model import GaussianReadoutNoise
from convergence import ConvergenceMonitor
from run_active_slam import run_active_slam

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

# Ground-zero prior: diffuse, centered away from truth (we are NOT
# supposed to know E_c/alpha/coupling in advance -- ground-zero claim).
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
monitor = ConvergenceMonitor(
    p_conf=0.9, epsilon=0.05, N=5,
    target_occupation=np.array([1.0, 1.0, 1.0, 1.0]),
    require_actuation_quiescence=True,
)

t0 = time.time()
result = run_active_slam(
    pf, ground_truth, noise, monitor,
    w_discrete=1.0, w_continuous=1.0,
    vg_range=(0.0, 15.0), n_candidates=10, max_steps=20,
    run_phase0=True, phase0_n_points_per_line=25,
    rng=rng, verbose=True,
)
if result.coarse_sweep is not None:
    print(f"\nPhase 0: {result.coarse_sweep.n_measurements} measurements, "
          f"{result.coarse_sweep.n_transitions_found} transitions found")
elapsed = time.time() - t0
print(f"\nDone in {elapsed:.1f}s ({elapsed/len(result.history):.3f}s/step)")
print(f"converged={result.converged}  n_measurements={result.n_measurements}")
print(f"final mu (weighted mean, true=[2.1,2.1,0.06,0.06,0.05]):")
print(pf.weighted_mean_mu())
