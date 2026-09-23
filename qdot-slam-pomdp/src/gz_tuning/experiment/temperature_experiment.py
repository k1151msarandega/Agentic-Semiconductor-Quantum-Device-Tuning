"""
Compares the project's original hardcoded T=0.05 against a physically
calibrated T, over a full addition-voltage period.

WHY T MATTERS AND WHY IT IS NOT A FREE TUNING KNOB:
The only dimensionless quantity governing thermal softening is
E_c / (kB*T). QArray computes kB_T = 8.617333262145e-5 * T internally
(eV/K * K), so E_c is interpreted in eV. This project picked E_c=2.1 as
an effectively dimensionless feasibility-driven number (Primer 4b) and
T=0.05 as if it meant "50 mK". Combined, those give
E_c/kB_T ~ 4.9e5 -- versus a REAL semiconductor dot (E_c ~ 1-5 meV at
50-300 mK) which sits at E_c/kB_T ~ 80-600, typically ~230.

At 4.9e5 the Boltzmann factor is exp(-5e5): the ground state is an exact
step function, the transition region is ~9e-16 V wide (machine epsilon),
and a measured 0.5% of uniformly-sampled candidates carry ANY usable
gradient. That is the real reason the acquisition landscape has been all
zeros-with-isolated-spikes, and it is upstream of both the Jacobian
precision issue and the candidate-proposal issue.

T is a DEVICE/EXPERIMENT property, not a knob to crank until the
algorithm converges -- raising it widens transitions and makes tuning
easier, but past a point it destroys the discrete charge structure the
POMDP is built on (at T=3000, transitions smear across ~37% of an
addition voltage). So T is set here to match a real device's
E_c/kB_T ratio (~230 -> T ~ 100 in this project's E_c units), NOT to
whatever maximizes measured performance.

Also corrected here: vg_range. The addition voltage is V_add = e/C_g =
1/alpha = 16.67 V, but the project's default vg_range was (0, 15) --
less than ONE full charge-stability period, so parts of the state space
were unreachable by construction.
"""

import time
import numpy as np

from qarray_env import DeviceParams, QArrayEnv
from particle_init import SharedPriorSpec, CouplingStratum, init_particle_filter
from noise_model import GaussianReadoutNoise
from convergence import ConvergenceMonitor
from run_active_slam import run_active_slam

TRUE_VEC = np.array([2.1, 2.1, 0.06, 0.06, 0.05])
V_ADD = 1.0 / 0.06  # = 16.67, one full addition voltage


def run_cfg(T, n_particles, max_steps, seed, vg_range):
    rng = np.random.default_rng(seed)
    gt_params = DeviceParams(
        E_c1=2.1, E_c2=2.1, alpha1=0.06, alpha2=0.06,
        E_cm_intra1=0.3, E_cm_intra2=0.3, cross_capacitance=0.05,
    )
    ground_truth = QArrayEnv(gt_params, T=T)

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
        T=T, require_actuation_quiescence=True,
    )

    t0 = time.time()
    result = run_active_slam(
        pf, ground_truth, noise, monitor,
        w_discrete=1.0, w_continuous=1.0,
        vg_range=vg_range, n_candidates=12, max_steps=max_steps,
        run_phase0=True, phase0_n_points_per_line=15,
        T=T, rng=rng, verbose=False,
    )
    elapsed = time.time() - t0

    final_mu = pf.weighted_mean_mu()
    rel_err = np.abs((final_mu - TRUE_VEC) / TRUE_VEC)
    ig = np.array([h.best_ig for h in result.history])
    return {
        "T": T,
        "n_particles": n_particles,
        "elapsed_s": round(elapsed, 1),
        "phase0_transitions": result.coarse_sweep.n_transitions_found if result.coarse_sweep else 0,
        "final_mu": np.round(final_mu, 4),
        "rel_err_pct": np.round(rel_err * 100, 2),
        "mean_rel_err_pct": round(float(np.mean(rel_err) * 100), 2),
        "ig_mean": round(float(ig.mean()), 4),
        "ig_zero_frac": round(float(np.mean(ig < 1e-9)), 3),
        "converged": result.converged,
    }


if __name__ == "__main__":
    print("true params:", TRUE_VEC, "\n")
    for T, label in [(0.05, "OLD (unphysical, E_c/kB_T~4.9e5)"),
                     (100.0, "CALIBRATED (E_c/kB_T~243, real-device-like)")]:
        r = run_cfg(T=T, n_particles=20, max_steps=25, seed=11, vg_range=(0.0, V_ADD))
        print(f"=== T={T}  {label} ===")
        for k, v in r.items():
            print(f"  {k}: {v}")
        print()
