import sys, copy
for d in ["acquisition", "belief", "simulator", "control"]:
    sys.path.insert(0, d)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kalman import ParticleKalmanFilter, N_PARAMS
from particle_filter import OccupationParticle, RBParticleFilter
from fim import compute_ig_fim
from qarray_env import DeviceParams, QArrayEnv, N_GATE
from noise_model import GaussianReadoutNoise
from phase_switch import PhaseSwitchController, Phase
from convergence import ConvergenceMonitor

DEFAULT_PARAMS = DeviceParams(
    E_c1=2.5, E_c2=2.7, alpha1=0.05, alpha2=0.05,
    E_cm_intra1=0.3, E_cm_intra2=0.3, cross_capacitance=0.05,
)

rng_init = np.random.default_rng(1)
particles = []
for _ in range(5):
    E_c1 = 2.5 + rng_init.normal(scale=0.35)
    alpha1 = 0.05 + rng_init.normal(scale=0.0035)
    mu = np.array([E_c1, 2.7, alpha1, 0.05, 0.05])
    Sigma = np.eye(N_PARAMS) * 0.05
    kf = ParticleKalmanFilter(mu=mu, Sigma=Sigma, E_cm_intra1=0.3, E_cm_intra2=0.3)
    particles.append(OccupationParticle(kalman=kf, weight=1.0))
pf = RBParticleFilter(particles=particles)

ground_truth = QArrayEnv(DEFAULT_PARAMS)
noise = GaussianReadoutNoise(sigma=np.sqrt(0.02))
R = np.eye(4) * 0.02
rng = np.random.default_rng(7)

# FIM-only candidate selection, restricted to the known-informative band
# found earlier (v~9.99 has real alpha-sensitivity) -- deliberately
# isolating "does the Kalman/particle machinery converge given informative
# measurements" from the separate, already-documented w(t)/stalling issue.
N_CAND = 15
cand_axis = np.linspace(0.0, 30.0, N_CAND)
# NOTE: an earlier version of this script reused v~9.99 as a
# known-informative candidate band -- that point was found via a
# DIFFERENT slice (all 4 gates swept together), not this script's
# slice (gate2=gate3 fixed at 0). Widened to the full domain already
# confirmed (via two_dial_landscape.png) to contain real, sparse FIM
# spikes in THIS specific slice.
Vc0, Vc1 = np.meshgrid(cand_axis, cand_axis, indexing="ij")

N_STEPS = 15
mean_mu_trace = []
within_cov_trace = []
between_cov_trace = []
chosen_vgs = []
pf_snapshots = []

initial_particle_cloud = np.array([[p.kalman.mu[0], p.kalman.mu[2]] for p in pf.particles])

print("Running FIM-driven belief convergence trajectory...")
for step in range(N_STEPS):
    fim_grid = np.zeros((N_CAND, N_CAND))
    for i in range(N_CAND):
        for j in range(N_CAND):
            vg = np.array([Vc0[i, j], Vc1[i, j], 0.0, 0.0])
            fim_grid[i, j] = compute_ig_fim(pf, vg, R)
    best = np.unravel_index(np.argmax(fim_grid), fim_grid.shape)
    vg_star = np.array([Vc0[best], Vc1[best], 0.0, 0.0])

    pf_snapshots.append(copy.deepcopy(pf))
    mean_mu_trace.append(pf.weighted_mean_mu().copy())
    within_cov_trace.append(pf.within_particle_covariance())
    between_cov_trace.append(pf.between_particle_covariance())
    chosen_vgs.append(vg_star.copy())

    true_reading = ground_truth.query(vg_star)
    measured = noise.sample(true_reading, rng)
    pf.update(measured, vg_star, R, T=0.05)

    print(f"  step {step+1}/{N_STEPS}: vg=({vg_star[0]:.2f},{vg_star[1]:.2f}), "
          f"max_FIM={fim_grid[best]:.4f}, E_c1_est={mean_mu_trace[-1][0]:.3f}, "
          f"alpha1_est={mean_mu_trace[-1][2]:.4f}, within_cov={within_cov_trace[-1]:.4f}")

final_particle_cloud = np.array([[p.kalman.mu[0], p.kalman.mu[2]] for p in pf.particles])
mean_mu_trace = np.array(mean_mu_trace)
chosen_vgs = np.array(chosen_vgs)

print(f"\nwithin_cov range: {min(within_cov_trace):.3f} to {max(within_cov_trace):.3f}")
print(f"between_cov range: {min(between_cov_trace):.3f} to {max(between_cov_trace):.3f}")

# ============================================================
# Threshold choice for the phase-switch/convergence replay,
# made AFTER seeing the real trajectory's actual covariance
# range (not guessed blind) -- cheap to replay against the
# stored deepcopy snapshots without re-running the FIM sweep.
# ============================================================
wc = np.array(within_cov_trace)
threshold_low = wc.min() + 0.25 * (wc.max() - wc.min())
threshold_high = wc.min() + 0.55 * (wc.max() - wc.min())
print(f"Chosen threshold_low={threshold_low:.3f}, threshold_high={threshold_high:.3f}")

controller = PhaseSwitchController(threshold_low=threshold_low, threshold_high=threshold_high,
                                    mass_fraction_required=0.9)
phase_trace = []
blue_frac_trace = []
red_line_trace = []
for snap in pf_snapshots:
    diag = controller.step(snap)
    phase_trace.append(controller.phase)
    blue_frac_trace.append(diag.blue_mass_fraction_below_low)
    red_line_trace.append(diag.red_line)

target_vg = np.array([11.0, 11.0, 0.0, 0.0])
target_occ = ground_truth.query(target_vg)
monitor = ConvergenceMonitor(p_conf=0.7, epsilon=2.0, N=3, target_occupation=target_occ)
belief_stable_trace = []
converged_trace = []
consecutive_trace = []
for k, snap in enumerate(pf_snapshots):
    step_vec = (chosen_vgs[k + 1] - chosen_vgs[k]) if k + 1 < len(chosen_vgs) else np.zeros(N_GATE)
    status = monitor.step(snap, chosen_vgs[k], step_vec)
    belief_stable_trace.append(status.belief_stable)
    converged_trace.append(status.converged)
    consecutive_trace.append(status.consecutive_count)

# ============================================================
# Figure
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
steps = np.arange(1, N_STEPS + 1)

axes[0, 0].scatter(initial_particle_cloud[:, 0], initial_particle_cloud[:, 1],
                    c="tab:red", label="initial particles", s=60, alpha=0.8)
axes[0, 0].scatter(final_particle_cloud[:, 0], final_particle_cloud[:, 1],
                    c="tab:blue", label="final particles", s=60, alpha=0.8)
axes[0, 0].axvline(DEFAULT_PARAMS.E_c1, color="black", linestyle="--", linewidth=1, label="true E_c1")
axes[0, 0].axhline(DEFAULT_PARAMS.alpha1, color="black", linestyle=":", linewidth=1, label="true alpha1")
axes[0, 0].set_xlabel("E_c1")
axes[0, 0].set_ylabel("alpha1")
axes[0, 0].set_title("Particle cloud: initial vs final")
axes[0, 0].legend(fontsize=7)

axes[0, 1].plot(steps, np.abs(mean_mu_trace[:, 0] - DEFAULT_PARAMS.E_c1), "-o", label="|E_c1 error|")
axes[0, 1].plot(steps, np.abs(mean_mu_trace[:, 2] - DEFAULT_PARAMS.alpha1), "-s", label="|alpha1 error|")
axes[0, 1].set_xlabel("step")
axes[0, 1].set_ylabel("|weighted-mean error|")
axes[0, 1].set_title("Belief error vs ground truth over the run")
axes[0, 1].legend(fontsize=8)
axes[0, 1].set_yscale("log")

axes[0, 2].plot(steps, within_cov_trace, "-o", label="within-particle (blue line, aggregate)")
finite_between = np.array(between_cov_trace)
if np.any(np.isfinite(finite_between)):
    axes[0, 2].plot(steps, finite_between, "-s", label="between-particle (red line)")
axes[0, 2].axhline(threshold_low, color="green", linestyle="--", linewidth=1, label="threshold_low")
axes[0, 2].axhline(threshold_high, color="orange", linestyle="--", linewidth=1, label="threshold_high")
axes[0, 2].set_xlabel("step")
axes[0, 2].set_ylabel("normalized logdet")
axes[0, 2].set_title("Covariance traces + phase-switch thresholds")
axes[0, 2].legend(fontsize=7)

phase_numeric = [1 if p is Phase.PHASE_2_FEATURE_BASED else 0 for p in phase_trace]
axes[1, 0].step(steps, phase_numeric, where="post", color="purple")
axes[1, 0].set_yticks([0, 1])
axes[1, 0].set_yticklabels(["PHASE_1", "PHASE_2"])
axes[1, 0].set_xlabel("step")
axes[1, 0].set_title("Phase-switch state over the run")

axes[1, 1].plot(steps, blue_frac_trace, "-o", label="blue mass fraction below threshold_low")
axes[1, 1].axhline(0.9, color="gray", linestyle="--", linewidth=1, label="mass_fraction_required")
axes[1, 1].set_xlabel("step")
axes[1, 1].set_ylabel("fraction")
axes[1, 1].set_title("Mass-based blue-line diagnostic")
axes[1, 1].legend(fontsize=7)

ax_guard = axes[1, 2]
ax_guard.step(steps, [1 if b else 0 for b in belief_stable_trace], where="post", label="belief_stable", color="tab:blue")
ax_guard.step(steps, [1 if c else 0 for c in converged_trace], where="post", label="converged", color="tab:red", linewidth=2)
ax_guard2 = ax_guard.twinx()
ax_guard2.plot(steps, consecutive_trace, "-o", color="tab:green", label="consecutive_count")
ax_guard.set_yticks([0, 1])
ax_guard.set_xlabel("step")
ax_guard.set_title(f"Convergence guards (target vg={target_vg[:2]}, target_occ={target_occ})")
lines1, labels1 = ax_guard.get_legend_handles_labels()
lines2, labels2 = ax_guard2.get_legend_handles_labels()
ax_guard.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="center left")

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/belief_convergence_and_guards.png", dpi=140)
plt.close()
print("Saved belief_convergence_and_guards.png")
