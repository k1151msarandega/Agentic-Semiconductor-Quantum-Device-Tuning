import sys
for d in ["acquisition", "belief", "simulator"]:
    sys.path.insert(0, d)

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kalman import ParticleKalmanFilter, N_PARAMS, params_to_vector
from particle_filter import OccupationParticle, RBParticleFilter
from bald import compute_ig_bald
from fim import compute_ig_fim
from two_dial import compute_ig_total
from qarray_env import DeviceParams, QArrayEnv, N_GATE
from noise_model import GaussianReadoutNoise

DEFAULT_PARAMS = DeviceParams(
    E_c1=2.5, E_c2=2.7, alpha1=0.05, alpha2=0.05,
    E_cm_intra1=0.3, E_cm_intra2=0.3, cross_capacitance=0.05,
)

# Deliberately arbitrary equal weighting for this demo -- Primer Section
# 5/12's w(t) dynamic-range schedule is an explicitly OPEN item (re-derive
# only after the IG_BALD joint-entropy bound is settled), and
# compute_ig_total requires w_discrete/w_continuous explicitly for exactly
# this reason (no silent default). 1.0/1.0 here is NOT a recommendation,
# just a fixed, visible choice for this visualization.
W_DISCRETE = 1.0
W_CONTINUOUS = 1.0
R = np.eye(4) * 0.02


def make_initial_pf(n_particles=5, spread=0.35, rng=None):
    """A population with real disagreement in E_c1 (and correspondingly in
    alpha1, to also exercise FIM's alpha-sensitivity) -- deliberately NOT
    converged, since a converged population would make both BALD and FIM
    trivially flat (nothing left to learn) and defeat the point of a
    landscape visualization."""
    if rng is None:
        rng = np.random.default_rng(0)
    particles = []
    for _ in range(n_particles):
        E_c1 = 2.5 + rng.normal(scale=spread)
        alpha1 = 0.05 + rng.normal(scale=spread * 0.01)
        mu = np.array([E_c1, 2.7, alpha1, 0.05, 0.05])
        Sigma = np.eye(N_PARAMS) * 0.05
        kf = ParticleKalmanFilter(mu=mu, Sigma=Sigma, E_cm_intra1=0.3, E_cm_intra2=0.3)
        particles.append(OccupationParticle(kalman=kf, weight=1.0))
    return RBParticleFilter(particles=particles)


# ============================================================
# PART 1: Static landscape -- BALD, FIM, and combined IG_total
# across a 2D voltage slice, for ONE fixed (unconverged)
# particle population. Demonstrates Section 5's core premise:
# the two dials are genuinely different signals, not redundant.
# ============================================================
print("Building IG landscape (BALD vs FIM vs IG_total)...")
pf_landscape = make_initial_pf(n_particles=5, spread=0.35, rng=np.random.default_rng(1))

N = 45
v_range = np.linspace(0.0, 30.0, N)
# NOTE: an earlier version of this script used a narrow [6,14]V window
# "zoomed on the transition-rich region" -- this was a real methodological
# mistake, not a good zoom: both BALD and FIM peaked right at that
# window's EDGE, which is the signature of clipping off the actual
# richest part of the domain rather than finding a genuine interior
# peak. Widened to match the full range already characterized by the
# charge-stability-diagram and Jacobian sweeps.
V0, V1 = np.meshgrid(v_range, v_range, indexing="ij")

bald_grid = np.zeros((N, N))
fim_grid = np.zeros((N, N))
total_grid = np.zeros((N, N))

t0 = time.time()
for i in range(N):
    for j in range(N):
        vg = np.array([V0[i, j], V1[i, j], 0.0, 0.0])
        bald_grid[i, j] = compute_ig_bald(pf_landscape, vg)
        fim_grid[i, j] = compute_ig_fim(pf_landscape, vg, R)
        total_grid[i, j] = W_DISCRETE * bald_grid[i, j] + W_CONTINUOUS * fim_grid[i, j]
    if (i + 1) % 10 == 0:
        print(f"  row {i+1}/{N} ({time.time()-t0:.1f}s elapsed)")
print(f"Landscape done in {time.time()-t0:.1f}s")

bald_argmax = np.unravel_index(np.argmax(bald_grid), bald_grid.shape)
fim_argmax = np.unravel_index(np.argmax(fim_grid), fim_grid.shape)
total_argmax = np.unravel_index(np.argmax(total_grid), total_grid.shape)
bald_peak_vg = (V0[bald_argmax], V1[bald_argmax])
fim_peak_vg = (V0[fim_argmax], V1[fim_argmax])
total_peak_vg = (V0[total_argmax], V1[total_argmax])
peak_distance = np.hypot(bald_peak_vg[0] - fim_peak_vg[0], bald_peak_vg[1] - fim_peak_vg[1])
print(f"BALD peaks at {bald_peak_vg}, FIM peaks at {fim_peak_vg} "
      f"(distance apart: {peak_distance:.2f}V)")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, grid, title, peak, cmap in [
    (axes[0], bald_grid, "IG_BALD(v)", bald_peak_vg, "viridis"),
    (axes[1], fim_grid, "IG_FIM(v)", fim_peak_vg, "plasma"),
    (axes[2], total_grid, f"IG_total(v)\n(w_d={W_DISCRETE}, w_c={W_CONTINUOUS}, demo weights)", total_peak_vg, "inferno"),
]:
    im = ax.imshow(grid.T, origin="lower", extent=[v_range[0], v_range[-1], v_range[0], v_range[-1]],
                    cmap=cmap, aspect="auto")
    ax.scatter([peak[0]], [peak[1]], marker="*", color="white", s=250,
               edgecolor="black", linewidth=1, label="argmax")
    ax.set_title(title)
    ax.set_xlabel("Gate 0 (V)")
    ax.set_ylabel("Gate 1 (V)")
    ax.legend(fontsize=8, loc="upper right")
    plt.colorbar(im, ax=ax)

plt.suptitle(f"BALD and FIM peak {peak_distance:.2f}V apart -- the two dials genuinely disagree "
             f"about where to look next", y=1.02)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/two_dial_landscape.png", dpi=140, bbox_inches="tight")
plt.close()
print("Saved two_dial_landscape.png")


# ============================================================
# PART 2: Acquisition trajectory -- run several real steps of
# argmax(IG_total) -> simulate measurement -> pf.update(), and
# track where the policy looks, how belief spread shrinks, and
# how the achieved IG_total value trends over the run.
# ============================================================
print("\nRunning acquisition trajectory (real measurement updates)...")

pf_traj = make_initial_pf(n_particles=5, spread=0.35, rng=np.random.default_rng(1))
ground_truth = QArrayEnv(DEFAULT_PARAMS)
noise = GaussianReadoutNoise(sigma=np.sqrt(0.02))
rng = np.random.default_rng(42)

N_STEPS = 8
CANDIDATE_N = 18  # coarser than the landscape grid -- this runs N_STEPS times
candidate_axis = np.linspace(0.0, 30.0, CANDIDATE_N)
Vc0, Vc1 = np.meshgrid(candidate_axis, candidate_axis, indexing="ij")

chosen_vgs = []
achieved_ig = []
within_cov_trace = []
between_cov_trace = []

t0 = time.time()
for step in range(N_STEPS):
    total_c = np.zeros((CANDIDATE_N, CANDIDATE_N))
    for i in range(CANDIDATE_N):
        for j in range(CANDIDATE_N):
            vg = np.array([Vc0[i, j], Vc1[i, j], 0.0, 0.0])
            total_c[i, j] = compute_ig_total(pf_traj, vg, R, w_discrete=W_DISCRETE, w_continuous=W_CONTINUOUS)
    best = np.unravel_index(np.argmax(total_c), total_c.shape)
    vg_star = np.array([Vc0[best], Vc1[best], 0.0, 0.0])

    chosen_vgs.append(vg_star.copy())
    achieved_ig.append(total_c[best])
    within_cov_trace.append(pf_traj.within_particle_covariance())
    between_cov_trace.append(pf_traj.between_particle_covariance())

    true_reading = ground_truth.query(vg_star)
    measured = noise.sample(true_reading, rng)
    pf_traj.update(measured, vg_star, R, T=0.05)

    print(f"  step {step+1}/{N_STEPS}: chose vg=({vg_star[0]:.2f},{vg_star[1]:.2f}), "
          f"IG_total={achieved_ig[-1]:.4f}, "
          f"within_cov={within_cov_trace[-1]:.2f}, between_cov={between_cov_trace[-1]:.2f} "
          f"({time.time()-t0:.1f}s elapsed)")

chosen_vgs = np.array(chosen_vgs)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

im = axes[0].imshow(total_grid.T, origin="lower",
                     extent=[v_range[0], v_range[-1], v_range[0], v_range[-1]],
                     cmap="inferno", aspect="auto", alpha=0.6)
axes[0].plot(chosen_vgs[:, 0], chosen_vgs[:, 1], "-o", color="cyan", markersize=6,
             linewidth=1.5, markeredgecolor="black")
for k, (x, y) in enumerate(chosen_vgs[:, :2]):
    axes[0].annotate(str(k), (x, y), textcoords="offset points", xytext=(5, 5),
                      fontsize=8, color="white")
axes[0].set_title("Chosen candidates over the run\n(background: INITIAL IG_total landscape, for context only --\nthe true landscape moves as belief updates)")
axes[0].set_xlabel("Gate 0 (V)")
axes[0].set_ylabel("Gate 1 (V)")
plt.colorbar(im, ax=axes[0])

steps = np.arange(1, N_STEPS + 1)
axes[1].plot(steps, within_cov_trace, "-o", label="within-particle covariance\n(matrix-averaged, logdet)")
finite_between = np.array(between_cov_trace)
if np.any(np.isfinite(finite_between)):
    axes[1].plot(steps, finite_between, "-s", label="between-particle covariance\n(logdet, -inf if singular)")
axes[1].set_xlabel("step")
axes[1].set_ylabel("normalized logdet")
axes[1].set_title("Belief spread over the run\n(should trend down as particles agree/converge)")
axes[1].legend(fontsize=8)

axes[2].plot(steps, achieved_ig, "-o", color="tab:red")
axes[2].set_xlabel("step")
axes[2].set_ylabel("IG_total at chosen candidate")
axes[2].set_title("Achieved information gain per step\n(should trend down as easy information is exhausted)")

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/two_dial_trajectory.png", dpi=140)
plt.close()
print("Saved two_dial_trajectory.png")

print(f"\nSummary:")
print(f"  BALD/FIM peak separation in static landscape: {peak_distance:.2f}V")
print(f"  within_cov: {within_cov_trace[0]:.2f} -> {within_cov_trace[-1]:.2f}")
print(f"  IG_total achieved: {achieved_ig[0]:.4f} -> {achieved_ig[-1]:.4f}")
