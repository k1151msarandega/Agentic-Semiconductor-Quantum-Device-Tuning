import sys
for d in ["simulator"]:
    sys.path.insert(0, d)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from qarray_env import DeviceParams, QArrayEnv, InfeasibleDeviceParamsError, N_GATE

DEFAULT_PARAMS = DeviceParams(
    E_c1=2.5, E_c2=2.7, alpha1=0.05, alpha2=0.05,
    E_cm_intra1=0.3, E_cm_intra2=0.3, cross_capacitance=0.05,
)

# ============================================================
# FIGURE 1: Charge stability diagram, DQD-A (gates 0,1), swept
# with DQD-B (gates 2,3) held at two different fixed points --
# demonstrating the REAL cross-term perturbation this project
# deliberately does NOT compensate for (Primer Section 2/6/8:
# no virtual-gate compensation is built; DQD-B's occupation is
# allowed to shift DQD-A's transition lines).
# ============================================================
print("Building charge stability diagrams...")
env = QArrayEnv(DEFAULT_PARAMS)

N = 220
v_range = np.linspace(0.0, 30.0, N)
V0, V1 = np.meshgrid(v_range, v_range, indexing="ij")

# First, find a DQD-B setpoint that actually pushes it into a stable,
# occupied (1,1)-like state, so panel B genuinely differs from panel A
# (dot2,dot3 fixed at 0,0) rather than accidentally sitting at the same
# state.
probe = np.array([[0.0, 0.0, v, v] for v in np.linspace(0.0, 30.0, 200)])
probe_occ = env.query(probe)
b_occupied_idx = np.argmax(probe_occ[:, 2] + probe_occ[:, 3] >= 1.9)
v_b_setpoint = probe[b_occupied_idx, 2]
print(f"DQD-B setpoint chosen: {v_b_setpoint:.3f}V "
      f"(occupation there: dot2={probe_occ[b_occupied_idx,2]:.2f}, "
      f"dot3={probe_occ[b_occupied_idx,3]:.2f})")

def sweep_dqd_a(v2, v3):
    vg = np.stack([V0, V1, np.full_like(V0, v2), np.full_like(V0, v3)], axis=-1)
    occ = env.query(vg)
    total = occ.sum(axis=-1)
    return total

total_baseline = sweep_dqd_a(0.0, 0.0)
total_coupled = sweep_dqd_a(v_b_setpoint, v_b_setpoint)

def edge_map(total):
    gy, gx = np.gradient(total)
    return np.sqrt(gx**2 + gy**2)

edges_baseline = edge_map(total_baseline)
edges_coupled = edge_map(total_coupled)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

im0 = axes[0].imshow(edges_baseline.T, origin="lower", extent=[0, 30, 0, 30],
                      cmap="gray_r", aspect="auto")
axes[0].set_title("DQD-A stability diagram\n(DQD-B fixed at (0,0))")
axes[0].set_xlabel("Gate 0 (V)")
axes[0].set_ylabel("Gate 1 (V)")

im1 = axes[1].imshow(edges_coupled.T, origin="lower", extent=[0, 30, 0, 30],
                      cmap="gray_r", aspect="auto")
axes[1].set_title(f"DQD-A stability diagram\n(DQD-B fixed at ({v_b_setpoint:.1f}V, {v_b_setpoint:.1f}V), occupied)")
axes[1].set_xlabel("Gate 0 (V)")
axes[1].set_ylabel("Gate 1 (V)")

diff = total_coupled - total_baseline
im2 = axes[2].imshow(diff.T, origin="lower", extent=[0, 30, 0, 30],
                      cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
axes[2].set_title("Occupation difference\n(coupled - baseline)\nnonzero = cross-term shift")
axes[2].set_xlabel("Gate 0 (V)")
axes[2].set_ylabel("Gate 1 (V)")
plt.colorbar(im2, ax=axes[2], label="Δ(total occupation)")

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/charge_stability_diagram.png", dpi=140)
plt.close()
print("Saved charge_stability_diagram.png")

shifted_fraction = np.mean(np.abs(diff) > 0.05)
print(f"Fraction of (gate0,gate1) grid where DQD-B's occupation state "
      f"measurably shifts DQD-A's total occupation: {shifted_fraction:.1%}")

# ============================================================
# FIGURE 2: alpha/E_c feasibility boundary map (Primer Section
# 12's blocking action item) -- via REAL QArrayEnv construction
# attempts across a grid, using the same InfeasibleDeviceParamsError
# path the rest of the codebase relies on (post-fix, unified for
# both diagonal and off-diagonal infeasibility).
# ============================================================
print("\nBuilding alpha/E_c feasibility boundary map (real construction sweep)...")

alpha_values = np.linspace(0.02, 1.0, 90)
Ec_values = np.linspace(0.1, 4.0, 90)

feasible = np.zeros((len(alpha_values), len(Ec_values)), dtype=bool)
min_diag = np.full((len(alpha_values), len(Ec_values)), np.nan)

for i, alpha in enumerate(alpha_values):
    for j, Ec in enumerate(Ec_values):
        params = DeviceParams(
            E_c1=Ec, E_c2=Ec, alpha1=alpha, alpha2=alpha,
            E_cm_intra1=0.3, E_cm_intra2=0.3, cross_capacitance=0.05,
        )
        try:
            e = QArrayEnv(params)
            feasible[i, j] = True
            min_diag[i, j] = np.min(np.diag(e._Cdd))
        except InfeasibleDeviceParamsError:
            feasible[i, j] = False

print(f"Feasible fraction of grid: {feasible.mean():.1%}")

# Direct check against the primer's own historical claim (Section 4b):
# "at alpha=1.0 with realistic coupling already in use, max achievable
# E_c ~ 0.81, not 2.5" -- verify this against the REAL sweep rather than
# re-asserting it from the document.
alpha_1_idx = np.argmin(np.abs(alpha_values - 1.0))
feasible_Ec_at_alpha1 = Ec_values[feasible[alpha_1_idx, :]]
max_Ec_at_alpha1 = feasible_Ec_at_alpha1.max() if len(feasible_Ec_at_alpha1) else float("nan")
print(f"At alpha={alpha_values[alpha_1_idx]:.3f}: max feasible E_c = {max_Ec_at_alpha1:.3f} "
      f"(primer's own historical claim: ~0.81)")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

cmap = ListedColormap(["#d9534f", "#5cb85c"])
axes[0].imshow(feasible.T, origin="lower",
               extent=[alpha_values[0], alpha_values[-1], Ec_values[0], Ec_values[-1]],
               aspect="auto", cmap=cmap, vmin=0, vmax=1)
axes[0].set_xlabel("alpha (lever arm)")
axes[0].set_ylabel("E_c (charging energy)")
axes[0].set_title("Feasibility (real QArrayEnv construction)\ngreen = feasible, red = infeasible")
axes[0].scatter([1.0], [2.5], marker="x", color="black", s=80,
                label="Primer's illustrative (alpha=1.0, E_c=2.5)\n-- confirmed infeasible")
axes[0].scatter([DEFAULT_PARAMS.alpha1], [DEFAULT_PARAMS.E_c1], marker="o",
                color="blue", s=60, label="This project's actual default\n(alpha=0.05, E_c=2.5)")
axes[0].legend(loc="upper right", fontsize=8)

im1 = axes[1].imshow(min_diag.T, origin="lower",
                      extent=[alpha_values[0], alpha_values[-1], Ec_values[0], Ec_values[-1]],
                      aspect="auto", cmap="viridis")
axes[1].set_xlabel("alpha (lever arm)")
axes[1].set_ylabel("E_c (charging energy)")
axes[1].set_title("Feasibility margin\n(min diagonal self-capacitance entry;\nNaN/blank = infeasible)")
plt.colorbar(im1, ax=axes[1], label="min diag(Cdd)")

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/feasibility_boundary_map.png", dpi=140)
plt.close()
print("Saved feasibility_boundary_map.png")

# Recommend concrete ground-truth values WITH real margin, not just
# barely-feasible ones -- Primer Section 4b/12's explicit ask ("pick
# final ground-truth values with real margin ... don't pick a
# replacement number by feel").
MARGIN_THRESHOLD = 0.05  # same order as this project's tracked-parameter scale
good_mask = feasible & (min_diag > MARGIN_THRESHOLD)

# NOTE: naively maximizing margin over the FULL (alpha, E_c) grid is a
# degenerate recommendation -- margin trivially grows as E_c shrinks
# toward 0.1, which is not a physically useful device scale (this
# project's own tests, and the primer's original intent, target E_c
# around 2.5). Two separate, more useful things to report instead:
#   (1) the largest FEASIBLE E_c actually achievable within this
#       project's real alpha range [0.05, 0.4], with margin, and
#   (2) the best-margin (alpha, E_c) pairs specifically NEAR the
#       primer's original target scale (E_c in [2.0, 3.0]), which is
#       what actually answers Section 12's blocking question: "can we
#       keep using E_c~2.5, and if so at what alpha, with real margin?"
alpha_lo_idx = np.searchsorted(alpha_values, 0.05)
alpha_hi_idx = np.searchsorted(alpha_values, 0.4, side="right")
sub_feasible = feasible[alpha_lo_idx:alpha_hi_idx, :]
sub_margin = min_diag[alpha_lo_idx:alpha_hi_idx, :]
sub_alpha = alpha_values[alpha_lo_idx:alpha_hi_idx]

good_sub = sub_feasible & (sub_margin > MARGIN_THRESHOLD)
if good_sub.any():
    idxs = np.argwhere(good_sub)
    max_Ec_row = max(idxs, key=lambda ij: Ec_values[ij[1]])
    print(f"\nLargest feasible E_c (with margin > {MARGIN_THRESHOLD}) within "
          f"alpha in [0.05, 0.4]: E_c={Ec_values[max_Ec_row[1]]:.3f} at "
          f"alpha={sub_alpha[max_Ec_row[0]]:.3f} "
          f"(margin={sub_margin[max_Ec_row[0], max_Ec_row[1]]:.4f})")
else:
    print(f"\nNo (alpha in [0.05,0.4], E_c) pair meets margin > {MARGIN_THRESHOLD} at all.")

Ec_lo_idx = np.searchsorted(Ec_values, 2.0)
Ec_hi_idx = np.searchsorted(Ec_values, 3.0, side="right")
near_target_mask = np.zeros_like(feasible)
near_target_mask[alpha_lo_idx:alpha_hi_idx, Ec_lo_idx:Ec_hi_idx] = True
good_near_target = feasible & near_target_mask & (min_diag > MARGIN_THRESHOLD)

if good_near_target.any():
    idxs = np.argwhere(good_near_target)
    candidates = [(alpha_values[i], Ec_values[j], min_diag[i, j]) for i, j in idxs]
    candidates.sort(key=lambda t: -t[2])
    print(f"\nBest-margin (alpha, E_c) candidates NEAR the primer's original "
          f"target scale (E_c in [2.0, 3.0], alpha in [0.05, 0.4]) -- this is "
          f"the actual answer to Section 12's blocking question:")
    for alpha, Ec, margin in candidates[:8]:
        print(f"  alpha={alpha:.3f}, E_c={Ec:.3f}  (margin={margin:.4f})")
else:
    print(f"\nNO feasible (alpha, E_c) pair exists near the primer's original "
          f"target scale (E_c in [2.0, 3.0], alpha in [0.05, 0.4]) with "
          f"margin > {MARGIN_THRESHOLD} -- the primer's E_c~2.5 target may "
          f"need to be revised downward, or the margin threshold relaxed, "
          f"or E_cm_intra/cross_capacitance reduced to free up headroom.")

print("\nDone.")
