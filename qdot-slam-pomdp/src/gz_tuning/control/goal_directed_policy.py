"""
gz_tuning.control.goal_directed_policy

NEW MODULE -- Primer Section 6b: once control/phase_switch.py's
PhaseSwitchController declares Phase 2, hard-switch the acquisition choice
from IG_total maximization (acquisition/candidate_search.py) to navigating
toward the elected champion particle's (phase_switch.py's MAP-proxy,
elect_champion()) target-occupation boundary. No new acquisition math --
this reuses control/convergence.py's target_occupation_probability, the
same per-dot p_flip machinery Section 6a already specifies for judging
"how close is this vg to the target occupation."

Deliberately NOT a redraw of candidate_search.py's global uniform/seeded
search: Phase 2 is local navigation toward one already-located target
particle, not the Phase-0/Phase-1 blank-start candidate-proposal problem
coarse_sweep.py and candidate_search.py exist to solve. A global candidate
redraw here would reintroduce exactly the thin-slice-of-4D-range proposal
gap Section 5 diagnosed (real transitions occupy a thin slice of the full
vg_range^N_GATE volume), for no benefit once a MAP estimate already exists
to walk toward -- so candidates here are a LOCAL batch around the current
voltage, not a fresh global draw.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from convergence import target_occupation_probability
from particle_filter import OccupationParticle
from qarray_env import N_GATE


@dataclass(frozen=True)
class GoalDirectedSearchResult:
    best_vg: np.ndarray
    best_p_target: float
    all_vg: np.ndarray
    all_p_target: np.ndarray


def select_goal_directed_measurement(
    champion: OccupationParticle,
    target_occupation: np.ndarray,
    *,
    v_cur: np.ndarray,
    vg_range: tuple[float, float] = (0.0, 15.0),
    n_candidates: int = 50,
    local_radius: float = 1.0,
    T: float = 0.05,
    rng: np.random.Generator | None = None,
) -> GoalDirectedSearchResult:
    """argmax_v target_occupation_probability(champion.kalman, v,
    target_occupation) over a LOCAL candidate batch around v_cur --
    a greedy hill-climb against the champion particle's own belief, not a
    global search (see module docstring).

    Candidate proposal: v_cur + Gaussian(0, local_radius) per gate,
    clipped to vg_range. Candidate 0 is always v_cur itself, so this
    policy can never score worse than staying put -- unlike a pure-random
    local batch, which has no such guarantee and could occasionally walk
    away from the target on an unlucky draw.

    champion is passed in explicitly (rather than this function reaching
    into a PhaseSwitchController itself) so it stays a pure function of
    its inputs, consistent with candidate_search.select_next_measurement's
    own posture -- the caller (experiment/run_active_slam.py) is
    responsible for only calling this while
    PhaseSwitchController.phase is PHASE_2_FEATURE_BASED, where
    .champion is guaranteed non-None.
    """
    if rng is None:
        rng = np.random.default_rng()
    v_cur = np.asarray(v_cur, dtype=np.float64)
    lo, hi = vg_range

    candidates = v_cur + rng.normal(0.0, local_radius, size=(n_candidates, N_GATE))
    candidates = np.clip(candidates, lo, hi)
    candidates[0] = v_cur

    p_values = np.empty(n_candidates, dtype=np.float64)
    for i, vg in enumerate(candidates):
        p_values[i] = target_occupation_probability(
            champion.kalman, vg, target_occupation, T=T
        )

    best_idx = int(np.argmax(p_values))
    return GoalDirectedSearchResult(
        best_vg=candidates[best_idx],
        best_p_target=float(p_values[best_idx]),
        all_vg=candidates,
        all_p_target=p_values,
    )
