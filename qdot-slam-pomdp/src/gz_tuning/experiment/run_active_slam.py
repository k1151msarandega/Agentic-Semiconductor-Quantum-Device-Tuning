"""
gz_tuning.experiment.run_active_slam

NEW MODULE -- the active-policy analog of baselines/raster_scan.py's
run_raster_scan: walks a real measurement loop against a ground-truth
QArrayEnv, but chooses each next vg via candidate_search.select_next_measurement
(IG_total argmax over a random candidate batch) instead of a fixed grid.
Nothing in the repo previously tied acquisition + belief update +
convergence into one runnable loop for the active method -- this is that
loop.

NOTE ON "PHASE" VOCABULARY -- two unrelated axes share the word "phase" in
this codebase; neither name was changed here to avoid touching unrelated
call sites, but they are genuinely distinct concepts:
  - `run_phase0` / coarse_sweep.py's "Phase 0" is the deterministic
    per-gate bootstrap sweep that closes the candidate-PROPOSAL gap before
    adaptive search starts (Primer Section 5's phase-0 fix).
  - `phase_controller.phase` (control/phase_switch.py's Phase enum) is
    Primer Section 6's raw-signal-vs-feature-based hysteresis state,
    entirely separate from the above and evaluated every step for the
    rest of this loop's lifetime, long after Phase 0 (the bootstrap) has
    finished.
They do not conflict in practice (`run_phase0` only gates the one-time
bootstrap below; `phase_controller.phase` gates the per-step acquisition
branch further down), but a reader skimming for "phase" should not
conflate them.

SECTION 6b WIRING -- once phase_controller.step(pf) reports
Phase.PHASE_2_FEATURE_BASED, this loop hard-switches the per-step
acquisition call from candidate_search.select_next_measurement (IG_total
argmax) to goal_directed_policy.select_goal_directed_measurement
(navigation toward phase_controller.champion's target-occupation
boundary), per the primer's Section 6b design. `monitor.target_occupation`
is reused as the navigation target rather than adding a second,
independently-specifiable target parameter to this function -- Phase 2
navigation and the convergence check it feeds into should not be able to
disagree about what "the target" is.

w_discrete/w_continuous: Primer Section 12 lists the w(t) dynamic-range
schedule as an explicit open item, re-derivable only after the IG_BALD
joint-entropy fix (done, see bald.py) lands -- which it has, but the
schedule itself was never derived. Rather than invent one silently, this
module takes them as required, fixed (not time-varying) scalars, same
posture two_dial.py already takes at the single-candidate level. Flagged
directly in run_active_slam's docstring below, not buried. These are only
used for the Phase-1, IG-based branch of the loop -- Phase 2 navigation
does not use w_discrete/w_continuous at all (see Section 6b wiring note
above).

envs reuse: in the Phase-1 branch, select_next_measurement() already
built one QArrayEnv per particle (at PRE-update params) to search
candidates; this loop reuses that exact same envs list for the subsequent
pf.update() call so the chosen candidate's belief update doesn't pay a
third redundant solve. In the Phase-2 branch, candidate_search's
build_particle_envs() is called directly (goal_directed_policy's own
search only needs the champion's env, not the full ensemble's, but
pf.update() still needs one env per particle) to preserve the same
reuse for every particle's subsequent belief update.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bald import compute_ig_bald
from candidate_search import build_particle_envs, select_next_measurement
from coarse_sweep import CoarseSweepResult, run_coarse_sweep
from convergence import ConvergenceMonitor, ConvergenceStatus
from goal_directed_policy import select_goal_directed_measurement
from noise_model import GaussianReadoutNoise
from particle_filter import RBParticleFilter
from phase_switch import Phase, PhaseSwitchController
from qarray_env import N_GATE, QArrayEnv


@dataclass
class StepLog:
    step: int
    vg: np.ndarray
    phase: Phase
    best_ig: float | None
    best_ig_bald: float | None
    best_p_target: float | None
    ess: float
    resampled: bool
    resample_reason: str | None
    within_particle_logdet: float
    between_particle_logdet: float
    blue_mass_fraction_low: float
    blue_mass_fraction_high: float
    red_below_low: bool
    red_below_high: bool
    beta: float
    belief_stable: bool
    actuation_quiescent: bool
    consecutive_count: int


@dataclass
class ActiveSlamResult:
    converged: bool
    n_measurements: int
    final_status: ConvergenceStatus | None
    history: list[StepLog] = field(default_factory=list)
    coarse_sweep: CoarseSweepResult | None = None


def run_active_slam(
    pf: RBParticleFilter,
    ground_truth: QArrayEnv,
    noise: GaussianReadoutNoise,
    monitor: ConvergenceMonitor,
    phase_controller: PhaseSwitchController,
    *,
    w_discrete: float,
    w_continuous: float,
    vg_range: tuple[float, float] = (0.0, 15.0),
    n_candidates: int = 50,
    max_steps: int = 500,
    T: float = 0.05,
    v_init: np.ndarray | None = None,
    ess_resample_frac: float = 0.5,
    mu_jitter_scale: float | None = 0.1,
    forced_resample_period: int | None = None,
    run_phase0: bool = True,
    phase0_n_points_per_line: int = 25,
    phase0_background_fractions: tuple[float, ...] = (1.0 / 3.0, 2.0 / 3.0),
    seed_fraction: float = 0.5,
    seed_jitter_sigma: float = 0.3,
    goal_directed_local_radius: float = 1.0,
    target_ess_frac: float | None = 0.5,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
) -> ActiveSlamResult:
    """Active POMDP+SLAM measurement loop, analogous to run_raster_scan
    but choosing each vg via IG_total argmax (Phase 1) or goal-directed
    navigation toward the elected champion particle (Phase 2, Section 6b)
    instead of a fixed schedule.

    forced_resample_period: if set, force a resample every N steps
    regardless of ESS -- independent of, and in addition to, the existing
    ess_resample_frac-triggered resample below. Added after a real run
    showed adaptive tempering pinning beta near 0 on ~93-97% of steps in
    BOTH IG-driven acquisition (Phase 1) and a blind deterministic sweep
    (Phase 0) -- ruling out "acquisition finding disagreement forces the
    throttle" as the cause, and pointing at this prior/noise combination
    keeping raw likelihood disagreement high enough that ESS-gated
    resampling never fires on its own. Deliberately NOT implemented as a
    change to ess_resample_frac itself: loosening that threshold directly
    fights the quantity tempering uses to judge "is a resample safe right
    now," and risks reopening the early-collapse failure tempering exists
    to prevent (pruning a still-genuinely-uncertain ensemble, just on a
    slower schedule). A periodic forced resample is decoupled from the
    belief-quality signal entirely -- its only failure mode is pruning at
    a locally bad moment, which is recoverable, not catastrophic, unlike
    collapsing prematurely. Treated as the lower-risk default; an annealed
    ess_resample_frac remains a fallback if this proves insufficient on
    its own, not something to build alongside it as a co-equal option.

    phase_controller: caller-constructed PhaseSwitchController (thresholds
    are not derived here -- see control/phase_switch.py's module
    docstring). Its .phase is advanced once per loop iteration via
    .step(pf), and its .champion drives the Phase-2 acquisition branch.
    Passing a fresh controller each run is the caller's responsibility,
    same posture as `monitor`.

    monitor must have require_actuation_quiescence=True (the default) --
    this is the active-policy case control/convergence.py's guard B is
    meant for; a require_actuation_quiescence=False monitor here would
    silently drop the Section 8 failure-mode-B guard for a policy that
    actually has a meaningful "proposed next step" signal to check,
    defeating the reason that guard exists. Enforced the same way
    raster_scan.py enforces the opposite constraint on its own monitor.
    """
    if not monitor.require_actuation_quiescence:
        raise ValueError(
            "run_active_slam requires a ConvergenceMonitor constructed "
            "with require_actuation_quiescence=True -- the active policy "
            "has a real 'proposed next step' signal (Section 8 guard B) "
            "that a fixed-schedule baseline doesn't; see "
            "control/convergence.py and baselines/raster_scan.py's "
            "module docstrings for the two cases this distinction covers."
        )

    if rng is None:
        rng = np.random.default_rng()
    if v_init is None:
        mid = float(np.mean(vg_range))
        v_init = np.full(N_GATE, mid)
    v_cur = np.asarray(v_init, dtype=np.float64)

    R = noise.R_matrix()
    n_particles = pf.n_particles
    history: list[StepLog] = []
    status: ConvergenceStatus | None = None

    coarse_result: CoarseSweepResult | None = None
    seed_points: np.ndarray | None = None
    if run_phase0:
        coarse_result = run_coarse_sweep(
            pf, ground_truth, noise,
            vg_range=vg_range, n_points_per_line=phase0_n_points_per_line,
            background_fractions=phase0_background_fractions, T=T, rng=rng,
            target_ess_frac=target_ess_frac, verbose=verbose,
        )
        if coarse_result.n_transitions_found > 0:
            seed_points = coarse_result.seed_points
        if verbose:
            print(
                f"Phase 0: {coarse_result.n_measurements} measurements, "
                f"{coarse_result.n_transitions_found} transitions found "
                f"(seed_points={'none' if seed_points is None else len(seed_points)})"
            )

    for step in range(max_steps):
        diag = phase_controller.step(pf)

        if phase_controller.phase is Phase.PHASE_2_FEATURE_BASED:
            envs, x0_bases = build_particle_envs(pf)
            goal = select_goal_directed_measurement(
                phase_controller.champion, monitor.target_occupation,
                v_cur=v_cur, vg_range=vg_range, n_candidates=n_candidates,
                local_radius=goal_directed_local_radius, T=T, rng=rng,
            )
            v_next = goal.best_vg
            best_ig = None
            best_ig_bald = None
            best_p_target = goal.best_p_target
        else:
            search = select_next_measurement(
                pf, R,
                w_discrete=w_discrete, w_continuous=w_continuous,
                vg_range=vg_range, n_candidates=n_candidates, T=T,
                seed_points=seed_points, seed_fraction=seed_fraction,
                seed_jitter_sigma=seed_jitter_sigma, rng=rng,
            )
            v_next = search.best_vg
            best_ig = search.best_ig
            # Isolate IG_BALD alone (mixture-disagreement entropy) at the
            # CHOSEN candidate, separate from best_ig = w_d*IG_BALD +
            # w_c*IG_FIM: the tempering-link hypothesis is specifically
            # about particle disagreement (BALD's territory), and IG_FIM
            # (unbounded, can dominate the sum numerically) could dilute
            # a real correlation if only the combined total is checked.
            # envs here are pf's PRE-update envs (built by select_next_
            # measurement above), matching pf.particles' current state --
            # must be computed before pf.update() mutates them below.
            best_ig_bald = compute_ig_bald(pf, v_next, T=T, envs=search.envs)
            best_p_target = None
            envs = search.envs

        proposed_step = v_next - v_cur

        true_reading = ground_truth.soft_prediction(v_next, T=T)
        measured = noise.sample(true_reading, rng)
        pf.update(measured, v_next, R, T=T, envs=envs, target_ess_frac=target_ess_frac)

        ess = pf.effective_sample_size()
        resampled = False
        resample_reason: str | None = None
        if ess < ess_resample_frac * n_particles:
            pf.resample(rng, mu_jitter_scale=mu_jitter_scale)
            resampled = True
            resample_reason = "ess"
        elif forced_resample_period is not None and (step + 1) % forced_resample_period == 0:
            # Decoupled from ESS/beta entirely -- fires on schedule even
            # though the ESS-based check above did not trigger. This is
            # the whole point: tempering can hold ESS at target forever
            # without this, per the Phase-0-vs-Phase-1 beta finding.
            pf.resample(rng, mu_jitter_scale=mu_jitter_scale)
            resampled = True
            resample_reason = "forced"

        status = monitor.step(pf, v_next, proposed_step)

        history.append(StepLog(
            step=step,
            vg=v_next.copy(),
            phase=phase_controller.phase,
            best_ig=best_ig,
            best_ig_bald=best_ig_bald,
            best_p_target=best_p_target,
            ess=ess,
            resampled=resampled,
            resample_reason=resample_reason,
            within_particle_logdet=pf.within_particle_covariance(),
            between_particle_logdet=pf.between_particle_covariance(),
            blue_mass_fraction_low=diag.blue_mass_fraction_below_low,
            blue_mass_fraction_high=diag.blue_mass_fraction_below_high,
            red_below_low=diag.red_below_low,
            red_below_high=diag.red_below_high,
            beta=pf.last_beta,
            belief_stable=status.belief_stable,
            actuation_quiescent=status.actuation_quiescent,
            consecutive_count=status.consecutive_count,
        ))
        if verbose:
            ig_str = f"IG={best_ig:.4f}" if best_ig is not None else f"p_tgt={best_p_target:.4f}"
            print(
                f"step {step:4d}  vg={np.round(v_next, 3)}  phase={phase_controller.phase.name}  "
                f"{ig_str}  ESS={ess:6.1f}"
                f"{'*F' if resample_reason == 'forced' else ('*E' if resample_reason == 'ess' else '  ')}  "
                f"blue_agg={history[-1].within_particle_logdet:7.3f}  "
                f"red={history[-1].between_particle_logdet:7.3f}  "
                f"blue_mass_low={diag.blue_mass_fraction_below_low:.3f}  "
                f"red<low={diag.red_below_low}  "
                f"beta={pf.last_beta:.4f}  "
                f"ig_bald={'n/a' if history[-1].best_ig_bald is None else f'{history[-1].best_ig_bald:.4f}'}  "
                f"stable={status.belief_stable}  quiescent={status.actuation_quiescent}  "
                f"consec={status.consecutive_count}"
            )

        v_cur = v_next
        if status.converged:
            return ActiveSlamResult(
                converged=True, n_measurements=step + 1, final_status=status,
                history=history, coarse_sweep=coarse_result,
            )

    return ActiveSlamResult(
        converged=False, n_measurements=max_steps, final_status=status,
        history=history, coarse_sweep=coarse_result,
    )
