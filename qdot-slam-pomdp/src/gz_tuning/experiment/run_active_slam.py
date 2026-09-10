"""
gz_tuning.experiment.run_active_slam

NEW MODULE -- the active-policy analog of baselines/raster_scan.py's
run_raster_scan: walks a real measurement loop against a ground-truth
QArrayEnv, but chooses each next vg via candidate_search.select_next_measurement
(IG_total argmax over a random candidate batch) instead of a fixed grid.
Nothing in the repo previously tied acquisition + belief update +
convergence into one runnable loop for the active method -- this is that
loop.

w_discrete/w_continuous: Primer Section 12 lists the w(t) dynamic-range
schedule as an explicit open item, re-derivable only after the IG_BALD
joint-entropy fix (done, see bald.py) lands -- which it has, but the
schedule itself was never derived. Rather than invent one silently, this
module takes them as required, fixed (not time-varying) scalars, same
posture two_dial.py already takes at the single-candidate level. Flagged
directly in run_active_slam's docstring below, not buried.

envs reuse: select_next_measurement() already built one QArrayEnv per
particle (at PRE-update params) to search candidates; this loop reuses
that exact same envs list for the subsequent pf.update() call so the
chosen candidate's belief update doesn't pay a third redundant solve
(kalman.update's own solve, particle_filter.update's likelihood solve, and
now the acquisition search's solve are all the same envs where possible).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from candidate_search import select_next_measurement
from coarse_sweep import CoarseSweepResult, run_coarse_sweep
from convergence import ConvergenceMonitor, ConvergenceStatus
from noise_model import GaussianReadoutNoise
from particle_filter import RBParticleFilter
from qarray_env import N_GATE, QArrayEnv


@dataclass
class StepLog:
    step: int
    vg: np.ndarray
    best_ig: float
    ess: float
    resampled: bool
    within_particle_logdet: float
    between_particle_logdet: float
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
    run_phase0: bool = True,
    phase0_n_points_per_line: int = 25,
    phase0_background_fractions: tuple[float, ...] = (1.0 / 3.0, 2.0 / 3.0),
    seed_fraction: float = 0.5,
    seed_jitter_sigma: float = 0.3,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
) -> ActiveSlamResult:
    """Active POMDP+SLAM measurement loop, analogous to run_raster_scan
    but choosing each vg via IG_total argmax instead of a fixed schedule.

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
            verbose=verbose,
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
        search = select_next_measurement(
            pf, R,
            w_discrete=w_discrete, w_continuous=w_continuous,
            vg_range=vg_range, n_candidates=n_candidates, T=T,
            seed_points=seed_points, seed_fraction=seed_fraction,
            seed_jitter_sigma=seed_jitter_sigma, rng=rng,
        )
        v_next = search.best_vg
        proposed_step = v_next - v_cur

        true_reading = ground_truth.soft_prediction(v_next, T=T)
        measured = noise.sample(true_reading, rng)
        pf.update(measured, v_next, R, T=T, envs=search.envs)

        ess = pf.effective_sample_size()
        resampled = False
        if ess < ess_resample_frac * n_particles:
            pf.resample(rng, mu_jitter_scale=mu_jitter_scale)
            resampled = True

        status = monitor.step(pf, v_next, proposed_step)

        history.append(StepLog(
            step=step,
            vg=v_next.copy(),
            best_ig=search.best_ig,
            ess=ess,
            resampled=resampled,
            within_particle_logdet=pf.within_particle_covariance(),
            between_particle_logdet=pf.between_particle_covariance(),
            belief_stable=status.belief_stable,
            actuation_quiescent=status.actuation_quiescent,
            consecutive_count=status.consecutive_count,
        ))
        if verbose:
            print(
                f"step {step:4d}  vg={np.round(v_next, 3)}  IG={search.best_ig:.4f}  "
                f"ESS={ess:6.1f}{'*' if resampled else ' '}  "
                f"blue={history[-1].within_particle_logdet:7.3f}  "
                f"red={history[-1].between_particle_logdet:7.3f}  "
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
