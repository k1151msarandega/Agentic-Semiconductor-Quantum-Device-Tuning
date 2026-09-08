"""
gz_tuning.control.convergence

Primer Section 8: dual convergence criterion under coupling, distinguishing
two genuinely different failure modes that would otherwise get conflated
into a single "looks stable" counter:
  A -- statistical false positive: a noisy measurement momentarily favors
       the target occupation; the next reading swings back. An artifact of
       observation noise alone.
  B -- genuine physical perturbation: navigating the OTHER DQD shifts this
       DQD's chemical potential via the cross-term enough to actually
       change its occupation -- a real, in-scope consequence of NOT
       building virtual-gate compensation (Primer Section 2).

Both guards are required, sustained for N consecutive steps, before
declaring convergence -- for an ACTIVE policy (the SLAM method):
  1. belief_stable(...)   -- guards A. Posterior mass on the target joint
     occupation exceeds p_conf.
  2. actuation_quiescent(...) -- guards B. The policy's own chosen next
     step magnitude is below epsilon on EVERY gate, not just ones near the
     target this DQD cares about -- if the OTHER DQD is still being
     actively navigated, its cross-term is still a live risk to this DQD,
     so this DQD isn't safe to call "done" yet even if it is momentarily
     at target. Reuses the acquisition function's own chosen step as the
     signal -- no new machinery, per Primer Section 8's explicit framing.

REVISION -- guard B disabled for non-adaptive baselines
(require_actuation_quiescence=False), found necessary during review:
guard B's ORIGINAL raster-baseline operationalization (baselines/
raster_scan.py measuring distance to the next SCHEDULED grid point) was a
real bug, not a design choice -- a fixed raster grid moves by a
non-trivial, constant step between every consecutive point BY
CONSTRUCTION, almost certainly larger than any sensible epsilon (which
should be noise-floor scale, per Section 6a). That made guard B false on
essentially every step except the last few, meaning the baseline could
only ever "converge" in the tail of the entire grid regardless of how fast
its belief actually stabilized -- structurally incapable of stopping
early, which is exactly backwards from what Section 8's fairness
requirement was protecting (a real early-stop comparison, not an
artificially inflated SLAM-method advantage from a baseline that can't
stop).

Re-reading Section 8's own text, the literal requirement is narrower than
what the original dual-guard raster integration assumed: "give the
baseline THE SAME BELIEF-CONFIDENCE-THRESHOLD STOPPING RULE" -- this
centers on guard A, not an unqualified dual-guard criterion. Guard B's
physical justification (an ACTIVE policy's chosen next step signaling live
cross-term perturbation risk) has no coherent analog for a fixed,
non-adaptive schedule: "distance to the next scheduled point" measures
"is the predetermined scan still going," not "has physical perturbation
risk actually settled." So require_actuation_quiescence=False for a
non-adaptive baseline is not a weaker or less-fair criterion -- it is the
correct operationalization of the SAME belief-confidence stopping rule
Section 8 asks for, for a policy shape (fixed schedule) guard B was never
meant to describe. This was a real decision point (an alternative --
redefining guard B via the SLAM method's own quiescence check evaluated
against the raster's current belief -- was considered and rejected as
importing SLAM-specific machinery into what is supposed to be the
classical baseline, muddying the comparison) -- flagged here rather than
silently resolved, since it changes what numbers actually get compared in
Claim 1.

FAIRNESS REQUIREMENT, still load-bearing: p_conf, epsilon (when guard B is
active), N, and T must be IDENTICAL across whatever ConvergenceMonitor
instances back the SLAM method and the raster baseline -- only
require_actuation_quiescence is expected to differ (True for the active
policy, False for a fixed schedule), and that difference is justified
above, not an inconsistency to paper over.

DISCRETE-STATE / POINT-ESTIMATE FIX, found during review -- a real
recurrence of a bug pattern already caught three times elsewhere in this
project (Section 6's mass-based blue-line fix over
particle_filter.py's matrix-averaged aggregate; noise_model.py's
boundary-distance/sigma_eff separation; Section 8's own p_flip
machinery): the original version of belief_stable/target_occupation_mass
computed "posterior mass on target" via round(particle's mean
soft_prediction) == target_occupation -- a hard 0/1 indicator using ONLY
each particle's mean estimate, ignoring that particle's own Sigma
entirely. A particle whose MEAN happens to round to the target contributed
its FULL weight even when its own continuous-parameter covariance is large
enough that it's genuinely unsure whether it's at target or an adjacent
state -- concentrated exactly at interdot transitions (Section 4c), where
a small shift in mu flips the rounded occupation, and exactly where a
convergence declaration matters most.

This is NOT the same design choice as particle_filter.py's own
implicit-discrete-state interpretation (soft_prediction + rounding), which
remains fine for its own actual use in this project: bald.py uses rounding
only to ENUMERATE a local candidate superset, then computes an EXACT joint
entropy over that enumeration via free_energy_at_candidates -- rounding is
never the final judgment there. This module's old version used rounding
AS the final yes/no judgment (mass or no mass), which is precisely the
point-estimate-masks-uncertainty trap.

FIX, without reintroducing the intractability Section 2 already rejected:
proper Bayesian marginalization over each particle's OWN Sigma (sampling
params ~ N(mu, Sigma) and averaging the resulting discrete distribution)
is EXACTLY the nested-Monte-Carlo computation Section 2 rejected for the
unified-BALD acquisition metric ("entropy of a mixture of discrete-labeled
Gaussians has no closed form, forcing nested Monte Carlo"). Reintroducing
that here would be the same mistake in a new location. Instead,
target_occupation_probability reuses noise_model.py's ALREADY-BUILT
closed-form machinery -- per_dot_distance_to_boundary, per_dot_sigma_eff,
p_flip_per_dot -- exactly as Section 8's own p_flip framework does:
propagate Sigma through the observation Jacobian H (H @ Sigma @ H.T) to
get an effective per-dot spread, no sampling required. The one difference
from noise_model.py's own use of this machinery: there, sigma_eff also
includes sensor readout noise R (relevant when asking "will the NEXT noisy
reading flip"); here there is no pending sensor reading being evaluated --
this is an internal belief-confidence question ("how sure is this
particle, on its own terms, that it is at target right now") -- so R is
omitted (equivalently R=0), and only the parameter-uncertainty term
diag(H @ Sigma @ H.T) contributes to the effective spread.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from noise_model import per_dot_sigma_eff, p_flip_per_dot
from particle_filter import RBParticleFilter
from kalman import ParticleKalmanFilter
from qarray_env import N_DOT, N_GATE, QArrayEnv, observation_jacobian


def target_occupation_probability(
    kalman: ParticleKalmanFilter,
    vg: np.ndarray,
    target_occupation: np.ndarray,
    *,
    T: float = 0.05,
) -> float:
    """Per-particle probability that the TRUE occupation at vg equals
    target_occupation, accounting for this particle's OWN continuous-
    parameter uncertainty (Sigma) -- not just whether its mean prediction
    happens to round to the target. See module docstring for the full
    derivation and why this avoids the nested-Monte-Carlo trap Section 2
    already rejected.

    Returns the product, across dots, of each dot's probability of sitting
    on the target-consistent side of its nearest boundary -- i.e.
    P(all 4 dots' true occupation matches target), assuming independence
    across dots. This is the SAME independence assumption
    noise_model.p_flip_total already makes (no cross-dot noise correlation
    is modeled anywhere else in this project either), applied here to a
    different question (internal confidence, not sensor-flip risk).
    """
    vg = np.asarray(vg, dtype=np.float64)
    target_occupation = np.asarray(target_occupation, dtype=np.float64)
    if target_occupation.shape != (N_DOT,):
        raise ValueError(
            f"target_occupation must have shape ({N_DOT},), got "
            f"{target_occupation.shape}"
        )

    env = QArrayEnv(kalman.params)
    H = observation_jacobian(kalman.params, vg, T=T)
    prediction = env.soft_prediction(vg, T=T)

    # Signed distance to this dot's own nearest boundary: positive when the
    # mean prediction sits on the target-consistent side, negative when it
    # doesn't. per_dot_distance_to_boundary (noise_model.py) is unsigned
    # (always in [0, 0.5]) and doesn't distinguish "confidently at target"
    # from "confidently at the WRONG adjacent state" -- both matter here,
    # so the sign is recovered explicitly rather than reusing that
    # function's return value directly.
    frac = prediction - np.floor(prediction)
    on_target_side = np.round(prediction) == target_occupation
    unsigned_distance = np.abs(frac - 0.5)

    # R=0: no pending sensor reading is being evaluated here -- see module
    # docstring. Only the parameter-uncertainty term propagated through H
    # contributes to the effective spread.
    zero_R = np.zeros((H.shape[0], H.shape[0]))
    sigma_eff = per_dot_sigma_eff(zero_R, H, kalman.Sigma)

    unsigned_p_flip = p_flip_per_dot(unsigned_distance, sigma_eff)
    # On the target-consistent side: probability of REMAINING correct is
    # (1 - flip probability). On the wrong side: probability of actually
    # BEING correct is exactly the (mirrored) flip-back probability --
    # the same two-sided Gaussian tail calculation, just naming the
    # complementary event.
    p_correct_per_dot = np.where(on_target_side, 1.0 - unsigned_p_flip, unsigned_p_flip)

    return float(np.prod(p_correct_per_dot))


def target_occupation_mass(
    pf: RBParticleFilter, vg: np.ndarray, target_occupation: np.ndarray, *, T: float = 0.05
) -> float:
    """Weighted EXPECTED probability mass on target_occupation: each
    particle contributes weight * target_occupation_probability(...), NOT
    a hard 0/1 indicator of whether its mean happens to round to target.
    See module docstring's "DISCRETE-STATE / POINT-ESTIMATE FIX" for why
    the hard-indicator version this replaces was a real bug, not a
    simplification.
    """
    vg = np.asarray(vg, dtype=np.float64)
    target_occupation = np.asarray(target_occupation, dtype=np.float64)
    if target_occupation.shape != (N_DOT,):
        raise ValueError(
            f"target_occupation must have shape ({N_DOT},), got "
            f"{target_occupation.shape}"
        )

    mass = 0.0
    for p in pf.particles:
        mass += p.weight * target_occupation_probability(p.kalman, vg, target_occupation, T=T)
    return mass


def belief_stable(
    pf: RBParticleFilter,
    vg: np.ndarray,
    target_occupation: np.ndarray,
    *,
    p_conf: float,
    T: float = 0.05,
) -> bool:
    """Guard A: does posterior mass on target_occupation exceed p_conf?"""
    return target_occupation_mass(pf, vg, target_occupation, T=T) > p_conf


def actuation_quiescent(proposed_step: np.ndarray, *, epsilon: float) -> bool:
    """Guard B: is the policy's own chosen next step magnitude below
    epsilon on EVERY gate? proposed_step is whatever the acquisition
    function/policy was about to do next (e.g. |v_next - v_current| per
    gate, across the FULL joint voltage space -- Primer Section 5's
    joint-4D-space acquisition already evaluates over all 4 gates at once,
    so this naturally covers gates belonging to the other DQD too, per the
    module docstring's point about DQD-2 perturbing DQD-1).

    Only meaningful for an ACTIVE policy -- see module docstring for why
    this guard is disabled (ConvergenceMonitor.require_actuation_quiescence
    =False) for a fixed, non-adaptive raster schedule.
    """
    proposed_step = np.asarray(proposed_step, dtype=np.float64)
    if proposed_step.shape != (N_GATE,):
        raise ValueError(
            f"proposed_step must have shape ({N_GATE},), got "
            f"{proposed_step.shape}"
        )
    return bool(np.all(np.abs(proposed_step) < epsilon))


@dataclass(frozen=True)
class ConvergenceStatus:
    """Snapshot of one step() call -- both guards reported SEPARATELY
    (Section 8's own two-failure-mode framing: A vs B need different
    guards, so a caller diagnosing a stalled run should be able to tell
    which one is failing, same two-line-diagnostic posture as
    control/phase_switch.py).

    actuation_quiescence_checked: False whenever the monitor was
    constructed with require_actuation_quiescence=False -- makes it
    explicit in the log that guard B was NOT evaluated for this step
    (actuation_quiescent is reported as True by convention in that case,
    but this field is what a caller should check before trusting that
    True as a meaningful guard result).
    """

    belief_stable: bool
    actuation_quiescent: bool
    actuation_quiescence_checked: bool
    consecutive_count: int
    converged: bool


@dataclass
class ConvergenceMonitor:
    """Stateful dual-criterion convergence monitor, Primer Section 8.
    Tracks a running count of CONSECUTIVE steps where the required
    guard(s) held; resets to 0 the moment any required guard fails (not a
    leaky/decaying counter -- Section 8 asks for N CONSECUTIVE steps, and
    a decaying counter would silently weaken that guarantee).

    p_conf/epsilon/N: all caller-supplied, not derived here. N in
    particular should come from noise_model.derive_consecutive_count, fed
    by a p_flip computed from the SAME noise/belief state this monitor is
    being run against (not a universal constant -- matches noise_model
    .py's own explicit warning about threshold_low/N not being universal
    constants).

    require_actuation_quiescence: True (default) for an ACTIVE policy
    (the SLAM method) -- guard B is evaluated normally. False for a fixed,
    non-adaptive schedule (baselines/raster_scan.py) -- see module
    docstring for why this is the correct operationalization of Section
    8's stopping rule for that case, not a weaker one. When False,
    proposed_step's SHAPE is still validated on every step() call (a
    caller passing garbage is still caught), but its VALUE never gates
    convergence.
    """

    p_conf: float
    epsilon: float
    N: int
    target_occupation: np.ndarray
    T: float = 0.05
    require_actuation_quiescence: bool = True
    _consecutive: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.target_occupation = np.asarray(self.target_occupation, dtype=np.float64)
        if self.target_occupation.shape != (N_DOT,):
            raise ValueError(
                f"target_occupation must have shape ({N_DOT},), got "
                f"{self.target_occupation.shape}"
            )
        if self.N < 1:
            raise ValueError(f"N must be >= 1, got {self.N}")

    def step(
        self, pf: RBParticleFilter, vg: np.ndarray, proposed_step: np.ndarray
    ) -> ConvergenceStatus:
        """Evaluate the required guard(s) for this step, update the
        consecutive counter, and report whether convergence has now been
        declared (counter reached N). Does NOT reset the counter to 0 on
        convergence -- once N is reached the run is done; callers should
        stop calling step() rather than rely on this method to keep
        counting past declared convergence."""
        stable = belief_stable(
            pf, vg, self.target_occupation, p_conf=self.p_conf, T=self.T
        )

        if self.require_actuation_quiescence:
            quiescent = actuation_quiescent(proposed_step, epsilon=self.epsilon)
        else:
            # Guard B not evaluated -- see module docstring. Still
            # validate proposed_step's shape so a caller passing garbage
            # is caught, but don't let its VALUE gate convergence.
            proposed_step = np.asarray(proposed_step, dtype=np.float64)
            if proposed_step.shape != (N_GATE,):
                raise ValueError(
                    f"proposed_step must have shape ({N_GATE},), got "
                    f"{proposed_step.shape}"
                )
            quiescent = True

        both_held = stable and quiescent

        if both_held:
            self._consecutive += 1
        else:
            self._consecutive = 0

        converged = self._consecutive >= self.N
        return ConvergenceStatus(
            belief_stable=stable,
            actuation_quiescent=quiescent,
            actuation_quiescence_checked=self.require_actuation_quiescence,
            consecutive_count=self._consecutive,
            converged=converged,
        )

    def reset(self) -> None:
        """Explicit reset of the consecutive-step counter -- e.g. for
        reusing one monitor instance across multiple sweep runs (Primer
        Section 7a) rather than constructing a fresh one each time. Not
        called automatically anywhere in this module."""
        self._consecutive = 0
