"""
gz_tuning.control.phase_switch

Primer Section 6: dual-line hard threshold with hysteresis, gating the
raw-signal/GP-style Phase 1 -> feature-based Phase 2 transition, once
enough distinct transitions have been located to actually FIT E_c/lever-arm
from spacing (not assume it) -- Natalia Ares's working-hypothesis framing,
demoted from headline-claim status per Primer Section 1.

Two diagnostic lines, both required to clear before switching:
  - "Blue line" -- within-particle certainty.
  - "Red line" -- between-particle agreement (belief/particle_filter.py's
    between_particle_covariance -- genuinely population-level; no
    per-particle version of this quantity exists).

MASS-BASED AGGREGATION -- a real, deliberate departure from
belief/particle_filter.py's within_particle_covariance(), not an oversight:

particle_filter.py's within_particle_covariance() computes Sigma_avg =
sum_i(w_i * Sigma_i) FIRST, then takes ONE normalized_logdet of that
averaged matrix -- this matches Section 6's literal notation ("Sigma_i
w_i*Sigma_i") and remains useful as a single logged diagnostic number. But
Section 6 is explicit that the SWITCH DECISION itself must NOT use this:
"Aggregation must be mass-based, not simple particle-weighted mean, for the
switch decision specifically: switch only once some fraction (e.g. >=90%)
of particle weight has both normalized-trace terms below threshold -- more
conservative, matches the actual claim 'belief has genuinely converged'
rather than 'the average happened to cross a line.'" A single averaged
matrix can have small logdet even when a meaningful minority of particle
mass is still individually uncertain (matrix-averaging-first can mask a
long tail); Section 6's mass-based fraction check cannot be fooled the same
way. So this module computes each particle's OWN normalized_logdet(Sigma_i)
individually and asks what FRACTION of total weight clears a threshold --
a strictly different (and, in the cases that matter, strictly more
conservative) computation than particle_filter.py's aggregate blue line,
despite both being called "the blue line" in the primer. Both remain
available: the mass-based fraction (this module) drives the actual switch;
particle_filter.py's population aggregate stays available directly as a
supplementary logged number, per Section 6's own preference for several
separately-inspectable diagnostics over one combined number.

No equivalent tension exists for the red line: between-particle covariance
of the means is inherently a single population-level quantity (there is no
"one particle's own between-particle disagreement"), so it is used exactly
as computed by particle_filter.py's between_particle_covariance(), a single
number.

BUG FIX (found via a real Phase-2 smoke run, confirmed structurally against
this file): an earlier version of this docstring's last sentence claimed
red was "compared against the same thresholds as the blue line's fraction
check." That was the actual shipped behavior -- one threshold_low/
threshold_high pair, used for both lines -- and it is wrong, not just
under-argued. A real run showed why directly: blue's mass fraction climbed
0.176 -> 0.826 over 60 steps while red sat flat around -4.1, against a
threshold_low of -57.36 -- roughly 53 nats away, on a quantity that isn't
trending toward it at all. The root cause isn't a design flaw in THIS
file's logic (splitting one threshold into two is a trivial field change,
which is what this version does); it's that the only threshold-producing
function that existed (noise_model.calibrate_threshold_low) constructs a
single ParticleKalmanFilter, not an ensemble -- between-particle covariance
is mathematically undefined for one particle, so that function was
structurally incapable of yielding a red-scaled number, and anyone who fed
its output to both lines (exactly as this file's old docstring instructed)
would hit this same wall. The fix has two parts: this file now takes
SEPARATE threshold pairs for each line (blue_threshold_low/high,
red_threshold_low/high), and noise_model.py gained
calibrate_red_threshold(), which actually instantiates a population (via
init_particle_filter) and reads off pf.between_particle_covariance() --
the only way a red-scaled reference number can exist at all.

THRESHOLDS ARE NOT DERIVED HERE. blue_threshold_low/high are expected to
come from simulator.noise_model.derive_phase_thresholds(calibrate_
threshold_low(...)); red_threshold_low/high from derive_phase_thresholds(
calibrate_red_threshold(...)) -- two independent calls, per the bug fix
above, not the same call reused. This module remains deliberately agnostic
to HOW those numbers were produced, mirroring belief/kalman.py and
belief/particle_filter.py's own separation of "correct recursion" from
"where the numeric constant comes from."

SECTION 6b ADDITION -- champion election for goal-directed navigation:
once Phase 2 triggers, control/goal_directed_policy.py needs a single
"current best hypothesis" particle to navigate toward (the primer's "the
current MAP estimate's target-occupation boundary"). elect_champion()
below returns the highest-weight particle -- the actual MAP-estimate
proxy for a particle filter -- deliberately NOT a weighted mean of
particles, which is the MMSE estimate, a different quantity: because
adaptive tempering (see particle_filter.py's _find_tempering_beta)
deliberately keeps multiple live discrete hypotheses alive rather than
collapsing early, an arithmetic mean of mu across particles that still
disagree can land in a physically meaningless region between two real
hypotheses (a transition boundary is not an average-able quantity the way
a scalar estimate is). PhaseSwitchController holds the elected champion
fixed across steps (avoiding target jitter between near-tied top-weight
particles), re-electing only on Phase 1 -> Phase 2 entry or when the held
particle object no longer lives in the current particle list -- see
PhaseSwitchController.champion's docstring for why that specific trigger
(not per-step re-election, not permanent stickiness) is the right scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from kalman import DEFAULT_SCALE, normalized_logdet
from particle_filter import OccupationParticle, RBParticleFilter


class Phase(Enum):
    PHASE_1_RAW_SIGNAL = 1
    PHASE_2_FEATURE_BASED = 2


def within_particle_mass_fraction_below(
    pf: RBParticleFilter,
    threshold: float,
    *,
    scale: np.ndarray = DEFAULT_SCALE,
) -> float:
    """Fraction of TOTAL PARTICLE WEIGHT belonging to particles whose OWN
    normalized_logdet(Sigma_i) is below `threshold`. See module docstring
    for why this differs from particle_filter.py's
    within_particle_covariance() (which averages matrices first) and why
    Section 6 requires this mass-based version specifically for the switch
    decision. normalized_logdet's own -inf case (a genuinely singular,
    valid PSD particle covariance) is handled correctly by ordinary `<`
    comparison -- -inf is always below any finite threshold, no special
    case needed here.
    """
    total = 0.0
    for p in pf.particles:
        logdet_i = normalized_logdet(p.kalman.Sigma, scale=scale)
        if logdet_i < threshold:
            total += p.weight
    return total


def elect_champion(pf: RBParticleFilter) -> OccupationParticle:
    """MAP-estimate proxy for a particle filter: the single highest-weight
    particle -- NOT a weighted mean of particles (that is the MMSE
    estimate, a different quantity; see module docstring for why the
    distinction is load-bearing here, not just terminological). Ties are
    broken by first occurrence; given the red-line precondition Phase 2
    entry already requires (between-particle agreement below
    red_threshold_low), an exact tie between substantively different
    hypotheses is not expected in practice.
    """
    return max(pf.particles, key=lambda p: p.weight)


def _is_live(particle: OccupationParticle, pf: RBParticleFilter) -> bool:
    """Identity check, NOT equality. OccupationParticle and
    ParticleKalmanFilter are undecorated @dataclass, so the generated
    `__eq__` compares fields -- including `mu`, a numpy array. Any
    `in`-style or `==` check against a held particle reference hits numpy
    array comparison and raises `ValueError: truth value of an array is
    ambiguous` the first time it actually runs, not at review time. `is`
    identity is the only safe way to ask "does this exact object still
    live in pf.particles" (e.g. to detect that a resample() has replaced
    the entire particle list since this reference was captured).
    """
    return any(particle is p for p in pf.particles)


@dataclass(frozen=True)
class PhaseSwitchDiagnostics:
    """Snapshot of both diagnostic lines against both thresholds, for
    logging -- Section 6's explicit preference for two separately-
    inspectable numbers over one combined signal: 'if the switch misfires,
    the two-line log immediately shows whether it's within-particle
    overconfidence or genuine between-particle disagreement at fault.'
    """

    blue_mass_fraction_below_low: float
    blue_mass_fraction_below_high: float
    red_line: float
    red_below_low: bool
    red_below_high: bool


@dataclass
class PhaseSwitchController:
    """Stateful hysteresis controller for Primer Section 6's Phase 1 <->
    Phase 2 transition. Holds the CURRENT phase across calls -- this is
    genuinely stateful (hysteresis requires remembering which side of the
    band was last committed to), unlike the pure functions above, which
    depend only on the current belief.

    blue_threshold_low/blue_threshold_high, red_threshold_low/
    red_threshold_high: see module docstring -- not computed here; each
    pair must come from its own calibration call (calibrate_threshold_low
    for blue, calibrate_red_threshold for red -- NOT the same call reused
    for both, which was the bug this split fixes).
    mass_fraction_required: Section 6's ">=90%" figure -- kept as an
    explicit parameter (default 0.9) rather than hardcoded, since the
    primer's own phrasing ("e.g.") flags it as illustrative, not fixed.
    """

    blue_threshold_low: float
    blue_threshold_high: float
    red_threshold_low: float
    red_threshold_high: float
    mass_fraction_required: float = 0.9
    scale: np.ndarray = field(default_factory=lambda: DEFAULT_SCALE.copy())
    phase: Phase = Phase.PHASE_1_RAW_SIGNAL
    _champion: OccupationParticle | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.blue_threshold_high <= self.blue_threshold_low:
            raise ValueError(
                f"blue_threshold_high ({self.blue_threshold_high}) must "
                f"exceed blue_threshold_low ({self.blue_threshold_low}) -- "
                f"see noise_model.derive_phase_thresholds, which enforces "
                f"this via an additive (not multiplicative) hysteresis gap."
            )
        if self.red_threshold_high <= self.red_threshold_low:
            raise ValueError(
                f"red_threshold_high ({self.red_threshold_high}) must "
                f"exceed red_threshold_low ({self.red_threshold_low})."
            )
        if not (0.0 < self.mass_fraction_required <= 1.0):
            raise ValueError(
                f"mass_fraction_required must be in (0, 1], got "
                f"{self.mass_fraction_required}"
            )

    @property
    def champion(self) -> OccupationParticle | None:
        """The elected MAP-proxy particle (see elect_champion) driving
        Section 6b's goal-directed navigation. None while in Phase 1.

        Held fixed across steps within Phase 2 to avoid the navigation
        target jittering between near-tied top-weight particles step to
        step -- EXCEPT across a resample() event, which forces
        re-election: resample() replaces `pf.particles` wholesale and its
        own mu-jitter step deliberately breaks exact particle lineage by
        design (see particle_filter.py's resample() docstring), so "the
        same particle survived resampling" is not a claim worth trying to
        preserve across that boundary -- and a real ensemble-level belief
        shift large enough to trigger a resample is a legitimate reason to
        reconsider the navigation target, unlike identity churn alone.

        Practical consequence, worth being explicit about rather than
        leaving implicit in the mechanism: the Phase-2 navigation target
        CAN visibly jump immediately after a resample. That is the
        intended tradeoff (see above), not a bug -- but it is a real,
        user-visible behavior of the tuning loop.
        """
        return self._champion

    def diagnostics(self, pf: RBParticleFilter) -> PhaseSwitchDiagnostics:
        """Compute both lines against both thresholds without mutating
        controller state -- exposed separately from step() so callers can
        log/inspect before (or without) actually advancing the hysteresis
        state machine."""
        blue_mass_low = within_particle_mass_fraction_below(
            pf, self.blue_threshold_low, scale=self.scale
        )
        blue_mass_high = within_particle_mass_fraction_below(
            pf, self.blue_threshold_high, scale=self.scale
        )
        red_line = pf.between_particle_covariance(scale=self.scale)
        return PhaseSwitchDiagnostics(
            blue_mass_fraction_below_low=blue_mass_low,
            blue_mass_fraction_below_high=blue_mass_high,
            red_line=red_line,
            red_below_low=red_line < self.red_threshold_low,
            red_below_high=red_line < self.red_threshold_high,
        )

    def step(self, pf: RBParticleFilter) -> PhaseSwitchDiagnostics:
        """Evaluate the current belief and advance the hysteresis state
        machine per Section 6:
          - PHASE_1 -> PHASE_2 requires BOTH lines below their own
            threshold_low (blue: mass-based fraction >= mass_fraction_
            required against blue_threshold_low; red: pf.between_
            particle_covariance() below red_threshold_low).
          - PHASE_2 -> PHASE_1 (revert) requires EITHER line climbing back
            above its own threshold_high -- a real, reportable regression (Primer
            Section 8's failure-mode-B framing: a genuine physical
            perturbation, not noise, can legitimately push the belief back
            into Phase 1; this is a feature of the hysteresis band, not a
            bug to suppress).

        Also advances champion election (Section 6b, see the `champion`
        property's docstring): elects on Phase 1 -> Phase 2 entry, holds
        fixed while the held particle is still live, force-re-elects if a
        resample has replaced it, and clears to None on any reversion to
        Phase 1.

        Returns the diagnostics computed for this call; self.phase (and
        self.champion) reflect the (possibly updated) state after this
        call.
        """
        diag = self.diagnostics(pf)

        if self.phase is Phase.PHASE_1_RAW_SIGNAL:
            if (
                diag.blue_mass_fraction_below_low >= self.mass_fraction_required
                and diag.red_below_low
            ):
                self.phase = Phase.PHASE_2_FEATURE_BASED
        else:  # PHASE_2_FEATURE_BASED
            if (
                diag.blue_mass_fraction_below_high < self.mass_fraction_required
                or not diag.red_below_high
            ):
                self.phase = Phase.PHASE_1_RAW_SIGNAL

        if self.phase is Phase.PHASE_2_FEATURE_BASED:
            if self._champion is None or not _is_live(self._champion, pf):
                self._champion = elect_champion(pf)
        else:
            self._champion = None

        return diag
