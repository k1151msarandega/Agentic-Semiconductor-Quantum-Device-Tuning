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
number compared against the same thresholds as the blue line's fraction
check.

THRESHOLDS ARE NOT DERIVED HERE. threshold_low/threshold_high are expected
to come from simulator.noise_model.derive_phase_thresholds (fed by
calibrate_threshold_low) -- this module is deliberately agnostic to HOW
those numbers were produced, mirroring belief/kalman.py and
belief/particle_filter.py's own separation of "correct recursion" from
"where the numeric constant comes from."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from kalman import DEFAULT_SCALE, normalized_logdet
from particle_filter import RBParticleFilter


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

    threshold_low/threshold_high: see module docstring -- not computed
    here.
    mass_fraction_required: Section 6's ">=90%" figure -- kept as an
    explicit parameter (default 0.9) rather than hardcoded, since the
    primer's own phrasing ("e.g.") flags it as illustrative, not fixed.
    """

    threshold_low: float
    threshold_high: float
    mass_fraction_required: float = 0.9
    scale: np.ndarray = field(default_factory=lambda: DEFAULT_SCALE.copy())
    phase: Phase = Phase.PHASE_1_RAW_SIGNAL

    def __post_init__(self) -> None:
        if self.threshold_high <= self.threshold_low:
            raise ValueError(
                f"threshold_high ({self.threshold_high}) must exceed "
                f"threshold_low ({self.threshold_low}) -- see "
                f"noise_model.derive_phase_thresholds, which enforces this "
                f"via an additive (not multiplicative) hysteresis gap."
            )
        if not (0.0 < self.mass_fraction_required <= 1.0):
            raise ValueError(
                f"mass_fraction_required must be in (0, 1], got "
                f"{self.mass_fraction_required}"
            )

    def diagnostics(self, pf: RBParticleFilter) -> PhaseSwitchDiagnostics:
        """Compute both lines against both thresholds without mutating
        controller state -- exposed separately from step() so callers can
        log/inspect before (or without) actually advancing the hysteresis
        state machine."""
        blue_mass_low = within_particle_mass_fraction_below(
            pf, self.threshold_low, scale=self.scale
        )
        blue_mass_high = within_particle_mass_fraction_below(
            pf, self.threshold_high, scale=self.scale
        )
        red_line = pf.between_particle_covariance(scale=self.scale)
        return PhaseSwitchDiagnostics(
            blue_mass_fraction_below_low=blue_mass_low,
            blue_mass_fraction_below_high=blue_mass_high,
            red_line=red_line,
            red_below_low=red_line < self.threshold_low,
            red_below_high=red_line < self.threshold_high,
        )

    def step(self, pf: RBParticleFilter) -> PhaseSwitchDiagnostics:
        """Evaluate the current belief and advance the hysteresis state
        machine per Section 6:
          - PHASE_1 -> PHASE_2 requires BOTH lines below threshold_low
            (blue: mass-based fraction >= mass_fraction_required; red:
            single value below threshold_low).
          - PHASE_2 -> PHASE_1 (revert) requires EITHER line climbing back
            above threshold_high -- a real, reportable regression (Primer
            Section 8's failure-mode-B framing: a genuine physical
            perturbation, not noise, can legitimately push the belief back
            into Phase 1; this is a feature of the hysteresis band, not a
            bug to suppress).
        Returns the diagnostics computed for this call; self.phase reflects
        the (possibly updated) state after this call.
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

        return diag
