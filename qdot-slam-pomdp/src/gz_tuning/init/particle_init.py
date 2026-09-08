"""
gz_tuning.init.particle_init

Primer Section 7: particle initialization for the Rao-Blackwellized filter
(belief/particle_filter.py), covering the two genuinely separate decisions
Section 7 insists be kept separate in implementation:

  (a) GROUND-TRUTH SWEEP (Claim 2): a range of TRUE cross_capacitance
      values the simulator is configured with, including exactly zero as a
      control. Nothing to do with what any particle BELIEVES -- this
      decides what's actually true for a given run.
  (b) PRIOR SEEDING (this module's main content): what particles believe
      at t=0, within any single run. Section 7b requires the coupling term
      specifically (not the other 4 tracked parameters) to use a
      STRATIFIED draw across particles -- some seeded near-zero, some
      moderate, some strong -- rather than one shared diffuse Gaussian
      identical across every particle. Reason (Section 7b, in full): if
      every particle shares an identical prior, continuous-parameter
      diversity only emerges from particles disagreeing about DISCRETE
      transitions (data-association divergence), which can stay degenerate
      early on -- exactly when BALD is near-flat (the primer's documented
      aliasing failure mode) -- silently gutting the between-particle
      ("red line", control/phase_switch.py) diagnostic before it has
      anything real to catch. Coupling specifically warrants this because
      zero is a real, physically plausible, INTERESTING hypothesis
      boundary -- unlike E_c, which is always positive and bounded away
      from zero by construction, so a shared prior for E_c/alpha carries no
      equivalent risk.

METHODOLOGICAL INVARIANT, enforced here with a runtime check rather than
left as a documentation-only promise (matching this project's established
pattern -- e.g. kalman.py's feasibility guard, noise_model.py's PSD check):
the prior scheme from (b) MUST be IDENTICAL across every point in the sweep
from (a). Narrowing the prior based on knowledge of a specific run's ground
truth (even unintentionally -- Section 7's own example: "reusing a prior
'tuned while debugging on one test device'") would mean testing "recovery
when the prior happens to bracket the truth" rather than genuine ground-
zero recovery, undermining Claim 1 (not just Claim 2).
validate_prior_consistency_across_sweep below is the enforcement mechanism
-- call it once, across every (SharedPriorSpec, list[CouplingStratum]) pair
actually used in a sweep run, before trusting the sweep's results.

FEASIBILITY, reusing the same guard kalman.py's update() already has: a
sampled draw of (E_c1, E_c2, alpha1, alpha2, cross_capacitance) is not
guaranteed constructible (Primer Section 4b's real, alpha-dependent
capacitance-budget constraint -- see qarray_env.py). This module checks
feasibility by attempting to actually construct a QArrayEnv for each
particle's initial draw (the same mechanism kalman.py's update() feasibility
guard uses), and RE-DRAWS ALL FIVE parameters fresh (not just the
infeasible-seeming ones -- E_c/alpha/coupling jointly determine
feasibility, so a "fix just the offending dimension" patch would not be
principled) up to max_resample_attempts times before raising.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kalman import N_PARAMS, ParticleKalmanFilter
from particle_filter import OccupationParticle, RBParticleFilter
from qarray_env import DeviceParams, InfeasibleDeviceParamsError, QArrayEnv, TRACKED_PARAM_FIELDS

# Section 7b's stratification applies to this one dimension only. Derived
# from the shared constant rather than hardcoded as index 4, so this stays
# correct if TRACKED_PARAM_FIELDS' order ever changes.
_COUPLING_IDX = TRACKED_PARAM_FIELDS.index("cross_capacitance")
_SHARED_FIELDS = tuple(f for f in TRACKED_PARAM_FIELDS if f != "cross_capacitance")


@dataclass(frozen=True)
class SharedPriorSpec:
    """Prior for the 4 tracked parameters NOT subject to Section 7b's
    stratification (E_c1, E_c2, alpha1, alpha2). Diagonal Gaussian, IDENTICAL
    across all particles AND across every point in the coupling-strength
    sweep (Section 7's methodological invariant) -- construct ONE instance
    per experiment configuration and reuse it, unchanged, across the whole
    sweep; do not vary it per sweep point (enforced by
    validate_prior_consistency_across_sweep, not just documented here).
    """

    mean: dict[str, float]  # keys: exactly _SHARED_FIELDS
    std: dict[str, float]   # same keys -- diagonal only, no init-time
                             # cross-parameter correlation assumed

    def __post_init__(self) -> None:
        for d, name in ((self.mean, "mean"), (self.std, "std")):
            missing = set(_SHARED_FIELDS) - set(d)
            extra = set(d) - set(_SHARED_FIELDS)
            if missing or extra:
                raise ValueError(
                    f"SharedPriorSpec.{name} keys must be exactly "
                    f"{_SHARED_FIELDS}; missing={missing}, extra={extra}"
                )
        if any(s <= 0 for s in self.std.values()):
            raise ValueError("SharedPriorSpec.std values must all be > 0")


@dataclass(frozen=True)
class CouplingStratum:
    """One stratum of Section 7b's stratified cross_capacitance draw --
    e.g. "near-zero", "moderate", "strong". `weight` is this stratum's
    fraction of the total particle population; weights across every
    stratum passed to init_particle_filter must sum to 1.0 (checked
    there, not here, since a lone stratum can't validate its siblings)."""

    mean: float
    std: float
    weight: float

    def __post_init__(self) -> None:
        if self.std <= 0:
            raise ValueError(f"CouplingStratum.std must be > 0, got {self.std}")
        if not (0.0 < self.weight <= 1.0):
            raise ValueError(f"CouplingStratum.weight must be in (0, 1], got {self.weight}")


def _allocate_stratum_counts(n_particles: int, strata: list[CouplingStratum]) -> list[int]:
    """Largest-remainder allocation of n_particles across strata
    proportional to `weight`, guaranteeing the counts sum EXACTLY to
    n_particles (simple floor+round-robin can under/overshoot by a few
    particles at small n_particles, which matters here since Section 7b's
    stratification is meant to guarantee real representation in EVERY
    stratum, not just approximately)."""
    weights = np.array([s.weight for s in strata], dtype=np.float64)
    if not np.isclose(weights.sum(), 1.0, atol=1e-8):
        raise ValueError(
            f"CouplingStratum weights must sum to 1.0, got {weights.sum()}"
        )
    raw = weights * n_particles
    floors = np.floor(raw).astype(int)
    remainder = n_particles - floors.sum()
    fractional = raw - floors
    # give the remaining `remainder` particles to the strata with the
    # largest fractional part (standard largest-remainder / Hamilton method)
    order = np.argsort(-fractional)
    counts = floors.copy()
    for idx in order[:remainder]:
        counts[idx] += 1
    return counts.tolist()


def _sample_particle_mu(
    shared_prior: SharedPriorSpec, stratum: CouplingStratum, rng: np.random.Generator
) -> np.ndarray:
    """One draw of the 5-vector (E_c1, E_c2, alpha1, alpha2,
    cross_capacitance) in TRACKED_PARAM_FIELDS order: the 4 shared fields
    from shared_prior (identical distribution for every particle, per
    Section 7b), cross_capacitance from THIS particle's assigned stratum.
    """
    vec = np.zeros(N_PARAMS, dtype=np.float64)
    for i, field_name in enumerate(TRACKED_PARAM_FIELDS):
        if field_name == "cross_capacitance":
            vec[i] = rng.normal(stratum.mean, stratum.std)
        else:
            vec[i] = rng.normal(shared_prior.mean[field_name], shared_prior.std[field_name])
    return vec


def init_particle_filter(
    n_particles: int,
    shared_prior: SharedPriorSpec,
    coupling_strata: list[CouplingStratum],
    prior_sigma_diag: np.ndarray,
    *,
    E_cm_intra1: float,
    E_cm_intra2: float,
    rng: np.random.Generator | None = None,
    max_resample_attempts: int = 50,
    max_backoffs: int = 5,
) -> RBParticleFilter:
    """Draw n_particles initial hypotheses per Section 7b and assemble an
    RBParticleFilter, each particle's OWN Sigma initialized to
    diag(prior_sigma_diag) -- IDENTICAL across particles (only the MEAN
    draw differs; see module docstring for why an identical Sigma alongside
    a stratified mu is exactly the scenario Section 6's red-line diagnostic
    is designed to catch, not a shortcoming of this function).

    E_cm_intra1/E_cm_intra2: fixed, non-tracked structural parameters,
    shared by every particle (matches ParticleKalmanFilter's own treatment
    -- these are not part of the 5-parameter Kalman state; see kalman.py).

    Infeasible draws (Primer Section 4b) are re-drawn FROM SCRATCH (all 5
    dimensions, not just the offending one -- see module docstring) up to
    max_resample_attempts times per particle before raising
    InfeasibleDeviceParamsError to the caller, unmodified, so the caller
    sees the same error type/message qarray_env.py's own feasibility check
    would raise directly.
    """
    if rng is None:
        rng = np.random.default_rng()
    prior_sigma_diag = np.asarray(prior_sigma_diag, dtype=np.float64)
    if prior_sigma_diag.shape != (N_PARAMS,):
        raise ValueError(
            f"prior_sigma_diag must have shape ({N_PARAMS},), got "
            f"{prior_sigma_diag.shape}"
        )
    Sigma0 = np.diag(prior_sigma_diag)

    counts = _allocate_stratum_counts(n_particles, coupling_strata)

    particles: list[OccupationParticle] = []
    for stratum, count in zip(coupling_strata, counts):
        for _ in range(count):
            last_error: Exception | None = None
            for _attempt in range(max_resample_attempts):
                mu = _sample_particle_mu(shared_prior, stratum, rng)
                kwargs = dict(zip(TRACKED_PARAM_FIELDS, mu))
                candidate_params = DeviceParams(
                    E_cm_intra1=E_cm_intra1, E_cm_intra2=E_cm_intra2, **kwargs
                )
                try:
                    QArrayEnv(candidate_params)  # feasibility check via construction
                except InfeasibleDeviceParamsError as e:
                    last_error = e
                    continue
                pkf = ParticleKalmanFilter(
                    mu=mu, Sigma=Sigma0.copy(),
                    E_cm_intra1=E_cm_intra1, E_cm_intra2=E_cm_intra2,
                    max_backoffs=max_backoffs,
                )
                particles.append(OccupationParticle(kalman=pkf, weight=1.0))
                break
            else:
                raise InfeasibleDeviceParamsError(
                    f"Could not draw a feasible particle from stratum "
                    f"(mean={stratum.mean}, std={stratum.std}) after "
                    f"{max_resample_attempts} attempts. Last error: "
                    f"{last_error}"
                )

    return RBParticleFilter(particles=particles)


def validate_prior_consistency_across_sweep(
    specs: list[tuple[SharedPriorSpec, list[CouplingStratum]]]
) -> None:
    """Enforce Section 7's methodological invariant across an actual sweep:
    every (SharedPriorSpec, coupling_strata) pair used across the sweep
    from (a) must be IDENTICAL. Raises ValueError naming the first
    mismatch found, rather than silently allowing per-sweep-point prior
    drift -- this is exactly the mistake Section 7 warns would undermine
    Claim 1, so it's checked with a real assertion, not left as a comment.
    """
    if len(specs) < 2:
        return
    ref_prior, ref_strata = specs[0]
    for i, (prior, strata) in enumerate(specs[1:], start=1):
        if prior.mean != ref_prior.mean or prior.std != ref_prior.std:
            raise ValueError(
                f"Sweep point {i}'s SharedPriorSpec differs from sweep "
                f"point 0's -- Section 7's methodological invariant "
                f"requires an IDENTICAL prior scheme across every point "
                f"in the ground-truth coupling sweep (Claim 1's "
                f"credibility depends on this: a prior that happens to "
                f"vary with knowledge of each run's ground truth would "
                f"mean testing 'recovery when the prior brackets the "
                f"truth', not genuine ground-zero recovery)."
            )
        if len(strata) != len(ref_strata) or any(
            (s.mean, s.std, s.weight) != (r.mean, r.std, r.weight)
            for s, r in zip(strata, ref_strata)
        ):
            raise ValueError(
                f"Sweep point {i}'s coupling_strata differ from sweep "
                f"point 0's -- see SharedPriorSpec check above; the same "
                f"invariant applies to the stratification scheme itself."
            )


@dataclass(frozen=True)
class CouplingSweepPoint:
    """One point in Primer Section 7a's ground-truth coupling-strength
    sweep (Claim 2) -- entirely about what's TRUE for this run, unrelated
    to what any particle believes at init (see module docstring)."""

    true_cross_capacitance: float
    label: str  # e.g. "zero_control", "weak", "moderate", "strong"


def build_coupling_sweep(
    base_params: DeviceParams, sweep_values: list[float], labels: list[str] | None = None
) -> list[tuple[CouplingSweepPoint, DeviceParams]]:
    """Ground-truth DeviceParams for each point in a coupling-strength
    sweep, holding every other parameter fixed at base_params. Section 7a
    requires EXACTLY ZERO be included as a control point -- checked here,
    not left to the caller to remember, since Claim 2's stated result
    ("coupling-aware costs nothing when there's no coupling") depends on
    it being present.
    """
    if 0.0 not in sweep_values:
        raise ValueError(
            "Section 7a requires an exact-zero coupling control point in "
            "the sweep -- 0.0 not found in sweep_values."
        )
    if labels is None:
        labels = [f"coupling_{v:g}" for v in sweep_values]
    if len(labels) != len(sweep_values):
        raise ValueError("labels must have the same length as sweep_values")

    out = []
    for v, label in zip(sweep_values, labels):
        point = CouplingSweepPoint(true_cross_capacitance=v, label=label)
        params = base_params.replace(cross_capacitance=v)
        out.append((point, params))
    return out
