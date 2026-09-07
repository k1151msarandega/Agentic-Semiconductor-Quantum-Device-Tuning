"""
gz_tuning.simulator.qarray_env

Thin wrapper around qarray.DotArray for the joint 4-dot (two coupled DQDs)
ground-zero environment.

=== v2 CHANGES -- read before trusting v1 numbers anywhere ===

This replaces an earlier version that had two real bugs, both found by
reading QArray's source directly and verifying against actual DotArray
behavior (not from docstrings or hand-derivation):

  1. E_c was never wired into the physics at all (Cdd diagonal hardcoded to
     0.0). An attempted fix using a closed-form energy-matrix inversion
     (E_matrix -> Cdd_maxwell = inv(E_matrix) -> back out raw Cdd/Cgd) turned
     out to be structurally infeasible for ANY genuinely coupled system,
     independent of parameter magnitude: Cdd_maxwell's off-diagonal entries
     are necessarily negative for a physically sensible E_matrix (standard
     Maxwell capacitance sign convention), and QArray's raw Cgd input is
     required strictly non-negative -- so Cgd_raw = Cdd_maxwell @ alpha_matrix
     comes out infeasible whenever there's real inter-dot coupling, at ANY
     value of alpha (verified numerically across alpha in [0.01, 1.0], 8/12
     off-diagonal entries stayed negative throughout). This is not a
     calibration problem, it's a structural mismatch -- abandoned, not
     patched.
  2. t_c1/t_c2 were listed as tracked continuous parameters, but QArray's
     constant-interaction ground-state model is a pure bilinear quadratic
     form in occupation -- there's no mechanism for a state-conditional
     correction (which is what quantum tunnel coupling actually is). What
     the off-diagonal intra-DQD Cdd term captures is classical mutual
     charging energy, a different physical quantity. Renamed accordingly:
     E_cm_intra1/E_cm_intra2. True t_c estimation is deferred to the QDarts
     cross-check tier (Primer Section 3), where it's a real, identifiable
     quantity via QDarts's many-body Hamiltonian -- not implemented here.

CURRENT (v2) approach -- verified working, not just derived:

Parameters split into two groups:
  - Direct, always-feasible physical quantities (no inversion, non-negative
    by construction): E_cm_intra1, E_cm_intra2, cross_capacitance (all raw
    Cdd off-diagonal entries), alpha1, alpha2 (raw Cgd diagonal entries).
  - E_c1, E_c2: NOT set directly. All 4 dots' raw Cdd DIAGONAL entries
    (self-capacitance terms, previously hardcoded to 0.0) are solved
    JOINTLY (scipy.optimize.root, 4 equations/4 unknowns simultaneously)
    so the resulting Maxwell-form charging energy (diag(inv(Cdd_maxwell)))
    matches each dot's target E_c, holding the direct parameters above
    fixed. An earlier version solved this sequentially, one dot at a time
    -- found to be biased by up to 18% at this file's alpha=0.05 regime,
    since each dot's independent solve ignored how its own self-capacitance
    choice shifts its coupled neighbors' achieved E_c. The joint solve is
    verified to hit target to ~1e-15 residual (see build_capacitance_matrices
    tests) -- treat this as exact, not approximate.

FEASIBILITY IS REAL AND ALPHA-DEPENDENT -- this is physics, not a bug:
lever arm and charging energy are NOT independent free parameters. Larger
alpha eats into the same capacitance budget that determines E_c's ceiling
(E_c ~ 1/C_total, and the gate itself contributes to C_total). Verified
numerically: alpha=1.0 (used throughout earlier conversation and the
Section 9 benchmark script) caps achievable E_c at ~0.8 given the coupling
values in use here -- E_c=2.5 was never achievable at alpha=1.0, at any
self-capacitance value.

alpha=0.2 was tried as a first correction and is ALSO NOT SAFE -- verified
numerically to have only ~6-18% margin against E_c=2.5/2.7 at this file's
default (weak) coupling values, and to become catastrophically infeasible
(E_c ceiling collapses to ~0.1) once cross_capacitance reaches values
comparable to E_cm_intra (a "strong coupling" sweep endpoint, Primer
Section 7a) -- meaning alpha=0.2 would make the coupling-strength sweep
central to Claim 2 impossible to run at its intended strong-coupling end.

DEFAULT IS NOW alpha=0.05 (within the literature-typical 0.05-0.4 range,
at the conservative end deliberately). Verified: ceiling on E_c1 stays
above 4.3 even at cross_capacitance=1.0 (a genuinely strong coupling point,
~3x E_cm_intra), comfortable margin over the E_c=2.5/2.7 targets used
throughout. This is still a corrected placeholder, not a calibrated device
constant -- if you have real device-specific lever-arm values, use those
instead. But it's chosen with an explicit, checked margin against the
sweep this project actually needs to run, not picked by feel.

OPEN, NOT YET CHASED DOWN: constructing the DotArray at these parameter
values triggers QArray's own internal warning ("cdd matrix contains off
diagonal elements which are sufficiently strong that it cannot be treated
as an approximately diagonal matrix... may produce distortions"). This
suggests `algorithm='thresholded'` (a DotArray constructor option) may be
more appropriate than the `'default'` used throughout -- not yet
investigated. Flagged, not fixed.

Dot/gate index convention (unchanged from v1, matches Section 9 benchmark):
  dots 0,1 = DQD-A ; dots 2,3 = DQD-B
  Cdd[0,1]: E_cm_intra1 (DQD-A mutual charging energy)
  Cdd[2,3]: E_cm_intra2 (DQD-B mutual charging energy)
  Cdd[1,2]: cross_capacitance (inter-DQD coupling)
  gates 0..3 each control the same-numbered dot only (alpha1,alpha1,alpha2,alpha2)
  -- cross-gate leakage (GATE_LEAKAGE_FRAC in v1) has been DROPPED, not
  carried forward: it was an arbitrary constant with no physical grounding,
  and the v2 parameterization has no natural place to reintroduce it
  without reopening the infeasibility problem this file exists to avoid.
  Add it back deliberately (as a fixed, small, explicit alpha_matrix
  off-diagonal, with its own feasibility check) if cross-gate leakage turns
  out to matter for your results -- don't silently reintroduce it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import root

from qarray import DotArray

N_DOT = 4
N_GATE = 4

# The 5 continuous parameters each particle's Kalman filter tracks, per
# Primer Section 4's correction (t_c dropped -- see module docstring).
# This exact order is load-bearing: observation_jacobian's columns and
# belief/kalman.py's state vector must agree on it. Import this constant
# rather than hardcoding the tuple a second time anywhere.
TRACKED_PARAM_FIELDS = ("E_c1", "E_c2", "alpha1", "alpha2", "cross_capacitance")


@dataclass(frozen=True)
class DeviceParams:
    """One hypothesis for the joint 4-dot device's continuous parameters.

    t_c1/t_c2 deliberately absent -- see module docstring point 2. This is
    now a 5-parameter state (E_c1, E_c2, alpha1, alpha2, cross_capacitance),
    matching the Primer Section 4 correction each particle's Kalman filter
    tracks under QArray as primary simulator.
    """

    E_c1: float  # DQD-A charging energy (dots 0,1). Root-solved into Cdd diagonal -- see build_capacitance_matrices.
    E_c2: float  # DQD-B charging energy (dots 2,3). Same.
    alpha1: float  # DQD-A lever arm -> Cgd diagonal, dots 0,1. Real devices: ~0.05-0.4.
    alpha2: float  # DQD-B lever arm -> Cgd diagonal, dots 2,3.
    E_cm_intra1: float  # DQD-A mutual charging energy -> Cdd[0,1]. Renamed from t_c1 -- classical, not tunnel coupling.
    E_cm_intra2: float  # DQD-B mutual charging energy -> Cdd[2,3]. Renamed from t_c2.
    cross_capacitance: float  # Inter-DQD coupling -> Cdd[1,2]. Primer Section 7b's
    # near-zero-is-a-real-hypothesis parameter -- unchanged in role from v1.

    def replace(self, **kwargs) -> "DeviceParams":
        return replace(self, **kwargs)


class InfeasibleDeviceParamsError(ValueError):
    """Raised when a requested E_c cannot be achieved given fixed alpha/
    coupling values, at any non-negative self-capacitance. This is a real
    physical constraint (E_c ~ 1/C_total, gate is part of C_total), not a
    numerical-tolerance issue -- see module docstring."""


def _solve_self_capacitance_joint(
    target_E_c: np.ndarray,
    Cdd_offdiag_fixed: np.ndarray,
    Cgd_fixed: np.ndarray,
    *,
    tol: float = 1e-13,
    x0: np.ndarray | None = None,
) -> np.ndarray:
    """Jointly root-find all 4 raw Cdd diagonal entries (self-capacitances)
    such that the resulting Maxwell-form charging energy diag(inv(Cdd_maxwell))
    matches target_E_c elementwise, holding off-diagonal Cdd and all of Cgd
    fixed. This is a simultaneous 4-equation, 4-unknown solve -- NOT the
    per-dot-sequential approximation an earlier version of this file used,
    which was found to be biased by up to ~18% at this file's alpha=0.05
    parameter regime (each dot's independent solve ignored how its own
    self-capacitance choice shifts its coupled neighbors' achieved E_c).
    Verified (empirically, in the conversation that produced this fix) to
    hit target to ~1e-15 residual with scipy.optimize.root's default 'hybr'
    method and an initial guess derived from the old sequential estimate.

    Mirrors QArray's own convert_to_maxwell construction exactly (verified
    against qarray.DotArrays._helper_functions.convert_to_maxwell's source,
    not assumed) so the solve is checking the same quantity DotArray will
    actually construct internally.

    x0: optional warm-start override. PERFORMANCE NOTE, verified directly:
    this joint solve (not DotArray construction, and not the
    update_capacitance_matrices swap) is the dominant cost of building a
    capacitance matrix pair -- ~516us/call cold-started, measured directly,
    which makes Primer Section 9's ~270us/particle swap benchmark STALE (that
    benchmark predates E_c being wired in at all -- Cdd's diagonal was
    hardcoded to 0 then, so no joint solve existed to measure). For small
    parameter perturbations (e.g. observation_jacobian's finite-difference
    step), warm-starting from a nearby already-solved self-capacitance
    vector cuts this to ~209us/call with residual ~1e-15 (verified) --
    essentially free correctness, real speedup. observation_jacobian uses
    this; any other caller doing repeated small-perturbation solves should
    too, rather than re-deriving the generic heuristic guess from scratch
    each time.
    """

    def residual(c_self_vec: np.ndarray) -> np.ndarray:
        Cdd_raw = Cdd_offdiag_fixed.copy()
        np.fill_diagonal(Cdd_raw, c_self_vec)
        cdd_sum = Cdd_raw.sum(axis=1)  # row sum BEFORE zeroing diagonal -- matches QArray's convert_to_maxwell exactly
        cgd_sum = Cgd_fixed.sum(axis=1)
        offdiag_only = Cdd_raw.copy()
        np.fill_diagonal(offdiag_only, 0.0)
        Cdd_maxwell = np.diag(cdd_sum + cgd_sum) - offdiag_only
        achieved_E_c = np.diag(np.linalg.inv(Cdd_maxwell))
        return achieved_E_c - target_E_c

    if x0 is None:
        # Generic heuristic guess (the old sequential-solve estimate) --
        # only used when no better (warm-start) guess is available.
        cgd_sum = Cgd_fixed.sum(axis=1)
        offdiag_sum = Cdd_offdiag_fixed.sum(axis=1)
        x0 = np.maximum(1.0 / target_E_c - cgd_sum - offdiag_sum, 1e-6)
        used_warm_start = False
    else:
        used_warm_start = True

    sol = root(residual, x0, method="hybr", tol=tol)

    if not sol.success and used_warm_start:
        # Found empirically: hybr's internal convergence check can fail at
        # specific small perturbations even when the warm-start is already
        # very close to the true root (verified case: alpha2 decreased by
        # 1e-4 -- reported "solution" was numerically indistinguishable
        # from x0, suggesting a scaling/step-size quirk in hybr's internal
        # finite-difference Jacobian at that point, not a genuine
        # infeasibility). Fall back to a cold start before concluding
        # infeasibility -- preserves warm-start speed in the common case
        # without sacrificing correctness when it occasionally misfires.
        cgd_sum = Cgd_fixed.sum(axis=1)
        offdiag_sum = Cdd_offdiag_fixed.sum(axis=1)
        x0_cold = np.maximum(1.0 / target_E_c - cgd_sum - offdiag_sum, 1e-6)
        sol = root(residual, x0_cold, method="hybr", tol=tol)

    if not sol.success or np.any(sol.x < 0):
        raise InfeasibleDeviceParamsError(
            f"Could not jointly solve for self-capacitances achieving "
            f"target_E_c={target_E_c.tolist()} given fixed alpha/coupling "
            f"structure (solver success={sol.success}, solution={sol.x}). "
            f"This usually means alpha is too large relative to the "
            f"requested E_c -- lever arm and charging energy share a "
            f"capacitance budget (E_c ~ 1/C_total, gate is part of "
            f"C_total). Try a smaller alpha or a smaller target_E_c."
        )
    return sol.x


def build_capacitance_matrices(
    params: DeviceParams, *, x0: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Map a DeviceParams hypothesis to QArray's (Cdd, Cgd) raw matrices.

    Raises InfeasibleDeviceParamsError if the requested E_c1/E_c2 cannot be
    achieved given alpha1/alpha2 and the coupling terms -- this is a real
    physical constraint, not a bug; see module docstring.

    x0: optional warm-start for the joint self-capacitance solve -- see
    _solve_self_capacitance_joint's docstring. Pass the self-capacitance
    diagonal from a nearby already-solved DeviceParams when doing repeated
    small perturbations (verified ~2.4x speedup).
    """
    Cdd_offdiag = np.zeros((N_DOT, N_DOT), dtype=np.float64)
    Cdd_offdiag[0, 1] = Cdd_offdiag[1, 0] = params.E_cm_intra1
    Cdd_offdiag[2, 3] = Cdd_offdiag[3, 2] = params.E_cm_intra2
    Cdd_offdiag[1, 2] = Cdd_offdiag[2, 1] = params.cross_capacitance

    Cgd = np.zeros((N_GATE, N_DOT), dtype=np.float64)
    lever_arms = (params.alpha1, params.alpha1, params.alpha2, params.alpha2)
    for i in range(N_GATE):
        Cgd[i, i] = lever_arms[i]

    target_E_c = np.array([params.E_c1, params.E_c1, params.E_c2, params.E_c2])
    c_self_vec = _solve_self_capacitance_joint(target_E_c, Cdd_offdiag, Cgd, x0=x0)
    Cdd_raw = Cdd_offdiag.copy()
    np.fill_diagonal(Cdd_raw, c_self_vec)

    return Cdd_raw, Cgd


def _soft_prediction_direct(
    params: DeviceParams, vg: np.ndarray, *, T: float, x0: np.ndarray | None = None
) -> np.ndarray:
    """Solve + construct a DotArray at the target T directly, in one step --
    used internally by observation_jacobian to avoid QArrayEnv's own T=0
    construction (wasted there, since only soft_prediction's T=0.05 result
    is ever used). Found and fixed as part of the same perf investigation
    that added x0 warm-starting: after fixing the redundant double-solve,
    this leftover ~345us/call DotArray-construction waste became the
    dominant remaining cost (10 calls x 345us = 3.45ms, larger than the
    now-optimized 10 x 209us = 2.09ms warm-started solve cost).
    """
    Cdd, Cgd = build_capacitance_matrices(params, x0=x0)
    model = DotArray(
        Cdd=Cdd, Cgd=Cgd, algorithm="default", implementation="rust",
        charge_carrier="electron", T=T, max_charge_carriers=None,
    )
    vg = np.asarray(vg, dtype=np.float64).reshape(1, -1)
    return np.asarray(model.ground_state_open(vg))[0]


def observation_jacobian(
    params: DeviceParams,
    vg: np.ndarray,
    *,
    step: float = 1e-4,
    T: float = 0.05,
    x0_base: np.ndarray | None = None,
) -> np.ndarray:
    """Finite-difference Jacobian of a THERMALLY-SOFTENED ground-state
    prediction with respect to the 5 tracked continuous parameters (E_c1,
    E_c2, alpha1, alpha2, cross_capacitance), at a fixed voltage point vg.

    CORRECTNESS NOTE, kept here because it was caught mid-implementation and
    is easy to reintroduce by "simplifying": an earlier version of this
    function differentiated continuous_prediction (the linear relaxation
    v_dash = Cgd @ vg) instead. That is WRONG for this purpose -- v_dash is
    exactly linear in vg with coefficients from Cgd alone, and structurally
    cannot depend on Cdd_inv (hence cannot depend on E_c or any coupling
    term) at ANY vg, not just at "uninformative" points. Using it here would
    make H's E_c columns identically zero everywhere, silently telling
    IG_FIM that no measurement anywhere can ever inform E_c -- wrong, and a
    much worse failure than the (real, documented separately) fact that
    single-dot transitions specifically don't inform E_c. The fix is to
    differentiate the actual (thermally-softened, hence differentiable)
    ground_state_open output, which genuinely depends on Cdd_inv through
    QArray's softargmin -- verified empirically (see the conversation that
    produced this fix): a same-dot transition showed exactly zero E_c
    sensitivity, matching the real, textbook V_add=e/Cg result, while an
    interdot transition (occupation trading between coupled dots) showed
    real, smoothly-varying E_c sensitivity, as physically expected.

    Returns a (4, 5) array: H[i, k] = d(prediction_i) / d(param_k), param
    order = (E_c1, E_c2, alpha1, alpha2, cross_capacitance) -- matches
    DeviceParams field order minus E_cm_intra1/E_cm_intra2, which are not
    tracked as Kalman state (t_c dropped per Primer Section 4 correction).

    PRACTICAL CONSEQUENCE for callers: pick vg near an interdot/multi-dot
    transition boundary if you need E_c-sensitivity in the returned H. A vg
    deep inside a stable plateau, or near a same-dot-only transition, will
    correctly return a near-zero E_c column -- this is real physics
    (single-dot addition voltage V_add=e/Cg is charging-energy-independent),
    not a bug, but easy to mistake for one.

    CORRECTNESS NOTE #2, also kept here because it was a genuine, verified
    bug and is easy to reintroduce: evaluating this Jacobian at a vg point
    sitting exactly on a symmetry line (e.g. vg0=vg1 when dot0 and dot1
    share the same target E_c and coupling structure) can silently return
    near-zero sensitivity for a parameter that IS genuinely observable
    nearby -- NOT because of thermal-softening/T miscalibration (that
    hypothesis was tested and disproven: the thermally-smoothed transition
    window was directly measured to be wide, not narrow, at T=0.05), and
    NOT a finite-difference-vs-autodiff issue either (verified: even a
    perturbation 1000x larger than the finite-difference step showed the
    same near-zero result, which rules out step-size/precision artifacts --
    autodiff would faithfully reproduce the same near-zero true derivative
    at such a point, not fix it). The real cause, verified algebraically and
    confirmed to match an independently-bisected transition voltage to 6
    decimal places: at vg0=vg1 with dot0/dot1 sharing identical target E_c
    (A00=A11 in Cdd_inv), the free-energy difference between occupation
    states (1,0,..) and (0,1,..) reduces to a term that is IDENTICALLY ZERO
    for any E_c1, independent of vg -- a structural degeneracy of that
    specific comparison, not a numerical artifact. The genuinely
    E_c-sensitive transition nearby (the (0,0,..) vs tied-{(1,0,..),(0,1,..)}
    boundary) sits at a different, precisely computable location:
    v = Cdd_inv[0,0] / (2*alpha1*(Cdd_inv[0,0]+Cdd_inv[0,1])) along that
    line -- evaluating the Jacobian there (rather than at an arbitrary
    nearby guess) recovers large, smooth, physically sensible sensitivity
    (~2000-2500 in this file's default parameter regime, decreasing
    monotonically as T increases and thermally smooths the transition --
    exactly the expected behavior, unlike the earlier symmetric-point
    result).

    PRACTICAL CONSEQUENCE: candidate vg points that sit on an exact
    symmetry of the tracked-parameter structure (equal target E_c for two
    coupled dots being probed via a vg0=vg1-style sweep is the case found
    here; other symmetries may exist depending on which dots/parameters a
    candidate probes) can silently and misleadingly report near-zero
    sensitivity for a parameter that's genuinely observable a short
    distance away. This is a real trap for any future candidate-selection
    logic (Section 5's IG_FIM) -- worth an explicit awareness note there,
    not just here.
    """
    tracked_fields = TRACKED_PARAM_FIELDS
    n_params = len(tracked_fields)
    vg = np.asarray(vg, dtype=np.float64)
    H = np.zeros((N_DOT, n_params), dtype=np.float64)

    # Solve the base (unperturbed) params once -- its self-capacitance
    # vector is an excellent warm-start for all 10 small perturbations
    # below (verified: ~2.4x speedup per solve, machine-precision
    # convergence, since a step-size-1e-4 perturbation barely moves the
    # solution). x0_base lets a caller that ALREADY has a solved
    # environment for these exact params (e.g. kalman.py's update(), which
    # was found to independently re-solve current_params a second time for
    # its own soft_prediction call) skip this redundant solve entirely --
    # pass that environment's own self-capacitance diagonal instead of
    # letting this function re-derive it from scratch.
    if x0_base is None:
        Cdd_base, _ = build_capacitance_matrices(params)
        x0_base = np.diag(Cdd_base).copy()

    for k, field in enumerate(tracked_fields):
        base_val = getattr(params, field)
        params_plus = params.replace(**{field: base_val + step})
        params_minus = params.replace(**{field: base_val - step})

        pred_plus = _soft_prediction_direct(params_plus, vg, T=T, x0=x0_base)
        pred_minus = _soft_prediction_direct(params_minus, vg, T=T, x0=x0_base)

        H[:, k] = (pred_plus - pred_minus) / (2 * step)

    return H


class QArrayEnv:
    """Wraps a single qarray.DotArray instance for the joint 4-dot system.

    Unchanged from v1's interface/role -- see original docstring for the
    ground-truth-vs-per-particle-scratch-model usage note. Only
    build_capacitance_matrices' internals changed.
    """

    def __init__(
        self,
        params: DeviceParams,
        *,
        T: float = 0.0,
        charge_carrier: str = "electron",
        implementation: str = "rust",
        algorithm: str = "default",
        x0: np.ndarray | None = None,
    ) -> None:
        self._params = params
        Cdd, Cgd = build_capacitance_matrices(params, x0=x0)
        self._Cdd = Cdd  # cached -- soft_prediction reuses these rather than
        self._Cgd = Cgd  # re-solving; T doesn't affect the capacitance matrices at all.
        self._model = DotArray(
            Cdd=Cdd,
            Cgd=Cgd,
            algorithm=algorithm,
            implementation=implementation,
            charge_carrier=charge_carrier,
            T=T,
            max_charge_carriers=None,
        )

    @property
    def params(self) -> DeviceParams:
        return self._params

    @property
    def n_dot(self) -> int:
        return N_DOT

    @property
    def n_gate(self) -> int:
        return N_GATE

    def update_params(self, params: DeviceParams) -> None:
        """Swap this instance's device-parameter hypothesis in place.

        Per-particle-per-step operation, Primer Section 9. Warm-starts the
        joint self-capacitance solve from this instance's OWN currently-cached
        self-capacitance (self._Cdd's diagonal) -- verified ~2.4x speedup
        for small perturbations (see _solve_self_capacitance_joint's
        docstring). This matters here specifically: update_params is called
        repeatedly on the SAME instance as a particle's continuous estimate
        evolves step-by-step, so consecutive calls are typically small
        perturbations of each other -- exactly the case warm-starting helps.

        PERFORMANCE NOTE, corrected from an earlier, now-stale version of
        this docstring: Primer Section 9's original ~270us/particle swap
        benchmark predates E_c being wired in at all (Cdd's diagonal was
        hardcoded to 0 then, no joint solve existed to measure). The
        current joint solve costs ~500us cold-started, ~200us warm-started
        (both measured directly) -- re-benchmark this specific operation at
        realistic per-step parameter-change magnitudes if per-step cost
        becomes a concern; don't rely on the old 270us figure for capacity
        planning.
        """
        x0 = np.diag(self._Cdd).copy()
        Cdd, Cgd = build_capacitance_matrices(params, x0=x0)
        self._Cdd = Cdd
        self._Cgd = Cgd
        self._model.update_capacitance_matrices(Cdd, Cgd)
        self._params = params

    def query(self, vg: np.ndarray) -> np.ndarray:
        """Ground-state occupation query. vg shape (..., 4) -> returns (..., 4)."""
        vg = np.asarray(vg, dtype=np.float64)
        if vg.shape[-1] != N_GATE:
            raise ValueError(
                f"Expected last dimension {N_GATE} (n_gate), got shape {vg.shape}"
            )
        return np.asarray(self._model.ground_state_open(vg))

    def continuous_prediction(self, vg: np.ndarray) -> np.ndarray:
        """The unconstrained (real-valued, not rounded to integers) relaxed
        ground-state prediction: v_dash = model.cgd @ vg. This is the
        quantity F's continuous minimizer sits at before rounding to the
        nearest integer occupation -- well-defined at any T (including
        T=0), unlike the hard rounded ground state.

        This is the right quantity for a "distance to decision boundary"
        calculation: e.g. distance = 0.5 - |frac(continuous_prediction) - 0.5|
        per dot. Verified empirically that this behaves sensibly (a
        prediction of 0.1 -- far from the n=0/n=1 boundary at 0.5 -- rounds
        to occupation 0, consistent with the actual ground_state_open output
        at the same vg).
        """
        vg = np.asarray(vg, dtype=np.float64)
        if vg.shape[-1] != N_GATE:
            raise ValueError(
                f"Expected last dimension {N_GATE} (n_gate), got shape {vg.shape}"
            )
        return np.einsum("ij,...j->...i", self._model.cgd, vg)

    def free_energy_at_candidates(self, candidates: np.ndarray, vg: np.ndarray) -> np.ndarray:
        """F(n, vg) = (n - Cgd@vg)^T @ Cdd_inv @ (n - Cgd@vg), evaluated at
        each candidate integer occupation vector, for a fixed vg. This is
        QArray's own free-energy quadratic form (verified against
        qarray.python_implementations.helper_functions.free_energy's source
        earlier in this project), evaluated using THIS instance's own
        Cdd_inv/Cgd -- i.e. one particle's own hypothesis, not any shared
        or global quantity.

        candidates: shape (K, N_DOT) -- a small enumerated set of nearby
        integer occupation states (see acquisition/bald.py's local
        candidate enumeration for how these are chosen).
        vg: shape (N_GATE,) -- a single voltage point (not batched).

        Returns shape (K,): F evaluated at each candidate.

        This is the machinery that makes an exact (not summed-marginal)
        joint entropy over nearby discrete states cheap and closed-form --
        no hidden QArray internals need exposing, no MC sampling: enumerate
        a handful of candidates, evaluate this, softmax(-F/T).
        """
        candidates = np.asarray(candidates, dtype=np.float64)
        vg = np.asarray(vg, dtype=np.float64)
        v_dash = self._model.cgd @ vg  # shape (N_DOT,)
        diff = candidates - v_dash  # shape (K, N_DOT), broadcasts over K
        cdd_inv = self._model.cdd_inv
        return np.einsum("ki,ij,kj->k", diff, cdd_inv, diff)

    def soft_prediction(self, vg: np.ndarray, *, T: float) -> np.ndarray:
        """Thermally-softened ground-state prediction at temperature T --
        this is the DIFFERENTIABLE quantity (via QArray's softargmin) that
        observation_jacobian differentiates, and the quantity a Kalman
        filter's innovation term should compare a noisy near-boundary
        reading against (NOT continuous_prediction, which is linear in vg
        and structurally cannot depend on E_c -- see observation_jacobian's
        docstring for the full explanation of why that distinction matters).

        Rebuilds a DotArray at the requested T rather than reusing self._model
        (which may have been constructed at a different T, typically T=0 for
        ground-truth generation) -- this is deliberately explicit rather than
        silently reusing whatever T this instance happened to be built with.
        Reuses this instance's already-solved Cdd/Cgd (cached in __init__)
        rather than re-solving -- T does not affect the capacitance matrices
        at all, only DotArray's internal thermal averaging, so re-solving
        here was pure waste (found and fixed: this was silently paying the
        full ~500us joint-solve cost a second time on every call, on top of
        __init__'s own solve, for identical Cdd/Cgd).
        """
        model = DotArray(
            Cdd=self._Cdd, Cgd=self._Cgd, algorithm="default", implementation="rust",
            charge_carrier="electron", T=T, max_charge_carriers=None,
        )
        vg = np.asarray(vg, dtype=np.float64).reshape(1, -1)
        if vg.shape[-1] != N_GATE:
            raise ValueError(
                f"Expected last dimension {N_GATE} (n_gate), got shape {vg.shape}"
            )
        return np.asarray(model.ground_state_open(vg))[0]


def make_default_ground_truth(
    *,
    E_c1: float = 2.5,
    E_c2: float = 2.7,
    alpha1: float = 0.05,  # CHANGED from 1.0 (v1), then from 0.2 (an unsafe first fix) -- see module docstring.
    alpha2: float = 0.05,  # Verified feasible with E_c=2.5/2.7 even at strong-coupling sweep endpoints.
    E_cm_intra1: float = 0.3,
    E_cm_intra2: float = 0.3,
    cross_capacitance: float = 0.05,
) -> QArrayEnv:
    """Convenience constructor for a ground-truth environment. Defaults
    verified to construct successfully and achieve target E_c to ~1e-15
    (machine precision, via the joint solve -- see module docstring; an
    earlier version of this docstring said "~1%", which was true of the
    since-replaced sequential solve, not the current one) -- NOT a
    substitute for the actual coupling-strength sweep (Primer Section 7a)
    or for real device-specific parameter values if you have them.
    """
    params = DeviceParams(
        E_c1=E_c1,
        E_c2=E_c2,
        alpha1=alpha1,
        alpha2=alpha2,
        E_cm_intra1=E_cm_intra1,
        E_cm_intra2=E_cm_intra2,
        cross_capacitance=cross_capacitance,
    )
    return QArrayEnv(params)
