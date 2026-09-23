"""
gz_tuning.simulator.qarray_jax

Replaces qarray_env.py's finite-difference observation_jacobian with a
真 analytic (autodiff) Jacobian, per Primer Section 12's top-priority open
item and Section 4c's finding that FD Jacobians are unreliable exactly
where they matter most (near sharp transitions).

Validated by direct spike testing (not assumed) before this was written:
1. jax.config.update('jax_enable_x64', True) is REQUIRED -- JAX defaults
   to float32, which would silently reintroduce a precision problem in
   the same spirit as the FD noise-floor issue this module exists to fix
   (our physics routinely needs to resolve differences at the 1e-5-1e-7
   scale). Set at import time, below, before anything else touches jax.
2. QArray's own ground-state prediction (softargmin over enumerated
   occupations, with a jaxopt.BoxOSQP QP fallback for the boundary case)
   is cleanly differentiable via jax.grad with NO rewrite needed -- the
   public ground_state_open_default_jax() wrapper casts its output to
   plain numpy (breaks the trace), but the underlying jitted
   _ground_state_open_0d() does not. Confirmed: clean, finite gradients,
   no tracer errors, at a real point from an earlier run.
3. THE FINDING THAT MATTERED: naively swapping qarray_env.py's
   scipy.optimize.root self-capacitance solve for jaxopt.GaussNewton on
   the RAW self-capacitance variables converged to a spurious,
   UNPHYSICAL root (negative self-capacitance) at the very first test --
   completely different from scipy's correct answer, silently "converged"
   with no error. The nonlinear system has multiple roots; unconstrained
   Gauss-Newton with a naive initial guess found the wrong one.

   Fix adopted here: solve in LOG-SPACE (`c_self = exp(y)`, solve for
   `y`) so positivity holds by construction regardless of what the
   optimizer does, rather than hoping the optimizer stays in the
   physical region. This is the same posture the project already takes
   elsewhere (assert_feasible, the Kalman-update feasibility backoff) --
   don't trust a solver's "converged" flag without an independent
   physicality/consistency check, which validate_against_scipy() below
   performs once at import-adjacent test time, not per-call (too slow).
"""

from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jaxopt import LevenbergMarquardt

from qarray.jax_implementations.default_jax.open import _ground_state_open_0d
from qarray_env import DeviceParams, TRACKED_PARAM_FIELDS, N_DOT, N_GATE


def _maxwell_form(
    c_self: jnp.ndarray, cross_cap: jnp.ndarray, alpha1: jnp.ndarray, alpha2: jnp.ndarray,
    E_cm_intra1: float, E_cm_intra2: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Replicates the Maxwell-form Cdd conversion used internally by a
    real QArray DotArray instance (checked directly against a live
    instance's own .cdd_inv/.cgd attributes, not just re-derived from the
    convert_to_maxwell source in isolation -- see the note below the
    Cdd_maxwell_inv computation for why that distinction mattered):
      cdd_maxwell = diag(row_sum(Cdd_raw) + row_sum(Cgd_raw)) - offdiag(Cdd_raw)
    Inverting the RAW Cdd directly (skipping this) is a different, wrong
    matrix -- a first attempt that did exactly that produced a residual
    of -10.39 at scipy's own known-correct solution instead of ~0, which
    is what caught the need for this conversion in the first place.
    Returns (Cdd_maxwell_inv, Cgd_raw) -- exactly the two objects
    _ground_state_open_0d expects as `cdd_inv` and `cgd`.
    """
    Cdd_raw = jnp.zeros((N_DOT, N_DOT), dtype=jnp.float64)
    Cdd_raw = Cdd_raw.at[0, 1].set(E_cm_intra1).at[1, 0].set(E_cm_intra1)
    Cdd_raw = Cdd_raw.at[2, 3].set(E_cm_intra2).at[3, 2].set(E_cm_intra2)
    Cdd_raw = Cdd_raw.at[1, 2].set(cross_cap).at[2, 1].set(cross_cap)
    Cdd_raw = Cdd_raw.at[jnp.arange(N_DOT), jnp.arange(N_DOT)].set(c_self)

    Cgd_raw = jnp.diag(jnp.array([alpha1, alpha1, alpha2, alpha2]))

    cdd_sum = Cdd_raw.sum(axis=1)
    cgd_sum = Cgd_raw.sum(axis=1)
    cdd_offdiag_only = Cdd_raw.at[jnp.arange(N_DOT), jnp.arange(N_DOT)].set(0.0)
    Cdd_maxwell = jnp.diag(cdd_sum + cgd_sum) - cdd_offdiag_only
    Cdd_maxwell_inv = jnp.linalg.inv(Cdd_maxwell)
    # NOTE: convert_to_maxwell's source also returns a NEGATED cgd
    # (`cgd_negative = -cgd_non_maxwell`), and an earlier version of this
    # function replicated that. Checking directly against a real DotArray
    # instance's own .cgd attribute (not just the conversion function in
    # isolation) showed the negation is NOT what the live open-array
    # ground-state path actually uses -- model.cgd matched the RAW,
    # un-negated Cgd exactly, while cdd_inv matched this function's
    # Maxwell-converted form exactly. Caught by comparing against a real
    # DotArray instance's internals rather than trusting the conversion
    # function's docstring/return signature alone -- same "check against
    # actual behavior, not just source" pattern as everywhere else this
    # project has been debugged.
    return Cdd_maxwell_inv, Cgd_raw


def _residual_logspace(
    log_c_self: jnp.ndarray, target_E_c: jnp.ndarray,
    cross_cap: jnp.ndarray, alpha1: jnp.ndarray, alpha2: jnp.ndarray,
    E_cm_intra1: float, E_cm_intra2: float,
) -> jnp.ndarray:
    c_self = jnp.exp(log_c_self)
    Cdd_maxwell_inv, _ = _maxwell_form(c_self, cross_cap, alpha1, alpha2, E_cm_intra1, E_cm_intra2)
    achieved_E_c = jnp.diag(Cdd_maxwell_inv)
    return achieved_E_c - target_E_c


_gn_solver = LevenbergMarquardt(
    residual_fun=_residual_logspace, tol=1e-13, xtol=1e-13, gtol=1e-13,
    maxiter=200, implicit_diff=True,
)


def _log_c_self_x0(target_E_c: np.ndarray, alpha1: float, alpha2: float,
                    E_cm_intra1: float, E_cm_intra2: float) -> jnp.ndarray:
    cgd_sum = np.array([alpha1, alpha1, alpha2, alpha2])
    offdiag_sum = np.array([E_cm_intra1, E_cm_intra1, E_cm_intra2, E_cm_intra2])
    c_self0 = np.maximum(1.0 / target_E_c - cgd_sum - offdiag_sum, 1e-6)
    return jnp.log(jnp.array(c_self0))


def solve_c_self_jax(
    target_E_c: jnp.ndarray, cross_cap: jnp.ndarray, alpha1: jnp.ndarray, alpha2: jnp.ndarray,
    E_cm_intra1: float, E_cm_intra2: float, log_x0: jnp.ndarray,
) -> jnp.ndarray:
    """Differentiable self-capacitance solve. Returns c_self (always > 0
    by construction, regardless of solver behavior)."""
    sol = _gn_solver.run(log_x0, target_E_c, cross_cap, alpha1, alpha2, E_cm_intra1, E_cm_intra2)
    return jnp.exp(sol.params)


_KB_EV_PER_K = 8.617333262145e-5  # matches qarray's own DotArrays/DotArray.py:_ground_state_open
# exactly -- kB_T = 8.617333262145e-5 * model.T is computed by QArray's
# dispatcher BEFORE calling the low-level rust/jax function, converting T
# from the same units QArrayEnv/DeviceParams use everywhere else in this
# project into an energy scale (eV). BUG FIX (found via a real
# discrepancy, not assumed): calling _ground_state_open_0d directly
# (bypassing that dispatcher, which is what this whole module has to do
# to get gradients) means WE have to apply this conversion ourselves --
# an earlier version of this module passed T=0.05 straight through as if
# it were already kB_T, a ~23,000x scale error in the thermal-softening
# temperature. This produced predictions that were close-looking but
# measurably wrong (e.g. [0.540, 0.460, 0.460, 0.540] instead of QArray's
# own [0.5, 0.5, 0.5, 0.5] at a symmetric point) rather than obviously
# broken, which is what made it easy to miss on the first two test
# points and only show up once enough points were checked.


def _full_predict(
    E_c1, E_c2, alpha1, alpha2, cross_cap, *,
    E_cm_intra1: float, E_cm_intra2: float, vg: jnp.ndarray, T: float, log_x0: jnp.ndarray,
) -> jnp.ndarray:
    target_E_c = jnp.array([E_c1, E_c1, E_c2, E_c2])
    c_self = solve_c_self_jax(target_E_c, cross_cap, alpha1, alpha2, E_cm_intra1, E_cm_intra2, log_x0)
    Cdd_maxwell_inv, Cgd_raw = _maxwell_form(
        c_self, cross_cap, alpha1, alpha2, E_cm_intra1, E_cm_intra2
    )
    kB_T = _KB_EV_PER_K * T
    return _ground_state_open_0d(vg, cgd=Cgd_raw, cdd_inv=Cdd_maxwell_inv, T=kB_T)


_full_predict_jit = jax.jit(_full_predict, static_argnames=("E_cm_intra1", "E_cm_intra2", "T"))
_jac_fn = jax.jit(
    jax.jacobian(_full_predict, argnums=(0, 1, 2, 3, 4)),
    static_argnames=("E_cm_intra1", "E_cm_intra2", "T"),
)


def jax_observation_jacobian(
    params: DeviceParams, vg: np.ndarray, *, T: float = 0.05,
    log_x0: np.ndarray | None = None,
) -> np.ndarray:
    """Drop-in analytic replacement for qarray_env.observation_jacobian.
    Same signature shape (returns a (N_DOT, 5) array in the same
    TRACKED_PARAM_FIELDS column order), computed via jax.jacobian instead
    of finite differences.
    """
    vg_j = jnp.array(vg, dtype=jnp.float64)
    target_E_c = np.array([params.E_c1, params.E_c1, params.E_c2, params.E_c2])
    if log_x0 is None:
        log_x0_j = _log_c_self_x0(
            target_E_c, params.alpha1, params.alpha2,
            params.E_cm_intra1, params.E_cm_intra2,
        )
    else:
        log_x0_j = jnp.array(log_x0, dtype=jnp.float64)

    grads = _jac_fn(
        params.E_c1, params.E_c2, params.alpha1, params.alpha2, params.cross_capacitance,
        E_cm_intra1=params.E_cm_intra1, E_cm_intra2=params.E_cm_intra2,
        vg=vg_j, T=T, log_x0=log_x0_j,
    )
    H = np.stack([np.asarray(g) for g in grads], axis=1)
    return H


def jax_soft_prediction(
    params: DeviceParams, vg: np.ndarray, *, T: float = 0.05,
    log_x0: np.ndarray | None = None,
) -> np.ndarray:
    """Same prediction qarray_env.QArrayEnv.soft_prediction gives, computed
    through the jax path -- useful for cross-checking the two pipelines
    agree, not intended to replace QArrayEnv.soft_prediction in the hot
    loop (that one still uses the Rust backend, which is faster for pure
    prediction; only the JACOBIAN needed the jax rewrite)."""
    vg_j = jnp.array(vg, dtype=jnp.float64)
    target_E_c = np.array([params.E_c1, params.E_c1, params.E_c2, params.E_c2])
    if log_x0 is None:
        log_x0_j = _log_c_self_x0(
            target_E_c, params.alpha1, params.alpha2,
            params.E_cm_intra1, params.E_cm_intra2,
        )
    else:
        log_x0_j = jnp.array(log_x0, dtype=jnp.float64)
    out = _full_predict_jit(
        params.E_c1, params.E_c2, params.alpha1, params.alpha2, params.cross_capacitance,
        E_cm_intra1=params.E_cm_intra1, E_cm_intra2=params.E_cm_intra2,
        vg=vg_j, T=T, log_x0=log_x0_j,
    )
    return np.asarray(out)


def validate_against_scipy(params: DeviceParams, *, rtol: float = 1e-6) -> tuple[bool, np.ndarray, np.ndarray]:
    """One-time consistency check between the jax log-space solve and the
    existing scipy.optimize.root solve, for a given DeviceParams -- NOT
    called per-step (too slow; this is for the correctness test, and any
    future re-validation after touching this module), same spirit as
    qarray_env.py's own build_capacitance_matrices being checked directly
    against source rather than trusted on derivation alone.
    """
    from qarray_env import build_capacitance_matrices
    Cdd_scipy, _ = build_capacitance_matrices(params)
    c_self_scipy = np.diag(Cdd_scipy)

    target_E_c = jnp.array([params.E_c1, params.E_c1, params.E_c2, params.E_c2])
    log_x0 = _log_c_self_x0(
        np.array(target_E_c), params.alpha1, params.alpha2,
        params.E_cm_intra1, params.E_cm_intra2,
    )
    c_self_jax = np.asarray(solve_c_self_jax(
        target_E_c, params.cross_capacitance, params.alpha1, params.alpha2,
        params.E_cm_intra1, params.E_cm_intra2, log_x0,
    ))
    ok = bool(np.allclose(c_self_scipy, c_self_jax, rtol=rtol)) and bool(np.all(c_self_jax > 0))
    return ok, c_self_scipy, c_self_jax
