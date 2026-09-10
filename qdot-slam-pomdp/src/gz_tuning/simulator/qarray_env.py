"""
gz_tuning.simulator.qarray_env  (sandbox copy, unchanged from repo)
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import root

from qarray import DotArray

N_DOT = 4
N_GATE = 4

TRACKED_PARAM_FIELDS = ("E_c1", "E_c2", "alpha1", "alpha2", "cross_capacitance")


@dataclass(frozen=True)
class DeviceParams:
    E_c1: float
    E_c2: float
    alpha1: float
    alpha2: float
    E_cm_intra1: float
    E_cm_intra2: float
    cross_capacitance: float

    def replace(self, **kwargs) -> "DeviceParams":
        return replace(self, **kwargs)


class InfeasibleDeviceParamsError(ValueError):
    pass


def _construct_dot_array(Cdd: np.ndarray, Cgd: np.ndarray, *, T: float,
                          algorithm: str = "default", implementation: str = "rust",
                          charge_carrier: str = "electron") -> DotArray:
    try:
        return DotArray(
            Cdd=Cdd, Cgd=Cgd, algorithm=algorithm, implementation=implementation,
            charge_carrier=charge_carrier, T=T, max_charge_carriers=None,
        )
    except ValueError as e:
        raise InfeasibleDeviceParamsError(
            f"QArray rejected the constructed (Cdd, Cgd) pair as physically "
            f"infeasible. Original QArray error: {e}"
        ) from e


def _solve_self_capacitance_joint(
    target_E_c: np.ndarray,
    Cdd_offdiag_fixed: np.ndarray,
    Cgd_fixed: np.ndarray,
    *,
    tol: float = 1e-13,
    x0: np.ndarray | None = None,
) -> np.ndarray:
    def residual(c_self_vec: np.ndarray) -> np.ndarray:
        Cdd_raw = Cdd_offdiag_fixed.copy()
        np.fill_diagonal(Cdd_raw, c_self_vec)
        cdd_sum = Cdd_raw.sum(axis=1)
        cgd_sum = Cgd_fixed.sum(axis=1)
        offdiag_only = Cdd_raw.copy()
        np.fill_diagonal(offdiag_only, 0.0)
        Cdd_maxwell = np.diag(cdd_sum + cgd_sum) - offdiag_only
        achieved_E_c = np.diag(np.linalg.inv(Cdd_maxwell))
        return achieved_E_c - target_E_c

    if x0 is None:
        cgd_sum = Cgd_fixed.sum(axis=1)
        offdiag_sum = Cdd_offdiag_fixed.sum(axis=1)
        x0 = np.maximum(1.0 / target_E_c - cgd_sum - offdiag_sum, 1e-6)
        used_warm_start = False
    else:
        used_warm_start = True

    sol = root(residual, x0, method="hybr", tol=tol)

    if not sol.success and used_warm_start:
        cgd_sum = Cgd_fixed.sum(axis=1)
        offdiag_sum = Cdd_offdiag_fixed.sum(axis=1)
        x0_cold = np.maximum(1.0 / target_E_c - cgd_sum - offdiag_sum, 1e-6)
        sol = root(residual, x0_cold, method="hybr", tol=tol)

    if not sol.success or np.any(sol.x < 0):
        raise InfeasibleDeviceParamsError(
            f"Could not jointly solve for self-capacitances achieving "
            f"target_E_c={target_E_c.tolist()} (solver success={sol.success}, "
            f"solution={sol.x})."
        )
    return sol.x


def build_capacitance_matrices(
    params: DeviceParams, *, x0: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
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
    Cdd, Cgd = build_capacitance_matrices(params, x0=x0)
    model = _construct_dot_array(Cdd, Cgd, T=T)
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
    tracked_fields = TRACKED_PARAM_FIELDS
    n_params = len(tracked_fields)
    vg = np.asarray(vg, dtype=np.float64)
    H = np.zeros((N_DOT, n_params), dtype=np.float64)

    if x0_base is None:
        Cdd_base, _ = build_capacitance_matrices(params)
        x0_base = np.diag(Cdd_base).copy()

    for k, field in enumerate(tracked_fields):
        H[:, k] = _adaptive_fd_column(
            params, field, vg, T=T, x0_base=x0_base, step0=step
        )

    return H


def _adaptive_fd_column(
    params: DeviceParams,
    field: str,
    vg: np.ndarray,
    *,
    T: float,
    x0_base: np.ndarray,
    step0: float,
    min_step: float = 1e-8,
    max_refinements: int = 6,
    convergence_rtol: float = 0.1,
) -> np.ndarray:
    """BUG FIX (found via a real run at a different particle count than
    previously tested): perturbing a parameter by +/- step to build the FD
    Jacobian can itself land in an INFEASIBLE (Cdd, Cgd) region -- e.g. a
    particle whose mean cross_capacitance sits close to the feasibility
    boundary (Section 4b) will have one perturbed direction reject with
    InfeasibleDeviceParamsError. This wasn't caught anywhere, so it crashed
    the whole particle-filter update (confirmed: crashed a real run at
    n_particles=20, not at n_particles=10, purely because more particles
    means a higher chance one of them randomly drifted near the boundary).

    Fix: treat an infeasible perturbation the same way an oversized/
    saturated finite difference is already treated -- shrink the step and
    retry. If it's still infeasible at min_step (i.e. the particle's mean
    itself is right at the edge of the feasible region in this
    direction), fall back to whatever column was already computed at a
    larger step, or a zero column on the very first attempt -- treating
    this parameter as locally unobservable at this point rather than
    crashing. This is the same physically-motivated fallback the adaptive
    refinement loop already uses for numerical saturation, just extended
    to cover infeasibility as another failure mode.
    """
    base_val = getattr(params, field)
    cur_step = step0
    prev_col: np.ndarray | None = None
    col = None

    for _ in range(max_refinements + 1):
        params_plus = params.replace(**{field: base_val + cur_step})
        params_minus = params.replace(**{field: base_val - cur_step})
        try:
            pred_plus = _soft_prediction_direct(params_plus, vg, T=T, x0=x0_base)
            pred_minus = _soft_prediction_direct(params_minus, vg, T=T, x0=x0_base)
        except InfeasibleDeviceParamsError:
            cur_step /= 2.0
            if cur_step < min_step:
                return col if col is not None else np.zeros(N_DOT)
            continue

        diff = pred_plus - pred_minus
        col = diff / (2 * cur_step)

        saturated = bool(np.any(np.abs(diff) > 0.9))
        if not saturated:
            break
        if prev_col is not None:
            denom = np.maximum(np.abs(col), np.abs(prev_col))
            denom = np.maximum(denom, 1e-12)
            if np.all(np.abs(col - prev_col) <= convergence_rtol * denom):
                break

        prev_col = col
        cur_step /= 2.0
        if cur_step < min_step:
            break

    return col


class QArrayEnv:
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
        self._Cdd = Cdd
        self._Cgd = Cgd
        self._model = _construct_dot_array(
            Cdd, Cgd, T=T, algorithm=algorithm, implementation=implementation,
            charge_carrier=charge_carrier,
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
        x0 = np.diag(self._Cdd).copy()
        Cdd, Cgd = build_capacitance_matrices(params, x0=x0)
        self._Cdd = Cdd
        self._Cgd = Cgd
        self._model.update_capacitance_matrices(Cdd, Cgd)
        self._params = params

    def query(self, vg: np.ndarray) -> np.ndarray:
        vg = np.asarray(vg, dtype=np.float64)
        if vg.shape[-1] != N_GATE:
            raise ValueError(f"Expected last dimension {N_GATE}, got shape {vg.shape}")
        return np.asarray(self._model.ground_state_open(vg))

    def continuous_prediction(self, vg: np.ndarray) -> np.ndarray:
        vg = np.asarray(vg, dtype=np.float64)
        if vg.shape[-1] != N_GATE:
            raise ValueError(f"Expected last dimension {N_GATE}, got shape {vg.shape}")
        return np.einsum("ij,...j->...i", self._model.cgd, vg)

    def free_energy_at_candidates(self, candidates: np.ndarray, vg: np.ndarray) -> np.ndarray:
        candidates = np.asarray(candidates, dtype=np.float64)
        vg = np.asarray(vg, dtype=np.float64)
        v_dash = self._model.cgd @ vg
        diff = candidates - v_dash
        cdd_inv = self._model.cdd_inv
        return np.einsum("ki,ij,kj->k", diff, cdd_inv, diff)

    def soft_prediction(self, vg: np.ndarray, *, T: float) -> np.ndarray:
        model = _construct_dot_array(self._Cdd, self._Cgd, T=T)
        vg = np.asarray(vg, dtype=np.float64).reshape(1, -1)
        if vg.shape[-1] != N_GATE:
            raise ValueError(f"Expected last dimension {N_GATE}, got shape {vg.shape}")
        return np.asarray(model.ground_state_open(vg))[0]


def make_default_ground_truth(
    *,
    E_c1: float = 2.5,
    E_c2: float = 2.7,
    alpha1: float = 0.05,
    alpha2: float = 0.05,
    E_cm_intra1: float = 0.3,
    E_cm_intra2: float = 0.3,
    cross_capacitance: float = 0.05,
) -> QArrayEnv:
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
