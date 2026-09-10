from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from convergence import ConvergenceMonitor, ConvergenceStatus
from noise_model import GaussianReadoutNoise
from particle_filter import RBParticleFilter
from qarray_env import N_GATE, QArrayEnv


def _joint_grid(vg_range: tuple[float, float], n_points_per_gate: int) -> np.ndarray:
    axis = np.linspace(vg_range[0], vg_range[1], n_points_per_gate)
    return np.array(list(itertools.product(axis, repeat=N_GATE)), dtype=np.float64)


def _alternating_grid(
    vg_range: tuple[float, float], n_points_per_gate: int, *, n_alternations: int = 4
) -> np.ndarray:
    axis = np.linspace(vg_range[0], vg_range[1], n_points_per_gate)
    mid = float(np.mean(vg_range))
    grid_2d = np.array(list(itertools.product(axis, repeat=2)), dtype=np.float64)

    points = []
    for _ in range(n_alternations):
        for (v0, v1) in grid_2d:
            points.append([v0, v1, mid, mid])
        for (v2, v3) in grid_2d:
            points.append([mid, mid, v2, v3])
    return np.array(points, dtype=np.float64)


def build_raster_grid(
    vg_range: tuple[float, float] = (0.0, 15.0),
    n_points_per_gate: int = 20,
    *,
    mode: str = "alternating",
    n_alternations: int = 4,
) -> np.ndarray:
    if mode == "joint":
        return _joint_grid(vg_range, n_points_per_gate)
    elif mode == "alternating":
        return _alternating_grid(vg_range, n_points_per_gate, n_alternations=n_alternations)
    else:
        raise ValueError(f"Unknown mode {mode!r}")


@dataclass
class RasterScanResult:
    converged: bool
    n_measurements: int
    final_status: ConvergenceStatus | None


def run_raster_scan(
    pf: RBParticleFilter,
    ground_truth: QArrayEnv,
    grid: np.ndarray,
    noise: GaussianReadoutNoise,
    monitor: ConvergenceMonitor,
    *,
    T: float = 0.05,
    rng: np.random.Generator | None = None,
) -> RasterScanResult:
    if monitor.require_actuation_quiescence:
        raise ValueError(
            "run_raster_scan requires require_actuation_quiescence=False"
        )

    if rng is None:
        rng = np.random.default_rng()
    R = noise.R_matrix()

    status: ConvergenceStatus | None = None
    for i, vg in enumerate(grid):
        true_reading = ground_truth.soft_prediction(vg, T=T)
        measured = noise.sample(true_reading, rng)
        pf.update(measured, vg, R, T=T)

        if i + 1 < len(grid):
            proposed_step = grid[i + 1] - vg
        else:
            proposed_step = np.zeros(N_GATE)

        status = monitor.step(pf, vg, proposed_step)
        if status.converged:
            return RasterScanResult(converged=True, n_measurements=i + 1, final_status=status)

    return RasterScanResult(converged=False, n_measurements=len(grid), final_status=status)
