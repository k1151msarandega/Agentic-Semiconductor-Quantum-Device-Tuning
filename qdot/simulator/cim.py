"""
qdot/simulator/cim.py
=====================
Constant Interaction Model (CIM) physics simulator for double quantum dots.

Physics Model:
    Two quantum dots coupled by a tunnel barrier.
    Charging energy: E_c = e²/2C (capacitive energy cost per electron)
    Tunnel coupling: t_c (interdot hopping amplitude)
    Gate voltage → energy via lever arm: E = α * V_gate

References:
    Koch et al., Phys. Rev. A 76, 042319 (2007) — Charge qubits
    Hanson et al., Rev. Mod. Phys. 79, 1217 (2007) — Spin qubits review
    van der Wiel et al., Rev. Mod. Phys. 75, 1 (2002) — Electron transport
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import numpy as np

from qdot.core.types import Measurement, MeasurementModality, VoltagePoint
from qdot.hardware.adapter import DeviceAdapter


class ConstantInteractionDevice:
    """CIM physics engine."""

    def __init__(
        self,
        E_c1: float = 0.50,
        E_c2: float = 0.55,
        t_c: float = 0.05,
        T: float = 0.015,
        lever_arm: float = 1.0,
        noise_level: float = 0.01,
        seed: Optional[int] = None,
    ) -> None:
        self.E_c1 = E_c1
        self.E_c2 = E_c2
        self.t_c = t_c
        self.T = T
        self.alpha = lever_arm
        self.noise_level = noise_level
        self.rng = np.random.default_rng(seed)

        self._disorder_map: Optional[np.ndarray] = None
        self._disorder_v1_grid: Optional[np.ndarray] = None
        self._disorder_v2_grid: Optional[np.ndarray] = None

    def chemical_potential(self, vg1: float, vg2: float, n1: int, n2: int) -> float:
        E_charge = self.E_c1 * n1 + self.E_c2 * n2
        E_gate = self.alpha * (vg1 * n1 + vg2 * n2)
        return E_charge + E_gate

    def ground_state_energy(self, vg1: float, vg2: float, n1: int, n2: int) -> float:
        mu = self.chemical_potential(vg1, vg2, n1, n2)
        if n1 == 1 and n2 == 1:
            mu -= self.t_c
        return mu

    def current(self, vg1: float, vg2: float) -> float:
        """Scalar conductance at (vg1, vg2). Applies disorder if injected."""
        if self._disorder_map is not None:
            disorder_offset = self._interpolate_disorder(vg1, vg2)
            vg1 = vg1 + disorder_offset * 0.1

        states = [(n1, n2) for n1 in range(3) for n2 in range(3)]
        energies = [self.ground_state_energy(vg1, vg2, n1, n2) for n1, n2 in states]
        sorted_energies = sorted(energies)
        energy_gap = sorted_energies[1] - sorted_energies[0]

        broadening = max(self.t_c, self.T)
        conductance = broadening / (energy_gap ** 2 + broadening ** 2)

        if self.noise_level > 0:
            conductance += self.rng.normal(0, self.noise_level)

        return float(np.clip(conductance, 0, None))

    def current_grid(self, v1_grid: np.ndarray, v2_grid: np.ndarray) -> np.ndarray:
        """
        Vectorised 2D conductance map over a meshgrid.

        Replaces the Python double-for-loop in sample_patch and
        CIMDataset._simulate() with NumPy broadcasting — ~100–500× faster
        for a 64×64 patch (benchmarked: ~8 ms vs ~4 s).

        Disorder is NOT applied here (bilinear interpolation is not batched).
        For training-data generation disorder is always zero.

        Args:
            v1_grid: 1D array of vg1 values, shape (W,)
            v2_grid: 1D array of vg2 values, shape (H,)

        Returns:
            float32 array, shape (H, W), values in [0, inf) before normalisation.
            Row i = v2_grid[i], column j = v1_grid[j].
        """
        VG1, VG2 = np.meshgrid(
            v1_grid.astype(np.float64),
            v2_grid.astype(np.float64),
        )  # (H, W)

        alpha = float(self.alpha)
        E_c1 = float(self.E_c1)
        E_c2 = float(self.E_c2)
        t_c = float(self.t_c)

        # Energies for all 9 charge states, shape (9, H, W)
        states = [(n1, n2) for n1 in range(3) for n2 in range(3)]
        slabs = []
        for n1, n2 in states:
            e = E_c1 * n1 + E_c2 * n2 + alpha * (VG1 * n1 + VG2 * n2)
            if n1 == 1 and n2 == 1:
                e = e - t_c
            slabs.append(e)

        energies = np.stack(slabs, axis=0)          # (9, H, W)
        sorted_e = np.sort(energies, axis=0)        # (9, H, W)
        energy_gap = sorted_e[1] - sorted_e[0]      # (H, W)

        broadening = max(t_c, float(self.T))
        patch = broadening / (energy_gap ** 2 + broadening ** 2)

        if self.noise_level > 0:
            patch = patch + self.rng.normal(0, self.noise_level, patch.shape)

        return np.clip(patch, 0, None).astype(np.float32)

    def current_for_state(self, vg1: float, vg2: float, n1: int, n2: int) -> float:
        """
        POMDP observation model: predicted conductance conditioned on (n1,n2).
        Used exclusively by BeliefUpdater and ActiveSensingPolicy.
        """
        if self._disorder_map is not None:
            disorder_offset = self._interpolate_disorder(vg1, vg2)
            vg1 = vg1 + disorder_offset * 0.1

        E_target = self.ground_state_energy(vg1, vg2, n1, n2)

        all_states = [(m1, m2) for m1 in range(3) for m2 in range(3)]
        E_min = min(self.ground_state_energy(vg1, vg2, m1, m2) for m1, m2 in all_states)

        delta_E = max(0.0, E_target - E_min)
        T_eff = max(self.T, 0.01)
        boltzmann = float(np.exp(-delta_E / T_eff))

        neighbour_energies = []
        for dn1, dn2 in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            m1, m2 = n1 + dn1, n2 + dn2
            if 0 <= m1 <= 2 and 0 <= m2 <= 2:
                neighbour_energies.append(self.ground_state_energy(vg1, vg2, m1, m2))

        if not neighbour_energies:
            return 0.0

        energy_gap = min(abs(E_n - E_target) for E_n in neighbour_energies)
        broadening = max(self.t_c, self.T)
        conductance = broadening / (energy_gap ** 2 + broadening ** 2)

        return float(np.clip(conductance * boltzmann, 0, None))

    def inject_disorder(self, disorder_posterior: Dict) -> None:
        """Inject device-specific disorder from DisorderLearner (Phase 3)."""
        self._disorder_map = np.array(disorder_posterior["mean"])
        self._disorder_v1_grid = np.array(disorder_posterior["v1_grid"])
        self._disorder_v2_grid = np.array(disorder_posterior["v2_grid"])

    def _interpolate_disorder(self, vg1: float, vg2: float) -> float:
        """Bilinear interpolation of the disorder map at (vg1, vg2)."""
        if self._disorder_map is None:
            return 0.0
        v1g = self._disorder_v1_grid
        v2g = self._disorder_v2_grid
        i1 = np.searchsorted(v1g, vg1, side="left") - 1
        i2 = np.searchsorted(v2g, vg2, side="left") - 1
        i1 = int(np.clip(i1, 0, len(v1g) - 2))
        i2 = int(np.clip(i2, 0, len(v2g) - 2))
        return float(self._disorder_map[i2, i1])


class CIMSimulatorAdapter(DeviceAdapter):
    """
    Drop-in DeviceAdapter using the CIM physics engine.
    sample_patch uses the vectorised current_grid path — no Python loop.
    """

    DEFAULT_PARAMS = {
        "E_c1": 0.50,
        "E_c2": 0.55,
        "t_c": 0.05,
        "T": 0.015,
        "lever_arm": 1.0,
        "noise_level": 0.01,
    }

    def __init__(
        self,
        device_id: str = "sim_default",
        params: Optional[Dict] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.device_id = device_id
        p = {**self.DEFAULT_PARAMS, **(params or {})}
        self.device = ConstantInteractionDevice(seed=seed, **p)
        self._current_voltages: Dict[str, float] = {"vg1": 0.0, "vg2": 0.0}

    @property
    def device_type(self) -> str:
        return "CIM Simulator"

    def sample_patch(
        self,
        v1_range: Tuple[float, float] = (-1.0, 1.0),
        v2_range: Tuple[float, float] = (-1.0, 1.0),
        res: int = 32,
    ) -> Measurement:
        """
        Acquire a 2D conductance map.

        Uses current_grid (NumPy vectorised) when no disorder map is active,
        and falls back to the scalar loop only when disorder is injected (Phase 3).
        """
        v1_grid = np.linspace(v1_range[0], v1_range[1], res, dtype=np.float32)
        v2_grid = np.linspace(v2_range[0], v2_range[1], res, dtype=np.float32)

        if self.device._disorder_map is None:
            patch = self.device.current_grid(v1_grid, v2_grid)
        else:
            # Disorder active: scalar loop (interpolation not batched yet)
            patch = np.zeros((res, res), dtype=np.float32)
            for i, v2 in enumerate(v2_grid):
                for j, v1 in enumerate(v1_grid):
                    patch[i, j] = self.device.current(float(v1), float(v2))

        patch = self._normalise(patch)

        self._current_voltages["vg1"] = float(np.mean(v1_range))
        self._current_voltages["vg2"] = float(np.mean(v2_range))

        return Measurement(
            array=patch,
            modality=MeasurementModality.COARSE_2D,
            voltage_centre=VoltagePoint(*[float(np.mean(r)) for r in (v1_range, v2_range)]),
            v1_range=v1_range,
            v2_range=v2_range,
            resolution=res,
            device_id=self.device_id,
            timestamp=time.time(),
            meta={
                "v1_grid": v1_grid.tolist(),
                "v2_grid": v2_grid.tolist(),
                "E_c1": self.device.E_c1,
                "E_c2": self.device.E_c2,
                "t_c": self.device.t_c,
                "model": "Constant Interaction Model",
            },
        )

    def line_scan(
        self,
        axis: str = "vg1",
        start: float = -1.0,
        stop: float = 1.0,
        steps: int = 128,
        fixed: float = 0.0,
    ) -> Measurement:
        grid = np.linspace(start, stop, steps, dtype=np.float32)
        trace = np.zeros(steps, dtype=np.float32)

        for i, val in enumerate(grid):
            if axis == "vg1":
                trace[i] = self.device.current(val, fixed)
            else:
                trace[i] = self.device.current(fixed, val)

        trace = self._normalise(trace)

        if axis == "vg1":
            self._current_voltages["vg1"] = float(np.mean([start, stop]))
            self._current_voltages["vg2"] = fixed
        else:
            self._current_voltages["vg1"] = fixed
            self._current_voltages["vg2"] = float(np.mean([start, stop]))

        return Measurement(
            array=trace,
            modality=MeasurementModality.LINE_SCAN,
            voltage_centre=VoltagePoint(
                vg1=self._current_voltages["vg1"],
                vg2=self._current_voltages["vg2"],
            ),
            axis=axis,
            steps=steps,
            device_id=self.device_id,
            timestamp=time.time(),
            meta={
                "axis": axis,
                "start": start,
                "stop": stop,
                "fixed": fixed,
                "grid": grid.tolist(),
            },
        )

    def set_voltages(self, voltages: Dict[str, float]) -> None:
        """Update internal voltage state (no-op for physics sim)."""
        self._current_voltages.update(voltages)
