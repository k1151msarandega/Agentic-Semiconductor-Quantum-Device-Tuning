"""
qdot/simulator/cim.py
=====================
Constant Interaction Model (CIM) physics simulator for double quantum dots.

Physics Model:
    Two quantum dots coupled capacitively and by a tunnel barrier.
    Charging energy: E_c = e²/2C (capacitive energy cost per electron)
    Inter-dot coupling: E_m (capacitive cross-coupling between dots)
    Tunnel coupling: t_c (interdot hopping amplitude)
    Gate voltage → energy via lever arm: E = α * V_gate

References:
    van der Wiel et al., Rev. Mod. Phys. 75, 1 (2002) — Electron transport in double dots
    Hanson et al., Rev. Mod. Phys. 79, 1217 (2007) — Spin qubits review
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import numpy as np

from qdot.core.types import Measurement, MeasurementModality, VoltagePoint
from qdot.hardware.adapter import DeviceAdapter


class ConstantInteractionDevice:
    """Corrected Double Quantum Dot CIM Physics Engine."""

    def __init__(
        self,
        E_c1: float = 1.20,       # Self-charging energy Dot 1 (increased for clear scaling)
        E_c2: float = 1.25,       # Self-charging energy Dot 2
        E_m: float = 0.25,        # CRITICAL FIX: Inter-dot capacitive cross-coupling energy
        t_c: float = 0.04,        # Tunnel coupling hybridization factor
        T: float = 0.020,         # Effective electron temperature
        lever_arm: float = 1.0,   # Voltage-to-energy conversion arm
        noise_level: float = 0.005,
        seed: Optional[int] = None,
    ) -> None:
        self.E_c1 = E_c1
        self.E_c2 = E_c2
        self.E_m = E_m            # Inter-dot interaction energy
        self.t_c = t_c
        self.T = T
        self.alpha = lever_arm
        self.noise_level = noise_level
        self.rng = np.random.default_rng(seed)

        self._disorder_map: Optional[np.ndarray] = None
        self._disorder_v1_grid: Optional[np.ndarray] = None
        self._disorder_v2_grid: Optional[np.ndarray] = None

    def ground_state_energy(self, vg1: float, vg2: float, n1: int, n2: int) -> float:
        """
        Calculates the total free energy for a given charge configuration (n1, n2).
        Uses a classical quadratic charging model with capacitive cross-coupling.
        """
        # 1. Electrostatic charging energy matrix (quadratic cost + mutual interaction)
        E_charging = 0.5 * self.E_c1 * (n1**2) + 0.5 * self.E_c2 * (n2**2) + self.E_m * n1 * n2
        
        # 2. Electrostatic potential from the gates (negative because positive V pulls energy down)
        E_gate = -self.alpha * (vg1 * n1 + vg2 * n2)
        
        E_total = E_charging + E_gate

        # 4. Tunnel-coupling quantum correction near triple-point boundaries
        if n1 > 0 and n2 > 0:
            E_total -= self.t_c
            
        return E_total

    def current(self, vg1: float, vg2: float) -> float:
        """Scalar conductance at (vg1, vg2) based on ground state transition gaps."""
        if self._disorder_map is not None:
            disorder_offset = self._interpolate_disorder(vg1, vg2)
            vg1 = vg1 + disorder_offset * 0.1

        # Evaluate energy for all accessible discrete charge configurations (up to 3 e- per dot)
        states = [(n1, n2) for n1 in range(4) for n2 in range(4)]
        energies = [self.ground_state_energy(vg1, vg2, n1, n2) for n1, n2 in states]
        sorted_energies = sorted(energies)
        
        # Transport occurs when the lowest two energy states are nearly degenerate
        energy_gap = sorted_energies[1] - sorted_energies[0]

        broadening = max(self.t_c, self.T)
        conductance = broadening / (energy_gap ** 2 + broadening ** 2)

        if self.noise_level > 0:
            conductance += self.rng.normal(0, self.noise_level)

        return float(np.clip(conductance, 0, None))

    def current_grid(self, v1_grid: np.ndarray, v2_grid: np.ndarray) -> np.ndarray:
        """Vectorised 2D conductance map over a grid for fast CNN patch generation."""
        VG1, VG2 = np.meshgrid(
            v1_grid.astype(np.float64),
            v2_grid.astype(np.float64),
        )

        alpha = float(self.alpha)
        E_c1 = float(self.E_c1)
        E_c2 = float(self.E_c2)
        E_m = float(self.E_m)
        t_c = float(self.t_c)

        states = [(n1, n2) for n1 in range(4) for n2 in range(4)]
        slabs = []
        for n1, n2 in states:
            # Broadcast electrostatic equations over the grid arrays
            E_charging = 0.5 * E_c1 * (n1**2) + 0.5 * E_c2 * (n2**2) + E_m * n1 * n2
            E_gate = -alpha * (VG1 * n1 + VG2 * n2)
            e = E_charging + E_gate
            if n1 > 0 and n2 > 0:
                e = e - t_c
            slabs.append(e)

        energies = np.stack(slabs, axis=0)          
        sorted_e = np.sort(energies, axis=0)        
        energy_gap = sorted_e[1] - sorted_e[0]      

        broadening = max(t_c, float(self.T))
        patch = broadening / (energy_gap ** 2 + broadening ** 2)

        if self.noise_level > 0:
            patch = patch + self.rng.normal(0, self.noise_level, patch.shape)

        return np.clip(patch, 0, None).astype(np.float32)

    def current_for_state(self, vg1: float, vg2: float, n1: int, n2: int) -> float:
        """POMDP observation model: predicted conductance conditioned on occupancy."""
        if self._disorder_map is not None:
            disorder_offset = self._interpolate_disorder(vg1, vg2)
            vg1 = vg1 + disorder_offset * 0.1

        E_target = self.ground_state_energy(vg1, vg2, n1, n2)

        all_states = [(m1, m2) for m1 in range(4) for m2 in range(4)]
        E_min = min(self.ground_state_energy(vg1, vg2, m1, m2) for m1, m2 in all_states)

        delta_E = max(0.0, E_target - E_min)
        T_eff = max(self.T, 0.01)
        boltzmann = float(np.exp(-delta_E / T_eff))

        neighbour_energies = []
        for dn1, dn2 in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            m1, m2 = n1 + dn1, n2 + dn2
            if 0 <= m1 <= 3 and 0 <= m2 <= 3:
                neighbour_energies.append(self.ground_state_energy(vg1, vg2, m1, m2))

        if not neighbour_energies:
            return 0.0

        energy_gap = min(abs(E_n - E_target) for E_n in neighbour_energies)
        broadening = max(self.t_c, self.T)
        conductance = broadening / (energy_gap ** 2 + broadening ** 2)

        return float(np.clip(conductance * boltzmann, 0, None))

    def inject_disorder(self, disorder_posterior: Dict) -> None:
        self._disorder_map = np.array(disorder_posterior["mean"])
        self._disorder_v1_grid = np.array(disorder_posterior["v1_grid"])
        self._disorder_v2_grid = np.array(disorder_posterior["v2_grid"])

    def _interpolate_disorder(self, vg1: float, vg2: float) -> float:
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
    """Drop-in DeviceAdapter mapping to the updated CIM physics engine."""

    DEFAULT_PARAMS = {
        "E_c1": 1.20,
        "E_c2": 1.25,
        "E_m": 0.25,
        "t_c": 0.04,
        "T": 0.020,
        "lever_arm": 1.0,
        "noise_level": 0.005,
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
        return "CIM Double-Dot Simulator"

    def sample_patch(
        self,
        v1_range: Tuple[float, float] = (0.0, 4.0),
        v2_range: Tuple[float, float] = (0.0, 4.0),
        res: int = 64,
    ) -> Measurement:
        v1_grid = np.linspace(v1_range[0], v1_range[1], res, dtype=np.float32)
        v2_grid = np.linspace(v2_range[0], v2_range[1], res, dtype=np.float32)

        if self.device._disorder_map is None:
            patch = self.device.current_grid(v1_grid, v2_grid)
        else:
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
                "E_m": self.device.E_m,
                "t_c": self.device.t_c,
                "model": "Constant Interaction Double-Dot Model",
            },
        )

    def line_scan(
        self,
        axis: str = "vg1",
        start: float = 0.0,
        stop: float = 4.0,
        steps: int = 128,
        fixed: float = 2.0,
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
        self._current_voltages.update(voltages)
