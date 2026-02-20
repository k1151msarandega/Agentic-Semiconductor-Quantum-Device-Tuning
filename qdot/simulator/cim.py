"""
qdot/simulator/cim.py
=====================
Constant Interaction Model (CIM) physics simulator for double quantum dots.

Ported directly from the hackathon ConstantInteractionDevice class —
the physics was solid, so it's preserved almost as-is per the blueprint.

Physics Model:
    Two quantum dots coupled by a tunnel barrier.
    Charging energy: E_c = e²/2C (capacitive energy cost per electron)
    Tunnel coupling: t_c (interdot hopping amplitude)
    Gate voltage → energy via lever arm: E = α * V_gate
    Current from Fermi-Dirac statistics at temperature T

References:
    Koch et al., Phys. Rev. A 76, 042319 (2007) — Charge qubits
    Hanson et al., Rev. Mod. Phys. 79, 1217 (2007) — Spin qubits review
    van der Wiel et al., Rev. Mod. Phys. 75, 1 (2002) — Electron transport

This is a prototyping simulator for control algorithm development.
Hardware deployment requires system identification from real measurements.
"""

from __future__ import annotations

import time
import uuid
from typing import Dict, Optional, Tuple

import numpy as np

from qdot.core.types import Measurement, MeasurementModality, VoltagePoint
from qdot.hardware.adapter import DeviceAdapter


# ---------------------------------------------------------------------------
# Physics core — ConstantInteractionDevice
# ---------------------------------------------------------------------------

class ConstantInteractionDevice:
    """
    The CIM physics engine.

    Separated from the adapter so the physics can be used directly
    by the POMDP belief updater (Phase 2) without going through the
    adapter layer.
    """

    def __init__(
        self,
        E_c1: float = 2.0,
        E_c2: float = 2.2,
        t_c: float = 0.30,
        T: float = 0.05,
        lever_arm: float = 0.80,
        noise_level: float = 0.008,
        seed: Optional[int] = None,
    ) -> None:
        """
        Args:
            E_c1: Charging energy dot 1 (meV)
            E_c2: Charging energy dot 2 (meV)
            t_c: Tunnel coupling (meV)
            T: Temperature (meV; ~0.6 K for 0.05 meV)
            lever_arm: Gate voltage to energy conversion (dimensionless)
            noise_level: Gaussian noise standard deviation
            seed: Random seed for reproducibility

        Parameter rationale
        -------------------
        Defaults are chosen so the CIM produces a detectable conductance gradient
        within the ±1.0 V scan window used by the benchmark, while remaining within
        the training-data distribution (CIMDataset dd_E_c_range=(1.8,5.5),
        dd_lever_range=(0.35,0.85)).

        The charge transition for dot 1 falls at vg1 = -E_c1/lever_arm = -2.5 V
        (outside the ±1.0 V window, as expected for real devices). The conductance
        gradient approaching that transition from within the scan window gives
        SNR ≈ 26 dB, well above the DQC HIGH threshold of 20 dB.

        Previous defaults (E_c1=2.3, lever_arm=0.5) placed the transition at
        -4.6 V, yielding SNR ≈ 5 dB within ±1.0 V → DQC LOW on every measurement.
        """
        self.E_c1 = E_c1
        self.E_c2 = E_c2
        self.t_c = t_c
        self.T = T
        self.alpha = lever_arm
        self.noise_level = noise_level
        self.rng = np.random.default_rng(seed)

        # Disorder offset (injected by DisorderLearner in Phase 3)
        self._disorder_map: Optional[np.ndarray] = None
        self._disorder_v1_grid: Optional[np.ndarray] = None
        self._disorder_v2_grid: Optional[np.ndarray] = None

    # -----------------------------------------------------------------------
    # Physics calculations
    # -----------------------------------------------------------------------

    def chemical_potential(
        self, vg1: float, vg2: float, n1: int, n2: int
    ) -> float:
        """
        Chemical potential for charge state (n1, n2).

            μ(n1, n2) = E_c1·n1 + E_c2·n2 + α·(V_g1·n1 + V_g2·n2)
        """
        E_charge = self.E_c1 * n1 + self.E_c2 * n2
        E_gate = self.alpha * (vg1 * n1 + vg2 * n2)
        return E_charge + E_gate

    def ground_state_energy(
        self, vg1: float, vg2: float, n1: int, n2: int
    ) -> float:
        """
        Ground state energy for charge configuration (n1, n2).
        Includes tunnel coupling for (1,1) stabilisation.
        """
        mu = self.chemical_potential(vg1, vg2, n1, n2)
        # Tunnel coupling lowers (1,1) state energy (anti-crossing)
        if n1 == 1 and n2 == 1:
            mu -= self.t_c
        return mu

    def current(self, vg1: float, vg2: float) -> float:
        """
        Measure current at given gate voltages.

        Returns normalised current ∈ [0, ∞) (clipped to ≥ 0) before
        the adapter normalises to [0, 1].

        High current occurs at charge degeneracy points (Coulomb lines).
        """
        # Apply disorder offset if fitted (Phase 3)
        if self._disorder_map is not None:
            disorder_offset = self._interpolate_disorder(vg1, vg2)
            vg1 = vg1 + disorder_offset * 0.1  # small perturbation

        # Evaluate all relevant charge states (n1, n2) ∈ {0,1,2}²
        states = [(n1, n2) for n1 in range(3) for n2 in range(3)]
        energies = [self.ground_state_energy(vg1, vg2, n1, n2) for n1, n2 in states]

        # Energy gap to next excited state
        sorted_energies = sorted(energies)
        energy_gap = sorted_energies[1] - sorted_energies[0]

        # Conductance: Lorentzian peak at charge degeneracy
        broadening = max(self.t_c, self.T)
        conductance = broadening / (energy_gap ** 2 + broadening ** 2)

        # Add Gaussian noise
        if self.noise_level > 0:
            conductance += self.rng.normal(0, self.noise_level)

        return float(np.clip(conductance, 0, None))

    def current_for_state(self, vg1: float, vg2: float, n1: int, n2: int) -> float:
        """
        POMDP observation model: predicted conductance conditioned on (n1, n2)
        being the hypothesised charge state.

        Unlike current(), which finds the true quantum ground state by minimising
        over all charge configurations, this method evaluates conductance from the
        perspective of a specific hypothesis (n1, n2). It is used exclusively by:
            - BeliefUpdater  (particle likelihood computation)
            - ActiveSensingPolicy (simulated measurement for IG estimation)
            - BeliefUpdater.uncertainty_map

        It must NOT be called from the hardware adapter layer — only current()
        should be used there, so real measurements always reflect true physics.

        Physics
        -------
        Conductance peaks at charge degeneracy: boundaries between stability
        regions where two adjacent charge states have equal energy. For the
        hypothesis (n1, n2), we compute the energy gap to each nearest neighbour
        {(n1±1,n2), (n1,n2±1)}, use the minimum gap for the Lorentzian, and
        weight by a Boltzmann factor exp(-ΔE/T) where ΔE = E(n1,n2) - E_min ≥ 0.

        When (n1,n2) IS the ground state, ΔE=0 → weight=1, same result as current().
        When it is NOT, the Boltzmann factor penalises the inconsistent hypothesis,
        making particles distinguishable by the voltage regions where they are stable.

        Args:
            vg1, vg2: Gate voltages (V).
            n1, n2:   Hypothesised charge occupation.

        Returns:
            Predicted conductance ∈ [0, ∞), deterministic (no noise added).
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
        """
        Inject device-specific disorder from DisorderLearner (Phase 3).

        Args:
            disorder_posterior: dict with keys "mean" (2D array),
                                "v1_grid", "v2_grid"
        """
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


# ---------------------------------------------------------------------------
# CIM Simulator Adapter — wraps device behind the DeviceAdapter interface
# ---------------------------------------------------------------------------

class CIMSimulatorAdapter(DeviceAdapter):
    """
    Drop-in DeviceAdapter implementation using the CIM physics engine.

    Maintains the same API as the hackathon's PhysicsSimulator, but
    now returns typed Measurement objects instead of raw tuples.

    Default parameters are tuned for clear double-dot features:
        E_c = 3.5 meV, t_c = 0.4 meV, T = 0.05 meV, noise = 0.01
    """

    DEFAULT_PARAMS = {
        "E_c1": 2.0,
        "E_c2": 2.2,
        "t_c": 0.30,
        "T": 0.05,
        "lever_arm": 0.80,
        "noise_level": 0.008,
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

    # -----------------------------------------------------------------------
    # DeviceAdapter interface
    # -----------------------------------------------------------------------

    def sample_patch(
        self,
        v1_range: Tuple[float, float] = (-1.0, 1.0),
        v2_range: Tuple[float, float] = (-1.0, 1.0),
        res: int = 32,
    ) -> Measurement:
        v1_grid = np.linspace(v1_range[0], v1_range[1], res, dtype=np.float32)
        v2_grid = np.linspace(v2_range[0], v2_range[1], res, dtype=np.float32)

        patch = np.zeros((res, res), dtype=np.float32)
        for i, v2 in enumerate(v2_grid):
            for j, v1 in enumerate(v1_grid):
                patch[i, j] = self.device.current(v1, v2)

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
