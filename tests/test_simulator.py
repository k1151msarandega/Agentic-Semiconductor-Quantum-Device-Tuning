"""
tests/test_simulator.py
=======================
Tests for the CIM physics engine and CIMSimulatorAdapter.
"""

import pytest
import numpy as np

from qdot.core.types import MeasurementModality
from qdot.simulator.cim import ConstantInteractionDevice, CIMSimulatorAdapter


class TestConstantInteractionDevice:
    def setup_method(self):
        self.dev = ConstantInteractionDevice(seed=42)

    def test_current_returns_non_negative(self):
        for vg1 in np.linspace(-1, 1, 10):
            for vg2 in np.linspace(-1, 1, 10):
                c = self.dev.current(vg1, vg2)
                assert c >= 0, f"Negative current at ({vg1:.2f}, {vg2:.2f}): {c}"

    def test_chemical_potential_depends_on_n(self):
        mu_01 = self.dev.chemical_potential(0.0, 0.0, 0, 1)
        mu_10 = self.dev.chemical_potential(0.0, 0.0, 1, 0)
        # With different E_c values they should differ
        assert mu_01 != mu_10

    def test_tunnel_coupling_lowers_11_state(self):
        """(1,1) state energy should be lowered by t_c vs no coupling."""
        dev_no_coupling = ConstantInteractionDevice(t_c=0.0, seed=42)
        dev_coupled = ConstantInteractionDevice(t_c=0.4, seed=42)

        e_no = dev_no_coupling.ground_state_energy(0.0, 0.0, 1, 1)
        e_coupled = dev_coupled.ground_state_energy(0.0, 0.0, 1, 1)
        assert e_coupled < e_no


class TestCIMSimulatorAdapter:
    def setup_method(self):
        self.adapter = CIMSimulatorAdapter(device_id="test_device", seed=42)

    def test_sample_patch_shape(self):
        m = self.adapter.sample_patch((-1, 1), (-1, 1), res=16)
        assert m.array.shape == (16, 16)
        assert m.modality == MeasurementModality.COARSE_2D
        assert m.device_id == "test_device"

    def test_sample_patch_normalised(self):
        m = self.adapter.sample_patch((-1, 1), (-1, 1), res=16)
        assert m.array.min() >= 0.0 - 1e-6
        assert m.array.max() <= 1.0 + 1e-6

    def test_line_scan_shape(self):
        m = self.adapter.line_scan(axis="vg1", start=-1, stop=1, steps=64, fixed=0.0)
        assert m.array.shape == (64,)
        assert m.modality == MeasurementModality.LINE_SCAN
        assert m.steps == 64
        assert m.axis == "vg1"

    def test_line_scan_normalised(self):
        m = self.adapter.line_scan(axis="vg2", start=-0.5, stop=0.5, steps=32)
        assert m.array.min() >= 0.0 - 1e-6
        assert m.array.max() <= 1.0 + 1e-6

    def test_device_type_string(self):
        assert "Simulator" in self.adapter.device_type

    def test_set_voltages_does_not_raise(self):
        self.adapter.set_voltages({"vg1": 0.3, "vg2": -0.2})

    def test_health_check(self):
        assert self.adapter.health_check() is True
