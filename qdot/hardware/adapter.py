"""
qdot/hardware/adapter.py
========================
DeviceAdapter — hardware-agnosticism by contract.

The Executive Agent never touches hardware directly.
It calls this interface. The hardware-specific implementation
lives entirely inside each adapter subclass.

Swapping GaAs for Si/SiGe requires zero changes above Layer 4.

To add a new device type:
    1. Subclass DeviceAdapter
    2. Implement sample_patch(), line_scan(), set_voltages()
    3. Register in your config under "adapter_class"
    Zero changes required anywhere else.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np

from qdot.core.types import Measurement, MeasurementModality, VoltagePoint


class DeviceAdapter(ABC):
    """
    Abstract base class for all device interfaces.

    The contract: every method returns normalised data (arrays in [0, 1]).
    Hardware-specific normalisation lives inside each adapter.
    """

    # -----------------------------------------------------------------------
    # Abstract primitives — every adapter must implement these three
    # -----------------------------------------------------------------------

    @abstractmethod
    def sample_patch(
        self,
        v1_range: Tuple[float, float],
        v2_range: Tuple[float, float],
        res: int = 32,
    ) -> Measurement:
        """
        Measure a 2D charge stability diagram patch.

        Args:
            v1_range: (min, max) for gate 1 in Volts
            v2_range: (min, max) for gate 2 in Volts
            res: number of pixels per side (res × res grid)

        Returns:
            Measurement with array of shape (res, res), normalised to [0, 1]
        """

    @abstractmethod
    def line_scan(
        self,
        axis: str,
        start: float,
        stop: float,
        steps: int = 128,
        fixed: float = 0.0,
    ) -> Measurement:
        """
        1D conductance scan along one gate axis.

        Args:
            axis: "vg1" or "vg2"
            start: start voltage (V)
            stop: stop voltage (V)
            steps: number of measurement points
            fixed: voltage of the orthogonal gate

        Returns:
            Measurement with array of shape (steps,), normalised to [0, 1]
        """

    @abstractmethod
    def set_voltages(self, voltages: Dict[str, float]) -> None:
        """
        Apply a gate voltage configuration to the device.

        Args:
            voltages: e.g. {"vg1": 0.25, "vg2": -0.10}
        """

    # -----------------------------------------------------------------------
    # Optional: adapter metadata
    # -----------------------------------------------------------------------

    @property
    def device_type(self) -> str:
        """Human-readable device type string (e.g. 'CIM Simulator', 'Si/SiGe HW')."""
        return self.__class__.__name__

    def health_check(self) -> bool:
        """
        Optional: verify the device is responding correctly.
        Override in hardware adapters for real connectivity checks.
        Default: always returns True (suitable for simulators).
        """
        return True

    # -----------------------------------------------------------------------
    # Shared normalisation utility — adapters should call this
    # -----------------------------------------------------------------------

    @staticmethod
    def _normalise(arr: np.ndarray) -> np.ndarray:
        """Normalise a conductance array to [0, 1]. Handles flat arrays."""
        arr = arr.astype(np.float32)
        lo, hi = arr.min(), arr.max()
        if hi - lo > 1e-12:
            return (arr - lo) / (hi - lo)
        return np.full_like(arr, 0.5)
