"""
qdot/agent/translator.py
=========================
Translation Agent — converts MeasurementPlan to DeviceAdapter calls.

The ExecutiveAgent plans *what* to measure using the Active Sensing Policy.
This agent converts that plan into *how* to measure it, via the DeviceAdapter.

Pipeline:
    MeasurementPlan (from sensing.py)
        → TranslationAgent.execute(plan, adapter)
        → Measurement (from DeviceAdapter)

Self-reflection error loop:
    1. Generate code string from plan
    2. AST-validate (syntax + safety — no exec, eval, imports)
    3. Execute via eval with {adapter} namespace
    4. If execution fails, log and return None

Naming conventions (must match DeviceAdapter exactly):
    Gate axes:  "vg1", "vg2"  (not "V_g1")
    Voltage dict keys: "vg1", "vg2"
    adapter.sample_patch(v1_range, v2_range, res)
    adapter.line_scan(axis, start, stop, steps, fixed)
    adapter.set_voltages({"vg1": ..., "vg2": ...})

Blueprint reference: §2.1 (Layer 2 — Translation Agent)
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Optional

# Phase 0 types — always import, never redefine
from qdot.core.types import Measurement, MeasurementModality, MeasurementPlan
from qdot.hardware.adapter import DeviceAdapter


@dataclass
class TranslationResult:
    """Result of translating and executing a MeasurementPlan."""
    success: bool
    code: str
    measurement: Optional[Measurement]   # None on failure
    error_message: str = ""
    rationale: str = ""


class TranslationAgent:
    """
    Converts MeasurementPlan objects into DeviceAdapter API calls.

    This is a deterministic template-based translator for Phase 2.
    Phase 3 may add LLM-based translation for complex instructions.
    """

    def __init__(self, adapter: DeviceAdapter):
        """
        Args:
            adapter: Any DeviceAdapter implementation (CIMSimulatorAdapter,
                     or real hardware adapter). The adapter is passed in, not
                     constructed here, to maintain dependency injection.
        """
        self.adapter = adapter

    def execute(self, plan: MeasurementPlan) -> TranslationResult:
        """
        Translate a MeasurementPlan into a DeviceAdapter call and execute it.

        Args:
            plan: MeasurementPlan from ActiveSensingPolicy.select().

        Returns:
            TranslationResult with Measurement on success, None on failure.
        """
        if plan.modality == MeasurementModality.NONE:
            return TranslationResult(
                success=True,
                code="# No measurement: belief peaked",
                measurement=None,
                rationale=plan.rationale,
            )

        # Generate code string
        code, rationale = self._generate_code(plan)

        # Validate before execution
        validation_error = self._validate(code)
        if validation_error:
            return TranslationResult(
                success=False,
                code=code,
                measurement=None,
                error_message=f"Validation failed: {validation_error}",
            )

        # Execute
        try:
            result = eval(code, {"__builtins__": {}}, {"adapter": self.adapter})
            return TranslationResult(
                success=True,
                code=code,
                measurement=result,
                rationale=rationale,
            )
        except Exception as exc:
            return TranslationResult(
                success=False,
                code=code,
                measurement=None,
                error_message=f"Execution error: {exc}",
            )

    def execute_voltage_move(
        self, vg1: float, vg2: float
    ) -> TranslationResult:
        """
        Apply a voltage move via adapter.set_voltages().

        Called by ExecutiveAgent after SafetyCritic.clip() approves a move.

        Args:
            vg1, vg2: Target absolute voltages (AFTER safety clipping).
        """
        code = f'adapter.set_voltages({{"vg1": {vg1:.6f}, "vg2": {vg2:.6f}}})'
        validation_error = self._validate(code)
        if validation_error:
            return TranslationResult(
                success=False,
                code=code,
                measurement=None,
                error_message=validation_error,
            )
        try:
            eval(code, {"__builtins__": {}}, {"adapter": self.adapter})
            return TranslationResult(
                success=True,
                code=code,
                measurement=None,
                rationale=f"Moved to vg1={vg1:.4f}, vg2={vg2:.4f}",
            )
        except Exception as exc:
            return TranslationResult(
                success=False,
                code=code,
                measurement=None,
                error_message=str(exc),
            )

    # ------------------------------------------------------------------
    # Code generation (private)
    # ------------------------------------------------------------------

    def _generate_code(self, plan: MeasurementPlan) -> tuple[str, str]:
        """Generate executable code string from a MeasurementPlan."""

        if plan.modality == MeasurementModality.LINE_SCAN:
            axis = plan.axis or "vg1"
            start = plan.start if plan.start is not None else -1.0
            stop = plan.stop if plan.stop is not None else 1.0
            steps = plan.steps
            fixed = 0.0  # Default fixed gate voltage

            code = (
                f"adapter.line_scan("
                f"axis={axis!r}, "
                f"start={start:.6f}, "
                f"stop={stop:.6f}, "
                f"steps={steps}, "
                f"fixed={fixed:.6f})"
            )
            rationale = f"Line scan {axis} from {start:.4f} to {stop:.4f} ({steps} points)"

        elif plan.modality in (
            MeasurementModality.COARSE_2D,
            MeasurementModality.LOCAL_PATCH,
            MeasurementModality.FINE_2D,
        ):
            v1_range = plan.v1_range or (-1.0, 1.0)
            v2_range = plan.v2_range or (-1.0, 1.0)
            res = plan.resolution

            code = (
                f"adapter.sample_patch("
                f"v1_range=({v1_range[0]:.6f}, {v1_range[1]:.6f}), "
                f"v2_range=({v2_range[0]:.6f}, {v2_range[1]:.6f}), "
                f"res={res})"
            )
            rationale = (
                f"{plan.modality.value} scan {res}×{res} "
                f"v1={v1_range}, v2={v2_range}"
            )

        else:
            code = "None"
            rationale = "Unknown modality"

        return code, rationale

    def _validate(self, code: str) -> Optional[str]:
        """
        Validate generated code before execution.

        Returns:
            Error string if invalid, None if safe.
        """
        # Syntax check
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return f"Syntax error: {exc}"

        # Safety check: only allow adapter.* calls
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return "Import statements not allowed"
            if isinstance(node, ast.Name) and node.id in (
                "exec", "eval", "compile", "open", "__import__"
            ):
                return f"Dangerous built-in: {node.id}"
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute):
                    if not (
                        isinstance(func.value, ast.Name)
                        and func.value.id == "adapter"
                    ):
                        return f"Only adapter.* calls allowed, got: {ast.dump(func)}"
                elif isinstance(func, ast.Name) and func.id != "None":
                    return f"Only adapter.* calls allowed, got function: {func.id}"

        return None
