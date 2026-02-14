"""
qdot/hardware/safety.py
=======================
SafetyCritic — the last gate before any voltage reaches hardware.

Consolidates the old six overlapping safety checks into three clean,
non-redundant constraints (blueprint Section 5.7).

Cannot be bypassed by any agent component. The Executive Agent must
call verify() and receive SafetyVerdict.all_passed == True before
applying any voltage move.

Three checks:
    1. Absolute voltage bounds  — V_min ≤ V_proposed ≤ V_max
    2. Slew rate (L1 norm)      — Σ|ΔV_i| ≤ l1_max per step
    3. Voltage margin           — feeds Risk Score; triggers HITL if < M_min

The per-axis cap from the hackathon is REMOVED (redundant with L1 for 2D).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from qdot.core.types import (
    ActionProposal,
    SafetyCheckResult,
    SafetyVerdict,
    VoltagePoint,
)


class SafetyCritic:
    """
    Enforces hard safety constraints on every proposed voltage move.

    Usage:
        critic = SafetyCritic(voltage_bounds=..., l1_max=0.10, M_min=0.02)
        proposal = critic.clip(proposal)           # clip to safe region
        verdict = critic.verify(current, proposal) # check all constraints
        if not verdict.all_passed:
            raise RuntimeError("Move rejected by safety critic")
    """

    def __init__(
        self,
        voltage_bounds: Optional[Dict[str, Dict[str, float]]] = None,
        l1_max: float = 0.10,
        M_min: float = 0.02,
    ) -> None:
        """
        Args:
            voltage_bounds: per-gate min/max in Volts.
                            e.g. {"vg1": {"min": -1.0, "max": 1.0}, ...}
            l1_max: maximum L1 norm of ΔV per control step (Volts)
            M_min: minimum voltage margin below which HITL is recommended
        """
        if voltage_bounds is None:
            voltage_bounds = {
                "vg1": {"min": -1.0, "max": 1.0},
                "vg2": {"min": -1.0, "max": 1.0},
            }
        self.voltage_bounds = voltage_bounds
        self.l1_max = l1_max
        self.M_min = M_min

    # -----------------------------------------------------------------------
    # Clip (non-destructive: returns modified proposal)
    # -----------------------------------------------------------------------

    def clip(self, proposal: ActionProposal, current: VoltagePoint) -> ActionProposal:
        """
        Clip the proposed ΔV to respect all hard constraints.

        Returns a new ActionProposal with safe_delta_v set.
        The original delta_v is preserved for logging.
        """
        dv = proposal.delta_v
        warnings = list(proposal.clip_warnings)
        clipped = False

        # Step 1: enforce absolute bounds per gate
        adj_vg1 = dv.vg1
        adj_vg2 = dv.vg2

        for gate, dv_val, current_val, attr in [
            ("vg1", dv.vg1, current.vg1, "adj_vg1"),
            ("vg2", dv.vg2, current.vg2, "adj_vg2"),
        ]:
            bounds = self.voltage_bounds.get(gate, {"min": -1.0, "max": 1.0})
            lo, hi = bounds["min"], bounds["max"]
            proposed_abs = current_val + dv_val
            if proposed_abs < lo:
                adj = lo - current_val
                if gate == "vg1":
                    adj_vg1 = adj
                else:
                    adj_vg2 = adj
                warnings.append(f"{gate}_lower_bound")
                clipped = True
            elif proposed_abs > hi:
                adj = hi - current_val
                if gate == "vg1":
                    adj_vg1 = adj
                else:
                    adj_vg2 = adj
                warnings.append(f"{gate}_upper_bound")
                clipped = True

        # Step 2: enforce L1 norm cap
        l1 = abs(adj_vg1) + abs(adj_vg2)
        if l1 > self.l1_max:
            scale = self.l1_max / (l1 + 1e-9)
            adj_vg1 = adj_vg1 * scale
            adj_vg2 = adj_vg2 * scale
            warnings.append("l1_clip")
            clipped = True

        safe_delta = VoltagePoint(vg1=adj_vg1, vg2=adj_vg2)
        new_abs = VoltagePoint(
            vg1=current.vg1 + adj_vg1,
            vg2=current.vg2 + adj_vg2,
        )

        return ActionProposal(
            delta_v=proposal.delta_v,
            safe_delta_v=safe_delta,
            expected_new_voltage=new_abs,
            info_gain=proposal.info_gain,
            clipped=clipped,
            clip_warnings=warnings,
        )

    # -----------------------------------------------------------------------
    # Verify (returns structured verdict — never raises)
    # -----------------------------------------------------------------------

    def verify(self, current: VoltagePoint, proposal: ActionProposal) -> SafetyVerdict:
        """
        Run all three safety checks against the safe_delta_v in proposal.

        Always use this after clip(). Calling verify() on an unclipped
        proposal that violates bounds will correctly report the violation.
        """
        dv = proposal.safe_delta_v or proposal.delta_v

        voltage_check = self._check_voltage_bounds(current, dv)
        slew_check = self._check_slew_rate(dv)
        margin_check = self._check_voltage_margin(current, dv)

        return SafetyVerdict(
            voltage_bounds=voltage_check,
            slew_rate=slew_check,
            voltage_margin=margin_check,
        )

    # -----------------------------------------------------------------------
    # Individual checks
    # -----------------------------------------------------------------------

    def _check_voltage_bounds(
        self, current: VoltagePoint, dv: VoltagePoint
    ) -> SafetyCheckResult:
        """Check 1: absolute bounds V_min ≤ V_proposed ≤ V_max."""
        per_gate: Dict[str, float] = {}
        passed = True

        for gate, current_val, dv_val in [
            ("vg1", current.vg1, dv.vg1),
            ("vg2", current.vg2, dv.vg2),
        ]:
            bounds = self.voltage_bounds.get(gate, {"min": -1.0, "max": 1.0})
            lo, hi = bounds["min"], bounds["max"]
            proposed_abs = current_val + dv_val
            margin = min(hi - proposed_abs, proposed_abs - lo)
            per_gate[gate] = float(margin)
            if margin < 0:
                passed = False

        overall_margin = min(per_gate.values()) if per_gate else 0.0
        return SafetyCheckResult(
            check_name="voltage_bounds",
            passed=passed,
            margin=overall_margin,
            per_gate=per_gate,
        )

    def _check_slew_rate(self, dv: VoltagePoint) -> SafetyCheckResult:
        """Check 2: Σ|ΔV_i| ≤ l1_max per control step."""
        l1 = dv.l1_norm
        margin = float(self.l1_max - l1)
        per_gate = {
            "vg1": float(self.l1_max - abs(dv.vg1)),
            "vg2": float(self.l1_max - abs(dv.vg2)),
        }
        return SafetyCheckResult(
            check_name="slew_rate",
            passed=margin >= 0,
            margin=margin,
            per_gate=per_gate,
            notes=f"L1={l1:.4f} vs cap={self.l1_max:.4f}",
        )

    def _check_voltage_margin(
        self, current: VoltagePoint, dv: VoltagePoint
    ) -> SafetyCheckResult:
        """Check 3: margin to bounds > M_min (contributes to Risk Score)."""
        per_gate: Dict[str, float] = {}

        for gate, current_val, dv_val in [
            ("vg1", current.vg1, dv.vg1),
            ("vg2", current.vg2, dv.vg2),
        ]:
            bounds = self.voltage_bounds.get(gate, {"min": -1.0, "max": 1.0})
            lo, hi = bounds["min"], bounds["max"]
            proposed_abs = current_val + dv_val
            margin = min(hi - proposed_abs, proposed_abs - lo)
            per_gate[gate] = float(margin)

        min_margin = min(per_gate.values()) if per_gate else 0.0
        return SafetyCheckResult(
            check_name="voltage_margin",
            passed=min_margin >= self.M_min,
            margin=min_margin,
            per_gate=per_gate,
            notes=f"min margin={min_margin:.4f} vs M_min={self.M_min:.4f}",
        )
