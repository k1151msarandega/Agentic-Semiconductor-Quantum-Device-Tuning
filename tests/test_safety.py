"""
tests/test_safety.py
====================
Unit tests and fuzz tests for qdot/hardware/safety.py.

The CI pipeline must run these on every commit.
Zero violations after clip() is a hard requirement.
"""

import pytest
import numpy as np

from qdot.core.types import ActionProposal, VoltagePoint
from qdot.hardware.safety import SafetyCritic


DEFAULT_BOUNDS = {
    "vg1": {"min": -1.0, "max": 1.0},
    "vg2": {"min": -1.0, "max": 1.0},
}


def make_proposal(dvg1: float, dvg2: float) -> ActionProposal:
    return ActionProposal(delta_v=VoltagePoint(dvg1, dvg2))


def make_critic(M_min: float = 0.02, l1_max: float = 0.10) -> SafetyCritic:
    return SafetyCritic(voltage_bounds=DEFAULT_BOUNDS, l1_max=l1_max, M_min=M_min)


# ---------------------------------------------------------------------------
# Clipping
# ---------------------------------------------------------------------------

class TestClip:
    def test_small_step_not_clipped(self):
        critic = make_critic()
        current = VoltagePoint(0.0, 0.0)
        prop = make_proposal(0.04, 0.04)
        result = critic.clip(prop, current)
        assert not result.clipped
        assert result.safe_delta_v is not None
        assert abs(result.safe_delta_v.vg1 - 0.04) < 1e-6

    def test_l1_excess_clipped(self):
        critic = make_critic()
        current = VoltagePoint(0.0, 0.0)
        prop = make_proposal(0.08, 0.08)  # l1 = 0.16 > 0.10
        result = critic.clip(prop, current)
        assert result.clipped
        assert "l1_clip" in result.clip_warnings
        l1 = result.safe_delta_v.l1_norm
        assert l1 <= 0.10 + 1e-6

    def test_upper_bound_clip(self):
        critic = make_critic()
        current = VoltagePoint(0.95, 0.0)
        prop = make_proposal(0.10, 0.0)  # would put vg1 at 1.05 > 1.0
        result = critic.clip(prop, current)
        assert result.clipped
        assert "vg1_upper_bound" in result.clip_warnings
        new_vg1 = current.vg1 + result.safe_delta_v.vg1
        assert new_vg1 <= 1.0 + 1e-6

    def test_lower_bound_clip(self):
        critic = make_critic()
        current = VoltagePoint(-0.95, 0.0)
        prop = make_proposal(-0.10, 0.0)
        result = critic.clip(prop, current)
        assert result.clipped
        assert "vg1_lower_bound" in result.clip_warnings
        new_vg1 = current.vg1 + result.safe_delta_v.vg1
        assert new_vg1 >= -1.0 - 1e-6


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

class TestVerify:
    def test_clean_move_passes_all(self):
        critic = make_critic()
        current = VoltagePoint(0.0, 0.0)
        prop = make_proposal(0.04, 0.03)
        prop = critic.clip(prop, current)
        verdict = critic.verify(current, prop)
        assert verdict.all_passed

    def test_out_of_bounds_fails_voltage_check(self):
        critic = make_critic()
        current = VoltagePoint(0.95, 0.0)
        # Deliberately don't clip — verify should catch it
        prop = ActionProposal(
            delta_v=VoltagePoint(0.10, 0.0),
            safe_delta_v=VoltagePoint(0.10, 0.0),  # force unclipped value
        )
        verdict = critic.verify(current, prop)
        assert not verdict.voltage_bounds.passed

    def test_slew_violation_detected(self):
        critic = make_critic()
        current = VoltagePoint(0.0, 0.0)
        prop = ActionProposal(
            delta_v=VoltagePoint(0.08, 0.08),
            safe_delta_v=VoltagePoint(0.08, 0.08),  # l1 = 0.16
        )
        verdict = critic.verify(current, prop)
        assert not verdict.slew_rate.passed

    def test_min_margin_check(self):
        critic = make_critic(M_min=0.10)
        current = VoltagePoint(0.92, 0.0)
        prop = ActionProposal(
            delta_v=VoltagePoint(0.05, 0.0),
            safe_delta_v=VoltagePoint(0.05, 0.0),
        )
        verdict = critic.verify(current, prop)
        # vg1 will be at 0.97, margin = 0.03 < M_min=0.10
        assert not verdict.voltage_margin.passed


# ---------------------------------------------------------------------------
# Fuzz test — clip then verify must always pass (5000 iterations)
# ---------------------------------------------------------------------------

class TestFuzz:
    def test_clip_then_verify_never_violates(self):
        """
        The safety guarantee: after clip(), verify() must always pass.
        This is the CI contract.
        """
        rng = np.random.default_rng(0)
        critic = make_critic()
        n_iterations = 5000

        for _ in range(n_iterations):
            # Random current position
            current = VoltagePoint(
                vg1=float(rng.uniform(-1.0, 1.0)),
                vg2=float(rng.uniform(-1.0, 1.0)),
            )
            # Random (potentially unsafe) proposed step
            raw_prop = make_proposal(
                dvg1=float(rng.uniform(-0.3, 0.3)),
                dvg2=float(rng.uniform(-0.3, 0.3)),
            )
            clipped = critic.clip(raw_prop, current)
            verdict = critic.verify(current, clipped)

            assert verdict.voltage_bounds.passed, (
                f"Voltage bounds violated after clip: "
                f"current={current}, delta={clipped.safe_delta_v}"
            )
            assert verdict.slew_rate.passed, (
                f"Slew rate violated after clip: "
                f"delta={clipped.safe_delta_v}, l1={clipped.safe_delta_v.l1_norm if clipped.safe_delta_v else 'N/A'}"
            )
