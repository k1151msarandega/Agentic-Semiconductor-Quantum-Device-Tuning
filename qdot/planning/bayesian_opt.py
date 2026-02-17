"""
qdot/planning/bayesian_opt.py
==============================
Multi-fidelity Bayesian Optimisation for voltage navigation.

Proposes ActionProposal objects (from qdot.core.types) using a Gaussian
Process surrogate model with CIM-informed prior mean.

Key types (from qdot.core.types — NOT redefined here):
    ActionProposal — proposed ΔV, passed to SafetyCritic before execution
    BOPoint        — single BO observation (stored in ExperimentState.bo_history)
    VoltagePoint   — (vg1, vg2) coordinate

Acquisition function: Upper Confidence Bound (UCB)
    UCB(V) = μ(V) + β·σ(V)

Blueprint reference: §5.1 (POMDP value computation / BO planner)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple

# Phase 0 types — never redefine
from qdot.core.types import ActionProposal, BOPoint, ChargeLabel, VoltagePoint
from qdot.core.state import BeliefState
from qdot.simulator.cim import ConstantInteractionDevice


class GaussianProcess:
    """
    Squared-exponential (RBF) kernel GP with physics-informed prior mean.

    Works in (vg1, vg2) space. All inputs/outputs are native VoltagePoint
    or float scalars — no numpy tuples in the public interface.
    """

    def __init__(
        self,
        length_scale: float = 0.05,
        signal_var: float = 0.5,
        noise_var: float = 0.01,
    ):
        self.length_scale = length_scale
        self.signal_var = signal_var
        self.noise_var = noise_var
        self._X: List[Tuple[float, float]] = []   # (vg1, vg2) pairs
        self._y: List[float] = []
        self._K_inv: Optional[np.ndarray] = None
        self.prior_mean_fn = lambda vg1, vg2: 0.0

    def set_prior_mean(self, fn) -> None:
        self.prior_mean_fn = fn

    def fit(self, bo_history: List[BOPoint]) -> None:
        """Fit GP to BO observation history from ExperimentState.bo_history."""
        if not bo_history:
            return
        self._X = [(p.voltage.vg1, p.voltage.vg2) for p in bo_history]
        self._y = [p.score for p in bo_history]
        n = len(self._X)
        K = np.array([[self._k(self._X[i], self._X[j]) for j in range(n)] for i in range(n)])
        K += self.noise_var * np.eye(n)
        try:
            self._K_inv = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            self._K_inv = np.linalg.inv(K + 1e-6 * np.eye(n))

    def predict(self, vg1: float, vg2: float) -> Tuple[float, float]:
        """Return (mean, variance) at (vg1, vg2)."""
        v_test = (vg1, vg2)
        prior = self.prior_mean_fn(vg1, vg2)
        if not self._X or self._K_inv is None:
            return prior, self.signal_var

        k_vec = np.array([self._k(xi, v_test) for xi in self._X])
        y_adj = np.array([
            self._y[i] - self.prior_mean_fn(self._X[i][0], self._X[i][1])
            for i in range(len(self._y))
        ])
        mean = prior + float(k_vec @ self._K_inv @ y_adj)
        var = max(0.0, float(self._k(v_test, v_test) - k_vec @ self._K_inv @ k_vec))
        return mean, var

    def _k(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        dist_sq = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
        return self.signal_var * np.exp(-dist_sq / (2 * self.length_scale ** 2))


class MultiResBO:
    """
    Multi-fidelity Bayesian Optimisation for voltage navigation.

    Usage:
        bo = MultiResBO(belief=state.belief, voltage_bounds=state.voltage_bounds)
        bo.update(state.bo_history)
        proposal = bo.propose(state.current_voltage, l1_max=state.step_caps["l1_max"])
        # → ActionProposal, pass to SafetyCritic.clip() before applying
    """

    def __init__(
        self,
        belief: BeliefState,
        voltage_bounds: Optional[Dict] = None,
        exploration_weight: float = 2.0,
        device: Optional[ConstantInteractionDevice] = None,
    ):
        """
        Args:
            belief: Current BeliefState (read-only — BO observes but doesn't update it).
            voltage_bounds: From ExperimentState.voltage_bounds.
                            Format: {"vg1": {"min": -1.0, "max": 1.0}, "vg2": ...}
            exploration_weight: UCB β parameter.
            device: CIM device for prior mean function.
        """
        self.belief = belief
        self.voltage_bounds = voltage_bounds or {
            "vg1": {"min": -1.0, "max": 1.0},
            "vg2": {"min": -1.0, "max": 1.0},
        }
        self.exploration_weight = exploration_weight
        self.device = device or ConstantInteractionDevice()

        self.gp = GaussianProcess()
        self._update_gp_prior()

    def update(self, bo_history: List[BOPoint]) -> None:
        """
        Re-fit GP to current BO history and refresh CIM prior.

        Call this after each new BOPoint is added to ExperimentState.bo_history.
        """
        self._update_gp_prior()
        self.gp.fit(bo_history)

    def propose(self, current: VoltagePoint, l1_max: float = 0.10) -> ActionProposal:
        """
        Propose a voltage move using UCB acquisition.

        Args:
            current: Current VoltagePoint from ExperimentState.current_voltage.
            l1_max: Step size cap from ExperimentState.step_caps["l1_max"].

        Returns:
            ActionProposal — pass to SafetyCritic.clip() before applying.
        """
        vg1_bounds = self.voltage_bounds.get("vg1", {"min": -1.0, "max": 1.0})
        vg2_bounds = self.voltage_bounds.get("vg2", {"min": -1.0, "max": 1.0})

        # Search bounds: intersection of global bounds and step cap
        search_bounds = [
            (max(vg1_bounds["min"], current.vg1 - l1_max / 2),
             min(vg1_bounds["max"], current.vg1 + l1_max / 2)),
            (max(vg2_bounds["min"], current.vg2 - l1_max / 2),
             min(vg2_bounds["max"], current.vg2 + l1_max / 2)),
        ]

        # Maximise UCB acquisition via L-BFGS-B
        def neg_ucb(xy: np.ndarray) -> float:
            mu, var = self.gp.predict(xy[0], xy[1])
            return -(mu + self.exploration_weight * np.sqrt(var))

        x0 = np.array([current.vg1, current.vg2])
        result = minimize(neg_ucb, x0, method="L-BFGS-B", bounds=search_bounds)

        if result.success:
            new_vg1, new_vg2 = float(result.x[0]), float(result.x[1])
        else:
            # Fallback: random within bounds
            new_vg1 = float(np.random.uniform(search_bounds[0][0], search_bounds[0][1]))
            new_vg2 = float(np.random.uniform(search_bounds[1][0], search_bounds[1][1]))

        delta = VoltagePoint(vg1=new_vg1 - current.vg1, vg2=new_vg2 - current.vg2)
        expected_new = VoltagePoint(vg1=new_vg1, vg2=new_vg2)

        # Info gain: current uncertainty - expected posterior uncertainty
        _, var_now = self.gp.predict(current.vg1, current.vg2)
        _, var_new = self.gp.predict(new_vg1, new_vg2)
        info_gain = max(0.0, float(var_now - var_new))

        return ActionProposal(
            delta_v=delta,
            expected_new_voltage=expected_new,
            info_gain=info_gain,
        )

    def make_bo_point(
        self,
        voltage: VoltagePoint,
        score: float,
        label: ChargeLabel = ChargeLabel.UNKNOWN,
        confidence: float = 0.0,
        step: int = 0,
    ) -> BOPoint:
        """
        Convenience factory for creating BOPoint objects to add to
        ExperimentState.bo_history. ExperimentState.add_classification()
        already creates BOPoints automatically, but this is available for
        explicit BO-driven observations.
        """
        return BOPoint(
            voltage=voltage,
            score=score,
            label=label,
            confidence=confidence,
            step=step,
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _update_gp_prior(self) -> None:
        """Set GP prior mean to CIM prediction weighted by current belief."""
        belief = self.belief

        def prior_fn(vg1: float, vg2: float) -> float:
            if not belief.charge_probs:
                return self.device.current(vg1, vg2)
            # Weighted sum over belief
            mu = 0.0
            for (n1, n2), prob in belief.charge_probs.items():
                mu += prob * self.device.current(vg1, vg2)
            return float(mu)

        self.gp.set_prior_mean(prior_fn)
