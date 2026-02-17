"""
qdot/planning/sensing.py
========================
Active Sensing Policy — information-theoretic measurement selection.

Replaces the hackathon's fixed Ct_low / Ct_high thresholds (blueprint §7.1: removed).

For each candidate modality, computes:
    score = I(belief; measurement) / cost(measurement)
Where I is estimated expected mutual information (entropy reduction).

Returns a typed MeasurementPlan (from qdot.core.types) that the
TranslationAgent converts into a DeviceAdapter call.

Cost model (blueprint §5.4):
    LINE_SCAN   → 128 points
    COARSE_2D   → 256 points  (16×16)
    LOCAL_PATCH → 1024 points (32×32)
    FINE_2D     → 4096 points (64×64)

Blueprint reference: §5.4 (Active Sensing Policy), Fig. 6
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple

# Always import from qdot.core.types — never redefine these
from qdot.core.types import MeasurementModality, MeasurementPlan
from qdot.core.state import BeliefState
from qdot.simulator.cim import ConstantInteractionDevice


# Cost model in measurement points (blueprint §5.4)
MODALITY_COST: Dict[MeasurementModality, int] = {
    MeasurementModality.LINE_SCAN:   128,
    MeasurementModality.COARSE_2D:   256,
    MeasurementModality.LOCAL_PATCH: 1024,
    MeasurementModality.FINE_2D:     4096,
}

MODALITY_RESOLUTION: Dict[MeasurementModality, int] = {
    MeasurementModality.LINE_SCAN:   128,
    MeasurementModality.COARSE_2D:   16,
    MeasurementModality.LOCAL_PATCH: 32,
    MeasurementModality.FINE_2D:     64,
}


class ActiveSensingPolicy:
    """
    Selects the next measurement to maximise information gain per cost.

    Monte Carlo information gain estimation:
    1. Sample N charge-state hypotheses from current belief
    2. For each, compute posterior entropy if that measurement were taken
    3. IG = H(prior) - E[H(posterior)]
    4. Return modality with highest IG / cost

    Line scans and 2D patches are both considered. The policy does NOT
    decide which will go through InspectionAgent — that is the Executive
    Agent's responsibility (line scans bypass InspectionAgent by design).
    """

    def __init__(
        self,
        device: Optional[ConstantInteractionDevice] = None,
        n_mc_samples: int = 8,
        info_gain_threshold: float = 1e-4,
    ):
        """
        Args:
            device: CIM physics device for simulating hypothetical measurements.
                    Uses default ConstantInteractionDevice if None.
            n_mc_samples: Monte Carlo samples for IG estimation (8 balances
                          accuracy vs speed — a key budget constraint).
            info_gain_threshold: Minimum IG/cost to justify any measurement.
        """
        self.device = device or ConstantInteractionDevice()
        self.n_mc_samples = n_mc_samples
        self.info_gain_threshold = info_gain_threshold

    def select(
        self,
        belief: BeliefState,
        v1_range: Tuple[float, float],
        v2_range: Tuple[float, float],
    ) -> MeasurementPlan:
        """
        Select the optimal next measurement.

        Args:
            belief: Current BeliefState (from ExperimentState.belief).
            v1_range: (min, max) for gate 1 in Volts.
            v2_range: (min, max) for gate 2 in Volts.

        Returns:
            MeasurementPlan — the typed output from qdot.core.types.
        """
        prior_entropy = belief.entropy()

        # Evaluate all non-NONE modalities
        best_score = -1.0
        best_plan = MeasurementPlan(
            modality=MeasurementModality.NONE,
            rationale="No measurement: information gain below threshold",
        )

        candidates = [
            MeasurementModality.LINE_SCAN,
            MeasurementModality.COARSE_2D,
            MeasurementModality.LOCAL_PATCH,
            MeasurementModality.FINE_2D,
        ]

        for modality in candidates:
            ig = self._estimate_ig(belief, modality, v1_range, v2_range)
            cost = MODALITY_COST[modality]
            score = ig / cost if cost > 0 else 0.0

            if score > best_score:
                best_score = score
                resolution = MODALITY_RESOLUTION[modality]

                if modality == MeasurementModality.LINE_SCAN:
                    # Scan the axis with higher uncertainty range
                    plan = MeasurementPlan(
                        modality=modality,
                        axis="vg1",
                        start=v1_range[0],
                        stop=v1_range[1],
                        steps=resolution,
                        rationale=f"Line scan: IG/cost={score:.6f}",
                        info_gain_per_cost=score,
                    )
                else:
                    plan = MeasurementPlan(
                        modality=modality,
                        v1_range=v1_range,
                        v2_range=v2_range,
                        resolution=resolution,
                        rationale=f"{modality.value}: IG/cost={score:.6f} (IG={ig:.4f}, cost={cost})",
                        info_gain_per_cost=score,
                    )

        if best_score < self.info_gain_threshold:
            return MeasurementPlan(
                modality=MeasurementModality.NONE,
                rationale=f"Max IG/cost={best_score:.6f} below threshold={self.info_gain_threshold:.6f}",
            )

        return best_plan

    # ------------------------------------------------------------------
    # Private: Monte Carlo IG estimation
    # ------------------------------------------------------------------

    def _estimate_ig(
        self,
        belief: BeliefState,
        modality: MeasurementModality,
        v1_range: Tuple[float, float],
        v2_range: Tuple[float, float],
    ) -> float:
        """Expected information gain = H(prior) - E[H(posterior)]."""
        prior_entropy = belief.entropy()
        if prior_entropy == 0.0:
            return 0.0  # Already fully certain

        resolution = MODALITY_RESOLUTION[modality]
        posterior_entropies = []

        # Sample hypothetical charge states from current belief
        states = list(belief.charge_probs.keys())
        probs = np.array([belief.charge_probs[s] for s in states], dtype=float)
        probs = probs / probs.sum()

        for _ in range(self.n_mc_samples):
            idx = np.random.choice(len(states), p=probs)
            n1, n2 = states[idx]

            # Simulate a measurement from this charge state
            if modality == MeasurementModality.LINE_SCAN:
                observed = self._sim_1d(n1, n2, v1_range, resolution)
            else:
                observed = self._sim_2d(n1, n2, v1_range, v2_range, resolution)

            # Compute posterior entropy after this hypothetical measurement
            post_ent = self._posterior_entropy(
                belief, observed, modality, v1_range, v2_range
            )
            posterior_entropies.append(post_ent)

        expected_post = float(np.mean(posterior_entropies))
        return max(0.0, prior_entropy - expected_post)

    def _sim_2d(self, n1: int, n2: int, v1_range, v2_range, resolution: int) -> np.ndarray:
        v1 = np.linspace(v1_range[0], v1_range[1], resolution)
        v2 = np.linspace(v2_range[0], v2_range[1], resolution)
        patch = np.zeros((resolution, resolution), dtype=np.float32)
        for i, vv2 in enumerate(v2):
            for j, vv1 in enumerate(v1):
                patch[i, j] = self.device.current(vv1, vv2)
        patch += np.random.normal(0, 0.02, patch.shape).astype(np.float32)
        return patch

    def _sim_1d(self, n1: int, n2: int, v_range, steps: int) -> np.ndarray:
        v = np.linspace(v_range[0], v_range[1], steps)
        trace = np.array([self.device.current(vv, 0.0) for vv in v], dtype=np.float32)
        trace += np.random.normal(0, 0.02, trace.shape).astype(np.float32)
        return trace

    def _posterior_entropy(
        self,
        belief: BeliefState,
        observed: np.ndarray,
        modality: MeasurementModality,
        v1_range: Tuple[float, float],
        v2_range: Tuple[float, float],
    ) -> float:
        """Approximate posterior entropy after observing `observed`."""
        noise_std = 0.05
        resolution = MODALITY_RESOLUTION[modality]
        log_weights: Dict[tuple, float] = {}

        for state, prior_prob in belief.charge_probs.items():
            if prior_prob <= 0:
                continue
            n1, n2 = state
            if modality == MeasurementModality.LINE_SCAN:
                v = np.linspace(v1_range[0], v1_range[1], len(observed))
                predicted = np.array([self.device.current(vv, 0.0) for vv in v])
            else:
                predicted = self._sim_2d(n1, n2, v1_range, v2_range, resolution)

            residuals = (observed - predicted) / (noise_std + 1e-8)
            ll = float(-0.5 * np.mean(residuals ** 2))
            log_weights[state] = np.log(prior_prob + 1e-12) + ll

        if not log_weights:
            return 0.0

        # Normalise
        log_vals = np.array(list(log_weights.values()))
        log_vals -= log_vals.max()
        weights = np.exp(log_vals)
        weights /= weights.sum() + 1e-12

        # Shannon entropy of posterior
        nonzero = weights[weights > 1e-10]
        if len(nonzero) == 0:
            return 0.0
        return float(-np.sum(nonzero * np.log(nonzero)))
