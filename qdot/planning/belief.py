"""
qdot/planning/belief.py
=======================
POMDP Belief Updater — particle filter over charge states (n1, n2).

Phase 2 implements the BeliefState stub declared in qdot/core/state.py.
This module adds the particle filter engine and CIM observation model that
the stub was designed to receive.

Key design decisions (from Phase 1 handoff):
  - Line scans (1D) → direct belief update via CIM model (NOT through InspectionAgent)
  - 2D patches → already classified by InspectionAgent; use Classification result to boost
  - physics_override == True → inflate uncertainty (reduce effective measurement weight)
  - DisorderLearner (Phase 3) will call inject_disorder() — stub preserved here

Blueprint reference: §5.1 (POMDP Executive Agent), Fig. 5 (Belief Update Cycle)
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from typing import Optional, Dict, Tuple

# Phase 0 types — never redefine these
from qdot.core.state import BeliefState
from qdot.core.types import ChargeLabel, Classification, Measurement, MeasurementModality

# Phase 0 simulator — reuse physics, don't reimplement it
from qdot.simulator.cim import ConstantInteractionDevice


# ---------------------------------------------------------------------------
# Particle filter internal state (separate from BeliefState dict)
# ---------------------------------------------------------------------------

class _ParticleSet:
    """
    Internal particle representation for the filter.

    BeliefState.charge_probs is the *public* interface (dict of probabilities).
    This class is the *private* computation engine that produces those probs.

    Particles represent hypotheses about (n1, n2) electron occupation.
    """

    def __init__(self, n_particles: int = 1000, n_max: int = 4):
        self.n_max = n_max
        # Sample particles uniformly from all (n1, n2) combinations
        states = [(n1, n2) for n1 in range(n_max + 1) for n2 in range(n_max + 1)]
        n_states = len(states)
        indices = np.random.choice(n_states, size=n_particles, replace=True)
        self.particles = np.array(states, dtype=np.int32)[indices]  # (N, 2)
        self.log_weights = np.full(n_particles, -np.log(n_particles))  # uniform

    @property
    def n_particles(self) -> int:
        return len(self.particles)

    @property
    def weights(self) -> np.ndarray:
        """Normalised weights, derived from log_weights."""
        lw = self.log_weights - logsumexp(self.log_weights)
        return np.exp(lw)

    def effective_sample_size(self) -> float:
        w = self.weights
        return 1.0 / float(np.sum(w ** 2))

    def resample_if_needed(self, threshold_fraction: float = 0.5) -> None:
        """Systematic resampling when ESS drops below threshold."""
        if self.effective_sample_size() < self.n_particles * threshold_fraction:
            self._systematic_resample()

    def _systematic_resample(self) -> None:
        n = self.n_particles
        w = self.weights
        cumsum = np.cumsum(w)
        positions = (np.arange(n) + np.random.uniform(0, 1)) / n
        new_particles = []
        for pos in positions:
            idx = int(np.searchsorted(cumsum, pos))
            new_particles.append(self.particles[min(idx, n - 1)])
        self.particles = np.array(new_particles, dtype=np.int32)
        self.log_weights = np.full(n, -np.log(n))

    def to_charge_probs(self) -> Dict[tuple, float]:
        """Aggregate particle weights into charge_probs dict (for BeliefState)."""
        w = self.weights
        probs: Dict[tuple, float] = {}
        for i, (n1, n2) in enumerate(self.particles):
            key = (int(n1), int(n2))
            probs[key] = probs.get(key, 0.0) + float(w[i])
        return probs


# ---------------------------------------------------------------------------
# CIM Observation Model (wraps ConstantInteractionDevice)
# ---------------------------------------------------------------------------

class CIMObservationModel:
    """
    Wraps ConstantInteractionDevice to compute P(measurement | n1, n2, voltages).

    Uses the existing physics engine from qdot/simulator/cim.py directly.
    Does NOT re-implement the physics — that would create a divergence risk.

    Assumes Gaussian observation noise around CIM-predicted conductance.
    """

    def __init__(
        self,
        device_params: Optional[Dict[str, float]] = None,
        noise_std: float = 0.05,
        seed: Optional[int] = None,
    ):
        """
        Args:
            device_params: CIM parameters (E_c1, E_c2, t_c, T, lever_arm, noise_level).
                           If None, uses ConstantInteractionDevice defaults.
            noise_std: Observation noise std dev for likelihood computation.
            seed: Random seed for the CIM device.
        """
        params = device_params or {}
        # Fallback values must match ConstantInteractionDevice defaults so the
        # observation model and the simulator agree when device_params is None.
        # Degeneracy check: -E_c/lever_arm must fall in [-1, 1] V.
        # With E_c≈0.5 and lever_arm=1.0, crossings are at ≈-0.5 V (in range).
        self.device = ConstantInteractionDevice(
            E_c1=params.get("E_c1", 0.50),
            E_c2=params.get("E_c2", 0.55),
            t_c=params.get("t_c", 0.05),
            T=params.get("T", 0.015),
            lever_arm=params.get("lever_arm", 1.0),
            noise_level=0.0,   # Noise is handled separately in likelihood
            seed=seed,
        )
        self.noise_std = noise_std

    def predicted_conductance_2d(
        self,
        n1: int,
        n2: int,
        v1_range: Tuple[float, float],
        v2_range: Tuple[float, float],
        resolution: int,
    ) -> np.ndarray:
        """
        Predict 2D conductance patch for charge state (n1, n2).

        Note: temporarily sets the device disorder to favour this state.
        This is a simplified version — Phase 3 disorder learning will refine it.
        
        PERFORMANCE NOTE: This is the critical bottleneck (called once per particle per measurement).
        The nested loop calling device.current() is slow. Future optimization: vectorize the CIM
        device to accept 2D voltage arrays and compute all points at once (10-50× speedup).
        """
        v1_vals = np.linspace(v1_range[0], v1_range[1], resolution)
        v2_vals = np.linspace(v2_range[0], v2_range[1], resolution)
        patch = np.zeros((resolution, resolution), dtype=np.float32)

        # BOTTLENECK: Nested loop with Python function calls
        # TODO: Replace with vectorized device.current_2d(v1_grid, v2_grid) method
        #
        # FIX: Use current_for_state() not current() so predictions differ per particle.
        # current() ignores (n1, n2) entirely — all particles would get identical
        # likelihoods and the filter would never update. current_for_state() conditions
        # on (n1, n2) being the ground state via a Boltzmann weight, making states
        # distinguishable by the regions of gate space where they are stable.
        for i, v2 in enumerate(v2_vals):
            for j, v1 in enumerate(v1_vals):
                patch[i, j] = self.device.current_for_state(v1, v2, n1, n2)

        return patch

    def log_likelihood_2d(
        self,
        observed: np.ndarray,
        n1: int,
        n2: int,
        v1_range: Tuple[float, float],
        v2_range: Tuple[float, float],
    ) -> float:
        """
        log P(observed_patch | n1, n2, voltages, CIM).

        Gaussian noise model: log L = -0.5 * sum((obs - pred)² / σ²)
        Normalised per pixel for numerical stability across resolutions.
        """
        resolution = observed.shape[0]
        predicted = self.predicted_conductance_2d(n1, n2, v1_range, v2_range, resolution)
        residuals = (observed - predicted) / (self.noise_std + 1e-8)
        return float(-0.5 * np.mean(residuals ** 2))

    def predicted_conductance_1d(
        self,
        n1: int,
        n2: int,
        axis: str,
        start: float,
        stop: float,
        steps: int,
        fixed: float,
    ) -> np.ndarray:
        """Predict 1D line scan conductance for charge state (n1, n2)."""
        voltages = np.linspace(start, stop, steps)
        trace = np.zeros(steps, dtype=np.float32)
        for i, v in enumerate(voltages):
            if axis == "vg1":
                # FIX: current_for_state() not current() — see predicted_conductance_2d note
                trace[i] = self.device.current_for_state(v, fixed, n1, n2)
            else:
                trace[i] = self.device.current_for_state(fixed, v, n1, n2)
        return trace

    def log_likelihood_1d(
        self,
        observed: np.ndarray,
        n1: int,
        n2: int,
        axis: str,
        start: float,
        stop: float,
        fixed: float,
    ) -> float:
        """log P(observed_trace | n1, n2, voltages, CIM)."""
        predicted = self.predicted_conductance_1d(n1, n2, axis, start, stop, len(observed), fixed)
        residuals = (observed - predicted) / (self.noise_std + 1e-8)
        return float(-0.5 * np.mean(residuals ** 2))


# ---------------------------------------------------------------------------
# Belief Updater — the main Phase 2 engine
# ---------------------------------------------------------------------------

class BeliefUpdater:
    """
    Bayesian belief updater using a particle filter.

    Implements the update cycle from Blueprint Fig. 5:
        belief_new = update(belief_old, measurement, classification)

    Works directly with BeliefState from qdot.core.state (the Phase 0 stub).
    After each update, writes back to belief.charge_probs so the rest of
    the system sees a consistent probability distribution.

    One BeliefUpdater instance persists for the full agent run.
    Carries internal particle state between updates.
    """

    def __init__(
        self,
        belief: BeliefState,
        n_particles: int = 1000,  # Original value - validate reductions with ablations
        n_max: int = 4,
        obs_model: Optional[CIMObservationModel] = None,
    ):
        """
        Args:
            belief: The BeliefState from ExperimentState (Phase 0 stub).
            n_particles: Number of filter particles. Trade-off: 500 is faster, 1000 is more accurate.
                        Run ablations to determine impact on benchmark metrics before reducing.
            n_max: Maximum electrons per dot to consider.
            obs_model: CIM observation model (created from belief.device_params if None).
        """
        self.belief = belief
        self.n_max = n_max

        # Initialise observation model from belief's device_params
        if obs_model is not None:
            self.obs_model = obs_model
        else:
            self.obs_model = CIMObservationModel(
                device_params=belief.device_params
            )

        # Initialise particle set
        self._particles = _ParticleSet(n_particles=n_particles, n_max=n_max)

        # Sync initial belief
        self._sync_belief()

    # ------------------------------------------------------------------
    # Public update methods
    # ------------------------------------------------------------------

    def update_from_2d(
        self,
        measurement: Measurement,
        classification: Optional[Classification] = None,
    ) -> None:
        """
        Bayesian update from a 2D stability diagram measurement.

        Args:
            measurement: 2D Measurement from DeviceAdapter (normalised to [0,1]).
            classification: Optional InspectionAgent output.
                If classification.physics_override == True, weight is halved
                (blueprint §5.1: physics override increases uncertainty).
        """
        if not measurement.is_2d:
            raise ValueError("update_from_2d requires a 2D measurement. Use update_from_1d for line scans.")

        observed = np.asarray(measurement.array, dtype=np.float32)
        v1_range = measurement.v1_range or (-1.0, 1.0)
        v2_range = measurement.v2_range or (-1.0, 1.0)

        # Measurement weight factor (reduced if physics override)
        weight_factor = 1.0
        if classification is not None and classification.physics_override:
            weight_factor = 0.5   # inflate uncertainty per blueprint §5.1

        # Update particle log-weights with CIM log-likelihood
        for i, (n1, n2) in enumerate(self._particles.particles):
            ll = self.obs_model.log_likelihood_2d(
                observed, int(n1), int(n2), v1_range, v2_range
            )
            self._particles.log_weights[i] += weight_factor * ll

        # Optional: boost particles matching CNN label
        if classification is not None and classification.label is not None:
            self._apply_classification_boost(classification)

        # Normalise
        self._normalise_log_weights()

        # Resample if needed
        self._particles.resample_if_needed()

        # Write back to BeliefState
        self._sync_belief()

    def update_from_1d(self, measurement: Measurement) -> None:
        """
        Bayesian update from a 1D line scan measurement.

        Line scans bypass InspectionAgent by design (blueprint §5.1).
        No classification object is passed here.

        Args:
            measurement: LINE_SCAN Measurement from DeviceAdapter.
        """
        if measurement.modality != MeasurementModality.LINE_SCAN:
            raise ValueError("update_from_1d requires a LINE_SCAN measurement.")

        observed = np.asarray(measurement.array, dtype=np.float32)
        axis = measurement.axis or "vg1"

        # Reconstruct scan parameters from Measurement metadata
        meta = measurement.meta or {}
        start = meta.get("start", -1.0)
        stop = meta.get("stop", 1.0)
        fixed = meta.get("fixed", 0.0)

        # Update particle log-weights
        for i, (n1, n2) in enumerate(self._particles.particles):
            ll = self.obs_model.log_likelihood_1d(
                observed, int(n1), int(n2), axis, start, stop, fixed
            )
            self._particles.log_weights[i] += ll

        # Normalise
        self._normalise_log_weights()

        # Resample if needed
        self._particles.resample_if_needed()

        # Write back
        self._sync_belief()

    def uncertainty_map(
        self,
        v1_range: Tuple[float, float],
        v2_range: Tuple[float, float],
        resolution: int = 32,
    ) -> np.ndarray:
        """
        Compute spatial uncertainty map over voltage space.

        For each point, computes weighted std dev of predicted conductance
        across all particle hypotheses. High variance = high uncertainty.

        Writes result to belief.uncertainty_map.

        Returns:
            Uncertainty map array of shape (resolution, resolution).
        """
        v1_vals = np.linspace(v1_range[0], v1_range[1], resolution)
        v2_vals = np.linspace(v2_range[0], v2_range[1], resolution)
        w = self._particles.weights
        umap = np.zeros((resolution, resolution), dtype=np.float32)

        for i, v2 in enumerate(v2_vals):
            for j, v1 in enumerate(v1_vals):
                predictions = []
                for n1, n2 in self._particles.particles:
                    # FIX: current_for_state() not current() — predictions must differ
                    # across particles for the variance to be non-zero. current() gives
                    # the same value for all (n1, n2), collapsing variance to 0 everywhere.
                    G = self.obs_model.device.current_for_state(v1, v2, int(n1), int(n2))
                    predictions.append(G)
                predictions = np.array(predictions)
                mean_G = float(np.sum(w * predictions))
                var_G = float(np.sum(w * (predictions - mean_G) ** 2))
                umap[i, j] = np.sqrt(max(0.0, var_G))

        self.belief.uncertainty_map = umap
        return umap

    def inject_disorder(self, disorder_posterior: Dict) -> None:
        """
        Phase 3 hook: update the CIM observation model with device disorder.

        Called by DisorderLearner when real device OOD score is elevated.
        No-op in Phase 2 (base CIM model used throughout).
        """
        self.obs_model.device.inject_disorder(disorder_posterior)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_classification_boost(self, classification: Classification) -> None:
        """
        Boost log-weights of particles consistent with CNN label.

        Only applied if not physics_override (already accounted for by weight_factor).
        Boost magnitude scales with classification confidence.
        """
        if classification.physics_override:
            return  # Already penalised above; don't double-count

        label = classification.label
        confidence = classification.confidence

        # Map ChargeLabel to expected charge state ranges
        # DOUBLE_DOT → (1,1) most likely; SINGLE_DOT → n1 or n2 = 0
        for i, (n1, n2) in enumerate(self._particles.particles):
            boost = 0.0
            if label == ChargeLabel.DOUBLE_DOT and n1 >= 1 and n2 >= 1:
                boost = confidence * 1.5
            elif label == ChargeLabel.SINGLE_DOT and (n1 == 0 or n2 == 0):
                boost = confidence * 1.0
            elif label == ChargeLabel.MISC and n1 + n2 == 0:
                boost = confidence * 0.5
            self._particles.log_weights[i] += boost

    def _normalise_log_weights(self) -> None:
        """Subtract log-sum-exp for numerical stability."""
        lse = logsumexp(self._particles.log_weights)
        self._particles.log_weights -= lse

    def _sync_belief(self) -> None:
        """Write current particle distribution to belief.charge_probs."""
        self.belief.charge_probs = self._particles.to_charge_probs()
