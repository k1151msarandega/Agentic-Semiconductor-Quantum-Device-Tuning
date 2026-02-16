"""
qdot/perception/inspector.py
============================
InspectionAgent — Layer 3 perception pipeline orchestrator.

Takes a quality-gated Measurement and produces a typed Classification
and OODResult for the Executive Agent. This is the single entry point
for all perception work in Phase 1.

Pipeline (blueprint §5.3):
    1. DQC check       — already done by Gatekeeper; verdict passed in
    2. Log-preprocess  — always first transformation
    3. CNN classify    — EnsembleCNN, 3-class softmax
    4. Physics validate— FFT + diagonal features; can set override_flag
    5. OOD detect      — Mahalanobis distance on penultimate features
    6. NL report       — structured JSON summary for Executive Agent

What this module does NOT do:
    - Process line scans (those go directly to Executive Agent)
    - Make voltage decisions (that is the POMDP planner's job)
    - Trigger HITL (the Executive Agent does this based on the report)

The NL report is a structured JSON dict in Phase 1 (no LLM call).
LLM calls are reserved for stage transitions and HITL triggers (§5.1).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from typing import Optional, Tuple
from uuid import UUID

import numpy as np

from qdot.core.types import (
    ChargeLabel,
    Classification,
    DQCQuality,
    DQCResult,
    Measurement,
    OODResult,
)
from qdot.perception.dqc import DQCGatekeeper
from qdot.perception.features import physics_features, physics_override_label
from qdot.perception.classifier import EnsembleCNN
from qdot.perception.ood import MahalanobisOOD


# Integer-to-label mapping (must match CIMDataset.LABEL_MAP)
INT_TO_LABEL = {
    0: ChargeLabel.DOUBLE_DOT,
    1: ChargeLabel.SINGLE_DOT,
    2: ChargeLabel.MISC,
}
LABEL_TO_INT = {v: k for k, v in INT_TO_LABEL.items()}


class InspectionAgent:
    """
    Perception pipeline for 2D charge stability diagrams.

    Accepts a raw Measurement (normalised conductance array), runs the
    full classification pipeline, and returns typed result objects that
    the Executive Agent consumes.

    Important: Only accepts 2D measurements. Line scans are excluded by
    design (blueprint §5.3): "The Inspection Agent classifies 2D stability
    diagrams only. Line scans are used for navigation by the Executive
    Agent directly."

    Usage:
        agent = InspectionAgent(ensemble=ensemble, ood_detector=ood)
        classification, ood_result = agent.inspect(measurement, dqc_result)

        # Access structured NL report
        print(classification.nl_summary)   # JSON string
    """

    # Minimum DQC quality to run the full pipeline.
    # LOW quality measurements must never reach the CNN.
    MIN_DQC_FOR_CNN = DQCQuality.MODERATE

    def __init__(
        self,
        ensemble: Optional[EnsembleCNN] = None,
        ood_detector: Optional[MahalanobisOOD] = None,
        gatekeeper: Optional[DQCGatekeeper] = None,
        # Physics validator thresholds (device-adaptive in Phase 2)
        peak_ratio_threshold: float = 3.5,
        diagonal_strength_min: float = 0.25,
        # OOD threshold override (uses detector's calibrated value if None)
        ood_threshold: Optional[float] = None,
    ) -> None:
        """
        Args:
            ensemble:     Trained EnsembleCNN. If None, uses untrained model
                          (for testing only — predictions will be random).
            ood_detector: Fitted MahalanobisOOD. If None, OOD detection is
                          skipped and flag is always False.
            gatekeeper:   DQCGatekeeper. If None, a default one is created.
            peak_ratio_threshold: FFT peak ratio above which SD signature is flagged.
            diagonal_strength_min: diagonal_strength below which transition lines absent.
            ood_threshold: Override the detector's calibrated threshold.
        """
        self.ensemble = ensemble or EnsembleCNN()
        self.ood_detector = ood_detector
        self.gatekeeper = gatekeeper or DQCGatekeeper()
        self.peak_ratio_threshold = peak_ratio_threshold
        self.diagonal_strength_min = diagonal_strength_min
        self.ood_threshold_override = ood_threshold

    # -----------------------------------------------------------------------
    # Primary interface
    # -----------------------------------------------------------------------

    def inspect(
        self,
        measurement: Measurement,
        dqc_result: Optional[DQCResult] = None,
    ) -> Tuple[Classification, OODResult]:
        """
        Run the full perception pipeline on a 2D measurement.

        Args:
            measurement: A Measurement from the Device Adapter.
                         Must be a 2D modality (COARSE_2D, LOCAL_PATCH, FINE_2D).
            dqc_result:  DQCResult from the Gatekeeper. If None, assessment is run here.
                         A LOW quality result raises RuntimeError to prevent CNN pollution.

        Returns:
            (Classification, OODResult)

        Raises:
            ValueError: if the measurement is not 2D.
            RuntimeError: if DQC quality is LOW (should have been stopped upstream).
        """
        # Validate modality
        if not measurement.is_2d:
            raise ValueError(
                f"InspectionAgent received a non-2D measurement "
                f"(modality={measurement.modality}). "
                "Line scans must be processed directly by the Executive Agent."
            )

        # Run DQC if not provided
        if dqc_result is None:
            dqc_result = self.gatekeeper.assess(measurement)

        # Guard: never pass LOW-quality data to the CNN
        if dqc_result.quality == DQCQuality.LOW:
            raise RuntimeError(
                f"InspectionAgent received LOW-quality data "
                f"(measurement_id={measurement.id}). "
                "This should have been stopped at the DQC Gatekeeper."
            )

        arr = np.asarray(measurement.array, dtype=np.float32)

        # ---- Step 1: CNN classification + ensemble UQ ----
        label_idx, confidence, disagreement = self.ensemble.classify(arr)
        cnn_label = INT_TO_LABEL[label_idx]

        # ---- Step 2: Physics feature extraction ----
        phys_feats = physics_features(arr)

        # ---- Step 3: Physics validator override ----
        override_str, override_reason = physics_override_label(
            cnn_label=cnn_label.value,
            features=phys_feats,
            peak_ratio_threshold=self.peak_ratio_threshold,
            diagonal_min=self.diagonal_strength_min,
        )
        physics_override = override_str is not None
        final_label = (
            ChargeLabel(override_str) if physics_override else cnn_label
        )
        if physics_override:
            # Confidence is penalised when we override (explicit uncertainty)
            confidence = min(confidence, 0.65)

        # ---- Step 4: OOD detection ----
        features_vec = self.ensemble.extract_features(arr)
        ood_result = self._run_ood(measurement.id, features_vec)

        # ---- Step 5: Build Classification ----
        all_features = {**phys_feats, "ensemble_disagreement": disagreement}

        classification = Classification(
            measurement_id=measurement.id,
            label=final_label,
            confidence=confidence,
            ensemble_disagreement=disagreement,
            features=all_features,
            physics_override=physics_override,
            nl_summary=self._generate_nl_report(
                measurement=measurement,
                cnn_label=cnn_label,
                final_label=final_label,
                confidence=confidence,
                disagreement=disagreement,
                ood_result=ood_result,
                dqc_result=dqc_result,
                phys_feats=phys_feats,
                override_reason=override_reason,
            ),
        )

        return classification, ood_result

    # -----------------------------------------------------------------------
    # Convenience: inspect array directly (useful for QFlow evaluation)
    # -----------------------------------------------------------------------

    def inspect_array(self, array: np.ndarray) -> Tuple[ChargeLabel, float, float]:
        """
        Quick classification of a raw array.

        Args:
            array: 2D normalised conductance array (H, W).

        Returns:
            (label, confidence, ensemble_disagreement)

        Does NOT compute OOD or physics override — use inspect() for the
        full pipeline with a Measurement object.
        """
        arr = np.asarray(array, dtype=np.float32)
        label_idx, confidence, disagreement = self.ensemble.classify(arr)
        return INT_TO_LABEL[label_idx], confidence, disagreement

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _run_ood(
        self, measurement_id: UUID, features: np.ndarray
    ) -> OODResult:
        """Run OOD detection if detector is fitted, else return safe default."""
        if self.ood_detector is None or not self.ood_detector._fitted:
            # No detector — return in-distribution result
            threshold = self.ood_threshold_override or 24.0
            return OODResult(
                measurement_id=measurement_id,
                score=0.0,
                threshold=threshold,
                flag=False,
            )

        result = self.ood_detector.score(measurement_id, features)

        # Apply override threshold if set
        if self.ood_threshold_override is not None:
            result = OODResult(
                measurement_id=measurement_id,
                score=result.score,
                threshold=self.ood_threshold_override,
                flag=result.score > self.ood_threshold_override,
            )

        return result

    def _generate_nl_report(
        self,
        measurement: Measurement,
        cnn_label: ChargeLabel,
        final_label: ChargeLabel,
        confidence: float,
        disagreement: float,
        ood_result: OODResult,
        dqc_result: DQCResult,
        phys_feats: dict,
        override_reason: str,
    ) -> str:
        """
        Generate a structured JSON NL report for the Executive Agent.

        This is a templated report in Phase 1 (no LLM call).
        The Executive Agent uses this to update its belief state
        and generate rationale text when triggered.

        Format matches blueprint §5.1 rationale format:
            {intent, observation_summary, physics_reasoning,
             proposed_action, expected_outcome}
        """
        # Confidence tier for human-readable summary
        if confidence >= 0.85:
            conf_desc = "high"
        elif confidence >= 0.60:
            conf_desc = "moderate"
        else:
            conf_desc = "low"

        # Uncertainty tier
        if disagreement > 0.30:
            unc_desc = "HIGH — ensemble disagreement above HITL threshold"
        elif disagreement > 0.15:
            unc_desc = "moderate"
        else:
            unc_desc = "low"

        report = {
            "timestamp": time.time(),
            "measurement_id": str(measurement.id),
            "intent": "classify_stability_diagram",
            "observation_summary": {
                "modality": measurement.modality.value,
                "resolution": measurement.resolution,
                "device_id": measurement.device_id,
                "dqc_quality": dqc_result.quality.value,
                "dqc_snr_db": round(dqc_result.snr_db, 2),
            },
            "classification": {
                "cnn_label": cnn_label.value,
                "final_label": final_label.value,
                "confidence": round(confidence, 4),
                "confidence_tier": conf_desc,
                "physics_override": override_reason if override_reason else None,
            },
            "uncertainty": {
                "ensemble_disagreement": round(disagreement, 4),
                "uncertainty_tier": unc_desc,
                "hitl_warranted": disagreement > 0.30,
            },
            "physics_reasoning": {
                "fft_peak_ratio": round(phys_feats.get("fft_peak_ratio", 0.0), 3),
                "diagonal_strength": round(phys_feats.get("diagonal_strength", 0.0), 3),
                "mean_conductance": round(phys_feats.get("mean_conductance", 0.0), 3),
                "conductance_std": round(phys_feats.get("conductance_std", 0.0), 3),
                "interpretation": self._physics_interpretation(phys_feats, final_label),
            },
            "ood": {
                "score": round(ood_result.score, 4),
                "threshold": round(ood_result.threshold, 4),
                "flag": ood_result.flag,
                "margin": round(ood_result.margin, 4),
                "action": (
                    "trigger_disorder_learner"
                    if ood_result.flag
                    else "continue_normal_loop"
                ),
            },
            "recommended_executive_action": self._recommend_action(
                final_label, confidence, disagreement, ood_result
            ),
        }
        return json.dumps(report, indent=2)

    def _physics_interpretation(
        self, features: dict, label: ChargeLabel
    ) -> str:
        """Human-readable interpretation of physics features."""
        pr = features.get("fft_peak_ratio", 0.0)
        ds = features.get("diagonal_strength", 0.0)

        if label == ChargeLabel.DOUBLE_DOT:
            return (
                f"Diagonal structure present (strength={ds:.2f}), "
                f"no single-dominant periodicity (peak_ratio={pr:.1f}). "
                "Consistent with honeycomb topology."
            )
        elif label == ChargeLabel.SINGLE_DOT:
            return (
                f"Dominant periodicity detected (peak_ratio={pr:.1f}). "
                f"Diagonal structure: {ds:.2f}. "
                "Consistent with Coulomb diamond pattern from single dot."
            )
        else:
            return (
                f"Featureless or ambiguous scan. "
                f"Diagonal strength={ds:.2f}, peak_ratio={pr:.1f}. "
                "Likely SC or pinch-off regime."
            )

    def _recommend_action(
        self,
        label: ChargeLabel,
        confidence: float,
        disagreement: float,
        ood_result: OODResult,
    ) -> str:
        """
        Recommend the next Executive Agent action based on this classification.
        This is advisory only — the POMDP planner makes the actual decision.
        """
        if ood_result.flag:
            return "trigger_disorder_learner"
        if disagreement > 0.30:
            return "request_hitl_classification_ambiguous"
        if label == ChargeLabel.DOUBLE_DOT and confidence >= 0.85:
            return "proceed_to_navigation_stage"
        if label == ChargeLabel.DOUBLE_DOT and confidence >= 0.60:
            return "refine_scan_local_patch"
        if label == ChargeLabel.SINGLE_DOT:
            return "adjust_barrier_voltages_increase_coupling"
        if label == ChargeLabel.MISC and confidence >= 0.70:
            return "backtrack_coarse_survey"
        return "take_coarse_2d_scan_for_more_information"
