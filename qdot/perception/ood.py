"""
qdot/perception/ood.py
======================
MahalanobisOOD — out-of-distribution detector for the Inspection Agent.

Uses Mahalanobis distance computed on PCA-projected penultimate-layer
features of TinyCNN model 0 (the reference model in the ensemble).

When a real device measurement is flagged as OOD, it means the scan
topology is genuinely outside the training distribution — not a quality
issue (that's DQC's job), but a device-specific signature the model
hasn't seen before. This triggers the DisorderLearner in Phase 3.

Important asymmetry in Phase 1:
    Training distribution = CIM-generated data
    OOD population        = QFlow real experimental data

This means QFlow scans *will* produce elevated OOD scores at test time
because real devices have charge disorder the CIM wasn't trained with.
That's the correct behaviour — the Phase 3 DisorderLearner is designed
to resolve exactly this gap. The OOD flag is not a failure; it's a
diagnostic that triggers the right module.

Calibration:
    Threshold is set at the 95th percentile of Mahalanobis distances on
    a held-out validation set from the CIM training distribution.
    This means ~5% false-positive rate on in-distribution data, which
    keeps the DisorderLearner from firing on normal CIM variation.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from qdot.core.types import OODResult
from uuid import UUID


class MahalanobisOOD:
    """
    OOD detector based on Mahalanobis distance in PCA feature space.

    The feature space is the 32-dimensional penultimate layer of TinyCNN.
    PCA reduces this to `n_components` dimensions where the covariance
    structure is better conditioned.

    Mahalanobis distance:
        d = sqrt( (x - μ)ᵀ Σ⁻¹ (x - μ) )
    where μ, Σ are the mean and covariance of the training distribution
    in PCA space.

    Usage:
        ood = MahalanobisOOD(n_components=16)

        # Fit on training features
        features = ensemble.extract_features_batch(X_train)   # (N, 32)
        ood.fit(features)

        # At test time
        feat = ensemble.extract_features(array)               # (32,)
        result = ood.score(measurement_id, feat)
        if result.flag:
            # trigger DisorderLearner
            ...
    """

    def __init__(
        self,
        n_components: int = 16,
        calibration_percentile: float = 95.0,
    ) -> None:
        """
        Args:
            n_components: PCA dimensionality. 16 retains > 90% variance
                          for typical TinyCNN penultimate features.
            calibration_percentile: Threshold = this percentile of
                          training distances. Default 95 → ~5% FPR.
        """
        self.n_components = n_components
        self.calibration_percentile = calibration_percentile

        # Fitted parameters (None until fit() is called)
        self._scaler: Optional[StandardScaler] = None
        self._pca: Optional[PCA] = None
        self._mu: Optional[np.ndarray] = None          # (n_components,)
        self._precision: Optional[np.ndarray] = None   # (n_components, n_components)
        self._threshold: Optional[float] = None
        self._fitted = False

    # -----------------------------------------------------------------------
    # Fitting
    # -----------------------------------------------------------------------

    def fit(self, features: np.ndarray) -> None:
        """
        Fit the OOD detector on penultimate-layer features from the
        training distribution (CIM-generated data).

        Args:
            features: float32/float64 array of shape (N, 32).
                      Extract with ensemble.extract_features_batch(X_train).
        """
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2:
            raise ValueError(f"Expected 2D feature array, got shape {features.shape}")

        n_samples, n_feat = features.shape
        n_comp = min(self.n_components, n_feat, n_samples - 1)

        # Step 1: Standardise features (zero mean, unit variance per dim)
        self._scaler = StandardScaler()
        scaled = self._scaler.fit_transform(features)

        # Step 2: PCA projection
        self._pca = PCA(n_components=n_comp, random_state=42)
        projected = self._pca.fit_transform(scaled)   # (N, n_comp)

        # Step 3: Compute mean + precision matrix of projected features
        self._mu = projected.mean(axis=0)
        cov = np.cov(projected.T) + np.eye(n_comp) * 1e-6   # regularise
        self._precision = np.linalg.inv(cov)

        # Step 4: Calibrate threshold on training distances
        train_distances = self._compute_distances(projected)
        self._threshold = float(
            np.percentile(train_distances, self.calibration_percentile)
        )

        explained = self._pca.explained_variance_ratio_.sum() if n_comp > 1 else 1.0
        self._fitted = True

        print(
            f"OOD detector fitted: n={n_samples}, "
            f"n_components={n_comp} ({explained:.1%} variance), "
            f"threshold={self._threshold:.3f} "
            f"({self.calibration_percentile:.0f}th percentile)"
        )

    # -----------------------------------------------------------------------
    # Scoring
    # -----------------------------------------------------------------------

    def score(self, measurement_id: UUID, features: np.ndarray) -> OODResult:
        """
        Compute OOD score for a single sample.

        Args:
            measurement_id: UUID to attach to the OODResult.
            features: float32/float64 array of shape (32,) or (1, 32).

        Returns:
            OODResult with score, threshold, and flag.
        """
        if not self._fitted:
            raise RuntimeError(
                "OOD detector has not been fitted. Call fit() first."
            )

        feat = np.asarray(features, dtype=np.float64).flatten()
        if feat.ndim == 0 or feat.shape[0] == 0:
            raise ValueError("Empty feature vector.")

        # Project to PCA space
        projected = self._project(feat.reshape(1, -1))   # (1, n_comp)

        # Mahalanobis distance
        diff = projected[0] - self._mu
        dist = float(np.sqrt(diff @ self._precision @ diff))

        flag = dist > self._threshold

        return OODResult(
            measurement_id=measurement_id,
            score=dist,
            threshold=self._threshold,
            flag=flag,
        )

    def score_batch(
        self, features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute OOD scores for a batch of samples.

        Args:
            features: (N, 32) feature matrix

        Returns:
            (scores, flags) — float64 (N,) and bool (N,)
        """
        if not self._fitted:
            raise RuntimeError("OOD detector not fitted.")

        features = np.asarray(features, dtype=np.float64)
        projected = self._project(features)   # (N, n_comp)
        distances = self._compute_distances(projected)
        flags = distances > self._threshold
        return distances, flags

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save fitted OOD detector to disk."""
        state = {
            "n_components": self.n_components,
            "calibration_percentile": self.calibration_percentile,
            "scaler": self._scaler,
            "pca": self._pca,
            "mu": self._mu,
            "precision": self._precision,
            "threshold": self._threshold,
            "fitted": self._fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> "MahalanobisOOD":
        """Load a previously fitted OOD detector."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(
            n_components=state["n_components"],
            calibration_percentile=state["calibration_percentile"],
        )
        obj._scaler = state["scaler"]
        obj._pca = state["pca"]
        obj._mu = state["mu"]
        obj._precision = state["precision"]
        obj._threshold = state["threshold"]
        obj._fitted = state["fitted"]
        return obj

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _project(self, features: np.ndarray) -> np.ndarray:
        """Scale then PCA-project features."""
        scaled = self._scaler.transform(features)
        return self._pca.transform(scaled)

    def _compute_distances(self, projected: np.ndarray) -> np.ndarray:
        """
        Vectorised Mahalanobis distance computation.
        projected: (N, n_components)
        Returns: (N,) distances
        """
        diff = projected - self._mu   # (N, n_comp)
        # d_i = sqrt( diff_i @ precision @ diff_i )
        # vectorised: right_term = diff @ precision, then sum
        right = diff @ self._precision   # (N, n_comp)
        distances_sq = (right * diff).sum(axis=1)   # (N,)
        distances_sq = np.maximum(distances_sq, 0.0)   # numerical safety
        return np.sqrt(distances_sq)


# ---------------------------------------------------------------------------
# Batch feature extraction helper (used during fitting)
# ---------------------------------------------------------------------------

def extract_features_batch(
    ensemble_or_model,
    X: np.ndarray,
    batch_size: int = 256,
    device: str = "cpu",
) -> np.ndarray:
    """
    Extract penultimate-layer features for a batch of preprocessed arrays.

    Args:
        ensemble_or_model: EnsembleCNN or TinyCNN with .extract_features()
        X: float32 array of shape (N, 1, 64, 64) — already preprocessed
        batch_size: batch size for inference

    Returns:
        float32 array of shape (N, 32)
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    dev = torch.device(device)

    # If EnsembleCNN, use model 0 as reference
    model = ensemble_or_model
    if hasattr(model, "models"):
        model = model.models[0]
    model.to(dev).eval()

    dataset = TensorDataset(torch.from_numpy(X).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_features = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(dev)
            feat = model.extract_features(batch)   # (B, 32)
            all_features.append(feat.cpu().numpy())

    return np.concatenate(all_features, axis=0).astype(np.float32)
