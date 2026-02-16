"""
qdot/perception/classifier.py
==============================
TinyCNN — 3-class primary classifier for 2D stability diagrams.
EnsembleCNN — 5-model ensemble with max-disagreement uncertainty.

Key change from hackathon:
    The CNN is now the PRIMARY CLASSIFIER, not an embedding extractor.
    It has a 3-class softmax head and is trained end-to-end on CIM data.
    Physics features (FFT, diagonal) are a *validator* layer, not the
    primary signal. See blueprint §7.1 for why this matters.

Architecture (TinyCNN):
    Input  → (1, 64, 64)   — log-preprocessed normalised conductance
    Conv1  → (16, 32, 32)  — 3×3, stride 2, BN, ReLU
    Conv2  → (32, 16, 16)  — 3×3, stride 2, BN, ReLU
    Conv3  → (64, 8, 8)    — 3×3, stride 2, BN, ReLU
    Conv4  → (64, 4, 4)    — 3×3, stride 2, BN, ReLU
    GAP    → (64,)          — global average pooling
    FC     → (32,)          — linear + ReLU   ← OOD detector attaches here
    Head   → (3,)           — linear (logits)

Ensemble:
    5 independent TinyCNN instances trained from different random seeds.
    uncertainty = max( max_j(p_j^(i)) - max_j(p_j^(k)) ) over all (i,k) pairs
    where p^(i) is the softmax output of model i.
    This is the max pairwise L∞ disagreement between models, feeds directly
    into the Risk Score formula (§4.1): disagreement > 0.3 → r += 0.35.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# TinyCNN architecture
# ---------------------------------------------------------------------------

class TinyCNN(nn.Module):
    """
    Compact CNN for 3-class stability diagram classification.

    Designed for fast inference on CPU during real-device experiments
    (target: < 5 ms per 64×64 patch on a modern laptop CPU).
    """

    N_CLASSES = 3

    def __init__(self, dropout_p: float = 0.2) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            # Block 1: 1×64×64 → 16×32×32
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # Block 2: 16×32×32 → 32×16×16
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Block 3: 32×16×16 → 64×8×8
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 4: 64×8×8 → 64×4×4
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Global average pooling: 64×4×4 → 64
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Penultimate layer — OOD detector extracts features here
        self.penultimate = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
        )

        # Classification head
        self.head = nn.Linear(32, self.N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits."""
        x = self.encoder(x)
        x = self.gap(x).squeeze(-1).squeeze(-1)   # (B, 64)
        x = self.penultimate(x)                    # (B, 32)
        return self.head(x)                        # (B, 3)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract penultimate layer features for OOD detection.
        Returns tensor of shape (B, 32).
        """
        with torch.no_grad():
            x = self.encoder(x)
            x = self.gap(x).squeeze(-1).squeeze(-1)
            x = self.penultimate(x)
        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Softmax probabilities. Shape: (B, 3)."""
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=-1)


# ---------------------------------------------------------------------------
# Ensemble wrapper
# ---------------------------------------------------------------------------

class EnsembleCNN:
    """
    5-model ensemble of TinyCNN instances.

    Provides:
        predict(x)          — majority-vote label + mean confidence
        predict_proba(x)    — mean softmax probabilities across ensemble
        uncertainty(x)      — max-disagreement metric ∈ [0, 1]

    The uncertainty metric feeds directly into the Risk Score:
        state.ensemble_disagreement > 0.30 → r += 0.35

    Usage:
        ensemble = EnsembleCNN.from_trained(model_dir)
        label, confidence, disagreement = ensemble.classify(array)
    """

    N_MODELS = 5

    def __init__(
        self,
        models: Optional[List[TinyCNN]] = None,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.models: List[TinyCNN] = models or [TinyCNN() for _ in range(self.N_MODELS)]
        for m in self.models:
            m.to(self.device)
            m.eval()

    # -----------------------------------------------------------------------
    # Primary interface
    # -----------------------------------------------------------------------

    def classify(
        self, array: np.ndarray
    ) -> Tuple[int, float, float]:
        """
        Classify a single 2D stability diagram.

        Args:
            array: float32 array of shape (H, W) or (1, H, W) or (1, 1, H, W).
                   Will be preprocessed automatically.

        Returns:
            (label_idx, confidence, disagreement)
            label_idx:    int ∈ {0, 1, 2}  (DOUBLE_DOT, SINGLE_DOT, MISC)
            confidence:   float ∈ [0, 1]    (mean max-prob across ensemble)
            disagreement: float ∈ [0, 1]    (max-disagreement metric)
        """
        x = self._prepare(array)
        all_probs = self._all_probabilities(x)   # (N_MODELS, 3)

        mean_probs = all_probs.mean(axis=0)       # (3,)
        label_idx = int(np.argmax(mean_probs))
        confidence = float(mean_probs[label_idx])
        disagreement = self._disagreement(all_probs)

        return label_idx, confidence, disagreement

    def predict_proba(self, array: np.ndarray) -> np.ndarray:
        """Mean softmax probabilities across ensemble. Shape: (3,)."""
        x = self._prepare(array)
        return self._all_probabilities(x).mean(axis=0)

    def uncertainty(self, array: np.ndarray) -> float:
        """Max-disagreement metric ∈ [0, 1]."""
        x = self._prepare(array)
        return self._disagreement(self._all_probabilities(x))

    def extract_features(self, array: np.ndarray) -> np.ndarray:
        """
        Extract penultimate-layer features from model 0.
        Used by MahalanobisOOD — we use a single reference model for OOD
        to keep the feature space stable, then the ensemble provides UQ.

        Returns: float32 array of shape (32,)
        """
        x = self._prepare(array)
        return self.models[0].extract_features(x).cpu().numpy().squeeze()

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    @classmethod
    def train_from_data(
        cls,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_epochs: int = 30,
        batch_size: int = 128,
        lr: float = 3e-4,
        device: str = "cpu",
        model_dir: Optional[str] = None,
        verbose: bool = True,
    ) -> "EnsembleCNN":
        """
        Train all 5 ensemble members from scratch.

        Args:
            X_train: float32 (N, 1, 64, 64)
            y_train: int64   (N,)
            X_val:   float32 (M, 1, 64, 64)
            y_val:   int64   (M,)
            model_dir: if provided, saves each model checkpoint here.

        Returns:
            Trained EnsembleCNN ready for inference.
        """
        ensemble = cls(device=device)

        for i, model in enumerate(ensemble.models):
            if verbose:
                print(f"\n=== Training model {i+1}/{cls.N_MODELS} ===")
            _train_single(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                n_epochs=n_epochs,
                batch_size=batch_size,
                lr=lr,
                device=torch.device(device),
                seed=i * 100 + 42,
                verbose=verbose,
            )
            if model_dir:
                Path(model_dir).mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), Path(model_dir) / f"model_{i}.pt")

        return ensemble

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save(self, model_dir: str) -> None:
        """Save all model weights."""
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), Path(model_dir) / f"model_{i}.pt")

    @classmethod
    def load(cls, model_dir: str, device: str = "cpu") -> "EnsembleCNN":
        """Load all model weights from a directory."""
        models = []
        for i in range(cls.N_MODELS):
            path = Path(model_dir) / f"model_{i}.pt"
            model = TinyCNN()
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            models.append(model)
        return cls(models=models, device=device)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _prepare(self, array: np.ndarray) -> torch.Tensor:
        """
        Prepare input array for inference.
        Handles arbitrary input shape → (1, 1, 64, 64) tensor.
        """
        from qdot.perception.features import log_preprocess
        from scipy.ndimage import zoom

        arr = np.asarray(array, dtype=np.float32)

        # Strip batch/channel dims if present
        while arr.ndim > 2 and arr.shape[0] == 1:
            arr = arr.squeeze(0)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array after squeezing, got shape {arr.shape}")

        # Log-preprocess
        arr = log_preprocess(arr)

        # Resize to 64×64 if needed
        if arr.shape != (64, 64):
            scale = 64.0 / arr.shape[0]
            arr = zoom(arr.astype(np.float64), scale, order=1).astype(np.float32)
            arr = np.clip(arr, 0.0, 1.0)

        # (1, 1, 64, 64) tensor
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)

    def _all_probabilities(self, x: torch.Tensor) -> np.ndarray:
        """
        Returns softmax probabilities from all models.
        Shape: (N_MODELS, N_CLASSES).
        """
        results = []
        for model in self.models:
            with torch.no_grad():
                probs = model.predict_proba(x).cpu().numpy()
            results.append(probs.squeeze())
        return np.stack(results, axis=0)

    @staticmethod
    def _disagreement(all_probs: np.ndarray) -> float:
        """
        Max-disagreement metric across ensemble.

        For each pair of models (i, j), compute the L∞ distance between
        their softmax vectors. Return the maximum over all pairs.

        This is more interpretable than entropy because it directly
        measures the worst-case disagreement between any two classifiers.
        """
        n = all_probs.shape[0]
        max_d = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.abs(all_probs[i] - all_probs[j]).max())
                if d > max_d:
                    max_d = d
        return max_d


# ---------------------------------------------------------------------------
# Single model training loop
# ---------------------------------------------------------------------------

def _train_single(
    model: TinyCNN,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int,
    verbose: bool,
) -> None:
    """Train a single TinyCNN with cosine-annealing LR and class-balanced sampling."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model.to(device).train()

    # Class-balanced sampler
    counts = np.bincount(y_train)
    class_weights = 1.0 / (counts + 1e-8)
    sample_weights = class_weights[y_train]
    sample_weights = sample_weights / sample_weights.sum()
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights.astype(np.float64)),
        num_samples=len(y_train),
        replacement=True,
    )

    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).long(),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=n_epochs, eta_min=lr * 0.01
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimiser.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimiser.step()

            train_loss += loss.item() * len(y_batch)
            train_correct += (logits.argmax(dim=1) == y_batch).sum().item()
            train_total += len(y_batch)

        scheduler.step()

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                val_correct += (logits.argmax(dim=1) == y_batch).sum().item()
                val_total += len(y_batch)

        val_acc = val_correct / val_total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose and (epoch + 1) % 5 == 0:
            train_acc = train_correct / train_total
            print(
                f"  Epoch {epoch+1:3d}/{n_epochs} | "
                f"train_loss={train_loss/train_total:.4f} | "
                f"train_acc={train_acc:.3f} | "
                f"val_acc={val_acc:.3f}"
            )

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    if verbose:
        print(f"  Best val_acc: {best_val_acc:.4f}")
    model.eval()
