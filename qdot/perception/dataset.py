"""
qdot/perception/dataset.py
==========================
CIMDataset — synthetic training data generator for the Inspection Agent.

Design decision (from architecture dialogue):
    Train on CIM-generated data. Benchmark on QFlow.

    QFlow is NEVER in the training loop. It is the held-out transfer test
    that proves the classifier generalises to real experimental data from
    a lab it has never seen. This makes the Phase 1 benchmark a genuine
    sim-to-real transfer claim, not a circular benchmark.

QFlow's role:
    • Label calibration: verify CIM "double-dot" looks like QFlow "double-dot"
    • Transfer benchmark only: 96% accuracy target is measured on QFlow

Dataset structure:
    Three classes (§5.3):
        DOUBLE_DOT — honeycomb pattern, Coulomb anti-crossings visible
        SINGLE_DOT — Coulomb diamonds from one dominant dot
        MISC       — featureless (SC/pinch-off) or ambiguous topology

Parameter sweep ranges (designed to cover the full device-type space):
    E_c    ∈ [1.5, 6.0] meV
    t_c    ∈ [0.02, 1.5] meV
    T      ∈ [0.01, 0.3] meV
    lever  ∈ [0.3, 0.9]
    noise  ∈ [0.005, 0.08]
    res    ∈ {16, 32, 64}  (multi-resolution)
    disorder ∈ [0, 0.5]    (amplitude of additive disorder)
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from qdot.core.types import ChargeLabel, Measurement, MeasurementModality, VoltagePoint


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """
    Controls the scope of the generated training dataset.

    Defaults produce ~50k samples at mixed resolutions. Reduce
    n_per_class for fast iteration during development.
    """
    # Samples per class (total = n_per_class × 3)
    n_per_class: int = 17_000           # → ~51k total

    # Resolution distribution (res: weight)
    resolutions: dict = field(default_factory=lambda: {16: 0.3, 32: 0.5, 64: 0.2})

    # Voltage scan range for all samples
    v_range: Tuple[float, float] = (-1.5, 1.5)

    # Random seed (None = non-reproducible)
    seed: Optional[int] = 42

    # Augmentation (applied to all samples at generation time)
    augment: bool = True
    noise_aug_sigma: float = 0.02       # additive Gaussian noise augmentation
    blur_aug_prob: float = 0.3          # probability of applying Gaussian blur

    # CIM parameter ranges for each class
    # Double-dot: balanced, moderate coupling
    dd_E_c_range: Tuple[float, float] = (1.8, 5.5)
    dd_t_c_range: Tuple[float, float] = (0.05, 0.6)
    dd_T_range: Tuple[float, float] = (0.01, 0.12)
    dd_lever_range: Tuple[float, float] = (0.35, 0.85)
    dd_asymmetry_max: float = 0.3       # max |E_c1 - E_c2| / mean(E_c)

    # Single-dot: strong coupling OR high asymmetry
    sd_E_c_range: Tuple[float, float] = (1.5, 5.5)
    sd_t_c_range: Tuple[float, float] = (0.6, 1.5)   # strong coupling
    sd_T_range: Tuple[float, float] = (0.01, 0.15)
    sd_lever_range: Tuple[float, float] = (0.3, 0.9)

    # Misc: featureless (SC: small E_c, or Barrier: large E_c / large noise)
    misc_E_c_range_sc: Tuple[float, float] = (0.3, 1.2)   # small → SC
    misc_E_c_range_barrier: Tuple[float, float] = (6.0, 12.0)  # large → pinch-off
    misc_noise_range: Tuple[float, float] = (0.05, 0.15)


# ---------------------------------------------------------------------------
# Dataset generator
# ---------------------------------------------------------------------------

class CIMDataset:
    """
    Generates labelled 2D stability diagrams by sweeping CIM parameters.

    Usage:
        cfg = DatasetConfig(n_per_class=5000, seed=42)
        dataset = CIMDataset(cfg)
        arrays, labels = dataset.generate()   # numpy arrays, integer labels

        # or: get typed Measurement objects + ChargeLabel
        samples = dataset.generate_measurements()
    """

    LABEL_MAP = {
        ChargeLabel.DOUBLE_DOT: 0,
        ChargeLabel.SINGLE_DOT: 1,
        ChargeLabel.MISC: 2,
    }
    INT_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

    def __init__(self, config: Optional[DatasetConfig] = None) -> None:
        self.cfg = config or DatasetConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

    # -----------------------------------------------------------------------
    # Primary generation methods
    # -----------------------------------------------------------------------

    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate the full dataset.

        Returns:
            arrays: float32 array of shape (N, 1, H, W) — ready for PyTorch.
                    H = W = 64 (all samples are resized to this for the CNN).
            labels: int64 array of shape (N,) — class indices.
        """
        samples = self.generate_measurements()
        arrays = np.stack([self._resize_to_64(s[0]) for s in samples], axis=0)
        arrays = arrays[:, np.newaxis, :, :]   # add channel dim → (N, 1, 64, 64)
        labels = np.array([self.LABEL_MAP[s[1]] for s in samples], dtype=np.int64)
        # Shuffle
        idx = self.rng.permutation(len(labels))
        return arrays[idx].astype(np.float32), labels[idx]

    def generate_measurements(self) -> List[Tuple[np.ndarray, ChargeLabel]]:
        """
        Generate raw (array, label) pairs at native resolution.
        Useful when you want to inspect samples before CNN resizing.
        """
        samples: List[Tuple[np.ndarray, ChargeLabel]] = []
        n = self.cfg.n_per_class

        print(f"Generating {n} double-dot samples...")
        for _ in range(n):
            samples.append(self._generate_double_dot())

        print(f"Generating {n} single-dot samples...")
        for _ in range(n):
            samples.append(self._generate_single_dot())

        print(f"Generating {n} misc samples...")
        for _ in range(n):
            samples.append(self._generate_misc())

        print(f"Dataset complete: {len(samples)} samples.")
        return samples

    # -----------------------------------------------------------------------
    # Per-class generators
    # -----------------------------------------------------------------------

    def _generate_double_dot(self) -> Tuple[np.ndarray, ChargeLabel]:
        """
        Double-dot regime: balanced charging energies, moderate tunnel coupling.
        The stability diagram shows a honeycomb pattern with Coulomb anti-crossings.
        """
        cfg = self.cfg
        E_c_mean = self.rng.uniform(*cfg.dd_E_c_range)
        asym = self.rng.uniform(0, cfg.dd_asymmetry_max) * E_c_mean
        E_c1 = E_c_mean + asym / 2
        E_c2 = E_c_mean - asym / 2
        t_c = self.rng.uniform(*cfg.dd_t_c_range)
        T = self.rng.uniform(*cfg.dd_T_range)
        lever = self.rng.uniform(*cfg.dd_lever_range)
        noise = self.rng.uniform(0.005, 0.05)
        res = self._sample_resolution()

        arr = self._simulate(E_c1, E_c2, t_c, T, lever, noise, res)
        if cfg.augment:
            arr = self._augment(arr)
        return arr, ChargeLabel.DOUBLE_DOT

    def _generate_single_dot(self) -> Tuple[np.ndarray, ChargeLabel]:
        """
        Single-dot regime: strong tunnel coupling merges the dots into one,
        or extreme asymmetry makes one dot's transitions invisible.
        Two sub-modes, selected randomly:
            Mode A: strong coupling (t_c >> E_c)
            Mode B: extreme asymmetry (E_c1 >> E_c2 × 5)
        """
        cfg = self.cfg
        mode = self.rng.choice(["strong_coupling", "asymmetric"])

        if mode == "strong_coupling":
            E_c1 = self.rng.uniform(*cfg.sd_E_c_range)
            E_c2 = self.rng.uniform(*cfg.sd_E_c_range)
            t_c = self.rng.uniform(*cfg.sd_t_c_range)   # t_c > 0.6, often > E_c
        else:
            # Asymmetric: one dot has very high E_c (pinched off)
            E_c1 = self.rng.uniform(1.5, 4.0)
            E_c2 = E_c1 * self.rng.uniform(4.0, 8.0)   # E_c2 >> E_c1
            t_c = self.rng.uniform(0.05, 0.4)           # coupling can be moderate

        T = self.rng.uniform(*cfg.sd_T_range)
        lever = self.rng.uniform(*cfg.sd_lever_range)
        noise = self.rng.uniform(0.005, 0.06)
        res = self._sample_resolution()

        arr = self._simulate(E_c1, E_c2, t_c, T, lever, noise, res)
        if cfg.augment:
            arr = self._augment(arr)
        return arr, ChargeLabel.SINGLE_DOT

    def _generate_misc(self) -> Tuple[np.ndarray, ChargeLabel]:
        """
        Misc / featureless regime. Two sub-modes:
            SC (short circuit): very small E_c → flat high conductance
            Barrier/pinch-off: very large E_c → flat low conductance
        Also includes high-noise samples to test DQC handoff.
        """
        cfg = self.cfg
        mode = self.rng.choice(["sc", "barrier", "high_noise"])

        if mode == "sc":
            E_c1 = self.rng.uniform(*cfg.misc_E_c_range_sc)
            E_c2 = self.rng.uniform(*cfg.misc_E_c_range_sc)
            t_c = self.rng.uniform(0.5, 2.0)
            T = self.rng.uniform(0.15, 0.5)              # high T → all levels populated
            lever = self.rng.uniform(0.2, 0.6)
            noise = self.rng.uniform(0.005, 0.04)

        elif mode == "barrier":
            E_c1 = self.rng.uniform(*cfg.misc_E_c_range_barrier)
            E_c2 = self.rng.uniform(*cfg.misc_E_c_range_barrier)
            t_c = self.rng.uniform(0.01, 0.1)            # weak coupling
            T = self.rng.uniform(0.01, 0.06)
            lever = self.rng.uniform(0.1, 0.4)           # small lever → need large V
            noise = self.rng.uniform(0.005, 0.04)

        else:  # high_noise — tests DQC handoff to InspectionAgent
            E_c1 = self.rng.uniform(1.5, 5.0)
            E_c2 = self.rng.uniform(1.5, 5.0)
            t_c = self.rng.uniform(0.1, 0.5)
            T = self.rng.uniform(0.01, 0.1)
            lever = self.rng.uniform(0.3, 0.8)
            noise = self.rng.uniform(*cfg.misc_noise_range)  # high noise

        res = self._sample_resolution()
        arr = self._simulate(E_c1, E_c2, t_c, T, lever, noise, res)
        if cfg.augment:
            arr = self._augment(arr)
        return arr, ChargeLabel.MISC

    # -----------------------------------------------------------------------
    # Physics simulation
    # -----------------------------------------------------------------------

    def _simulate(
        self,
        E_c1: float,
        E_c2: float,
        t_c: float,
        T: float,
        lever: float,
        noise: float,
        res: int,
    ) -> np.ndarray:
        """
        Simulate a 2D stability diagram using the CIM physics engine.
        Imports lazily to avoid circular imports during testing.
        """
        # Lazy import to keep this module independent of torch/CIM setup
        from qdot.simulator.cim import ConstantInteractionDevice

        device = ConstantInteractionDevice(
            E_c1=float(E_c1),
            E_c2=float(E_c2),
            t_c=float(t_c),
            T=float(T),
            lever_arm=float(lever),
            noise_level=float(noise),
            seed=int(self.rng.integers(0, 2**31)),
        )

        v_lo, v_hi = self.cfg.v_range
        v1_grid = np.linspace(v_lo, v_hi, res, dtype=np.float32)
        v2_grid = np.linspace(v_lo, v_hi, res, dtype=np.float32)

        patch = np.zeros((res, res), dtype=np.float32)
        for i, v2 in enumerate(v2_grid):
            for j, v1 in enumerate(v1_grid):
                patch[i, j] = device.current(float(v1), float(v2))

        # Normalise to [0, 1]
        lo, hi = patch.min(), patch.max()
        if hi - lo > 1e-12:
            patch = (patch - lo) / (hi - lo)
        else:
            patch = np.full_like(patch, 0.5)

        return patch

    # -----------------------------------------------------------------------
    # Augmentation
    # -----------------------------------------------------------------------

    def _augment(self, arr: np.ndarray) -> np.ndarray:
        """
        Light augmentation to improve generalisation to real hardware.

        Augmentations applied:
            - Additive Gaussian noise (always, small sigma)
            - Random 90° rotation (physics-consistent: both axes are gates)
            - Horizontal/vertical flip (symmetric under gate label swap)
            - Optional Gaussian blur (simulates finite measurement bandwidth)
        """
        arr = arr.copy()

        # Additive noise
        sigma = self.rng.uniform(0, self.cfg.noise_aug_sigma)
        arr += self.rng.normal(0, sigma, arr.shape).astype(np.float32)

        # Random 90° rotation
        k = int(self.rng.integers(0, 4))
        arr = np.rot90(arr, k=k)

        # Random flip
        if self.rng.random() > 0.5:
            arr = np.fliplr(arr)
        if self.rng.random() > 0.5:
            arr = np.flipud(arr)

        # Optional Gaussian blur (simulates bandwidth-limited measurement)
        if self.rng.random() < self.cfg.blur_aug_prob:
            from scipy.ndimage import gaussian_filter
            sigma_blur = self.rng.uniform(0.3, 1.2)
            arr = gaussian_filter(arr, sigma=sigma_blur).astype(np.float32)

        return np.clip(arr, 0.0, 1.0).astype(np.float32)

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def _sample_resolution(self) -> int:
        """Sample a resolution according to the configured distribution."""
        resolutions = list(self.cfg.resolutions.keys())
        weights = list(self.cfg.resolutions.values())
        # Normalise weights
        total = sum(weights)
        probs = [w / total for w in weights]
        idx = self.rng.choice(len(resolutions), p=probs)
        return resolutions[idx]

    @staticmethod
    def _resize_to_64(arr: np.ndarray) -> np.ndarray:
        """
        Resize a 2D array to 64×64 using bilinear interpolation.
        All training samples are standardised to 64×64 for the CNN.
        """
        if arr.shape == (64, 64):
            return arr.astype(np.float32)
        from scipy.ndimage import zoom
        scale = 64.0 / arr.shape[0]
        resized = zoom(arr.astype(np.float64), scale, order=1)   # bilinear
        return np.clip(resized, 0.0, 1.0).astype(np.float32)

    # -----------------------------------------------------------------------
    # Train/val split
    # -----------------------------------------------------------------------

    @staticmethod
    def split(
        arrays: np.ndarray,
        labels: np.ndarray,
        val_frac: float = 0.15,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Stratified train/val split.

        Returns:
            X_train, X_val, y_train, y_val
        """
        rng = np.random.default_rng(seed)
        classes = np.unique(labels)
        train_idx, val_idx = [], []

        for c in classes:
            idx = np.where(labels == c)[0]
            idx = rng.permutation(idx)
            n_val = max(1, int(len(idx) * val_frac))
            val_idx.extend(idx[:n_val].tolist())
            train_idx.extend(idx[n_val:].tolist())

        train_idx = np.array(train_idx)
        val_idx = np.array(val_idx)
        return arrays[train_idx], arrays[val_idx], labels[train_idx], labels[val_idx]
