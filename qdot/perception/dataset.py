"""
qdot/perception/dataset.py
==========================
CIMDataset — synthetic training data generator for the Inspection Agent.

KEY FIX (root cause of Phase 2 failure):
    _simulate() now centres each sample's scan window on the calculated
    charge transition voltage V_centre = -E_c_mean / lever_arm, spanning
    ±1.5 Coulomb periods.  The original fixed v_range = (-1.5, 1.5) V
    placed the transition entirely outside the window for all double-dot
    and single-dot samples (transition at -2.1 to -15.7 V), so the CNN
    learned gradient direction instead of charge morphology.

QFlow's role: transfer benchmark only — never in the training loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from qdot.core.types import ChargeLabel, Measurement, MeasurementModality, VoltagePoint


@dataclass
class DatasetConfig:
    n_per_class: int = 17_000
    resolutions: dict = field(default_factory=lambda: {16: 0.3, 32: 0.5, 64: 0.2})
    # v_range retained for reference; _simulate() ignores it in favour of
    # the per-sample transition-centred window.
    v_range: Tuple[float, float] = (-1.5, 1.5)
    seed: Optional[int] = 42
    augment: bool = True
    noise_aug_sigma: float = 0.02
    blur_aug_prob: float = 0.3
    # Double-dot
    dd_E_c_range: Tuple[float, float] = (1.0, 5.5)
    dd_t_c_range: Tuple[float, float] = (0.05, 0.6)
    dd_T_range: Tuple[float, float] = (0.01, 0.12)
    dd_lever_range: Tuple[float, float] = (0.35, 0.85)
    dd_asymmetry_max: float = 0.3
    # Single-dot
    sd_E_c_range: Tuple[float, float] = (1.5, 5.5)
    sd_t_c_range: Tuple[float, float] = (0.6, 1.5)
    sd_T_range: Tuple[float, float] = (0.01, 0.15)
    sd_lever_range: Tuple[float, float] = (0.3, 0.9)
    # Misc
    misc_E_c_range_sc: Tuple[float, float] = (0.3, 1.2)
    misc_E_c_range_barrier: Tuple[float, float] = (6.0, 12.0)
    misc_noise_range: Tuple[float, float] = (0.05, 0.15)


class CIMDataset:
    """
    Generates labelled 2D stability diagrams.

    Each sample's scan window is centred on V_centre = -E_c_mean / lever_arm
    with half-width δ = 1.5 × (one Coulomb period in voltage).  This
    guarantees honeycomb/diamond topology is visible in every training image.
    Uses the vectorised current_grid path (~100× faster than the old loop).
    """

    LABEL_MAP = {ChargeLabel.DOUBLE_DOT: 0, ChargeLabel.SINGLE_DOT: 1, ChargeLabel.MISC: 2}
    INT_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

    def __init__(self, config: Optional[DatasetConfig] = None) -> None:
        self.cfg = config or DatasetConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        samples = self.generate_measurements()
        arrays = np.stack([self._resize_to_64(s[0]) for s in samples], axis=0)
        arrays = arrays[:, np.newaxis, :, :]
        labels = np.array([self.LABEL_MAP[s[1]] for s in samples], dtype=np.int64)
        idx = self.rng.permutation(len(labels))
        return arrays[idx].astype(np.float32), labels[idx]

    def generate_measurements(self) -> List[Tuple[np.ndarray, ChargeLabel]]:
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

    def _generate_double_dot(self) -> Tuple[np.ndarray, ChargeLabel]:
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
        cfg = self.cfg
        mode = self.rng.choice(["strong_coupling", "asymmetric"])
        if mode == "strong_coupling":
            E_c1 = self.rng.uniform(*cfg.sd_E_c_range)
            E_c2 = self.rng.uniform(*cfg.sd_E_c_range)
            t_c = self.rng.uniform(*cfg.sd_t_c_range)
        else:
            E_c1 = self.rng.uniform(1.5, 4.0)
            E_c2 = E_c1 * self.rng.uniform(4.0, 8.0)
            t_c = self.rng.uniform(0.05, 0.4)
        T = self.rng.uniform(*cfg.sd_T_range)
        lever = self.rng.uniform(*cfg.sd_lever_range)
        noise = self.rng.uniform(0.005, 0.06)
        res = self._sample_resolution()
        arr = self._simulate(E_c1, E_c2, t_c, T, lever, noise, res)
        if cfg.augment:
            arr = self._augment(arr)
        return arr, ChargeLabel.SINGLE_DOT

    def _generate_misc(self) -> Tuple[np.ndarray, ChargeLabel]:
        cfg = self.cfg
        mode = self.rng.choice(["sc", "barrier", "high_noise"])
        if mode == "sc":
            E_c1 = self.rng.uniform(*cfg.misc_E_c_range_sc)
            E_c2 = self.rng.uniform(*cfg.misc_E_c_range_sc)
            t_c = self.rng.uniform(0.5, 2.0)
            T = self.rng.uniform(0.15, 0.5)
            lever = self.rng.uniform(0.2, 0.6)
            noise = self.rng.uniform(0.005, 0.04)
        elif mode == "barrier":
            E_c1 = self.rng.uniform(*cfg.misc_E_c_range_barrier)
            E_c2 = self.rng.uniform(*cfg.misc_E_c_range_barrier)
            t_c = self.rng.uniform(0.01, 0.1)
            T = self.rng.uniform(0.01, 0.06)
            lever = self.rng.uniform(0.1, 0.4)
            noise = self.rng.uniform(0.005, 0.04)
        else:
            E_c1 = self.rng.uniform(1.5, 5.0)
            E_c2 = self.rng.uniform(1.5, 5.0)
            t_c = self.rng.uniform(0.1, 0.5)
            T = self.rng.uniform(0.01, 0.1)
            lever = self.rng.uniform(0.3, 0.8)
            noise = self.rng.uniform(*cfg.misc_noise_range)
        res = self._sample_resolution()
        arr = self._simulate(E_c1, E_c2, t_c, T, lever, noise, res)
        if cfg.augment:
            arr = self._augment(arr)
        return arr, ChargeLabel.MISC

    def _simulate(
        self,
        E_c1: float, E_c2: float, t_c: float, T: float,
        lever: float, noise: float, res: int,
    ) -> np.ndarray:
        """
        Simulate a 2D stability diagram with a TRANSITION-CENTRED scan window.

        Window:  V_centre = -E_c_mean / lever  (charge degeneracy voltage)
                 half-width δ = 1.5 × Coulomb_period_in_voltage
                             = 1.5 × (E_c_mean / lever)

        This guarantees the honeycomb / Coulomb diamond features are present
        in every training image regardless of E_c and lever_arm values.
        Uses vectorised current_grid — no Python loop.
        """
        from qdot.simulator.cim import ConstantInteractionDevice

        device = ConstantInteractionDevice(
            E_c1=float(E_c1), E_c2=float(E_c2), t_c=float(t_c),
            T=float(T), lever_arm=float(lever), noise_level=float(noise),
            seed=int(self.rng.integers(0, 2**31)),
        )

        E_c_mean = (float(E_c1) + float(E_c2)) / 2.0
        V_centre = -E_c_mean / float(lever)
        coulomb_period_V = E_c_mean / float(lever)
        delta = max(1.5 * coulomb_period_V, 0.5)   # at least ±0.5 V

        v1_grid = np.linspace(V_centre - delta, V_centre + delta, res, dtype=np.float32)
        v2_grid = np.linspace(V_centre - delta, V_centre + delta, res, dtype=np.float32)

        patch = device.current_grid(v1_grid, v2_grid)

        lo, hi = patch.min(), patch.max()
        if hi - lo > 1e-12:
            patch = (patch - lo) / (hi - lo)
        else:
            patch = np.full_like(patch, 0.5)

        return patch.astype(np.float32)

    def _augment(self, arr: np.ndarray) -> np.ndarray:
        arr = arr.copy()
        sigma = self.rng.uniform(0, self.cfg.noise_aug_sigma)
        arr += self.rng.normal(0, sigma, arr.shape).astype(np.float32)
        k = int(self.rng.integers(0, 4))
        arr = np.rot90(arr, k=k)
        if self.rng.random() > 0.5:
            arr = np.fliplr(arr)
        if self.rng.random() > 0.5:
            arr = np.flipud(arr)
        if self.rng.random() < self.cfg.blur_aug_prob:
            from scipy.ndimage import gaussian_filter
            sigma_blur = self.rng.uniform(0.3, 1.2)
            arr = gaussian_filter(arr, sigma=sigma_blur).astype(np.float32)
        return np.clip(arr, 0.0, 1.0).astype(np.float32)

    def _sample_resolution(self) -> int:
        resolutions = list(self.cfg.resolutions.keys())
        weights = list(self.cfg.resolutions.values())
        total = sum(weights)
        probs = [w / total for w in weights]
        idx = self.rng.choice(len(resolutions), p=probs)
        return resolutions[idx]

    @staticmethod
    def _resize_to_64(arr: np.ndarray) -> np.ndarray:
        if arr.shape == (64, 64):
            return arr.astype(np.float32)
        from scipy.ndimage import zoom
        scale = 64.0 / arr.shape[0]
        resized = zoom(arr.astype(np.float64), scale, order=1)
        return np.clip(resized, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def split(
        arrays: np.ndarray, labels: np.ndarray,
        val_frac: float = 0.15, seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
