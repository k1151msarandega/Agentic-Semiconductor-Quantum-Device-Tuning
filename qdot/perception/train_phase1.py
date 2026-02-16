"""
experiments/train_phase1.py
============================
Phase 1 training script.

Generates CIM training data → trains 5-model ensemble → fits OOD detector
→ saves checkpoint → (optional) evaluates on QFlow.

Usage:
    python experiments/train_phase1.py                  # full 51k training run
    python experiments/train_phase1.py --fast           # 3k samples, 10 epochs (dev)
    python experiments/train_phase1.py --qflow PATH     # evaluate on QFlow after training

Outputs (in experiments/checkpoints/phase1/):
    model_0.pt ... model_4.pt  — ensemble weights
    ood_detector.pkl           — fitted Mahalanobis OOD detector
    training_log.json          — metrics, config snapshot, timestamps

Phase 1 benchmark targets (blueprint §8):
    ≥96% accuracy on QFlow held-out test set
    OOD detector flags 100% of QFlow test samples (since they're OOD by design)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description="Phase 1 training")
    parser.add_argument("--fast", action="store_true",
                        help="Use small dataset (3k samples, 10 epochs) for dev")
    parser.add_argument("--qflow", type=str, default=None,
                        help="Path to QFlow held-out test set (NPZ or directory)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device: 'cpu' or 'cuda'")
    parser.add_argument("--out", type=str, default="experiments/checkpoints/phase1",
                        help="Output directory for checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("PHASE 1 TRAINING — CIM → EnsembleCNN + OOD Detector")
    print(f"{'='*60}\n")

    # -----------------------------------------------------------------------
    # 1. Generate training data
    # -----------------------------------------------------------------------
    from qdot.perception.dataset import CIMDataset, DatasetConfig

    if args.fast:
        cfg = DatasetConfig(n_per_class=1_000, seed=args.seed, augment=True)
        n_epochs = 10
        print("FAST MODE: 3k samples, 10 epochs")
    else:
        cfg = DatasetConfig(n_per_class=17_000, seed=args.seed, augment=True)
        n_epochs = 30
        print("FULL MODE: ~51k samples, 30 epochs")

    t0 = time.time()
    dataset = CIMDataset(cfg)
    X_all, y_all = dataset.generate()
    print(f"Data generation: {time.time()-t0:.1f}s | shape={X_all.shape}")

    X_train, X_val, y_train, y_val = CIMDataset.split(
        X_all, y_all, val_frac=0.15, seed=args.seed
    )
    print(
        f"Split: {len(X_train)} train | {len(X_val)} val\n"
        f"Class counts (train): {np.bincount(y_train)}\n"
    )

    # -----------------------------------------------------------------------
    # 2. Train ensemble
    # -----------------------------------------------------------------------
    from qdot.perception.classifier import EnsembleCNN

    print("Training 5-model ensemble...")
    t1 = time.time()
    ensemble = EnsembleCNN.train_from_data(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        n_epochs=n_epochs,
        batch_size=128,
        lr=3e-4,
        device=args.device,
        model_dir=str(out_dir),
        verbose=True,
    )
    train_time = time.time() - t1
    print(f"\nEnsemble trained in {train_time:.1f}s")

    # Final validation accuracy
    val_preds = []
    for arr, true_label in zip(X_val, y_val):
        pred, _, _ = ensemble.classify(arr.squeeze())
        val_preds.append(pred)
    val_acc = float(np.mean(np.array(val_preds) == y_val))
    print(f"Final val accuracy: {val_acc:.4f}")

    # -----------------------------------------------------------------------
    # 3. Fit OOD detector on training features
    # -----------------------------------------------------------------------
    from qdot.perception.ood import MahalanobisOOD, extract_features_batch

    print("\nFitting OOD detector on training features...")
    t2 = time.time()
    train_features = extract_features_batch(ensemble, X_train, device=args.device)
    ood = MahalanobisOOD(n_components=16, calibration_percentile=95.0)
    ood.fit(train_features)

    # Sanity check FPR on validation set
    val_features = extract_features_batch(ensemble, X_val, device=args.device)
    _, val_flags = ood.score_batch(val_features)
    fpr = float(val_flags.mean())
    print(f"OOD detector fitted in {time.time()-t2:.1f}s | val FPR={fpr:.3f} (target ≤0.05)")

    ood.save(str(out_dir / "ood_detector.pkl"))

    # -----------------------------------------------------------------------
    # 4. (Optional) QFlow evaluation — sim-to-real transfer test
    # -----------------------------------------------------------------------
    qflow_acc = None
    qflow_ood_recall = None
    if args.qflow:
        print(f"\nEvaluating on QFlow: {args.qflow}")
        qflow_acc, qflow_ood_recall = _evaluate_qflow(
            args.qflow, ensemble, ood, args.device
        )
        print(f"QFlow accuracy:     {qflow_acc:.4f}  (target ≥0.96)")
        print(f"QFlow OOD recall:   {qflow_ood_recall:.4f}  (target =1.00 — all real data is OOD)")

    # -----------------------------------------------------------------------
    # 5. Save training log
    # -----------------------------------------------------------------------
    log = {
        "timestamp": time.time(),
        "config": {
            "n_per_class": cfg.n_per_class,
            "n_epochs": n_epochs,
            "batch_size": 128,
            "lr": 3e-4,
            "seed": args.seed,
            "device": args.device,
            "fast_mode": args.fast,
        },
        "results": {
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "val_accuracy_cim": val_acc,
            "ood_val_fpr": fpr,
            "qflow_accuracy": qflow_acc,
            "qflow_ood_recall": qflow_ood_recall,
            "train_time_s": train_time,
        },
        "benchmarks": {
            "val_acc_target": 0.96,
            "ood_fpr_target": 0.05,
            "qflow_acc_target": 0.96,
            "qflow_ood_recall_target": 1.0,
        },
    }
    with open(out_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nCheckpoints saved to: {out_dir}")
    print("Training complete.")
    print(f"\n{'='*60}")
    print("PHASE 1 BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"  Val accuracy (CIM):  {val_acc:.4f}  {'✓' if val_acc >= 0.96 else '✗'} (target ≥0.96)")
    print(f"  OOD FPR (CIM val):   {fpr:.4f}  {'✓' if fpr <= 0.05 else '✗'} (target ≤0.05)")
    if qflow_acc is not None:
        print(f"  QFlow accuracy:      {qflow_acc:.4f}  {'✓' if qflow_acc >= 0.96 else '✗'} (target ≥0.96)")
        print(f"  QFlow OOD recall:    {qflow_ood_recall:.4f}  {'✓' if qflow_ood_recall >= 0.95 else '✗'} (target ~1.0)")
    print(f"{'='*60}\n")


def _evaluate_qflow(
    qflow_path: str,
    ensemble,
    ood_detector,
    device: str,
) -> tuple[float, float]:
    """
    Evaluate ensemble on QFlow held-out test set.

    QFlow format: each sample is a 2D stability diagram labelled as
    one of {SC, Barrier, SD, DD}.

    Label mapping to our 3-class system:
        SC      → MISC   (2)
        Barrier → MISC   (2)
        SD      → SINGLE_DOT (1)
        DD      → DOUBLE_DOT (0)

    Returns:
        (accuracy, ood_recall)
        accuracy:   fraction of QFlow labels correctly predicted
        ood_recall: fraction of QFlow samples flagged as OOD (should be ~1.0)
    """
    from qdot.perception.ood import extract_features_batch
    from qdot.perception.dataset import CIMDataset
    import os

    qflow_path = Path(qflow_path)

    # Load QFlow — support both NPZ and directory of images
    if qflow_path.suffix == ".npz":
        data = np.load(qflow_path)
        X_qflow = data["arrays"].astype(np.float32)    # (N, H, W) or (N, 1, H, W)
        y_qflow = data["labels"].astype(np.int64)      # QFlow integer labels
        qflow_label_map = {0: 2, 1: 2, 2: 1, 3: 0}   # SC,Barrier→MISC; SD→SD; DD→DD
    else:
        raise NotImplementedError(
            "QFlow directory loading not yet implemented. "
            "Convert to NPZ format first: arrays (N, H, W), labels (N,) with "
            "SC=0, Barrier=1, SD=2, DD=3."
        )

    # Normalise shape to (N, 1, 64, 64)
    if X_qflow.ndim == 3:
        X_qflow = X_qflow[:, np.newaxis, :, :]
    if X_qflow.shape[-1] != 64:
        from scipy.ndimage import zoom
        n = X_qflow.shape[0]
        resized = np.zeros((n, 1, 64, 64), dtype=np.float32)
        for i in range(n):
            scale = 64.0 / X_qflow.shape[-1]
            resized[i, 0] = np.clip(
                zoom(X_qflow[i, 0].astype(np.float64), scale, order=1), 0, 1
            ).astype(np.float32)
        X_qflow = resized

    # Map QFlow labels to our 3-class system
    y_ours = np.array([qflow_label_map[int(l)] for l in y_qflow], dtype=np.int64)

    # Classify
    preds = []
    for arr in X_qflow:
        pred, _, _ = ensemble.classify(arr.squeeze())
        preds.append(pred)
    accuracy = float(np.mean(np.array(preds) == y_ours))

    # OOD: all QFlow samples are real hardware → should be flagged as OOD
    qflow_features = extract_features_batch(ensemble, X_qflow, device=device)
    _, ood_flags = ood_detector.score_batch(qflow_features)
    ood_recall = float(ood_flags.mean())

    return accuracy, ood_recall


if __name__ == "__main__":
    main()
