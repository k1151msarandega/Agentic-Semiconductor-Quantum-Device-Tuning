"""
qdot/perception/__init__.py
===========================
Phase 1 — Perception & Physics layer.

Public surface:
    DQCGatekeeper   — data quality classifier (Layer 3 entry point)
    InspectionAgent — full perception pipeline (DQC → features → CNN → OOD → report)
    TinyCNN         — 3-class primary classifier
    EnsembleCNN     — 5-model ensemble with uncertainty quantification
    MahalanobisOOD  — out-of-distribution detector
    CIMDataset      — synthetic training data generator
"""

from qdot.perception.dqc import DQCGatekeeper
from qdot.perception.classifier import TinyCNN, EnsembleCNN
from qdot.perception.ood import MahalanobisOOD
from qdot.perception.inspector import InspectionAgent
from qdot.perception.dataset import CIMDataset, DatasetConfig

__all__ = [
    "DQCGatekeeper",
    "TinyCNN",
    "EnsembleCNN",
    "MahalanobisOOD",
    "InspectionAgent",
    "CIMDataset",
    "DatasetConfig",
]
