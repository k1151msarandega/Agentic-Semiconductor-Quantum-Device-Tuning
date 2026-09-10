from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kalman import N_PARAMS, ParticleKalmanFilter
from particle_filter import OccupationParticle, RBParticleFilter
from qarray_env import DeviceParams, InfeasibleDeviceParamsError, QArrayEnv, TRACKED_PARAM_FIELDS

_COUPLING_IDX = TRACKED_PARAM_FIELDS.index("cross_capacitance")
_SHARED_FIELDS = tuple(f for f in TRACKED_PARAM_FIELDS if f != "cross_capacitance")


@dataclass(frozen=True)
class SharedPriorSpec:
    mean: dict[str, float]
    std: dict[str, float]

    def __post_init__(self) -> None:
        for d, name in ((self.mean, "mean"), (self.std, "std")):
            missing = set(_SHARED_FIELDS) - set(d)
            extra = set(d) - set(_SHARED_FIELDS)
            if missing or extra:
                raise ValueError(
                    f"SharedPriorSpec.{name} keys must be exactly "
                    f"{_SHARED_FIELDS}; missing={missing}, extra={extra}"
                )
        if any(s <= 0 for s in self.std.values()):
            raise ValueError("SharedPriorSpec.std values must all be > 0")


@dataclass(frozen=True)
class CouplingStratum:
    mean: float
    std: float
    weight: float

    def __post_init__(self) -> None:
        if self.std <= 0:
            raise ValueError(f"CouplingStratum.std must be > 0, got {self.std}")
        if not (0.0 < self.weight <= 1.0):
            raise ValueError(f"CouplingStratum.weight must be in (0, 1], got {self.weight}")


def _allocate_stratum_counts(n_particles: int, strata: list[CouplingStratum]) -> list[int]:
    weights = np.array([s.weight for s in strata], dtype=np.float64)
    if not np.isclose(weights.sum(), 1.0, atol=1e-8):
        raise ValueError(f"CouplingStratum weights must sum to 1.0, got {weights.sum()}")
    raw = weights * n_particles
    floors = np.floor(raw).astype(int)
    remainder = n_particles - floors.sum()
    fractional = raw - floors
    order = np.argsort(-fractional)
    counts = floors.copy()
    for idx in order[:remainder]:
        counts[idx] += 1
    return counts.tolist()


def _sample_particle_mu(
    shared_prior: SharedPriorSpec, stratum: CouplingStratum, rng: np.random.Generator
) -> np.ndarray:
    vec = np.zeros(N_PARAMS, dtype=np.float64)
    for i, field_name in enumerate(TRACKED_PARAM_FIELDS):
        if field_name == "cross_capacitance":
            vec[i] = rng.normal(stratum.mean, stratum.std)
        else:
            vec[i] = rng.normal(shared_prior.mean[field_name], shared_prior.std[field_name])
    return vec


def init_particle_filter(
    n_particles: int,
    shared_prior: SharedPriorSpec,
    coupling_strata: list[CouplingStratum],
    prior_sigma_diag: np.ndarray,
    *,
    E_cm_intra1: float,
    E_cm_intra2: float,
    rng: np.random.Generator | None = None,
    max_resample_attempts: int = 50,
    max_backoffs: int = 5,
) -> RBParticleFilter:
    if rng is None:
        rng = np.random.default_rng()
    prior_sigma_diag = np.asarray(prior_sigma_diag, dtype=np.float64)
    if prior_sigma_diag.shape != (N_PARAMS,):
        raise ValueError(f"prior_sigma_diag must have shape ({N_PARAMS},)")
    Sigma0 = np.diag(prior_sigma_diag)

    counts = _allocate_stratum_counts(n_particles, coupling_strata)

    particles: list[OccupationParticle] = []
    for stratum, count in zip(coupling_strata, counts):
        for _ in range(count):
            last_error: Exception | None = None
            for _attempt in range(max_resample_attempts):
                mu = _sample_particle_mu(shared_prior, stratum, rng)
                kwargs = dict(zip(TRACKED_PARAM_FIELDS, mu))
                candidate_params = DeviceParams(
                    E_cm_intra1=E_cm_intra1, E_cm_intra2=E_cm_intra2, **kwargs
                )
                try:
                    QArrayEnv(candidate_params)
                except InfeasibleDeviceParamsError as e:
                    last_error = e
                    continue
                pkf = ParticleKalmanFilter(
                    mu=mu, Sigma=Sigma0.copy(),
                    E_cm_intra1=E_cm_intra1, E_cm_intra2=E_cm_intra2,
                    max_backoffs=max_backoffs,
                )
                particles.append(OccupationParticle(kalman=pkf, weight=1.0))
                break
            else:
                raise InfeasibleDeviceParamsError(
                    f"Could not draw a feasible particle from stratum "
                    f"(mean={stratum.mean}, std={stratum.std}) after "
                    f"{max_resample_attempts} attempts. Last error: {last_error}"
                )

    return RBParticleFilter(particles=particles)


@dataclass(frozen=True)
class CouplingSweepPoint:
    true_cross_capacitance: float
    label: str


def build_coupling_sweep(
    base_params: DeviceParams, sweep_values: list[float], labels: list[str] | None = None
) -> list[tuple[CouplingSweepPoint, DeviceParams]]:
    if 0.0 not in sweep_values:
        raise ValueError("Section 7a requires an exact-zero coupling control point")
    if labels is None:
        labels = [f"coupling_{v:g}" for v in sweep_values]
    out = []
    for v, label in zip(sweep_values, labels):
        point = CouplingSweepPoint(true_cross_capacitance=v, label=label)
        params = base_params.replace(cross_capacitance=v)
        out.append((point, params))
    return out
