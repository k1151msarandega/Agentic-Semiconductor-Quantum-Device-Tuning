from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from noise_model import per_dot_sigma_eff, p_flip_per_dot
from particle_filter import RBParticleFilter
from kalman import ParticleKalmanFilter
from qarray_env import N_DOT, N_GATE, QArrayEnv, observation_jacobian


def target_occupation_probability(
    kalman: ParticleKalmanFilter,
    vg: np.ndarray,
    target_occupation: np.ndarray,
    *,
    T: float = 0.05,
) -> float:
    vg = np.asarray(vg, dtype=np.float64)
    target_occupation = np.asarray(target_occupation, dtype=np.float64)
    if target_occupation.shape != (N_DOT,):
        raise ValueError(f"target_occupation must have shape ({N_DOT},)")

    env = QArrayEnv(kalman.params)
    H = observation_jacobian(kalman.params, vg, T=T)
    prediction = env.soft_prediction(vg, T=T)

    frac = prediction - np.floor(prediction)
    on_target_side = np.round(prediction) == target_occupation
    unsigned_distance = np.abs(frac - 0.5)

    zero_R = np.zeros((H.shape[0], H.shape[0]))
    sigma_eff = per_dot_sigma_eff(zero_R, H, kalman.Sigma)

    unsigned_p_flip = p_flip_per_dot(unsigned_distance, sigma_eff)
    p_correct_per_dot = np.where(on_target_side, 1.0 - unsigned_p_flip, unsigned_p_flip)

    return float(np.prod(p_correct_per_dot))


def target_occupation_mass(
    pf: RBParticleFilter, vg: np.ndarray, target_occupation: np.ndarray, *, T: float = 0.05
) -> float:
    vg = np.asarray(vg, dtype=np.float64)
    target_occupation = np.asarray(target_occupation, dtype=np.float64)
    if target_occupation.shape != (N_DOT,):
        raise ValueError(f"target_occupation must have shape ({N_DOT},)")

    mass = 0.0
    for p in pf.particles:
        mass += p.weight * target_occupation_probability(p.kalman, vg, target_occupation, T=T)
    return mass


def belief_stable(
    pf: RBParticleFilter, vg: np.ndarray, target_occupation: np.ndarray, *, p_conf: float, T: float = 0.05
) -> bool:
    return target_occupation_mass(pf, vg, target_occupation, T=T) > p_conf


def actuation_quiescent(proposed_step: np.ndarray, *, epsilon: float) -> bool:
    proposed_step = np.asarray(proposed_step, dtype=np.float64)
    if proposed_step.shape != (N_GATE,):
        raise ValueError(f"proposed_step must have shape ({N_GATE},)")
    return bool(np.all(np.abs(proposed_step) < epsilon))


@dataclass(frozen=True)
class ConvergenceStatus:
    belief_stable: bool
    actuation_quiescent: bool
    actuation_quiescence_checked: bool
    consecutive_count: int
    converged: bool


@dataclass
class ConvergenceMonitor:
    p_conf: float
    epsilon: float
    N: int
    target_occupation: np.ndarray
    T: float = 0.05
    require_actuation_quiescence: bool = True
    _consecutive: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.target_occupation = np.asarray(self.target_occupation, dtype=np.float64)
        if self.target_occupation.shape != (N_DOT,):
            raise ValueError(f"target_occupation must have shape ({N_DOT},)")
        if self.N < 1:
            raise ValueError(f"N must be >= 1, got {self.N}")

    def step(self, pf: RBParticleFilter, vg: np.ndarray, proposed_step: np.ndarray) -> ConvergenceStatus:
        stable = belief_stable(pf, vg, self.target_occupation, p_conf=self.p_conf, T=self.T)

        if self.require_actuation_quiescence:
            quiescent = actuation_quiescent(proposed_step, epsilon=self.epsilon)
        else:
            proposed_step = np.asarray(proposed_step, dtype=np.float64)
            if proposed_step.shape != (N_GATE,):
                raise ValueError(f"proposed_step must have shape ({N_GATE},)")
            quiescent = True

        both_held = stable and quiescent
        if both_held:
            self._consecutive += 1
        else:
            self._consecutive = 0

        converged = self._consecutive >= self.N
        return ConvergenceStatus(
            belief_stable=stable,
            actuation_quiescent=quiescent,
            actuation_quiescence_checked=self.require_actuation_quiescence,
            consecutive_count=self._consecutive,
            converged=converged,
        )

    def reset(self) -> None:
        self._consecutive = 0
