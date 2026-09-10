from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kalman import DEFAULT_SCALE, N_PARAMS, ParticleKalmanFilter, normalized_logdet
from kalman import vector_to_params as _vec_to_params_for_feasibility
from qarray_env import N_DOT, N_GATE, QArrayEnv


@dataclass
class OccupationParticle:
    kalman: ParticleKalmanFilter
    weight: float = 1.0


def _gaussian_log_likelihood(residual: np.ndarray, cov: np.ndarray) -> float:
    n = residual.shape[0]
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError(f"Covariance not positive definite (slogdet sign={sign})")
    quad = residual @ np.linalg.solve(cov, residual)
    return float(-0.5 * (n * np.log(2 * np.pi) + logdet + quad))


@dataclass
class RBParticleFilter:
    particles: list[OccupationParticle]

    def __post_init__(self) -> None:
        self._normalize_weights()

    def _normalize_weights(self) -> None:
        weights = np.array([p.weight for p in self.particles])
        total = weights.sum()
        if total <= 0:
            raise ValueError(f"Particle weights sum to {total} <= 0.")
        for p in self.particles:
            p.weight = p.weight / total

    @property
    def n_particles(self) -> int:
        return len(self.particles)

    def predict(self, Q: np.ndarray | None = None) -> None:
        for p in self.particles:
            p.kalman.predict(Q)

    def update(
        self,
        measured_occupation: np.ndarray,
        vg: np.ndarray,
        R: np.ndarray,
        *,
        T: float = 0.05,
        envs: list["QArrayEnv"] | None = None,
    ) -> None:
        """Same as repo version, PLUS an optional `envs` param: if the
        caller (e.g. the acquisition candidate-search loop, which already
        built one QArrayEnv per particle for this step to evaluate IG_bald/
        IG_fim) already has fresh envs matching the particles' CURRENT
        (pre-update) params, pass them here to skip re-solving a third
        time. Caller is responsible for ensuring envs[i] matches
        particles[i].kalman.params -- not checked here."""
        measured_occupation = np.asarray(measured_occupation, dtype=np.float64)
        vg = np.asarray(vg, dtype=np.float64)

        log_weights = np.zeros(self.n_particles)
        built_envs = []
        for i, p in enumerate(self.particles):
            env = envs[i] if envs is not None else QArrayEnv(p.kalman.params)
            built_envs.append(env)
            predicted = env.soft_prediction(vg, T=T)
            residual = measured_occupation - predicted
            log_weights[i] = np.log(p.weight) + _gaussian_log_likelihood(residual, R)

        log_weights -= log_weights.max()
        new_weights = np.exp(log_weights)
        new_weights /= new_weights.sum()
        for p, w in zip(self.particles, new_weights):
            p.weight = w

        for p, env in zip(self.particles, built_envs):
            p.kalman.update(measured_occupation, vg, R, T=T, env=env)

    def effective_sample_size(self) -> float:
        weights = np.array([p.weight for p in self.particles])
        return float(1.0 / np.sum(weights ** 2))

    def resample(
        self,
        rng: np.random.Generator,
        *,
        mu_jitter_scale: float | None = 0.1,
        max_jitter_backoffs: int = 5,
    ) -> None:
        """Systematic resample, then roughen each new particle's `mu` (NOT
        Sigma) so that particles cloned from the same high-weight parent
        actually diversify.

        Why mu, not Sigma: without this, resampling copies `mu` exactly
        between clones. between_particle_covariance() is computed
        entirely from mu (Section 6's "red line"), so identical mu means
        an exactly-singular between-particle covariance matrix ->
        normalized_logdet returns -inf, permanently, from the first
        resample onward -- not a numerical edge case, guaranteed by
        construction. An earlier version of this fix routed jitter
        through predict(Q=...) instead; checking that against the actual
        predict() implementation (it only ever updates Sigma, never mu,
        by design -- Section 2's static-parameter-model choice) showed it
        cannot fix this: it would inflate every clone's Sigma while
        leaving their mu bit-for-bit identical, so the red line would
        still be -inf forever. Caught by reading the code, not by
        re-reasoning about the proposal -- consistent with how the rest
        of this project's error-catching has worked (Section 13).

        mu_jitter_scale: fraction of each particle's OWN (post-clone)
        Sigma used as the roughening covariance, i.e.
        mu_new ~ N(mu, mu_jitter_scale * Sigma). Reuses each particle's
        own belief-sharpness the same way Section 4's independent-per-
        particle-Kalman-filter design already does elsewhere, rather than
        introducing an untethered fixed-magnitude hyperparameter.
        Sigma itself is left untouched by this roughening step, so unlike
        a Q-based approach this does NOT trade off against how fast
        within_particle_covariance (the blue line) clears threshold_low --
        it only fixes the specific mu-duplication degeneracy. Pass None to
        disable (old behavior; between-particle diagnostics will degrade
        after any resample, same failure mode as the unroughened baseline
        that motivated this fix).
        """
        weights = np.array([p.weight for p in self.particles])
        n = self.n_particles
        positions = (np.arange(n) + rng.uniform()) / n
        cumsum = np.cumsum(weights)
        cumsum[-1] = 1.0
        indices = np.searchsorted(cumsum, positions)

        new_particles = []
        for idx in indices:
            src = self.particles[idx]
            new_kalman = ParticleKalmanFilter(
                mu=src.kalman.mu.copy(),
                Sigma=src.kalman.Sigma.copy(),
                E_cm_intra1=src.kalman.E_cm_intra1,
                E_cm_intra2=src.kalman.E_cm_intra2,
                max_backoffs=src.kalman.max_backoffs,
            )
            new_particles.append(OccupationParticle(kalman=new_kalman, weight=1.0 / n))

        if mu_jitter_scale is not None:
            for p in new_particles:
                jitter_cov = mu_jitter_scale * p.kalman.Sigma
                scale = 1.0
                for attempt in range(max_jitter_backoffs + 1):
                    perturbation = rng.multivariate_normal(
                        np.zeros(N_PARAMS), scale * jitter_cov
                    )
                    candidate_mu = p.kalman.mu + perturbation
                    candidate_params = _vec_to_params_for_feasibility(
                        candidate_mu,
                        E_cm_intra1=p.kalman.E_cm_intra1,
                        E_cm_intra2=p.kalman.E_cm_intra2,
                    )
                    try:
                        QArrayEnv(candidate_params)
                        p.kalman.mu = candidate_mu
                        break
                    except Exception:
                        scale *= 0.5
                        if attempt == max_jitter_backoffs:
                            pass  # keep exact clone (mu unchanged) rather than raise --
                            # a duplicate particle is a known-recoverable degeneracy,
                            # an infeasible one is not.

        self.particles = new_particles

    def weighted_mean_mu(self) -> np.ndarray:
        weights = np.array([p.weight for p in self.particles])
        mus = np.stack([p.kalman.mu for p in self.particles])
        return weights @ mus

    def within_particle_covariance(self, scale: np.ndarray = DEFAULT_SCALE) -> float:
        weights = np.array([p.weight for p in self.particles])
        Sigma_avg = np.zeros((N_PARAMS, N_PARAMS))
        for w, p in zip(weights, self.particles):
            Sigma_avg += w * p.kalman.Sigma
        return normalized_logdet(Sigma_avg, scale=scale)

    def between_particle_covariance(self, scale: np.ndarray = DEFAULT_SCALE) -> float:
        weights = np.array([p.weight for p in self.particles])
        mus = np.stack([p.kalman.mu for p in self.particles])
        grand_mean = weights @ mus
        deviations = mus - grand_mean
        Cov_between = (deviations * weights[:, None]).T @ deviations
        return normalized_logdet(Cov_between, scale=scale)
