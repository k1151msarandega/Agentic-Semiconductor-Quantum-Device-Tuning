"""
gz_tuning.belief.particle_filter

Rao-Blackwellized particle filter (FastSLAM-style, Primer Section 4): a
population of particles, each carrying its own independent
ParticleKalmanFilter (belief/kalman.py) over the 5 tracked continuous
device parameters. This module adds the genuinely NEW machinery on top of
kalman.py: particle WEIGHTS (from measurement likelihood), resampling, and
the two Section 6 diagnostic covariance lines.

DESIGN CHOICE, flagged rather than assumed -- read before trusting this
module's interpretation of "discrete occupation state":

The primer says the discrete part is "a particle filter over joint
occupation state (n1, n2, n3, n4)". This could mean an EXPLICIT discrete
label stored per particle (with its own resampling dynamics, possibly
decoupled from the continuous KF), or it could mean the discrete belief
emerges IMPLICITLY from each particle's own continuous-parameter estimate,
via soft_prediction's graded (fractional, near-boundary-aware) output.

This module implements the IMPLICIT interpretation: no separate discrete
label field exists on OccupationParticle. Each particle's soft_prediction
at a candidate vg already gives a natural, graded proxy for "which discrete
occupation state do I believe I'm in, and how confidently" (0.5 = maximally
uncertain between two adjacent integers, near 0 or 1 = confident) -- and
particle DIVERSITY over discrete hypotheses emerges naturally from
different particles' continuous-parameter estimates predicting different
most-likely discrete states near a boundary, without needing a second,
separately-maintained piece of state that would have to be kept consistent
with the continuous belief by hand. This directly reuses soft_prediction,
which is already built and tested (qarray_env.py) -- no new observation
machinery invented here.

If this interpretation turns out to be wrong for your purposes (e.g. you
need particles to represent genuinely discontinuous data-association
hypotheses -- "this measurement sequence implies we crossed transition A
before B" -- that soft_prediction's smooth interpolation can't capture),
that's a real, substantive redesign, not a small patch -- flag it rather
than silently reinterpreting this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kalman import DEFAULT_SCALE, N_PARAMS, ParticleKalmanFilter, normalized_logdet
from qarray_env import N_DOT, N_GATE, QArrayEnv


@dataclass
class OccupationParticle:
    """One particle: an independent continuous-parameter belief (its own
    ParticleKalmanFilter) plus a scalar importance weight. No separate
    discrete-state field -- see module docstring."""

    kalman: ParticleKalmanFilter
    weight: float = 1.0


def _gaussian_log_likelihood(residual: np.ndarray, cov: np.ndarray) -> float:
    """log N(residual; 0, cov), up to the usual normalizing constant that's
    the same across particles sharing the same cov (kept anyway for
    correctness rather than dropped as a premature optimization)."""
    n = residual.shape[0]
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError(f"Covariance not positive definite (slogdet sign={sign})")
    quad = residual @ np.linalg.solve(cov, residual)
    return float(-0.5 * (n * np.log(2 * np.pi) + logdet + quad))


@dataclass
class RBParticleFilter:
    """Population of OccupationParticles. Weights are maintained
    normalized (sum to 1) as an invariant -- checked, not just assumed, in
    tests.
    """

    particles: list[OccupationParticle]

    def __post_init__(self) -> None:
        self._normalize_weights()

    def _normalize_weights(self) -> None:
        weights = np.array([p.weight for p in self.particles])
        total = weights.sum()
        if total <= 0:
            raise ValueError(
                f"Particle weights sum to {total} <= 0 -- all particles "
                f"have collapsed to zero likelihood. This usually means "
                f"every particle's prediction was far from the observed "
                f"measurement (check R, or whether the true device is "
                f"outside every particle's prior support)."
            )
        for p in self.particles:
            p.weight = p.weight / total

    @property
    def n_particles(self) -> int:
        return len(self.particles)

    def predict(self, Q: np.ndarray | None = None) -> None:
        """Process step for every particle's continuous belief. See
        ParticleKalmanFilter.predict -- default Q=None is a no-op (static
        parameter estimation), matching Primer Section 4/2's rejection of
        treating device parameters as a moving target."""
        for p in self.particles:
            p.kalman.predict(Q)

    def update(
        self,
        measured_occupation: np.ndarray,
        vg: np.ndarray,
        R: np.ndarray,
        *,
        T: float = 0.05,
    ) -> None:
        """For each particle: compute its PRE-update predicted occupation
        (from its current mean), reweight by how well that prediction
        matches the actual measurement (standard particle-filter importance
        weighting), THEN run that particle's own Kalman update. Order
        matters -- weighting must use the pre-update prediction, not a
        prediction from parameters the update itself already moved.
        """
        measured_occupation = np.asarray(measured_occupation, dtype=np.float64)
        vg = np.asarray(vg, dtype=np.float64)

        log_weights = np.zeros(self.n_particles)
        envs = []  # kept for reuse below -- avoids re-solving the same
        # params a second time inside each particle's own kalman.update()
        # call (found during review: this was previously an independent
        # third solve of the same params, on top of the two already fixed
        # inside kalman.py's update() itself).
        for i, p in enumerate(self.particles):
            env = QArrayEnv(p.kalman.params)
            envs.append(env)
            predicted = env.soft_prediction(vg, T=T)
            residual = measured_occupation - predicted
            log_weights[i] = np.log(p.weight) + _gaussian_log_likelihood(residual, R)

        # normalize in log-space for numerical stability, then exponentiate
        log_weights -= log_weights.max()
        new_weights = np.exp(log_weights)
        new_weights /= new_weights.sum()
        for p, w in zip(self.particles, new_weights):
            p.weight = w

        for p, env in zip(self.particles, envs):
            p.kalman.update(measured_occupation, vg, R, T=T, env=env)

    def effective_sample_size(self) -> float:
        """Standard ESS = 1 / sum(w_i^2), with weights assumed normalized
        (checked via the class invariant, not re-verified here)."""
        weights = np.array([p.weight for p in self.particles])
        return float(1.0 / np.sum(weights ** 2))

    def resample(self, rng: np.random.Generator, *, jitter_Q: np.ndarray | None = None) -> None:
        """Systematic resampling. Replaces the particle population with
        n_particles draws proportional to weight, each becoming a fresh
        copy (independent mu/Sigma arrays, not shared references) with
        weight reset to 1/n_particles.

        jitter_Q: optional process noise applied to each resampled
        particle's Sigma (via predict) immediately after resampling.
        Resampling with replacement means duplicated particles start
        IDENTICAL -- without some diversity-reintroducing step, repeated
        resampling can collapse the population's diversity over many steps
        (a well-known particle-filter failure mode, "particle
        impoverishment"). NOT applied by default (jitter_Q=None) -- this is
        a real design choice with no primer-specified value; monitor
        effective_sample_size() and between_particle_covariance() over a
        real run to see if this matters in practice before picking a value.
        """
        weights = np.array([p.weight for p in self.particles])
        n = self.n_particles
        positions = (np.arange(n) + rng.uniform()) / n
        cumsum = np.cumsum(weights)
        cumsum[-1] = 1.0  # guard against floating-point sum slightly under 1
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

        self.particles = new_particles
        if jitter_Q is not None:
            self.predict(Q=jitter_Q)

    def weighted_mean_mu(self) -> np.ndarray:
        """Weighted mean of particle means, in raw (E_c1,E_c2,alpha1,alpha2,
        cross_capacitance) coordinates."""
        weights = np.array([p.weight for p in self.particles])
        mus = np.stack([p.kalman.mu for p in self.particles])
        return weights @ mus

    def within_particle_covariance(self, scale: np.ndarray = DEFAULT_SCALE) -> float:
        """Primer Section 6's "blue line": normalized_logdet of the
        weighted-average per-particle covariance, Sigma_avg = sum_i(w_i *
        Sigma_i). NOT the average of each particle's own logdet -- the
        primer's formula is explicit that the MATRICES are averaged first,
        then logdet taken once, and these are not interchangeable
        (logdet is not linear).
        """
        weights = np.array([p.weight for p in self.particles])
        Sigma_avg = np.zeros((N_PARAMS, N_PARAMS))
        for w, p in zip(weights, self.particles):
            Sigma_avg += w * p.kalman.Sigma
        return normalized_logdet(Sigma_avg, scale=scale)

    def between_particle_covariance(self, scale: np.ndarray = DEFAULT_SCALE) -> float:
        """Primer Section 6's "red line": normalized_logdet of Cov_i[mu_i],
        the weighted covariance of particle MEANS around the grand weighted
        mean. This is the diagnostic that catches the documented bug
        (Section 6): particles sharing an identical prior can each look
        individually confident (small within-particle covariance) while
        still disagreeing with each other about true device scale -- a
        false "converged" signal that within-particle covariance ALONE
        cannot catch. Requires this SEPARATE calculation, not derivable
        from within_particle_covariance.

        MINIMUM PARTICLE COUNT, a real statistical constraint, not an
        implementation detail -- verified directly: an empirical covariance
        estimated from k points has rank at most k-1, REGARDLESS OF HOW
        MUCH those points disagree. With N_PARAMS=5 tracked dimensions,
        fewer than 6 particles makes this matrix STRUCTURALLY singular
        (rank <= n_particles-1 < 5), so normalized_logdet will return -inf
        every time, even when particles disagree enormously -- this would
        misleadingly read as "particles perfectly agree" rather than "not
        enough particles to measure disagreement in every direction."
        Ensure n_particles > N_PARAMS (5) for this diagnostic to be
        meaningful; with exactly N_PARAMS+1=6 it's only meaningful if
        particles vary independently across all 5 dimensions, not just one.
        """
        weights = np.array([p.weight for p in self.particles])
        mus = np.stack([p.kalman.mu for p in self.particles])
        grand_mean = weights @ mus
        deviations = mus - grand_mean
        Cov_between = (deviations * weights[:, None]).T @ deviations
        return normalized_logdet(Cov_between, scale=scale)
