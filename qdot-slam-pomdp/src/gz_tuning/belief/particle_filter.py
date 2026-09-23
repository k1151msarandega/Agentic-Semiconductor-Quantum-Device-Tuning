from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kalman import DEFAULT_SCALE, N_PARAMS, ParticleKalmanFilter, normalized_logdet
from kalman import vector_to_params as _vec_to_params_for_feasibility
from local_candidates import local_candidate_states
from qarray_env import N_DOT, N_GATE, QArrayEnv


def _ess_from_log_weights(log_weights: np.ndarray) -> float:
    lw = log_weights - log_weights.max()
    w = np.exp(lw)
    w /= w.sum()
    return float(1.0 / np.sum(w ** 2))


def _find_tempering_beta(
    log_prior_weights: np.ndarray, log_lik: np.ndarray, target_ess: float,
    *, tol: float = 1e-3, max_iter: int = 40,
) -> float:
    """Adaptive-tempering line search (standard SMC-sampler technique):
    finds the largest beta in (0, 1] such that
    normalize(prior_weight * likelihood^beta) has ESS >= target_ess.
    ESS(beta) is monotonically non-increasing in beta (beta=0 recovers
    the prior weights untouched; beta=1 is the untempered, full-strength
    update), so this is a straightforward bisection.

    Motivation (found via a real trace, not assumed): a diffuse
    ground-zero prior evaluated against the very FIRST real measurement
    causes one lucky particle to vastly out-score the rest even with the
    mixture likelihood fix -- confirmed directly: original particles each
    received exactly ONE update() call before the first resample wiped
    out all 40 of them, within the first 1-2 Phase-0 measurements. Every
    resample after that is one surviving lineage's random walk (plus
    roughening jitter) replicated with cosmetic diversity, not real
    ensemble diversity -- which is how a whole 40-particle ensemble can
    end up confidently, unanimously wrong (E_c1 collapsing from ~2.0
    toward ~0.02) without ESS ever looking degenerate by the end.
    Reactively resampling after collapse already happened doesn't fix
    this; the fix has to stop any single step from collapsing the
    ensemble in the first place.

    If ESS at beta=1 already meets the target, returns 1.0 (no
    tempering needed) -- this only softens updates that would otherwise
    over-collapse the ensemble, it never artificially weakens an update
    that was already fine.
    """
    ess_at_1 = _ess_from_log_weights(log_prior_weights + log_lik)
    if ess_at_1 >= target_ess:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        ess_mid = _ess_from_log_weights(log_prior_weights + mid * log_lik)
        if ess_mid >= target_ess:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return lo


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
        candidate_epsilon: float = 0.02,
        target_ess_frac: float | None = None,
    ) -> None:
        """Same as repo version, PLUS an optional `envs` param: if the
        caller (e.g. the acquisition candidate-search loop, which already
        built one QArrayEnv per particle for this step to evaluate IG_bald/
        IG_fim) already has fresh envs matching the particles' CURRENT
        (pre-update) params, pass them here to skip re-solving a third
        time. Caller is responsible for ensuring envs[i] matches
        particles[i].kalman.params -- not checked here.

        BUG FIX (found via a real 15-run multi-seed sweep): the ORIGINAL
        likelihood scored each particle against a single deterministic
        point-prediction (soft_prediction, the softargmin mean), via a
        plain Gaussian residual. Occupations are quantized, so two
        particles that disagree about which side of a transition they're
        on can predict occupations ~1.0 apart while R's sigma is ~0.05 --
        a residual that costs `1^2/(2*sigma^2) ~= 200` nats of log-
        likelihood. Confirmed directly: ESS collapsed to exactly 1.0 in
        15/15 sweep runs, independent of particle count (10, 20, 40 all
        showed it) -- the filter was reduced to one effective hypothesis
        after essentially the first informative measurement, before any
        real evidence had accumulated. More particles couldn't help
        because there was never more than one particle's worth of signal
        surviving to begin with.

        This is NOT fixed by adding a stored discrete-state field to
        OccupationParticle (a schema change alone doesn't touch the
        likelihood that's actually causing the collapse) -- occupation
        here is a deterministic function of a particle's continuous
        params and vg, not an independent hidden variable. The real gap
        is that the likelihood conflated "is my point-prediction exactly
        right" with "is my hypothesis plausible": a particle whose
        continuous params are correct but sits just barely on the wrong
        side of a razor-thin transition boundary (a real, expected
        situation -- see Section 4c/5b's own discussion of how thin these
        regions are) was being treated as certainly wrong instead of
        given partial credit.

        Fix: score each particle against a MIXTURE over its own local
        discrete candidate states (reusing bald.py's exact
        local_candidate_states/softmax machinery, refactored into
        local_candidates.py so both modules can share it without a
        circular import) rather than a single point mean --
        P(measured | particle) = sum_k P(state=k | particle's continuous
        params) * N(measured; state_k, R). A particle near a boundary now
        gets weighted by how much of its own local probability mass is
        consistent with the measurement, not just whether its single
        point-estimate happened to land on the right side. The
        CONTINUOUS (Kalman) update is intentionally left using the
        existing point-prediction/Jacobian (an EKF moment-matching
        approximation against the mixture mean) -- that part was never
        the source of the collapse and doesn't need to change.

        Also added: optional `target_ess_frac` tempering (see
        _find_tempering_beta's docstring for the mechanism this fixes --
        found via tracing a real divergent run, where the mixture-
        likelihood fix alone wasn't sufficient: it reduces per-step
        over-collapse but doesn't prevent it on the very first
        measurement against a maximally diffuse ground-zero prior).
        Tempering affects only the WEIGHT update (how much of the
        ensemble's diversity a single measurement is allowed to
        discard), not the per-particle Kalman mean/covariance update --
        each particle's own continuous belief still updates at full
        strength; only how aggressively the ensemble PRUNES based on
        relative fit is throttled. Pass None (default) to disable and
        keep the untempered, backward-compatible likelihood.
        """
        measured_occupation = np.asarray(measured_occupation, dtype=np.float64)
        vg = np.asarray(vg, dtype=np.float64)

        log_prior_weights = np.log(np.array([p.weight for p in self.particles]))
        log_lik = np.zeros(self.n_particles)
        built_envs = []
        for i, p in enumerate(self.particles):
            env = envs[i] if envs is not None else QArrayEnv(p.kalman.params)
            built_envs.append(env)
            predicted = env.soft_prediction(vg, T=T)

            candidates = local_candidate_states(predicted[None, :], epsilon=candidate_epsilon)
            if candidates.shape[0] == 1:
                # No local ambiguity -- single candidate reduces exactly
                # to the old point-prediction likelihood, so this branch
                # changes nothing for particles that aren't near a
                # boundary (the common case away from transitions).
                residual = measured_occupation - candidates[0]
                log_lik[i] = _gaussian_log_likelihood(residual, R)
                continue

            F = env.free_energy_at_candidates(candidates, vg)
            neg_F_over_T = -F / T
            neg_F_over_T -= neg_F_over_T.max()
            local_w = np.exp(neg_F_over_T)
            local_w /= local_w.sum()  # P(state=k | this particle's continuous params)

            log_lik_per_candidate = np.array([
                _gaussian_log_likelihood(measured_occupation - c, R) for c in candidates
            ])
            # log-sum-exp over the mixture, weighted by local_w
            m = log_lik_per_candidate.max()
            log_lik[i] = m + np.log(np.sum(local_w * np.exp(log_lik_per_candidate - m)))

        if target_ess_frac is not None:
            beta = _find_tempering_beta(
                log_prior_weights, log_lik, target_ess=target_ess_frac * self.n_particles
            )
        else:
            beta = 1.0

        log_weights = log_prior_weights + beta * log_lik
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
