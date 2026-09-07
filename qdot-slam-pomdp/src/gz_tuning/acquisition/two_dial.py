"""
gz_tuning.acquisition.two_dial

Combines IG_BALD and IG_FIM per Primer Section 5:
    IG_total(v) = w_discrete(t) * IG_BALD(v) + w_continuous(t) * IG_FIM(v)

OPEN, NOT INVENTED HERE: the primer explicitly flags that w(t)'s job is
managing the dynamic-range mismatch between IG_BALD (bounded above by
min(log(n_particles), log(K)) -- a genuine mutual-information bound, see
bald.py's docstring; NOT N_DOT*log(2), which was specific to the buggy
summed-marginal-entropy construction bald.py has since been rewritten to
fix) and IG_FIM (unbounded, grows as posterior
covariance shrinks toward a point estimate) -- but does not specify the
functional form of w_discrete(t)/w_continuous(t). Primer Section 12 lists
several related derivations as still open. Rather than invent a specific
time-decay schedule now (which would be exactly the kind of unverified,
picked-by-feel numeric choice this project has repeatedly had to correct
elsewhere), compute_ig_total below takes w_discrete/w_continuous as
EXPLICIT, caller-supplied scalars, not something this module computes from
t internally. Pick and justify a schedule (or a fixed pair of weights) at
the call site, informed by real runs -- don't treat any default here as
validated.
"""

from __future__ import annotations

import numpy as np

from bald import compute_ig_bald
from fim import compute_ig_fim
from kalman import DEFAULT_SCALE
from particle_filter import RBParticleFilter


def compute_ig_total(
    pf: RBParticleFilter,
    vg: np.ndarray,
    R: np.ndarray,
    *,
    w_discrete: float,
    w_continuous: float,
    T: float = 0.05,
    scale: np.ndarray = DEFAULT_SCALE,
) -> float:
    """IG_total(v) = w_discrete * IG_BALD(v) + w_continuous * IG_FIM(v).

    w_discrete/w_continuous are required (no default) -- deliberately, so a
    caller cannot silently inherit an unexamined schedule (see module
    docstring). Both terms are in nats provided `scale` matches whatever
    normalization convention the rest of the belief (kalman.py) is using
    consistently -- this is asserted, not just assumed: see the runtime
    check below, which the primer calls for explicitly ("worth a runtime
    assertion, not just a one-time design note").
    """
    if scale.shape[0] != next(iter(pf.particles)).kalman.mu.shape[0]:
        raise ValueError(
            f"scale (len {scale.shape[0]}) does not match the tracked "
            f"parameter dimension (len {next(iter(pf.particles)).kalman.mu.shape[0]}) "
            f"-- IG_FIM's nats-unit claim only holds under a consistent "
            f"normalization; a mismatched scale here would silently break "
            f"that guarantee rather than raise."
        )

    ig_bald = compute_ig_bald(pf, vg, T=T)
    ig_fim = compute_ig_fim(pf, vg, R, T=T, scale=scale)
    return w_discrete * ig_bald + w_continuous * ig_fim
