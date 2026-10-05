# Ground-Zero POMDP+SLAM Charge Tuning — Coding Primer

**Purpose of this document:** this is a standalone project spinning out of a larger pipeline
(`Agentic-Semiconductor-Quantum-Device-Tuning`, path below) but deliberately kept separate.
Everything below was settled across an extended design conversation *before* any code was
written. This primer exists so a fresh chat focused on implementation doesn't need to
re-derive any of it. Treat every decision below as settled unless flagged "OPEN."

---

## 1. Scope and north star

**Claim 1 (primary, headline) — revised framing, precisely scoped after a real preemption
check:** active FastSLAM-style exploration (particle-filter belief + information-gain-driven
action selection) can tune **two capacitively-coupled double quantum dots (4 dots total) from a
true joint ground-zero start** — no assumed device parameters, no proportional patch sizing, no
pre-existing coarse localization, and critically, **the cross-DQD coupling is inferred live
rather than compensated away via virtual gates** — more efficiently (measurements-to-convergence)
than a raster-scan baseline. **Do not claim unqualified "ground-zero" as the novelty** — Schuff
et al. (arXiv:2402.03931, Ares/Zumbühl groups) demonstrated literal grounded-device-to-Rabi-
oscillations full autonomy on a single device. Single-device ground-zero is closed; this
project's actual, still-open claim is **joint ground-zero for coupled multi-device systems
without virtual-gate decoupling** — contrast specifically against MAVIS (Zwolak/Delft —
compensates cross-talk away) and Schuff (single device, not joint-coupled). State the claim this
specifically every time, not as bare "ground-zero." Anchored on an independently verified
literature gap: no existing work does explicit information-theoretic/particle-filter active
exploration for *ground-zero spatial* tuning specifically, as opposed to RL-reward-based
exploration (Nguyen et al. 2021) or methods that assume a coarse-tuning starting point.

**Verification status on this citation:** confirmed directly via search — arXiv:2402.03931,
"Fully autonomous tuning of a spin qubit," Schuff, Carballido, et al., Ares (Oxford) and
Zumbühl (Basel) groups, is real, and its abstract matches the claimed content ("first fully
autonomous tuning of a semiconductor qubit, from a grounded device to Rabi oscillations").
**Not yet independently confirmed: the specific journal venue ("Nature Electronics").** The
verified record shows it as an arXiv preprint; check the actual publication venue before citing
it that specifically in the paper, rather than carrying the venue claim forward unverified.

**Terminology, settled, stop relitigating:** say **"FastSLAM-style joint discrete-continuous
belief architecture,"** never bare "SLAM." This is the technically correct claim, not a
watered-down one — you're borrowing a specific, citable algorithmic factorization (Montemerlo et
al. 2002: particles for discrete multimodal state, per-particle Kalman for continuous
parameters), not claiming the full SLAM problem (localization + mapping + loop closure), which
this system genuinely doesn't have: voltage is a controlled, precisely-known input (no pose
drift to localize), and the "map" (Section 4's 5-parameter vector) is one small tightly-coupled
global calibration, not a spatial landmark set. The factorization still earns the FastSLAM
comparison — via conditional independence given discrete occupation state, not spatial/landmark
independence, which is the correct justification to give if pressed, not the robotics-standard
one. Loop closure is explicitly out of scope (Section 2) and should not be reframed as
hysteresis-correction to rescue the SLAM label — that mapping is imprecise (loop closure
corrects *pose*-drift under an assumed-static map; hysteresis is a *non-stationary map* problem,
a structurally different thing) and a careful technical reader (Havoutis specifically) will catch
the mismatch.

**Evidence status for the core mechanism — resolved, see Section 6b for the full account.** A
6-seed active-vs-raster belief-covariance comparison, independently re-run against this
project's own current code after an earlier version of the result was found to be
substantially inflated by the since-fixed discrete-state collapse bug (Section 4d), confirms a
real, more modest effect: active exploration reduces belief uncertainty faster than a raster
baseline, consistently across all 6 seeds (mean gap ≈10 in normalized log-covariance units,
versus the original, inflated claim's mean gap of ≈25). This is genuine, independently verified
evidence for the "active beats blind exploration" half of Claim 1 — cite the corrected numbers,
not the original ones.

**Claim 2 (supporting):** joint modeling of the cross-DQD coupling term outperforms a
naive-decoupled baseline (run the single-DQD ground-zero method twice, ignoring the coupling
term) — quantifies that coupling-awareness was worth the added complexity, with a coupling-
strength sweep including a **zero-coupling control** (joint method should gracefully reduce to
matching naive-decoupled performance there).

**Claim 3 (supporting):** the two-dial (BALD + FIM) acquisition design avoids the `O(N³)` GP
wall documented empirically in QADAPT (De Nicolo, Marchand, Carlsson, Vaidhyanathan, Ares,
arXiv:2607.09422, Oxford/Ares group) at exactly 4 dots — a citable, quantitative point against
a specific, verified, very recent paper from the same lab whose simulator (QArray) this project
also uses.

**Explicitly demoted from headline status:** the phase-switch trigger (Section 5) is a solid
methods detail, not a claim — it answers a question posed by *this* architecture, not literally
Natalia Ares's stated GP-over-raw-current framing (see Section 8, expert input).

**Calibrate ambition accordingly** — Natalia Ares stated directly that a working, published
ground-up result took her PhD student their entire PhD. This is a feasibility demonstration,
not a full pipeline.

---

## 2. What is explicitly OUT of scope, and why (don't reintroduce these)

- **Hysteresis / charge-latching / loop-closure / factor-graph SLAM.** Real physical effect,
  but not generated by any of the three simulators actually in use (QArray, QDarts, QDFlow all
  assume static electrostatics). Modeling it would mean solving a problem the simulation doesn't
  pose. Flagged as future work. A GraphSLAM/factor-graph implementation is also a materially
  heavier engineering lift than what's designed here (comparable to QADAPT's own implementation
  scale). **Update:** QArray+ (arXiv:2609.02736, Sept 2026 — see Section 3) can simulate a
  *related but physically distinct* history-dependent effect (non-equilibrium charge latching
  from slow tunnel rates vs. measurement rate, via a stochastic Markov-jump model) — not the
  same mechanism as oxide-charge-trap hysteresis discussed here, but a genuine, now-simulable
  source of scan-history-dependence. This does **not** retroactively reopen scope — the decision
  to exclude hysteresis stands, made deliberately (Natalia's PhD-timescale calibration warning),
  not simply because no tool could produce it. Revisit only as a deliberate scope decision if
  QArray+ becomes usable and timeline allows, not automatically because the capability now exists.
- **GP/BO for action selection.** Explicitly rejected, not merely deprioritized. The parent
  pipeline's `bayesian_opt.py` (`MultiResBO`, GP-UCB) is exactly the method class QADAPT found
  degrades catastrophically beyond 4 dots due to cubic GP complexity in accumulated
  observations. This project's whole justification (ground-zero, no assumed scale) is in
  tension with GP/BO's implicit justification (no physics-informed belief yet) — the tension
  resolves in favor of the physics-informed belief once you're building one on purpose.
- **A unified/generalized-BALD acquisition metric** that folds continuous-parameter uncertainty
  into one entropy term via marginalization. Tried and rejected: entropy of a mixture of
  discrete-labeled Gaussians has no closed form, forcing nested Monte Carlo (particles ×
  continuous-parameter samples) — the same intractability class GP-based EIG methods hit, i.e.
  reintroducing the exact problem the two-dial design exists to avoid. (Real alternative for
  later: amortized/neural EIG estimators, e.g. Deep Adaptive Design-style — not this paper.)
- **Shared / global Kalman filter** across particles for continuous device parameters. Rejected
  because of genuine data-association ambiguity: the same observed transition implies a
  different correction to `E_c`/`t_c`/cross-capacitance depending on which discrete occupation
  cell a particle believes it's near. A shared filter would average updates computed under
  contradictory discrete assumptions, contaminating good hypotheses. **Independent per-particle
  Kalman filters are required** (see Section 4).
- **Virtual-gate compensation** to decouple cross-talk between the two DQDs. Deliberately not
  built. This means DQD-2's navigation is allowed to physically perturb DQD-1's occupation via
  the cross-term — this is in scope *to happen*, not something to prevent. See Section 6.
  **Confirmed visually and numerically:** with DQD-B's occupation held fixed, DQD-A's entire
  charge-stability diagram shifts by a uniform, rigid translation across 100% of a swept 2D
  plane — expected, not a red flag: with the other DQD's occupation fixed, the cross-capacitance
  term is linear in that fixed occupation number, so it acts as a constant additive offset to
  DQD-A's effective chemical potential everywhere in the sweep. A perfectly rigid shift is the
  correct signature of a properly-wired bilinear cross-term, confirming Section 4b's capacitance
  recipe is behaving as designed for this specific case.
- **The parent pipeline's 6-stage state machine, DQC gatekeeper, HITL manager, governance
  logging, LLM narrator.** Confirmed via direct code read (not just inherited from the primer)
  that these are tightly coupled to a different phase's assumptions (fine-tuning, assumed-known
  scale, proportional window sizing). Do not import.
- **Any closed-form target-location formula without independently verifying sign/magnitude
  first.** Cautionary precedent: an earlier session used `V_centre = -E_c_mean/lever_arm`
  (wrong sign) to approximate the `(1,1)` cell center; correct is `+E_c_mean/lever_arm`. For
  `E_c1=2.5, E_c2=2.7, t_c=0.3, lever_arm=0.65`, true center ≈ `(+3.76, +3.85)`; the wrong-sign
  formula gave `(-4.00, -4.00)` — ~7.8V error, ~2 full periods off. A ground-zero method must
  never assume a closed-form target location — it has to earn its scale estimate from
  measurement.

---

## 3. Simulator choice

**Primary: QArray** (`pip install qarray`, package `qarray`, class `qarray.DotArray`).
Chosen because it's the one candidate simulator confirmed to satisfy both properties a
ground-zero method needs:
- **Point-queryable at arbitrary voltage locations**, not grid-locked: `DotArray.ground_state_open(vg)`
  takes a `(..., n_gate)` array (single point or batch) and returns `(..., n_dot)` ground-state
  occupations. No pre-generated grid required.
- **Does not require specifying max charge carriers a priori** (`max_charge_carriers=None`) —
  the one property among all candidates most directly consistent with "no assumed scale."
- Peer-reviewed (arXiv:2404.04994, van Straaten, Hickie, Schorling, Schuff, Fedele, Ares —
  Natalia Ares's own group), GPU/Rust-backed, constant-capacitance model (same physics family
  as the parent pipeline's `ConstantInteractionDevice`, so directly comparable, just validated
  and without the sign-bug risk documented in Section 2).

**Secondary / cross-check: QDarts.** Extends CIM with finite tunnel coupling (`t_c`) and
non-constant charging energy via a many-body Hamiltonian; also natively point-queryable
(polytope-finding algorithm handles arbitrary voltage combinations). Good mid-fidelity
validation layer. **Important correction (see Section 4a):** `t_c` is *not* actually in
QArray's ground-state formula — QArray's `F(n,vg)` is a pure bilinear quadratic form in `n`,
structurally incapable of representing the discrete, occupation-conditional bonding correction
tunnel coupling produces. QDarts's Hamiltonian formulation is the only one of the three primary
simulators that can represent `t_c` at all; this is *why* QDarts exists as a named cross-check
tier, not an optional nicety.

**Tracked for possible future adoption, not yet usable: QArray+** (van Straaten, Ares,
Marchand, De Nicolo, et al., arXiv:2609.02736, published Sept 2 2026 — same lab as QArray and
QADAPT). Extends QArray with a proper charge-basis Hamiltonian for tunnel coupling (structurally
resolves the `t_c` non-identifiability problem *within the QArray family*, no QDarts hand-off
needed), gate-voltage-dependent capacitances, a stochastic latching model (see Section 2), and
an open-quantum-system (Lindblad) formulation unifying both. GPU/JAX, scales to 64 dots. **As of
this check, no installable package (`qarray+`/`qarray-plus`) exists on PyPI or GitHub** — the
paper is two days old at time of writing. Do not switch simulators now. Worth re-checking for
public release before the writing phase; if available by then, it's a strong, specific
limitations/future-work citation for the `t_c`-drop decision below, not a reason to redo the
current architecture.

**Top-tier validation, small subset only: QTCAD and/or QDFlow.** QTCAD = full
Poisson-Schrödinger device electrostatics (commercial/licensed, slow per-query — not for the
inner loop). QDFlow = self-consistent Thomas-Fermi solver, physically deeper (dynamic
capacitance tied to actual charge density) but per-query expensive (self-consistent solve) —
better suited to mass-producing labeled synthetic data than serving as a cheap oracle inside a
tight active-learning loop.

**Explicitly ruled out for the inner loop: SimCATS.** Confirmed via QDarts's own paper: SimCATS
is a geometric (Bézier-curve) CSD generator built for pre-defined 2D sweeps, explicitly *not*
suited to arbitrary-voltage-combination queries. Architecturally wrong for ground-zero.

**Terminology trap to avoid:** NeuroQD's "cold-start" means simulating from device physical
turn-on (pinch-off → Coulomb peaks → charge stability), **not** "algorithm starts with zero
prior knowledge of device parameters." Different meaning than this project's "ground-zero" —
don't conflate them when reading related work.

---

## 4. Belief architecture: Rao-Blackwellized particle filter (FastSLAM-style)

- **Discrete part:** particle filter over joint occupation state `(n1, n2, n3, n4)` — genuinely
  multimodal, no meaningful gradient, entropy/BALD-style IG is the right tool.
- **Continuous part:** each particle carries its **own independent** Kalman filter over
  **5 continuous device parameters** — `E_c1, E_c2, α1 (lever arm 1), α2 (lever arm 2)`, and
  the **cross-capacitance term** between the two DQDs (`t_c1`/`t_c2` dropped — see 4a).
  Independent per particle, *not* shared (see Section 2 for why). This closes a real gap versus
  the parent pipeline: there, `CIMObservationModel` takes `device_params` as a fixed input,
  never estimated — here it's estimated live, per-hypothesis.
- **Why FastSLAM structure specifically:** textbook Rao-Blackwellized particle filtering
  (Montemerlo et al. 2002) — discrete states via particles, continuous states via per-particle
  Kalman filter, arrived at independently from the requirements (Step 1 of the acquisition
  redesign), not inherited from the parent repo. Structurally resembles QADAPT's Kalman-filtered
  Φ cross-capacitance matrix, but differs: QADAPT runs one global Kalman filter feeding
  decoupled per-agent control; this design runs one Kalman filter *per particle*, feeding a
  still-fully-joint belief.

**Joint, not decoupled, ground-zero.** The physics forces this: if DQDs are genuinely
capacitively coupled, DQD-2's occupation shifts where DQD-1's degeneracy lines sit (an
`E_cm·n1·n2`-style cross-term, same structural form as the existing intra-DQD interaction term
in the CIM formula). A particle filter tracking only `(n1,n2)` and ignoring DQD-2 is running
inference against a physics model missing a real term in the generative process — not a
simplification, a model mismatch that gets worse (not just slower) as coupling strength grows.

### 4a. `t_c` dropped from the tracked state — decided, not deferred

**Finding:** QArray's ground-state solver computes `F(n, vg) = (n − Cgd·vg)ᵀ·Cdd_inv·(n − Cgd·vg)`
— a pure bilinear quadratic form in `n`. Quantum tunnel coupling, per the parent pipeline's own
original physics (`ground_state_energy = chemical_potential − t_c`, **only if** `n1==1 and
n2==1`), is a discrete, occupation-conditional correction — not expressible as any choice of
`Cdd`/`Cgd`, no matter how tuned. This is a missing mechanism, not a units or convention issue.

**Decision (adopted):** drop `t_c1`/`t_c2` from the tracked continuous-parameter vector while
QArray is the primary simulator. Rename the intra-DQD off-diagonal `Cdd` term to `E_cm,intra`
(classical mutual charging energy — the physically real, QArray-representable quantity that was
being conflated with tunnel coupling). Tracked vector is now **5-dimensional**:
`(E_c1, E_c2, α1, α2, cross_capacitance)`. If `t_c` estimation becomes necessary later, either
add it back as QDarts-only (frozen/unobserved under QArray particles) or revisit once QArray+
(Section 3) is public and usable.

### 4b. Capacitance-matrix construction: `(E_matrix, α_matrix) → (Cdd, Cgd)`

QArray takes capacitances, not energies — every continuous parameter above needs mapping to
`Cdd`/`Cgd` before it can drive a simulated measurement. **Verified directly against QArray
source** (`_helper_functions.py`'s `convert_to_maxwell`, `python_implementations/helper_functions.py`'s
`free_energy`) — not re-derived on paper and trusted blindly, after an earlier hand-derived
version (which claimed a factor of 2) was checked empirically and found wrong:

```
E_matrix = Cdd_maxwell⁻¹                      # NOT 2·Cdd_inv — verified against the textbook
                                                 # single-electron-box result: 0→1 transition
                                                 # lands at E_c/(2·α) under this convention,
                                                 # matching exactly with no factor of 2.
```

Construction recipe, given a target `E_matrix` (diagonal = `E_c` per dot, off-diagonal =
`E_cm,intra`/cross-capacitance) and `α_matrix` (gate→dot lever arms, off-diagonal = deliberate
cross-gate leakage if any — not an arbitrary fixed fraction):

```python
def build_capacitance_matrices(E_matrix, alpha_matrix):
    Cdd_maxwell = np.linalg.inv(E_matrix)          # no factor of 2
    Cgd_raw = Cdd_maxwell @ alpha_matrix
    cgd_sum = Cgd_raw.sum(axis=1)

    Cdd_raw = -Cdd_maxwell.copy()
    np.fill_diagonal(Cdd_raw, 0)
    off_diag_row_sum = Cdd_raw.sum(axis=1)
    diag = np.diag(Cdd_maxwell) - cgd_sum - off_diag_row_sum
    np.fill_diagonal(Cdd_raw, diag)
    return Cdd_raw, Cgd_raw
```

**A real feasibility constraint exists — verified empirically, not just in theory.** `Cdd_raw`'s
diagonal can go negative for some `(E_matrix, α_matrix)` combinations, and QArray's constructor
rejects negative-diagonal input (`PositiveValuedMatrix` check → `ValueError`). This is not an
edge case: **confirmed structural** — lever arm and charging energy are not independent knobs,
because the gate is itself part of a dot's total capacitance (`E_c ~ 1/C_total`, gate included).
`α` needs real headroom below 1, with margin left over for coupling terms. **The illustrative
values used throughout this project's early discussion (`E_c≈2.5`, `α≈1.0`) are confirmed
infeasible** — at `α=1.0` with realistic coupling already in use, max achievable `E_c≈0.81`, not
2.5. Root cause: `α=1.0` was picked in the Section 9 benchmark script purely to make
`ground_state_open` return *something* fast, never claimed as calibrated, and quietly propagated
into every "ground truth" default downstream.

**Required guardrail, not optional** — add before accepting *any* new `(E_matrix, α_matrix)`
pair as ground truth, particle-initialization draw (Section 7), or coupling-sweep point
(Claim 2):

```python
def assert_feasible(E_matrix, alpha_matrix, margin=0.1):
    M = np.linalg.inv(E_matrix)
    diag = M @ (1 - alpha_matrix.sum(axis=1))
    if (diag < margin).any():
        raise ValueError(f"Infeasible: alpha too large relative to E_c/coupling for dots {np.where(diag<margin)[0]}")
```

**RESOLVED — real 90×90 grid sweep run against actual `QArrayEnv` construction, confirming by
measurement rather than formula:**
- The closed-form estimate above checks out closely: real sweep gives max `E_c≈0.757` at
  `α=1.0` (vs. the `~0.81` estimated analytically) — same conclusion, consistent numbers.
- **Final ground-truth values, locked in:** `α≈0.05–0.075`, `E_c≈2.0–2.2` — confirmed to sit
  with real margin (`~0.144`, minimum diagonal `Cdd` entry, comfortably above the `margin=0.1`
  guardrail). Use these going forward, not the originally-illustrative `E_c≈2.5, α≈1.0` values.
- **Secondary finding:** QArray's own condition-number check emits a warning that
  `algorithm='default'` isn't recommended at this device's coupling strength. The suggested
  alternative, `algorithm='brute_force'`, requires `max_charge_carriers` to be set explicitly —
  conflicting with Section 3's entire reason for choosing QArray — so switching wholesale isn't
  a real option. A 500-point spot check found zero disagreement between `default` and
  `brute_force`. **Not yet checked: 3+-state near-degenerate points (triple points)** — worth
  one targeted check before treating this warning as fully resolved.

### 4c. Observability: `E_c` is only learnable near interdot transitions — verified, load-bearing

**Finding, confirmed empirically and matching the textbook result exactly:** a same-dot
transition's addition voltage is `V_add = e/C_g` — independent of `E_c`. Differentiating the
model's occupation prediction with respect to `E_c` at a single-dot transition gives **exactly
zero** sensitivity; at an interdot/multi-dot boundary, sensitivity is real and nonzero.

**This is a hard constraint on the acquisition design, not a footnote:** `IG_FIM`'s candidate
search must actually reach interdot boundaries to learn `E_c` at all.

**Analytic Jacobian rewrite — DONE, validated directly against the real production (rust)
backend, not just derived on paper.** Implemented via `jax.jacobian` chained through QArray's
own JAX ground-state code (confirmed cleanly differentiable, no rewrite needed there) and a
log-space-reparameterized `jaxopt.LevenbergMarquardt` solve for the self-capacitance step
(guarantees positivity by construction — a naive `GaussNewton` attempt on the raw variables
converged to a spurious negative-capacitance root on the first real test). Getting this correct
required finding and fixing three separate bugs along the way, each caught by checking against
QArray's real behavior rather than trusting the derivation: (1) skipping the raw-to-Maxwell
capacitance conversion QArray applies internally, (2) a wrong sign convention on `Cgd`, (3)
forgetting QArray internally rescales `T` by the Boltzmann constant before use — the last one
being how the Section 4e temperature bug below was actually found. Final version matches the
rust backend exactly on every test point checked. See `qarray_jax.py`.

**Confirms the original motivation directly, not just in principle:** at a real point this
project's own acquisition search had used, the old adaptive-FD Jacobian returned three exactly-
zero rows and one absurd `8042`-magnitude spike in the fourth, while the analytic Jacobian gives
smooth, consistent values (`~1e-4` to `~1e-5`) across all four rows at the same point.

### 4d. Discrete occupation state: diagnosis confirmed, fix reconciled and validated — no longer blocking

**Diagnosis, unchanged and independently reconfirmed:** `OccupationParticle` had no discrete
label; occupation was computed on the fly as a deterministic function of each particle's
continuous parameters and voltage. Under readout noise `σ=0.05`, two particles predicting
categorically different occupations at the same voltage could differ by a residual costing
`1²/(2·0.0025) = 200` nats of log-likelihood — enough to collapse weights to `[1,0,0,...]`
after a single measurement. Confirmed directly: `min_ess = 1.0` in 15/15 sweep runs, mean
parameter error flat across particle counts (10/20/40 → 20.1%/17.2%/19.4%, no trend).

**A competing hypothesis was tested and refuted first, worth keeping as a documented negative
result so it isn't re-tried.** Before finding the real mechanism, a divergent run (`E_c1`
collapsing to ~0.02 against a true value of 2.1) was suspected to be driven by the particle
walking up against the `assert_feasible` capacitance-feasibility boundary. Traced directly via
`min_c_self` at every step of the divergent particle's lineage: it stayed at 20–56 throughout,
nowhere near the ~0.1 margin. **Refuted, not just unconfirmed.**

**Actual mechanism, found via GC-safe particle-lineage tracing** (an `id()`-keyed trace
initially gave a spurious result — Python reused a garbage-collected object's memory address,
making it look like one particle's `mu` jumped impossibly between steps; fixed by storing a
persistent trace ID on each object instead of trusting `id()`): a diffuse ground-zero prior,
evaluated against the very first real measurement, collapses to one surviving lineage almost
immediately — confirmed directly, every original particle received exactly one `update()` call
before the first resample discarded all of them. Everything after that is one trajectory's
random walk (plus cosmetic roughening jitter on resample) replicated across the ensemble, not
real diversity — which is how a 40-particle ensemble can end up unanimously, confidently wrong
without ESS ever looking degenerate by the end of a run.

**Reconciled fix — a lighter-weight implementation of the same underlying repair the original
diagnosis called for, not a different fix competing with it.** The root cause named in the
original diagnosis was "no mechanism for a particle to get credit across nearby discrete
hypotheses." Two changes address exactly that, using machinery already built for Section 5a's
`IG_BALD` joint-entropy fix rather than a new persistent schema field:

1. **Mixture likelihood over local discrete candidates.** `local_candidate_states` (the
   Section 5a machinery, extracted into a shared `local_candidates.py` so `particle_filter.py`
   can reuse it without a circular import with `bald.py`) enumerates the small local set of
   discrete occupation hypotheses each particle's continuous parameters are actually uncertain
   between, weights them by `softmax(-F/T)`, and scores the particle against that full mixture
   rather than a single point prediction. A particle just barely on the wrong side of a
   razor-thin transition boundary now gets partial credit instead of a catastrophic 200-nat
   penalty.
2. **Adaptive ESS-targeted likelihood tempering** (standard SMC-sampler technique, not a
   bespoke patch): per update, find the largest tempering exponent `β ≤ 1` via bisection such
   that the resulting ensemble ESS doesn't drop below a target fraction (`0.5`) of particle
   count. This was necessary *in addition to* the mixture likelihood — the mixture fix alone
   still let a maximally diffuse prior collapse on its very first real measurement, since
   softening helps a particle near *its own* boundary but not a particle whose entire predicted
   occupation pattern is simply wrong. Tempering affects only the weight update, not each
   particle's own Kalman mean/covariance update.

**Both were necessary; neither alone was sufficient** — confirmed by testing incrementally
(mixture likelihood alone still showed `min_ess=1.0` in 15/15 runs; mixture + tempering fixed
it) rather than assumed.

**Validated directly against a real 15-run multi-seed sweep** (10/20/40 particles × 5 seeds,
`stability_sweep.py`, raw results in `sweep_results_tempered.jsonl`):
- `ESS/n = 0.501` in every single run — hitting the tempering target exactly, not by luck.
- No more catastrophic single-seed divergence (the worst pre-fix case, 172% error on one
  parameter, is now a real ensemble with the same seed/settings recovering to reasonable
  values).
- For the first time this session, particle count behaves the way theory says it should under
  a fixed measurement budget: mean error 30.4% → 19.9% → 20.8% going 10 → 20 → 40 particles,
  with the flattening past 20 consistent with a fixed-information-budget ceiling rather than a
  filter artifact.

**Correction to a previous primer draft's specific wording:** it stated *"Standard mitigations
(tempered likelihood, inflated observation covariance, resample-move rejuvenation) would mask
the collapse, not fix it — agreed these are not the right move here."* That blanket claim is
now known false as written — it was a judgment call carried forward as if it were a proven
impossibility result. Tempering, specifically, demonstrably fixes the diagnosed mechanism on
real data. The corrected framing: a persistent discrete-state schema field *may still be worth
building eventually* as a more explicit, more directly inspectable architecture — but it is not
currently required to close the practical gap, since the lighter-weight fix already does,
verified.

**Status: downgraded from "highest-priority blocking item" to "resolved, verified."** The
persistent-discrete-state schema redesign moves to the open-items list as a possible future
architectural cleanup, not a blocker.

**Verification path, for anyone who wants to check this directly rather than trust this
summary:** `particle_filter.py` (mixture likelihood + `_find_tempering_beta`),
`local_candidates.py` (shared candidate enumeration), `coarse_sweep.py` (Phase-0 resampling,
which needed the same fix for the same reason), `run_active_slam.py` (wiring),
`stability_sweep.py` (the sweep script), `sweep_results_tempered.jsonl` (raw per-run output,
15 lines).

### 4e. Temperature convention — corrected, and it retroactively affects prior visualization magnitudes

**Bug found and fixed, empirically derived rather than picked by feel:** the only dimensionless
quantity governing thermal softening is `E_c/(kB·T)`. `E_c=2.1` was chosen purely as a
feasibility number (Section 4b) with no claim to being a literal energy in eV; `T=0.05` was
picked as if it meant 50 mK. QArray reads `E_c` as eV regardless of that intent (internally
computing `kB_T = 8.617333262145e-5 · T` before use — found while porting the Jacobian to JAX,
Section 4c), so the combination gave `E_c/(kB·T) ≈ 5×10⁵` — a real semiconductor dot sits around
`100–600` (1–5 meV over 50–300 mK). Result: the simulated device was ~2000× colder than any
physical one, producing transitions `~9×10⁻¹⁶ V` wide with only ~0.5% of candidates carrying any
gradient at all.

**Fix: `T≈100` in this project's `E_c` units** reproduces a realistic ratio. **Critical framing,
required wherever this constant appears:** this is *not* a claim the simulated device sits at
100 Kelvin — it's a correction for `E_c` itself being an arbitrary feasibility-driven number
rather than a literal eV value. The physically meaningful quantity is the *ratio*
`E_c/(kB·T)`, not either factor alone. `T` is a device property here, not a free tuning knob —
raising it further trades away the discrete charge structure the POMDP is built on (past
`T≈3000`, transitions smear across ~37% of an addition voltage).

**This means Section 4c's `T ≈ α·σ` recommendation was incomplete, not superseded.** It
correctly tied `T` to the noise floor but didn't account for `E_c`'s own convention mismatch.
The full derivation needs both steps: pick `T` so `E_c/(kB·T)` matches a realistic physical
ratio, *given whatever convention `E_c` is using* — not a single proportionality.

**Also found and fixed in the same pass:** `vg_range=(0,15)` doesn't cover a full addition-
voltage period (`V_add = 1/α ≈ 16.67V` at this project's `α` — confirmed by direct measurement,
matching the `1/α` prediction exactly) — part of the state space was unreachable by
construction.

**Retroactive consequence, real but bounded:** the earlier visualization round's severity
findings (`IG_FIM` near-zero almost everywhere except isolated single-pixel spikes; the 14-step
greedy trap) were generated at `T=0.05` — now known to be ~2000× too cold. The *qualitative*
claim (interdot-only observability, Section 4c) is real physics independent of temperature and
stands. The *magnitude* of how razor-thin the informative regions appeared may have been
exaggerated by the unphysical cold rather than being a pure property of the device. **Re-run
the landscape and trajectory visualizations at corrected `T` and `vg_range` before treating the
earlier severity numbers as final.**

---

## 5. Acquisition function: two-dial, closed-form

```
IG_total(v) = w_discrete(t) · IG_BALD(v)  +  w_continuous(t) · IG_FIM(v)
```

Evaluated over the **full joint 4D voltage space** in one shot (not two 2D spaces alternated) —
this is what avoids the alternating-navigation-doesn't-scale problem by construction, not by
looping back and forth. A query near DQD-1 that happens to be informative about the coupling
term gets credit for that jointly, in one evaluation.

- **`IG_BALD(v)`** — standard BALD over the particle mixture: `H[mixture prediction] − mean[per-particle
  entropy]`. High when hypotheses disagree about what a measurement at `v` would show; correctly
  goes to zero (not just low) when a window is uninformative for all live hypotheses. Needs an
  outer loop to relocate candidates when IG stalls (aliasing) rather than assuming IG alone
  always points somewhere useful. **Implementation trap, found and fixed (see 5a):** constructing
  the "prediction" as independent per-dot marginals and summing their entropies double-counts
  joint uncertainty near interdot transitions — use the true local joint entropy instead.
- **`IG_FIM(v)`** — Fisher-information-based, closed form, **no outcome-averaging needed** (the
  Kalman covariance update is deterministic given `v`, doesn't depend on what's actually
  observed): `IG_FIM(v) = Σᵢ wᵢ · 0.5·[logdet(Σᵢ) − logdet(Σᵢ'(v))]`, where `Σᵢ'(v)` is the
  post-update covariance from the standard Kalman covariance-update equation using the
  observation-model Jacobian `Hᵢ(v)`. **The `0.5` factor is required.**
- **Aggregation for acquisition purposes is weighted mean, not min/max** — between-particle
  disagreement in current parameter *means* is candidate-independent (doesn't vary with `v`),
  so it's a constant offset that drops out of `argmax_v` entirely. Weighted mean of
  *within*-particle terms is sufficient for choosing where to measure next.

**Candidate proposal — a real, separate gap, found and closed.** Section 5 as originally
written specifies how to *score* a candidate voltage but never how one gets *proposed* from a
literal blank start. Confirmed as a real, not hypothetical, gap by a live run: with candidates
drawn uniformly over the full 4-gate range, `IG_total` scored exactly `0.0` on 18/20 steps,
because interdot transitions occupy a thin slice of a 4D range that blind uniform sampling
essentially never lands near. **Fix, implemented: a deterministic Phase-0 bootstrap sweep**
(`coarse_sweep.py`) — per-gate 1D probes at multiple background settings, implementing Natalia
Ares's own original expert-input framing (Section 6) literally rather than assuming candidates
are already reachable. Every point is a real (noisy) measurement fed into the particle filter,
not a separate diagnostic pass. Detected transitions seed subsequent candidate proposal
(`candidate_search.py`). "Just draw more random candidates" is not a substitute — the
probability of a uniform sample landing near a lower-dimensional transition manifold shrinks
combinatorially with dimension; a deterministic sweep that crosses every gate's full range is
the only proposal method guaranteed to cross every transition line that gate participates in at
least once.

### 5a. `IG_BALD` construction: per-dot marginal sum is a real bug, not just an approximation

**The bug, confirmed against actual behavior:** constructing each particle's "prediction" as 4
independent per-dot Bernoulli marginals, then summing binary entropies across dots, **exactly
double-counts** joint uncertainty at a clean anti-correlated interdot boundary. True joint
entropy is `ln(2)`; summing two identical marginal entropies gives `2·ln(2)` — exactly double.
This is guaranteed by entropy subadditivity, and the bias is **largest exactly at interdot
boundaries** — the only place `E_c` is observable at all.

**Fix, implemented:** enumerate the small local candidate set, evaluate `F` at each,
`softmax(−F/T)`, compute the true joint entropy from that distribution directly. Cheap,
closed-form, no sampling. (This same machinery is what Section 4d's mixture-likelihood fix
reuses.)

**Consequence for `IG_BALD`'s dynamic-range bound:** the per-dot-marginal-sum construction is
bounded by `N_DOT·log(2) ≈ 2.77` nats; the joint-entropy fix changes this bound to
`log(n_states)` in the enumerated local set, not `N_DOT·log(2)` and not `log(N_particles)`
(both incorrect prior framings). Re-derive `w(t)`'s dynamic-range argument against this
corrected bound.

### 5b. Greedy single-step selection can trap on a locally-rich, globally-narrow vein

**Confirmed via a real FIM-only trajectory.** `IG_FIM` stayed genuinely nonzero for 14
consecutive steps while the policy repeatedly re-selected the same candidate, which had real
sensitivity to `α1` but exactly zero sensitivity to `E_c1` (same-dot-like point). `α1`'s error
collapsed almost immediately; `E_c1`'s error stayed flat the entire run. **Note (Section 4e):**
this trajectory was generated at the unphysically cold `T=0.05` — the qualitative finding
(greedy trapping is a real failure mode of single-step lookahead) stands independent of
temperature, but re-run at corrected `T`/`vg_range` before citing the specific 14-step figure.

**Why this matters:** a single-step-lookahead greedy policy has no structural pressure to seek
interdot boundaries once it's found *some* nonzero signal nearby. An outer relocation/
diversification mechanism needs to trigger on **information becoming narrow** (concentrated in
one parameter direction for many consecutive steps), not only on **information vanishing**
(`IG≈0`, Section 5's aliasing case) — two different triggers, still not implemented.

### Units — resolved, not an invented exchange rate

Both terms are in **nats**, exactly, provided continuous parameters are in **normalized/whitened
coordinates**:

```
IG_FIM = 0.5 · [logdet(Σ_prior,normalized) − logdet(Σ_posterior,normalized)]
```

**What normalization does *not* solve:** `IG_BALD` is bounded above (Section 5a); `IG_FIM`'s
differential-entropy term is unbounded as posterior covariance shrinks. `w(t)` still has a real
job: dynamic-range management between two commensurate-but-differently-bounded quantities.

---

## 6. Phase-switch trigger (methods detail, not a headline claim)

**Working hypothesis from expert input (Natalia Ares):** raw-signal/GP phase → feature-based
classification phase, switching once enough distinct transitions are located to *fit*
`E_c/lever_arm` from spacing (not assume it). This project's implementation of that switch:

**Dual-line hard threshold with hysteresis**, both conditions required to clear before
switching Phase 1 → Phase 2:
- **Blue line — within-particle certainty:** `Σᵢ wᵢΣᵢ` (weighted-average per-particle
  covariance, normalized coordinates).
- **Red line — between-particle agreement:** `Cov_i[μᵢ]` (covariance of particle means around
  the grand mean).

**Why both are needed:** if all particles started from an identical shared prior, each
particle's *own* covariance can shrink nicely while particles still substantially disagree with
each other about true device scale — a false "converged" signal. Law of total covariance:
`Σ_total = within + between`. Tracking them as two separate thresholds is more debuggable than
one combined number.

**Hysteresis banding**, not a knife-edge: enter Phase 2 when both lines drop below
`threshold_low`; only revert to Phase 1 if either climbs back above a higher `threshold_high`.

**Aggregation must be mass-based, not simple particle-weighted mean**: switch only once some
fraction (e.g. ≥90%) of particle weight has both normalized-trace terms below threshold.

**Threshold value derivation:** should come from the simulated sensor-noise model (Section 6a).

### 6a. Noise model and threshold derivation — final, corrected form

**Correct, final form** — distance from the mean estimate (a deterministic, positional fact),
with the covariance term correctly redeployed as noise *inflation*, not distance substitute:

```
distance_to_boundary(v*) = 0.5 − |frac(μ(v*)) − 0.5|      # from the current mean prediction
sigma_eff(v*)² = R + diag(H·Σ_total·Hᵀ)                    # sensor noise + parameter-uncertainty spread
p_flip = P(|noise| > distance_to_boundary),  noise ~ N(0, sigma_eff²)
```

**Must use `Σ_total` (within + between, Section 6), not a single particle's own `Σ`.**

**Dimensionality — flagged, not yet resolved:** per-dot vector combined via a union bound, or
collapsed to one scalar? Not yet decided.

**Threshold-derivation formula:** `Σ_floor ≈ (HᵀR⁻¹H)⁻¹` is not well-posed as a single-shot
formula (`H` is `4×5`, rank ≤4 against 5 parameters from any single query — generically
singular). **Adopted resolution: empirical calibration** — run a batch of deliberately-
informative measurements offline, read off the achieved `logdet(Σ)`, use that as
`threshold_low`'s reference scale. `threshold_low` is a function of
`(sigma, particle_count, prior_width)`, not a universal constant — recompute per experiment
configuration.

### 6b. Goal-directed policy — structural gap real; active-vs-raster claim now independently verified (with corrected numbers); goal-directed-criterion claim still unverified

**What actually checks out by inspection, independent of any run:** Section 5's acquisition
function, `IG_total(v) = w_d·IG_BALD(v) + w_c·IG_FIM(v)`, is purely information-maximizing —
nothing in that formula rewards proximity to a target occupation. Section 8's convergence
criterion presupposes goal-directed behavior (it checks belief mass *on a target* and
actuation quiescence *near that target*). This is a real, checkable-by-reading structural gap
between two sections of this primer, and stands regardless of everything below.

**A previous primer draft asserted two specific empirical claims from another session's
narrative summary with unwarranted confidence** ("Holds — validated directly," with no
corresponding script or result file traceable in this project's actual codebase at the time —
the same failure mode as a fabricated docstring quote caught earlier in this project's history,
just harder to catch because it read as careful, hedged analysis). Both have now been checked
against real, independently-supplied artifacts, with two different outcomes:

**Claim A — "6-seed active-vs-raster belief-covariance comparison" — now independently
verified, original numbers superseded.** The originating session supplied the actual script,
raw per-seed JSON traces, and a snapshot of the exact code state used, packaged specifically so
it could be checked rather than trusted (consistent with this project's own standing policy).
Checking the snapshot directly confirmed its own stated caveat: it predates the Section 4d
discrete-state fix, the Section 4c JAX Jacobian, and the Section 4e temperature correction —
`T=0.05` throughout, no mixture-likelihood/tempering machinery in `particle_filter.py`, no
JAX Jacobian module present. The comparison was then re-run directly against this project's
current code, in two stages:
- **Stage 1 — isolate just the 4d fix** (same `T=0.05`, same `vg_range` as the original run,
  tempering explicitly engaged): the dramatic gap **essentially vanished** — `0.08, 0.61, 0.74,
  0.00, -0.00, -0.00` across the 6 seeds, versus the original claim's `14.07` to `42.96`. This
  directly confirms the concern the originating session itself raised: the original result was
  substantially an artifact of the since-fixed discrete-state collapse, not a clean measurement
  of the active policy's advantage.
- **Stage 2 — fully current pipeline** (`T=100`, corrected `vg_range`, 4c/4d/4e all engaged): a
  real, positive, 6/6-consistent gap reappears — `10.12, 17.16, 6.72, 6.89, 10.84, 8.57`, mean
  ≈10. Active genuinely does beat raster under the trustworthy configuration — about 2.5×
  smaller than the original inflated claim (mean gap ≈25 → ≈10), consistent with the original
  effect being partly real signal and partly collapse artifact.

**Conclusion: Claim A holds, with corrected numbers.** Cite "active exploration reduces belief
uncertainty faster than raster, consistently across 6 seeds, mean gap ≈10 in normalized
log-covariance units, verified against the current pipeline" — not the original ≈25 figure.
Scripts and raw JSON for both stages are checked in alongside `stability_sweep.py`, same
verification standard.

**Claim B — "an attempt to build a goal-directed convergence criterion broke" — still
unverified.** This session's re-run checked only the belief-covariance comparison (Claim A).
No script, implementation attempt, or trace for the goal-directed-criterion claim has been
supplied or checked. It remains reported-by-another-session, not independently verified, and
should not be cited as settled. If the actual attempted implementation is supplied, it can be
checked the same way Claim A was.

**Process note worth keeping:** this is a case where "verify, don't debate" produced a more
interesting and more useful answer than either "the claim was right" or "the claim was wrong"
would have been — the original number was found to be part real effect, part artifact, and the
corrected, defensible number is now stronger evidence than an unchecked claim could ever be,
specifically because it survived an adversarial re-test.

**The fix proposal itself (hard-switch to MAP-navigation once Section 6's phase-switch
triggers) remains a reasonable, low-risk design sketch** — it doesn't depend on Claim B being
true, only on the structural gap (which does check out) being real. It should be treated as an
unbuilt proposal, not as validated by a test that still can't be traced to real code.

**Explicitly rejected regardless of the above: importing the parent pipeline's navigation
stage (`bayesian_opt.py`, `MultiResBO`) to "close the loop."** Two independent reasons: it's
exactly the GP/BO mechanism Claim 3 exists to avoid, and it's structurally mismatched
regardless — it consumes an already-coarsely-localized starting point, never built to take a
joint particle-filter-plus-per-particle-Kalman belief as input.

**Sequencing, updated:** the original blocker on this section (Section 4d's discrete-state fix
needing to land first, since the MAP estimate wasn't trustworthy) is resolved — 4d is now
verified fixed. This section's own blocker is now purely the unverified empirical claims above,
not a dependency on other unfinished work. If the active-vs-raster and goal-directed-criterion
results can be reproduced against real code, this section can move forward; until then, treat
it as an open design sketch, not a diagnosed-and-agreed fix.

---

## 7. Particle initialization for coupling strength

Two genuinely separate decisions — keep them separate in implementation:

**(a) Ground-truth sweep** (for Claim 2): run the algorithm across a range of *true*
cross-capacitance values, including **exactly zero as a control** (the joint method should
gracefully reduce to naive-decoupled performance there — state this explicitly as a result: "
coupling-aware costs nothing when there's no coupling"). Use QADAPT's tested coupling-strength
regime as an externally-calibrated reference for what counts as "strong" cross-talk.

**(b) Prior seeding, within any single run:** the coupling term specifically needs a
**stratified draw** across particles (some particles seeded near-zero, some moderate, some
strong coupling hypotheses) — NOT a shared diffuse Gaussian prior identical across all
particles. Coupling specifically warrants this because zero is a real, physically plausible,
interesting hypothesis boundary — unlike `E_c`, which is always positive and bounded away from
zero by construction. **Implemented:** `particle_init.py`'s `CouplingStratum`/
`init_particle_filter`.

**Methodological invariant — do not violate:** the prior scheme from (b) must be **identical
across every point in the sweep from (a)**, regardless of what true coupling value that run is
actually testing.

---

## 8. Convergence criterion under coupling

**Two failure modes, need different guards, not one combined counter:**
- **A — statistical false positive:** a noisy measurement momentarily favors `(1,1,1,1)`, next
  reading swings back. Artifact of observation noise.
- **B — genuine physical perturbation:** navigating DQD-2 shifts DQD-1's chemical potential via
  the cross-term enough to actually change its occupation.

**Dual criterion, both required, sustained for `N` consecutive steps:**
1. **Belief stability** (guards A): posterior mass on joint MAP `(1,1,1,1)` exceeds confidence
   threshold `p_conf`.
2. **Actuation quiescence** (guards B): the policy's own chosen step magnitude drops below `ε`
   on **any** gate. **Implemented:** `convergence.py`'s `ConvergenceMonitor`.

**`N` derivation:** smallest `N` such that `P(N consecutive spurious "stable" reads by chance)
< p` for some small target `p`.

**Fairness requirement for the baseline comparison:** the raster baseline must use a
**comparably rigorous stopping rule**, fed by raster-order measurements instead of IG-chosen
ones.

**Reportable, not hideable, result:** if failure-mode-B perturbation turns out severe at high
coupling strength, that's a real boundary of the method worth reporting directly.

---

## 9. Verified computational facts (QArray, benchmarked directly)

Environment: `pip install qarray --break-system-packages` (v1.6.0), `implementation='rust'`,
4-dot/4-gate `DotArray`, `max_charge_carriers=None`.

- **Single-point loop (naive, one call at a time):** ~26 µs/call.
- **Batched query, floors at batch size ≳50:** ~14.3 µs/point.
- **Per-particle capacitance-matrix swap cost:** ~270 µs/particle — dominates per-step cost,
  not the candidate query.
- **Realistic decision-step cost estimate:** 200 particles × 50 candidates ≈ 197 ms/step at
  that scale.

**RESOLVED — the "fresh QArrayEnv per particle per candidate" question, confirmed real and
fixed.** Early `bald.py`/`fim.py` implementations did reconstruct fresh per call inside the
acquisition loop — confirmed via direct code read. Fixed: `candidate_search.py`'s
`build_particle_envs` builds one `QArrayEnv` (and self-capacitance warm-start) per particle
*once per acquisition step*, reused across every candidate in that step's batch and handed
through to the subsequent belief update so it isn't solved a third time either.

**Not yet checked:** whether `implementation='jax'` does better on the swap-heavy access
pattern at higher particle counts than currently tested (10–40).

---

## 10. QADAPT (arXiv:2607.09422) — key facts to reuse, verified directly

De Nicolo, Marchand, Carlsson, Vaidhyanathan, Ares (Oxford, July 2026). Simulates using QArray.
Confirmed real by direct fetch of the paper and independently by the user having attended
Natalia Ares's presentation of these results in person.

- **Benchmarked joint GP-based Bayesian optimization** and found it "suffers greatly,"
  catastrophic slowdown beyond 4 dots — attributed to cubic complexity with accumulated
  observations. **Four dots is exactly this project's scale.**
- **Their actual architecture does not solve the joint problem jointly** — decouples agents via
  a learned factored action-space representation: a Kalman filter incrementally estimates the
  cross-capacitance matrix Φ online, then executes **decoupled** per-gate control in the
  resulting virtualized basis.
- **Ablation:** removing the Kalman-filtered virtualization drops convergence from ~93–95% to
  ~10–89% — online coupling estimation is load-bearing, not optional.
- **Not a ground-zero comparison** — QADAPT is trained via RL across many episodes with
  randomized device parameters, then zero-shot transferred. Cite as related work and as the
  source of the GP-wall benchmark; don't present as a directly comparable baseline.

---

## 11. Reference repo — what to reuse as pattern vs. what NOT to import

**Parent pipeline** (reference only, per Section 2): `C:\Users\ual-laptop\Documents\Agentic-Semiconductor-Quantum-Device-Tuning`

**This project's actual code:**
`C:\Users\ual-laptop\Documents\Agentic-Semiconductor-Quantum-Device-Tuning\qdot-slam-pomdp\src\gz_tuning\`
— `qarray_env.py`, `qarray_jax.py`, `kalman.py`, `particle_filter.py`, `local_candidates.py`,
`noise_model.py`, `particle_init.py`, `convergence.py`, `acquisition/` (`bald.py`, `fim.py`,
`two_dial.py`, `candidate_search.py`), `coarse_sweep.py`, `run_active_slam.py`,
`baselines/raster_scan.py`.

**Do NOT import:**
- `state_machine.py` — 6-stage orchestrator, HITL escalation, backtracking, DQC gatekeeper,
  governance logging. Assumes a scale-informed window from the start, exactly what ground-zero
  can't assume.
- `bayesian_opt.py` (`MultiResBO`) — GP/BO navigation, explicitly rejected (Section 2, 6b).
- `physics.py`'s `coulomb_centre()` — contains the documented wrong-sign formula.
- Any device-parameter values treated as fixed/known inputs to `CIMObservationModel` — this
  project estimates them live (Section 4).

---

## 12. Still open — carry into the coding chat, not yet resolved here

**Highest priority:**
- **Re-run Section 4c/5's landscape and trajectory visualizations** at corrected `T≈100` and
  full `vg_range` (Section 4e) — the earlier `IG_FIM`-near-zero and 14-step-trap *magnitudes*
  are provisional until regenerated; the *qualitative* interdot-observability finding stands.
- **Goal-directed policy layer (Section 6b)** — the active-vs-raster evidence blocking this is
  now resolved (independently re-verified, corrected numbers in). Still blocked on the
  goal-directed-criterion claim, which remains unverified — supply the actual attempted
  implementation to check it the same way. The structural gap and fix proposal are real and
  ready to build once that's resolved.
- **Verify Chat IV's (this session's) own claims get the same scrutiny going forward** —
  general process note, not a specific item: this round caught one fabricated quote and one
  overstated blanket claim from cross-chat narrative summaries. Any claim entering the primer
  from outside this session's own verified code/data should be marked provisional until checked
  the same way.

**Everything below is unchanged in substance:**
- **Outer relocation/diversification trigger (Sections 5, 5b)** — needed for two distinct
  failure signatures (IG→0 aliasing vs. narrow-but-nonzero greedy trapping). Not yet built.
- **Exact stratification scheme** for coupling-term particle seeding (Section 7b) — range,
  distribution shape, number of strata not yet numerically specified beyond current defaults.
- **`p_conf`/`ε` derivation** — `N`'s derivation is settled (Section 6a); `p_conf`/`ε`
  (Section 8) still need equivalent treatment.
- **Distance-to-boundary dimensionality** (Section 6a) — per-dot vector vs. single scalar, not
  yet decided.
- **`w(t)` dynamic-range schedule** (Section 5a) — needs re-derivation against the corrected
  joint-entropy bound; currently run with fixed equal weights as a placeholder.
- **`algorithm='default'` vs. `brute_force` at 3+-state near-degenerate points** (Section 4b) —
  500-point spot check passed but didn't cover triple points specifically.
- **QArray+ public release** (Section 3) — check periodically.
- **`raster_scan.py`'s actuation-quiescence signal** — likely non-functional on a real
  (non-repeated-point) grid; needs a decision between guard-A-only or a redefined guard-B
  signal.
- **Persistent discrete-state schema field** (downgraded from Section 4d, no longer blocking)
  — worth considering as a future architectural cleanup for explicit inspectability, not
  currently required.
- **Full-scale run** (200 particles, primer's original target) — not yet attempted; current
  validated runs are at 10–40 particles. JAX Jacobian rewrite (4c) makes this more tractable
  than the FD version would have been, but per-step cost at 200 particles × 50 candidates has
  not yet been re-benchmarked with the analytic Jacobian in place.
- **Claim 2's actual coupling-strength sweep and naive-decoupled baseline comparison** — not
  yet run; current work has focused on getting the single-run estimator itself trustworthy
  first.
- **Verification-hygiene:** confirm the "Nature Electronics" venue for Schuff et al.
  (Section 1) before citing it that specifically; a final pass cross-referencing this project's
  related-work list against QADAPT's own citations.

**Resolved since the last version of this primer** (changelog): the discrete-occupation-state
collapse, diagnosed *and* fixed *and* validated (4d, superseding the earlier "fix agreed, not
yet implemented" status); the feasibility-boundary hypothesis for particle divergence, tested
and refuted (4d); the analytic JAX Jacobian rewrite, done and validated against the real rust
backend (4c); the temperature-convention bug and `vg_range` coverage gap, fixed (4e); the
Section 5 candidate-proposal gap (deterministic Phase-0 sweep), fixed; the
fresh-QArrayEnv-per-candidate performance risk (Section 9), confirmed real and fixed; the
Schuff et al. citation, confirmed real (venue unconfirmed); Section 4d's blanket dismissal of
likelihood tempering, corrected after direct testing falsified it. **New this round:** Section
6b's active-vs-raster claim, independently re-verified against real supplied artifacts (script,
raw traces, code snapshot) and re-run directly against the current pipeline — original numbers
found to be ~2.5× inflated by the (now-fixed) 4d collapse bug, corrected numbers (mean gap ≈10,
6/6 seeds) now stand as genuine evidence for Claim 1. Section 6b's *other* empirical claim
(goal-directed-criterion failure) remains open — not yet supplied for checking, still marked
unverified rather than settled.

---

## 13. One process note worth carrying forward

Multiple design proposals in this project originated from consulting other chats/models.
Several were genuinely useful; others were overclaimed or structurally mismatched to this
project's actual state representation. The pattern that worked, and that this round
reconfirmed under harder conditions than before: treat proposals — and summaries of what
another session found — as hypotheses to be checked against the actual physics/math/code, not
adopted on the strength of how principled or careful they sound. This round specifically showed
that careful, hedged prose is not a safe signal to relax scrutiny on: a fabricated docstring
quote and an unverified "6-seed comparison, holds — validated directly" both made it into a
primer draft before being caught, and the second one specifically got through *because* it read
as measured analysis rather than a flashy claim. The fix each time was the same one this project
has used from the start: go read the actual file, run the actual sweep, check the actual number
— not re-reason about which summary sounds more credible.

**Worth noting the outcome isn't always "debunked" — sometimes it's "partly right, now
stronger."** When the active-vs-raster claim's originating session supplied its actual script,
raw data, and code snapshot rather than just a narrative summary, checking it directly didn't
just confirm or deny the claim — it separated a real effect from a collapse-bug artifact that
had been inflating it, and the corrected number is more defensible than the original unchecked
one could ever have been. The lesson isn't "distrust other sessions" — it's that verification
against real artifacts is what makes a claim usable at all, in either direction.
