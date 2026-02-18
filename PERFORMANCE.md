# Phase 2 Performance Guide

## TL;DR — The Right Way to Optimize

**DO NOT** blindly reduce computational budgets. Instead:

1. **Profile** to find the real bottleneck
2. **Optimize the bottleneck** (likely CIM vectorization)
3. **Validate any reductions** with ablation studies
4. **Document** empirical justification for paper

**Current status:**
- Baseline: 1000 particles, 8 MC samples
- CI fast mode: 10 trials, 20 steps (infrastructure testing only)
- Bottleneck: CIM forward model called 8000× per measurement (not vectorized)

---

## The Performance Problem

**Symptom:** Single trial takes 30 minutes in CI

**Math:**
- 8 MC samples × 1000 particles × ~100 steps × 256 CIM calls/patch = **204,800,000 forward model evaluations**
- At ~0.01ms per call (Python loop overhead), that's 34 minutes

**Root cause:** `CIMObservationModel.predicted_conductance_2d()` calls `device.current()` in a nested Python loop (256 times for 16×16 patch).

---

## The Right Fix: Vectorization

### Current Code (Slow)
```python
# belief.py line 156-158
for i, v2 in enumerate(v2_vals):
    for j, v1 in enumerate(v1_vals):
        patch[i, j] = self.device.current(v1, v2)  # Python loop + function call overhead
```

### Optimized Code (10-50× Faster)
```python
# Compute all voltage points at once with numpy
v1_grid, v2_grid = np.meshgrid(v1_vals, v2_vals)
patch = self.device.current_2d(v1_grid, v2_grid)  # Vectorized in numpy/C
```

**Why this matters:**
- Numpy operations are vectorized in C (SIMD instructions)
- Eliminates 256 Python function call overheads per patch
- Better CPU cache utilization
- Expected speedup: 10-50× for the forward model step

### Implementation Roadmap

1. Add `ConstantInteractionDevice.current_2d()` method (vectorized)
2. Update `CIMObservationModel.predicted_conductance_2d()` to use it
3. Benchmark: measure speedup on single trial
4. Validate: run ablation to confirm no accuracy loss

**This is the proper optimization** — improve efficiency without sacrificing scientific accuracy.

---

## Only After Vectorization: Consider Budget Reductions

If vectorization isn't enough, run ablations:

```bash
python experiments/ablation_phase2.py --n-trials 20
```

This tests:
- **baseline:** 1000 particles, 8 MC samples
- **reduced_particles:** 500 particles, 8 MC samples
- **reduced_mc:** 1000 particles, 4 MC samples
- **both_reduced:** 500 particles, 4 MC samples

Output:
```
Config               Success%   Reduction%   Duration(s)  Speedup
----------------------------------------------------------------------
baseline             90.0%      52.3% ± 3.1% 120.5 ± 12.3 -
reduced_particles    88.5%      51.1% ± 3.4%  65.2 ± 8.1  1.85x
reduced_mc           89.0%      50.8% ± 3.5%  68.1 ± 9.2  1.77x
both_reduced         86.5%      48.9% ± 4.1%  35.4 ± 6.7  3.40x

KEY FINDINGS:
reduced_particles:
  ✗ Performance differs from baseline (Δsuccess=1.5%, Δreduction=1.2%)
    → Not recommended despite 1.85× speedup
```

**Accept reductions only if:**
- Success rate Δ < 5%
- Measurement reduction Δ < 5%
- Documented in paper methods section

---

## Computational Bottlenecks

Phase 2 introduces several compute-intensive operations:

### 1. Particle Filter (BeliefUpdater)
**Cost:** O(n_particles × n_measurements)

Each measurement update:
- Computes likelihood for each particle (CIM forward model)
- Resamples when effective sample size drops below threshold
- Syncs to `belief.charge_probs` for other components

**Default:** 500 particles
**Trade-off:** 
- 100 particles: Fast but coarse uncertainty estimates
- 500 particles: Good balance (CI default)
- 1000 particles: High accuracy for critical experiments
- 2000+ particles: Overkill for most cases

### 2. Active Sensing Monte Carlo (ActiveSensingPolicy)
**Cost:** O(n_mc_samples × n_particles × n_candidate_plans)

Each sensing decision:
- Samples n_mc_samples hypothetical measurements
- For each sample, updates a copy of the particle filter
- Estimates information gain for each candidate plan
- Typically evaluates 3-5 candidate plans per decision

**Default:** 4 MC samples
**Trade-off:**
- 2 samples: Very rough IG estimates, fast
- 4 samples: Reasonable estimates (CI default)
- 8 samples: Good estimates (production)
- 16+ samples: Diminishing returns

**Combined cost:** 4 MC × 500 particles × 4 plans = 8,000 forward model evaluations per measurement selection

### 3. Bayesian Optimization (MultiResBO)
**Cost:** O(n_bo_history²) for GP fitting

Each BO proposal:
- Fits a Gaussian Process on growing BO history
- Optimizes acquisition function (UCB) over voltage space

**Grows over time:** More expensive as experiment progresses

### 4. CIM Forward Model
**Cost:** O(1) per call, but called repeatedly

The CIM simulator computes conductance at each voltage point:
- Chemical potential calculation (depends on charge state)
- Fermi-Dirac statistics
- Tunneling current formula

**Not a bottleneck** for single calls, but becomes significant when multiplied by particle filter and MC sampling.

---

## Profiling

### Quick Profile
```bash
python experiments/benchmark_phase2.py \
  --fast \
  --profile \
  --skip-missing-checkpoints
```

This will use Python's cProfile and print the top 20 slowest functions.

### Detailed Profile
```python
import cProfile
import pstats

from qdot.agent.executive import ExecutiveAgent
# ... setup state, adapter, etc.

profiler = cProfile.Profile()
profiler.enable()

summary = agent.run()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(50)
```

### Expected Hot Spots
Based on computational complexity:
1. `_ParticleSet.update()` - particle filter updates
2. `ActiveSensingPolicy._estimate_information_gain()` - MC sampling
3. `CIMObservationModel.log_likelihood_2d()` - forward model
4. `GaussianProcess.fit()` - GP kernel matrix inversion
5. `MultiResBO.propose()` - acquisition optimization

---

## Tuning Guidelines

### For CI (Fast Turnaround)
```python
# Already configured in benchmark_phase2.py --fast
# 10 trials, 20 steps, 512 measurement budget
# Runtime: 10-15 minutes
```

### For Development (Moderate Accuracy)
```python
agent = ExecutiveAgent(
    state=state,
    adapter=adapter,
    max_steps=50,
    measurement_budget=1024,
)
# BeliefUpdater uses 500 particles (default)
# ActiveSensingPolicy uses 4 MC samples (default)
# Runtime: ~20-30 minutes for 10 trials
```

### For Production (High Accuracy)
```python
# Create custom components with higher budgets
belief_updater = BeliefUpdater(
    belief=state.belief,
    n_particles=1000,  # 2x particles
)
sensing_policy = ActiveSensingPolicy(
    n_mc_samples=8,  # 2x MC samples
)

# Inject into ExecutiveAgent (Phase 3 feature)
# For Phase 2, edit the defaults in the source files
```

### For Benchmarking
```bash
# Full 100-trial evaluation with trained models
python experiments/benchmark_phase2.py \
  --n-trials 100 \
  --budget 2048 \
  --max-steps 100
# Runtime: 2-4 hours (depends on hardware)
```

---

## Performance Expectations

### Single Trial Timing (Intel i7, 8 cores)
- Bootstrap: ~5 seconds (line scan)
- Coarse Survey: ~30 seconds (coarse 2D scan)
- Charge ID: ~20 seconds (local patch + classification)
- Navigation: ~10 seconds per voltage move (BO + belief update)
- Verification: ~30 seconds (repeated measurements)

**Total per trial:** 1-3 minutes on average (depends on backtracking)

### 100-Trial Benchmark
- **Fast mode (CI):** 10 trials × 20 steps = 10-15 minutes
- **Full mode:** 100 trials × 100 steps = 2-4 hours

### Scaling Factors
- **Particles:** Linear scaling (2x particles = 2x runtime)
- **MC samples:** Linear scaling (2x samples = 2x runtime for sensing)
- **Step count:** Linear scaling (2x steps = 2x runtime)
- **BO history:** Quadratic scaling (2x history = 4x GP fit time)

---

## Optimization Strategies

### If BeliefUpdater is the Bottleneck
1. Reduce `n_particles` to 250-300
2. Increase `resample_threshold` to avoid frequent resampling
3. Use a coarser voltage grid for likelihood evaluation
4. Cache CIM forward model results for repeated voltage points

### If ActiveSensingPolicy is the Bottleneck
1. Reduce `n_mc_samples` to 2-3
2. Reduce the number of candidate plans considered
3. Skip active sensing for certain stages (e.g., bootstrap always uses line scan)
4. Use a heuristic policy (e.g., always take coarse 2D in survey stage)

### If BO is the Bottleneck
1. Limit BO history to last N points (e.g., 50 points)
2. Use a sparse GP approximation
3. Use simpler acquisition (e.g., probability of improvement vs UCB)
4. Skip BO optimization and use greedy search

### Parallelization (Future Work)
- Particle filter updates are embarrassingly parallel
- MC sampling can be parallelized across samples
- Multiple trials in benchmark can run in parallel

**Not implemented in Phase 2** - requires careful handling of NumPy random state and PyTorch device placement.

---

## Debugging Slow Runs

### Check if Agent is Stuck
```python
# Add verbose logging to ExecutiveAgent._step()
if self.state.step % 10 == 0:
    print(f"Step {self.state.step}: stage={self.state.stage}, "
          f"measurements={self.state.total_measurements}")
```

### Common Causes of Slowdown
1. **Backtracking loop:** State machine gets stuck retrying failed stages
2. **Low-quality measurements:** DQC repeatedly rejects measurements
3. **Poor BO convergence:** BO proposals don't improve, agent exhausts step budget
4. **HITL blocking:** HITL not in test mode, waiting for human input
5. **Excessive logging:** Governance logger writing large decision objects

### Quick Diagnosis
```bash
# Run with verbose output
python -u experiments/benchmark_phase2.py --fast 2>&1 | tee benchmark.log

# Check for repeated stage names (stuck in backtracking)
grep "stage=" benchmark.log | tail -50

# Check measurement count vs step count (efficiency)
grep "meas" benchmark.log | tail -20
```

---

## When to Profile

**Profile when:**
- CI timeout despite --fast mode
- Single trial takes >5 minutes
- Benchmark takes >1 hour for 10 trials
- Memory usage grows unbounded

**Don't profile when:**
- Runs complete successfully in expected time
- Small variance across trials (<2x)
- Just need to reduce accuracy for faster turnaround (adjust budgets directly)

---

## Summary

**The bottleneck is particle filter × MC sampling = 4 samples × 500 particles = 2000 forward model evaluations per measurement decision.**

**For CI:** Use --fast mode (10 trials, 20 steps, 4 MC, 500 particles) → 10-15 min
**For production:** Use full mode after training Phase 1 models → 2-4 hours
**For profiling:** Add --profile flag and check hot spots in cProfile output
