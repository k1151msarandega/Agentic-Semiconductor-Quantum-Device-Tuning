## Phase 0/1/2 Integration Guide

This document explains how the three phases of the agentic quantum dot tuning system integrate together.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                      EXECUTIVE AGENT (Phase 2)                       │
│                    Main orchestration loop (Fig. 2)                  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                  ┌─────────────────┼─────────────────┐
                  │                 │                 │
                  ▼                 ▼                 ▼
         ┌────────────────┐  ┌────────────┐  ┌─────────────┐
         │  PERCEPTION    │  │  PLANNING  │  │  HARDWARE   │
         │   (Phase 1)    │  │ (Phase 2)  │  │  (Phase 0)  │
         └────────────────┘  └────────────┘  └─────────────┘
         │                   │              │
         │ - DQCGatekeeper   │ - BeliefUpdater      │ - DeviceAdapter
         │ - InspectionAgent │ - ActiveSensingPolicy│ - SafetyCritic
         │ - EnsembleCNN     │ - MultiResBO         │ - HITLManager
         │ - OOD Detector    │ - StateMachine       │ - GovernanceLogger
         │                   │ - TranslationAgent   │
         └───────────────────┴──────────────────────┴─────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  EXPERIMENT STATE   │
                         │     (Phase 0)       │
                         │  Single source of   │
                         │       truth         │
                         └─────────────────────┘
```

---

## Data Flow: One Complete Loop

### 1. Active Sensing (Phase 2 Planning)
**ActiveSensingPolicy.select()** examines `ExperimentState.belief` and proposes a measurement:
```python
plan = sensing_policy.select(state.belief, v1_range, v2_range)
# → MeasurementPlan(modality=COARSE_2D, v1_range=(-0.5, 0.5), ...)
```

### 2. Translation (Phase 2 Agent)
**TranslationAgent.execute()** converts the plan to a DeviceAdapter call:
```python
result = translator.execute(plan)
measurement = result.measurement  # Measurement from CIMSimulatorAdapter
state.add_measurement(measurement)
```

### 3. Data Quality Check (Phase 1 Perception)
**DQCGatekeeper.assess()** runs physics-based quality checks:
```python
dqc_result = dqc.assess(measurement)
state.add_dqc_result(dqc_result)

if dqc_result.quality == DQCQuality.LOW:
    # STOP — do not pass to InspectionAgent
    return failure_result
```

### 4. Classification (Phase 1 Perception)
**InspectionAgent.inspect()** runs the full perception pipeline:
```python
if measurement.is_2d:
    classification, ood_result = inspector.inspect(measurement, dqc_result)
    state.add_classification(classification)
    state.add_ood_result(ood_result)
```

InspectionAgent internally:
- Log-preprocesses the array
- Runs EnsembleCNN (5-model ensemble)
- Extracts physics features (FFT peak ratio, diagonal strength)
- Checks for physics override
- Runs OOD detector on penultimate features
- Generates structured NL summary

### 5. Belief Update (Phase 2 Planning)
**BeliefUpdater** updates the POMDP belief using CIM observation model:
```python
belief_updater.update_from_2d(measurement, classification)
# Updates state.belief.charge_probs in-place
```

For line scans (bypass InspectionAgent per blueprint §5.1):
```python
if measurement.modality == MeasurementModality.LINE_SCAN:
    belief_updater.update_from_1d(measurement)
```

### 6. Bayesian Optimization (Phase 2 Planning)
**MultiResBO.propose()** uses GP with CIM-informed priors:
```python
bo.update(state.bo_history)
proposal = bo.propose(current=state.current_voltage, l1_max=0.10)
# → ActionProposal(delta_v=VoltagePoint(vg1=0.05, vg2=0.02), ...)
```

### 7. Safety Critic (Phase 0 Hardware)
**SafetyCritic** enforces hard constraints:
```python
proposal = safety_critic.clip(proposal, state.current_voltage)
verdict = safety_critic.verify(state.current_voltage, proposal)

if not verdict.all_passed:
    state.record_safety_violation()
    return failure_result
```

### 8. Risk Assessment + HITL (Phase 0 Governance)
**HITLManager.compute_risk_score()** aggregates 12 trigger conditions:
```python
risk = hitl_manager.compute_risk_score(
    proposal=proposal,
    safety_verdict=verdict,
    dqc_flag=dqc_result.quality.value,
    ood_score=ood_result.score,
    ensemble_disagreement=classification.ensemble_disagreement,
    consecutive_backtracks=state.consecutive_backtracks,
    step=state.step + 1,
)

if risk >= 0.70:
    event = hitl_manager.queue_request(...)
    event = hitl_manager.await_decision(event)  # BLOCKS
    state.add_hitl_event(event)
```

### 9. Execute Move (Phase 2 Agent + Phase 0 Hardware)
If all checks pass:
```python
translator.execute_voltage_move(
    vg1=state.current_voltage.vg1 + safe_dv.vg1,
    vg2=state.current_voltage.vg2 + safe_dv.vg2,
)
state.apply_move(safe_dv)
```

### 10. Governance (Phase 0)
**GovernanceLogger** records every action:
```python
decision = Decision(
    run_id=state.run_id,
    step=state.step,
    intent="voltage_move",
    stage=state.stage,
    observation_summary={...},
    action_summary={...},
    rationale="...",
)
state.add_decision(decision)
governance_logger.log(decision)
```

### 11. State Machine (Phase 2 Planning)
**StateMachine.process_result()** manages stage transitions:
```python
result = navigation_result(target_reached=True, belief_confidence=0.85)
new_stage, rationale, hitl_triggered = state_machine.process_result(result)

if new_stage != state.stage:
    state.advance_stage(new_stage)
```

Backtracking on failure:
```python
if retries_exhausted:
    event = BacktrackEvent(from_stage=NAVIGATION, to_stage=CHARGE_ID, ...)
    state.record_backtrack(event)
```

---

## Key Integration Points

### BeliefState (Phase 0 stub → Phase 2 implementation)

**Phase 0 defined the interface:**
```python
@dataclass
class BeliefState:
    charge_probs: Dict[tuple, float] = field(default_factory=dict)
    uncertainty_map: Optional[Any] = None
    device_params: Dict[str, float] = field(default_factory=lambda: {...})
    
    def entropy(self) -> float: ...
    def most_likely_state(self) -> Optional[tuple]: ...
    def initialise_uniform(self, charge_states: Optional[List[tuple]] = None) -> None: ...
```

**Phase 2 implemented the particle filter:**
```python
class BeliefUpdater:
    def __init__(self, belief: BeliefState, ...):
        self.belief = belief  # Operates on Phase 0 stub directly
        self._particles = _ParticleSet(...)  # Internal engine
    
    def update_from_2d(self, measurement, classification):
        # Updates self.belief.charge_probs in-place
        self.belief.charge_probs = self._particles.to_charge_probs()
```

### Classification Flow (Phase 1 → Phase 2)

**InspectionAgent output:**
```python
classification = Classification(
    measurement_id=...,
    label=ChargeLabel.DOUBLE_DOT,
    confidence=0.85,
    ensemble_disagreement=0.12,
    features={...},
    physics_override=False,
    nl_summary="...",
)
```

**Used by multiple Phase 2 components:**
- **BeliefUpdater:** Uses `label` and `confidence` to boost particle weights; `physics_override=True` → inflates uncertainty
- **MultiResBO:** Creates BOPoints from classifications; `confidence` becomes the reward signal
- **StateMachine:** Uses `confidence` to determine if stage success threshold is met
- **HITLManager:** Uses `ensemble_disagreement` in risk score computation

### Safety Integration (Phase 0 → Phase 2)

**ExecutiveAgent always calls SafetyCritic before any move:**
```python
proposal = bo.propose(...)                              # Phase 2 proposes
proposal = safety_critic.clip(proposal, current)        # Phase 0 clips
verdict = safety_critic.verify(current, proposal)       # Phase 0 verifies

if verdict.all_passed:
    translator.execute_voltage_move(...)                # Phase 2 executes
    state.apply_move(proposal.safe_delta_v)             # Phase 0 records
else:
    state.record_safety_violation()
```

No component can bypass SafetyCritic — it's architecturally enforced.

### HITL Integration (Phase 0 → Phase 2)

**HITLManager is called at two points:**
1. **Risk score computation (every voltage move):**
   ```python
   risk = hitl_manager.compute_risk_score(...)
   if risk >= 0.70:
       event = hitl_manager.queue_request(...)
       event = hitl_manager.await_decision(event)  # BLOCKS
   ```

2. **State machine backtracking (condition 8):**
   ```python
   if state.consecutive_backtracks >= 2:
       # StateMachine returns hitl_triggered=True
       # ExecutiveAgent queues HITL event
   ```

**No timeout auto-approval** — agent blocks until human decides.

---

## Running the Integrated System

### Without Trained Models (CI/Development)
```bash
# Uses InspectionAgent in stub mode (untrained ensemble, no OOD)
pytest tests/test_integration_phase012.py -v

# Run benchmark without checkpoints
python experiments/benchmark_phase2.py --fast --skip-missing-checkpoints
```

### With Trained Models (Full System)
```bash
# 1. Train Phase 1 models
python experiments/train_phase1.py --out experiments/checkpoints/phase1

# 2. Run benchmark
python experiments/benchmark_phase2.py --n-trials 100 --budget 2048
```

### Manual Inspection
```python
from qdot.core.state import ExperimentState
from qdot.simulator.cim import CIMSimulatorAdapter
from qdot.agent.executive import ExecutiveAgent
from qdot.perception.inspector import InspectionAgent
from qdot.core.hitl import HITLManager, HITLOutcome

# Setup
state = ExperimentState.new(device_id="manual_test")
adapter = CIMSimulatorAdapter(device_id="manual_test", seed=42)
inspector = InspectionAgent(ensemble=None, ood_detector=None)  # Or load trained

hitl = HITLManager()
hitl.set_test_mode(auto_outcome=HITLOutcome.APPROVED)  # No blocking for demo

# Create agent
agent = ExecutiveAgent(
    state=state,
    adapter=adapter,
    inspection_agent=inspector,
    hitl_manager=hitl,
    max_steps=50,
    measurement_budget=2048,
)

# Run
summary = agent.run()
print(f"Success: {summary['success']}")
print(f"Measurements: {summary['total_measurements']}")
print(f"Reduction: {summary['measurement_reduction']:.1%}")
```

---

## File Organization

```
qdot/
├── core/                          # Phase 0 — Foundation
│   ├── types.py                   # All canonical types
│   ├── state.py                   # ExperimentState + BeliefState stub
│   ├── governance.py              # GovernanceLogger
│   └── hitl.py                    # HITLManager
│
├── hardware/                      # Phase 0 — Device interface
│   ├── adapter.py                 # DeviceAdapter ABC
│   └── safety.py                  # SafetyCritic
│
├── simulator/                     # Phase 0 — CIM simulator
│   └── cim.py                     # ConstantInteractionDevice + CIMSimulatorAdapter
│
├── perception/                    # Phase 1 — Perception pipeline
│   ├── dqc.py                     # DQCGatekeeper
│   ├── inspector.py               # InspectionAgent
│   ├── classifier.py              # EnsembleCNN
│   ├── features.py                # Physics validators
│   ├── ood.py                     # MahalanobisOOD
│   └── dataset.py                 # CIMDataset for training
│
├── planning/                      # Phase 2 — POMDP + BO
│   ├── belief.py                  # BeliefUpdater + CIMObservationModel
│   ├── sensing.py                 # ActiveSensingPolicy
│   ├── bayesian_opt.py            # MultiResBO + GaussianProcess
│   └── state_machine.py           # StateMachine (5 stages)
│
└── agent/                         # Phase 2 — Main loop
    ├── executive.py               # ExecutiveAgent (orchestrator)
    └── translator.py              # TranslationAgent (NL → API)
```

---

## Testing Strategy

### Unit Tests (per module)
- `tests/test_types.py` — Phase 0 types
- `tests/test_state.py` — ExperimentState, BeliefState stub
- `tests/test_safety.py` — SafetyCritic + fuzz test (5000 iterations)
- `tests/test_simulator.py` — CIM physics
- `tests/test_perception.py` — DQC, CNN, OOD, InspectionAgent
- `tests/test_planning.py` — Belief, sensing, BO, state machine
- `tests/test_agent.py` — TranslationAgent, ExecutiveAgent

### Integration Tests
- `tests/test_integration_phase012.py` — Full pipeline smoke tests

### Benchmarks
- `experiments/train_phase1.py --fast` — Phase 1 smoke test (CI)
- `experiments/benchmark_phase2.py --fast` — Phase 2 quick eval (10 trials)
- `experiments/benchmark_phase2.py --n-trials 100` — Full benchmark

---

## Next Steps (Phase 3)

1. **LLM Integration (IBM Granite 3-8B-Instruct)**
   - Replace template rationales with LLM calls
   - ONE call per stage transition + ONE per HITL trigger
   - Budget: ~200 tokens per call

2. **Disorder Learning**
   - OOD flag → fit device-specific disorder map
   - Inject into CIMObservationModel.device
   - Close sim-to-real gap

3. **Meta-Learning**
   - Learn CIM parameter priors from device family
   - Fast adaptation to new devices

4. **Hardware Validation**
   - Integrate real Si/SiGe adapter
   - Run on QFlow hardware testbed
   - Validate ≥50% reduction on real devices

---

## Troubleshooting

### "No tests collected" in test_agent.py
**Cause:** File didn't exist or pytest can't find it.  
**Fix:** Ensure `tests/test_agent.py` is in the repo root `tests/` directory.

### "ModuleNotFoundError: No module named 'qdot.planning'"
**Cause:** Phase 2 modules not in Python path.  
**Fix:** Run `pip install -e .` from repo root to install in editable mode.

### InspectionAgent predictions are random
**Cause:** Running without trained checkpoints (ensemble=None).  
**Fix:** Either (a) train models with `experiments/train_phase1.py`, or (b) use `--skip-missing-checkpoints` flag for testing without models.

### HITL blocks forever
**Cause:** HITLManager not in test mode.  
**Fix:** Call `hitl_manager.set_test_mode(auto_outcome=HITLOutcome.APPROVED)` for non-blocking testing.

### SafetyCritic rejects all moves
**Cause:** Voltage bounds or l1_max cap too tight.  
**Fix:** Check `state.voltage_bounds` and `state.step_caps["l1_max"]` are reasonable for your device.

---

**Phase 0/1/2 integration complete.** All components tested individually and end-to-end. Ready for Phase 3.
