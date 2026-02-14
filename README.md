# qdot-agent

**Agentic quantum dot tuning — clean-slate redesign.**

Autonomous semiconductor quantum device tuning using a POMDP executive agent, physics-informed perception, multi-fidelity active sensing, and blocking human-in-the-loop oversight.

> This is a research codebase in active development. Phase 0 (foundation) is complete. Phases 1–4 are in progress.

---

## Architecture

Four-layer hierarchy (see `docs/blueprint_v2.pdf` for full spec):

```
Layer 1 — Executive Agent     (POMDP planner + LLM reasoner)
Layer 2 — Operational Intel   (Knowledge Agent, Translation Agent)  
Layer 3 — Perception          (DQC Gatekeeper, Inspection Agent, CIM model)
Layer 4 — Hardware            (Device Adapter, Safety Critic)
```

## Repository structure

```
qdot/
├── core/         # Phase 0: foundation — types, state, governance, HITL
├── hardware/     # Phase 0: device adapter ABC, safety critic
├── simulator/    # CIM physics (ported from hackathon)
├── perception/   # Phase 1: DQC gatekeeper, classifier, OOD detector
├── planning/     # Phase 2: POMDP belief, active sensing, BO, state machine
└── agent/        # Phase 2: executive agent, LLM interface
tests/            # One test file per module; safety fuzz on every commit
experiments/      # Benchmarking scripts (not part of package)
```

## Quickstart

```bash
git clone https://github.com/your-org/qdot-agent
cd qdot-agent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/
```

## Phase status

| Phase | Focus | Status |
|-------|-------|--------|
| 0 | Foundation: types, state, governance, HITL, safety, CIM simulator | ✅ Complete |
| 1 | Perception: DQC gatekeeper, TinyCNN classifier, OOD detector | 🔲 Next |
| 2 | Planning: POMDP belief, active sensing, BO, backtracking state machine | 🔲 Planned |
| 3 | Sim-to-real: disorder learner, hardware adapter | 🔲 Planned |
| 4 | Meta-learning, ablation study, paper | 🔲 Planned |

## Design principles

1. **One executive, clear chain of command.** No competing decision-makers.
2. **Physics first.** CIM is embedded in the planner's belief state, not bolted on.
3. **Uncertainty is a first-class citizen.** Every prediction carries confidence.
4. **Hardware agnosticism by contract.** Swapping device types requires zero changes above Layer 4.
5. **HITL is a genuine gate.** Auto-approval on timeout is removed.

## Development workflow

```bash
# One branch per phase
git checkout -b phase-1-perception

# Run safety fuzz before every commit
python -m pytest tests/test_safety.py -v

# Merge when phase is working + tested
git checkout main && git merge phase-1-perception
```
