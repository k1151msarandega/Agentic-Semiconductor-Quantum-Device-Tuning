---
title: SimQuantum Tuning Lab
emoji: ⚛
colorFrom: gray
colorTo: green
sdk: streamlit
sdk_version: 1.57.0
app_file: app.py
pinned: true
license: mit
short_description: Autonomous quantum dot tuning agent — AMD MI300X + Qwen2.5
---

# ⚛ SimQuantum Tuning Lab

**Autonomous quantum dot tuning · AMD Developer Hackathon 2025**

SimQuantum is a 6-stage POMDP agent that autonomously tunes a double quantum dot
device to the **(1,1) charge state** — one electron per dot — required for spin
qubit operation.

### What runs here (CPU demo)
- ✅ CIM physics simulator — real semiconductor physics
- ✅ Charge stability diagram visualisation
- ✅ Particle filter belief state
- ✅ 5-model CNN ensemble classifier (91.4% val acc, 51k training diagrams)
- ✅ Bayesian optimisation planner
- ✅ HITL safety critic
- ⬡ Dr. Q (Qwen2.5-1.5B) — connect your own vLLM endpoint in the sidebar

### Full GPU stack
The complete system runs on **AMD Instinct MI300X** via ROCm with
Qwen2.5-1.5B-Instruct served via vLLM. To connect Dr. Q, set the
vLLM endpoint in the sidebar to your MI300X instance.

### Architecture
```
BOOTSTRAPPING → COARSE_SURVEY → HYPERSURFACE_SEARCH → CHARGE_ID
     ↓               ↓                  ↓                  ↓
  DQC gate      32×32 sweep       16×16 local scan    CNN ensemble
                                                      + OOD detector
```
Navigation (→ VERIFICATION) is Phase 3 — active development.

### Links
- GitHub: https://github.com/k1151msarandega/Agentic-Semiconductor-Quantum-Device-Tuning
- AMD Developer Hackathon: https://lablab.ai/ai-hackathons/amd-developer
