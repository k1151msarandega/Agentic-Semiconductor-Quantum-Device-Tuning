# Related Papers

A running log of papers relevant to this project (agentic / RL-based tuning of semiconductor quantum devices).

---

## 1. QADAPT: Action-Factored Multi-Agent RL for Scalable Quantum Device Tuning

- **Authors:** E. De Nicolo, R. Marchand, C. Carlsson, P. Vaidhyanathan, N. Ares (University of Oxford)
- **arXiv:** [2607.09422](https://arxiv.org/abs/2607.09422) (v1, 10 Jul 2026)
- **Added:** 2026-08-12

**Summary:**
Introduces QADAPT, a cooperative multi-agent RL framework for tuning electrostatically-defined quantum dot arrays. One agent per gate (2N-1 agents for N dots). Key ideas:
- **Adaptive gate virtualization:** a lightweight CNN + Kalman filter online-estimates the gate-to-dot cross-capacitance matrix, used to construct a factored/decoupled "virtual" action basis that reduces cross-agent interference from capacitive cross-talk.
- **Modular actor-critic with parameter sharing:** one shared policy for all plunger gates, another for all barrier gates (CTDE paradigm) — keeps the number of learned policies independent of array size.
- **Bandit formulation:** discount factor γ=0 (treats tuning as a contextual bandit rather than a long-horizon RL problem), since the optimal action is always "move toward target."
- **Results:** zero-shot generalization from a 4-dot training array to 2/4/6/8-dot arrays with near-linear (O(N)) scaling in required measurements; outperforms IPPO, MAPPO, MADDPG, FACMAC, DreamerV3, Bayesian optimization, Nelder-Mead, L-BFGS, and random search baselines. Also sketched as extensible to superconducting transmon qubits (Appendix G).
- **Limitations noted by authors:** sim-to-real gap untested on hardware; strict plunger/barrier role assignment may not hold under fabrication imperfections; assumes "open" (reservoir-connected) dot regime.

**Relevance to this project:** Directly overlaps with agentic/multi-agent approaches to quantum device autotuning — closely related in both problem framing (quantum dot array tuning, cross-talk, virtual gates) and method (RL agents, CTDE, parameter sharing). Worth comparing against for baselines, and the virtualization/Kalman-filter approach may be relevant to any cross-talk compensation work here.

---
