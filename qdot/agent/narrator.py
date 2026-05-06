"""
qdot/agent/narrator.py
======================
LLM Narration layer — maintains episodic memory across a tuning run.

The narrator observes every stage transition and HITL trigger in real
time, accumulates them in a conversation thread, and generates
human-readable rationale for each event. Because it holds the full
run history, it can answer "what went wrong at step 7?" with genuine
context rather than inference from static logs.

For the AMD hackathon: set QDOT_LLM_BASE_URL to your vLLM endpoint
running Qwen-2.5-Coder-32B on MI300X, and set QDOT_LLM_MODEL to
"qwen2.5-coder-32b-instruct". Locally it defaults to the Anthropic API.

Environment variables:
    QDOT_LLM_BASE_URL   — OpenAI-compatible base URL (optional)
    QDOT_LLM_API_KEY    — API key (falls back to ANTHROPIC_API_KEY)
    QDOT_LLM_MODEL      — model name (default: claude-haiku-4-5-20251001)
    QDOT_LLM_ENABLED    — set to "0" to disable narration silently
"""

from __future__ import annotations

import json
import os
import time
import threading
from typing import Optional

SYSTEM_PROMPT = """You are Dr. Q, an expert experimental physicist specialising in \
semiconductor quantum dot devices. You are observing an autonomous AI agent tune a \
double quantum dot in real time and narrating its decisions to a lab operator.

The agent navigates a 2D gate voltage space (vg1, vg2) using a POMDP planner, \
Bayesian optimisation, and a CNN charge classifier to reach the (1,1) charge state — \
exactly one electron per dot. This is required for spin qubit operation.

Pipeline stages:
  BOOTSTRAPPING → COARSE_SURVEY → HYPERSURFACE_SEARCH → CHARGE_ID → NAVIGATION → VERIFICATION

Your style:
- Speak directly and naturally, like a physicist watching an experiment. Not bullet points, not headers.
- Reference the actual numbers you are given — voltages, SNR, confidence, budget remaining.
- Be concise: 2-3 sentences maximum per narration.
- When things go wrong, say so plainly and say what you think is happening.
- Never say "In Stage X" as your opening. Never use corporate language.
- You have memory of the full run so far — use it."""


class LLMNarrator:
    """
    Real-time LLM narrator for a single tuning run.

    Maintains a conversation thread so every narration call has full
    context of what happened before it in this run. Thread-safe: narration
    calls are fire-and-store so they never block the agent loop.
    """

    def __init__(self, run_id: str, enabled: bool = True) -> None:
        self.run_id = run_id
        self.enabled = enabled and os.environ.get("QDOT_LLM_ENABLED", "1") != "0"
        self._history: list[dict] = []
        self._lock = threading.Lock()
        self._pending: list[threading.Thread] = []

        # API configuration
        base_url = os.environ.get("QDOT_LLM_BASE_URL", "")
        api_key = os.environ.get("QDOT_LLM_API_KEY", "EMPTY")
        self._model = os.environ.get(
            "QDOT_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"
        )
        self._use_anthropic = False  # always use OpenAI-compatible (vLLM)

        if self.enabled:
            if not base_url:
                print("[Narrator] QDOT_LLM_BASE_URL not set — narration disabled.")
                self.enabled = False
            else:
                try:
                    import openai
                    self._client = openai.OpenAI(
                        base_url=base_url, api_key=api_key
                    )
                except ImportError:
                    print("[Narrator] openai package not installed — run: pip install openai")
                    self.enabled = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def narrate_transition(
        self,
        from_stage: str,
        to_stage: str,
        rationale: str,
        step: int,
        measurements_used: int,
        confidence: float,
        budget_total: int = 4096,
        snr_db: float = None,
        dqc_quality: str = None,
        belief_top_state: str = None,
        current_voltage: tuple = None,
    ) -> None:
        if not self.enabled:
            return
        budget_pct = int(100 * measurements_used / budget_total)
        details = []
        if snr_db is not None:
            details.append(f"SNR {snr_db:.1f}dB")
        if dqc_quality is not None:
            details.append(f"data quality {dqc_quality}")
        if belief_top_state is not None:
            details.append(f"most likely charge state {belief_top_state}")
        if current_voltage is not None:
            details.append(
                f"gates at vg1={current_voltage[0]:+.3f}V, vg2={current_voltage[1]:+.3f}V"
            )
        physics_context = ". ".join(details)

        user_msg = (
            f"Step {step} — {from_stage} → {to_stage} "
            f"(confidence {confidence:.2f}, {budget_pct}% of measurement budget used).\n"
            f"Physics: {physics_context}.\n"
            f"Agent log: {rationale}"
        )
        self._fire(user_msg, tag=f"transition:{from_stage}→{to_stage}")

    def narrate_hitl(
        self,
        stage: str,
        trigger_reason: str,
        risk_score: float,
        step: int,
    ) -> None:
        """Called when HITL is triggered. Non-blocking."""
        if not self.enabled:
            return
        user_msg = (
            f"[Step {step} | Stage: {stage}]\n"
            f"HITL triggered — risk score: {risk_score:.2f}\n"
            f"Trigger reason: {trigger_reason}\n"
            f"Explain the risk to the operator and what they should check."
        )
        self._fire(user_msg, tag=f"hitl:{stage}")

    def ask(self, question: str) -> str:
        """
        Synchronous query — 'what went wrong?' style questions.
        The LLM answers with full run history in context.
        """
        if not self.enabled:
            return "[Narrator disabled]"
        self._drain()  # wait for any pending async calls first
        response = self._call_llm(question)
        with self._lock:
            self._history.append({"role": "user", "content": question})
            self._history.append({"role": "assistant", "content": response})
        return response

    def drain(self) -> None:
        """Wait for all pending async narration calls to complete."""
        self._drain()

    def full_transcript(self) -> list[dict]:
        """Return the full conversation history for this run."""
        self._drain()
        with self._lock:
            return list(self._history)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fire(self, user_msg: str, tag: str) -> None:
        """Append to history and call LLM in a background thread."""
        with self._lock:
            self._history.append({"role": "user", "content": user_msg})

        t = threading.Thread(
            target=self._fire_and_store,
            args=(user_msg, len(self._history) - 1),
            daemon=True,
        )
        self._pending.append(t)
        t.start()

    def _fire_and_store(self, user_msg: str, history_idx: int) -> None:
        try:
            response = self._call_llm(user_msg)
        except Exception as exc:
            response = f"[Narrator error: {exc}]"
        with self._lock:
            # Insert assistant reply right after the user message
            self._history.insert(history_idx + 1, {"role": "assistant", "content": response})
        print(f"\n[Narrator] {response}\n")

    def _call_llm(self, latest_user_msg: str) -> str:
        with self._lock:
            history_snapshot = list(self._history)

        # Build messages — exclude the latest user msg (already in history)
        messages = history_snapshot[:-1]  # everything before latest user msg

        if self._use_anthropic:
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=150,
                system=SYSTEM_PROMPT,
                messages=messages + [{"role": "user", "content": latest_user_msg}],
            )
            return resp.content[0].text.strip()
        else:
            # OpenAI-compatible (vLLM on AMD MI300X)
            resp = self._client.chat.completions.create(
                model=self._model,
                max_tokens=150,
                messages=(
                    [{"role": "system", "content": SYSTEM_PROMPT}]
                    + messages
                    + [{"role": "user", "content": latest_user_msg}]
                ),
            )
            return resp.choices[0].message.content.strip()

    def _drain(self) -> None:
        for t in self._pending:
            t.join(timeout=30)
        self._pending = [t for t in self._pending if t.is_alive()]
