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

SYSTEM_PROMPT = """You are an expert quantum physicist overseeing an autonomous \
semiconductor quantum dot tuning experiment in real time.

The system is a 6-stage POMDP agent navigating a 2D voltage space to reach \
the (1,1) charge state — one electron per quantum dot — using a Constant \
Interaction Model simulator, Bayesian optimisation, and a CNN charge classifier.

Stages in order:
  BOOTSTRAPPING → COARSE_SURVEY → HYPERSURFACE_SEARCH → CHARGE_ID \
→ NAVIGATION → VERIFICATION → COMPLETE

Your role:
- Narrate each stage transition in 2-3 clear sentences a lab operator \
  can understand without a physics PhD.
- When HITL is triggered, explain the risk concisely and what the operator \
  should watch for.
- When something fails, diagnose why based on what you have observed so far \
  in this run — you have full episodic memory.
- Be precise, grounded, and honest. Do not speculate beyond what the data shows.
- Keep responses under 80 words."""


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
    ) -> None:
        """Called after every stage transition. Non-blocking."""
        if not self.enabled:
            return
        user_msg = (
            f"[Step {step} | {measurements_used} measurements used]\n"
            f"Stage transition: {from_stage} → {to_stage}\n"
            f"Agent rationale: {rationale}\n"
            f"Confidence: {confidence:.3f}\n"
            f"Please narrate this transition for the operator."
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
