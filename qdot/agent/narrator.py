"""
qdot/agent/narrator.py
======================
LLM layer for the quantum dot tuning agent — four distinct modes:

  1. EXCEPTION REPORTING  — fires only when something unexpected happens:
     OOD spike, DQC LOW, consecutive backtracks, stage failure.
     Silent during normal operation. Operators don't want a running
     commentary; they want to know when to pay attention.

  2. HITL DECISION SUPPORT — fires at every Human-in-the-Loop gate.
     Presents the situation, the risk, and a recommendation. The
     operator can approve, reject, or ask a follow-up question.

  3. ON-DEMAND INTERROGATION — synchronous .ask() method the operator
     can call any time: "what happened while I was gone?", "why did
     it backtrack?", "should I be worried about that OOD spike?"
     The narrator has full episodic memory of the run and answers
     from that context.

  4. POST-RUN SUMMARY — called once at the end of each trial.
     Produces a concise, physicist-readable summary: what happened,
     what the key decision points were, and what to watch next time.

Environment variables:
    QDOT_LLM_BASE_URL   — OpenAI-compatible base URL (vLLM on MI300X)
    QDOT_LLM_API_KEY    — API key (default: "EMPTY" for local vLLM)
    QDOT_LLM_MODEL      — model name (default: Qwen/Qwen2.5-1.5B-Instruct)
    QDOT_LLM_ENABLED    — set to "0" to disable silently
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are Dr. Q, an experimental physicist specialising in \
semiconductor quantum dot devices. You are the AI co-pilot for an autonomous \
tuning agent that navigates gate voltage space to reach the (1,1) charge \
state — one electron per dot — required for spin qubit operation.

The agent uses a POMDP planner, Bayesian optimisation, and a CNN charge \
classifier. It moves through these stages:
  BOOTSTRAPPING → COARSE_SURVEY → HYPERSURFACE_SEARCH → CHARGE_ID → NAVIGATION → VERIFICATION

You only speak when there is something worth saying: an anomaly, a decision \
point, a question from the operator, or a run summary. You do not narrate \
routine progress.

Your voice:
- Direct and precise. Like a physicist, not a chatbot.
- Reference actual numbers when you have them.
- 2-4 sentences unless a detailed answer is genuinely needed.
- When something goes wrong, say what you think is causing it.
- Never open with "In Stage X". Never say "Great news!" or "We've achieved".
- You remember everything that happened in this run. Use that memory."""


# ---------------------------------------------------------------------------
# Event log entry — what the narrator remembers
# ---------------------------------------------------------------------------

@dataclass
class RunEvent:
    """A single logged event in the run history."""
    kind: str          # "exception" | "hitl" | "summary" | "ask" | "transition"
    step: int
    measurements: int
    stage: str
    description: str   # structured context passed to LLM
    response: str = "" # LLM response (filled in asynchronously)
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# LLMNarrator
# ---------------------------------------------------------------------------

class LLMNarrator:
    """
    Four-mode LLM co-pilot for a quantum dot tuning run.

    Thread-safe. Exception reports and HITL support are non-blocking
    (fire-and-store). On-demand questions and post-run summaries are
    synchronous — they wait for the answer before returning.
    """

    def __init__(self, run_id: str, enabled: bool = True) -> None:
        self.run_id = run_id
        self.enabled = enabled and os.environ.get("QDOT_LLM_ENABLED", "1") != "0"

        self._events: list[RunEvent] = []
        self._history: list[dict] = []   # OpenAI message format
        self._lock = threading.Lock()
        self._pending: list[threading.Thread] = []

        base_url = os.environ.get("QDOT_LLM_BASE_URL", "")
        api_key  = os.environ.get("QDOT_LLM_API_KEY", "EMPTY")
        self._model = os.environ.get(
            "QDOT_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"
        )

        if self.enabled:
            if not base_url:
                print("[Dr. Q] QDOT_LLM_BASE_URL not set — co-pilot offline.")
                self.enabled = False
                return
            try:
                import openai
                self._client = openai.OpenAI(base_url=base_url, api_key=api_key)
            except ImportError:
                print("[Dr. Q] openai package not installed — run: pip install openai")
                self.enabled = False

    # ------------------------------------------------------------------
    # Mode 1 — Exception reporting
    # ------------------------------------------------------------------

    def report_exception(
        self,
        stage: str,
        exception_type: str,
        step: int,
        measurements_used: int,
        budget_total: int,
        details: dict,
    ) -> None:
        """
        Call when something anomalous happens mid-run.

        exception_type options:
            "ood_spike"          — OOD score exceeded threshold
            "dqc_low"            — measurement quality dropped to LOW
            "consecutive_backtracks"  — agent stuck in backtrack loop
            "stage_failure"      — stage exhausted retries, backtracking
            "budget_warning"     — <20% of measurement budget remaining

        details: dict with any relevant physics values.
        """
        if not self.enabled:
            return

        budget_pct = int(100 * measurements_used / budget_total)
        detail_str = "; ".join(f"{k}={v}" for k, v in details.items())

        msg = (
            f"[EXCEPTION — {exception_type}] "
            f"Step {step}, stage {stage}, {budget_pct}% budget used.\n"
            f"Details: {detail_str}\n"
            f"What is happening and should the operator intervene?"
        )
        self._fire_async(msg, kind="exception", step=step,
                         measurements=measurements_used, stage=stage)

    # ------------------------------------------------------------------
    # Mode 2 — HITL decision support
    # ------------------------------------------------------------------

    def support_hitl(
        self,
        stage: str,
        trigger_reason: str,
        risk_score: float,
        step: int,
        measurements_used: int,
        budget_total: int,
        proposal_summary: str,
        physics_context: dict,
    ) -> str:
        """
        Synchronous — blocks until Dr. Q responds.

        Returns the recommendation string so it can be shown to the
        operator before they make their HITL decision.

        proposal_summary: human-readable description of the proposed move.
        physics_context: dict with current voltage, DQC, OOD, belief state.
        """
        if not self.enabled:
            return ""

        budget_pct = int(100 * measurements_used / budget_total)
        physics_str = "; ".join(f"{k}={v}" for k, v in physics_context.items())

        msg = (
            f"[HITL GATE — risk score {risk_score:.2f}] "
            f"Step {step}, stage {stage}, {budget_pct}% budget used.\n"
            f"Trigger: {trigger_reason}\n"
            f"Proposed action: {proposal_summary}\n"
            f"Current state: {physics_str}\n"
            f"Give a 2-sentence recommendation: what is the risk, "
            f"and should the operator approve, modify, or reject?"
        )

        response = self._call_llm_sync(msg, kind="hitl",
                                       step=step, measurements=measurements_used,
                                       stage=stage)
        print(f"\n[Dr. Q — HITL] {response}\n")
        return response

    # ------------------------------------------------------------------
    # Mode 3 — On-demand interrogation
    # ------------------------------------------------------------------

    def ask(self, question: str, step: int = 0, stage: str = "unknown") -> str:
        """
        Synchronous. The operator asks anything; Dr. Q answers from
        full run memory.

        Example questions:
            "What happened while I was gone?"
            "Why did it backtrack at step 12?"
            "Should I be worried about that OOD spike?"
            "What's the most likely reason navigation is failing?"
        """
        if not self.enabled:
            return "[Dr. Q is offline — set QDOT_LLM_BASE_URL to enable]"

        self._drain()   # ensure all async events are stored first

        response = self._call_llm_sync(
            question, kind="ask", step=step, stage=stage,
            measurements=self._events[-1].measurements if self._events else 0,
        )
        print(f"\n[Dr. Q] {response}\n")
        return response

    # ------------------------------------------------------------------
    # Mode 4 — Post-run summary
    # ------------------------------------------------------------------

    def summarise_run(
        self,
        final_stage: str,
        success: bool,
        total_measurements: int,
        budget_total: int,
        total_steps: int,
        n_backtracks: int,
        n_hitl: int,
        n_exceptions: int,
    ) -> str:
        """
        Called once at the end of a trial. Synchronous.

        Returns a concise physicist-readable summary of the run.
        """
        if not self.enabled:
            return ""

        self._drain()

        budget_pct = int(100 * total_measurements / budget_total)
        outcome = "SUCCESS" if success else f"FAILED at {final_stage}"

        msg = (
            f"[RUN COMPLETE — {outcome}]\n"
            f"Used {total_measurements}/{budget_total} measurements ({budget_pct}%), "
            f"{total_steps} steps, {n_backtracks} backtracks, "
            f"{n_hitl} HITL gates, {n_exceptions} anomalies flagged.\n"
            f"Give a 3-4 sentence post-run summary a physicist would find useful: "
            f"what went well, what failed and why (based on what you observed), "
            f"and one concrete suggestion for the next run."
        )

        response = self._call_llm_sync(
            msg, kind="summary", step=total_steps,
            stage=final_stage, measurements=total_measurements,
        )
        print(f"\n[Dr. Q — Run Summary]\n{response}\n")
        return response

    # ------------------------------------------------------------------
    # Lightweight transition logging (no LLM call — just memory)
    # ------------------------------------------------------------------

    def log_transition(
        self,
        from_stage: str,
        to_stage: str,
        step: int,
        measurements_used: int,
        confidence: float,
        rationale: str,
        snr_db: float = None,
        dqc_quality: str = None,
        belief_top_state: str = None,
        current_voltage: tuple = None,
    ) -> None:
        """
        Records every stage transition into episodic memory WITHOUT
        calling the LLM. This keeps the run history rich for on-demand
        questions and the post-run summary, without generating noise.

        Only call narrate_transition (which triggers LLM) for exceptions.
        """
        details = []
        if snr_db is not None:
            details.append(f"SNR={snr_db:.1f}dB")
        if dqc_quality is not None:
            details.append(f"DQC={dqc_quality}")
        if belief_top_state is not None:
            details.append(f"belief={belief_top_state}")
        if current_voltage is not None:
            details.append(
                f"vg1={current_voltage[0]:+.3f}V vg2={current_voltage[1]:+.3f}V"
            )

        description = (
            f"Step {step}: {from_stage} → {to_stage} "
            f"(conf={confidence:.2f}). "
            f"{'; '.join(details)}. "
            f"Rationale: {rationale}"
        )

        event = RunEvent(
            kind="transition", step=step,
            measurements=measurements_used,
            stage=to_stage, description=description,
        )
        with self._lock:
            self._events.append(event)
            # Add to conversation history so LLM has context
            self._history.append({"role": "user", "content": description})
            self._history.append({
                "role": "assistant",
                "content": f"[logged — {from_stage}→{to_stage}]"
            })

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def drain(self) -> None:
        """Wait for all pending async calls to complete."""
        self._drain()

    def event_log(self) -> list[RunEvent]:
        """Return the full event log for this run."""
        with self._lock:
            return list(self._events)

    def n_exceptions(self) -> int:
        with self._lock:
            return sum(1 for e in self._events if e.kind == "exception")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fire_async(
        self, msg: str, kind: str, step: int, measurements: int, stage: str
    ) -> None:
        """Append event and call LLM in background thread."""
        event = RunEvent(kind=kind, step=step, measurements=measurements,
                         stage=stage, description=msg)
        with self._lock:
            self._events.append(event)
            self._history.append({"role": "user", "content": msg})

        idx = len(self._events) - 1

        t = threading.Thread(
            target=self._fire_and_store, args=(msg, idx), daemon=True
        )
        self._pending.append(t)
        t.start()

    def _fire_and_store(self, msg: str, event_idx: int) -> None:
        try:
            response = self._call_llm_raw(msg)
        except Exception as exc:
            response = f"[Dr. Q error: {exc}]"

        with self._lock:
            self._events[event_idx].response = response
            self._history.append({"role": "assistant", "content": response})

        print(f"\n[Dr. Q] {response}\n")

    def _call_llm_sync(
        self, msg: str, kind: str, step: int, measurements: int, stage: str
    ) -> str:
        """Synchronous LLM call — records to history and returns response."""
        event = RunEvent(kind=kind, step=step, measurements=measurements,
                         stage=stage, description=msg)
        with self._lock:
            self._events.append(event)
            self._history.append({"role": "user", "content": msg})

        try:
            response = self._call_llm_raw(msg)
        except Exception as exc:
            response = f"[Dr. Q error: {exc}]"

        idx = len(self._events) - 1
        with self._lock:
            self._events[idx].response = response
            self._history.append({"role": "assistant", "content": response})

        return response

    def _call_llm_raw(self, latest_msg: str) -> str:
        """Make the actual API call with full conversation history."""
        with self._lock:
            history = list(self._history)

        # history already contains the latest user message
        messages_without_last = history[:-1]

        resp = self._client.chat.completions.create(
            model=self._model,
            max_tokens=200,
            messages=(
                [{"role": "system", "content": SYSTEM_PROMPT}]
                + messages_without_last
                + [{"role": "user", "content": latest_msg}]
            ),
        )
        return resp.choices[0].message.content.strip()

    def _drain(self) -> None:
        for t in self._pending:
            t.join(timeout=30)
        self._pending = [t for t in self._pending if t.is_alive()]
