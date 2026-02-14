"""
qdot/core/hitl.py
=================
HITLManager — Human-in-the-Loop approval gate.

Critical departure from hackathon: auto-approval on timeout is REMOVED.
The agent cannot proceed past a HITL trigger until a human explicitly
approves, modifies, or rejects the proposed action.

The manager writes requests to disk so an external UI (Gradio, CLI, etc.)
can pick them up without needing a shared process. The agent polls.

If you're running in a fully automated test context, call
HITLManager.set_test_mode(auto_outcome=HITLOutcome.APPROVED) to bypass
the blocking wait — but never in production.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Optional, List, Dict, Any
from uuid import UUID

from qdot.core.types import (
    ActionProposal,
    HITLEvent,
    HITLOutcome,
    SafetyVerdict,
    TuningStage,
    VoltagePoint,
)


class HITLManager:
    """
    Manages the HITL approval queue.

    The queue is a directory of JSON files, one per pending request.
    An external UI polls this directory and writes decisions back.
    The agent polls via await_decision().

    Trigger conditions (from blueprint Section 4):
        1. Mission start (first voltage ever applied)
        2. Proximity to voltage limit: margin < M_min
        3. Proposed step exceeds slew cap
        4. DQC flag == 'moderate'
        5. DQC flag == 'low' (mandatory stop)
        6. OOD score > threshold
        7. Ensemble disagreement > 0.3
        8. Consecutive backtracks >= 2 at same stage
        9. Cumulative risk score >= 0.70
        10. Measurement modality escalation (1D → 2D)
        11. Stage transition: coarse → fine tuning
        12. Charge switch noise detected

    Risk weights for each trigger are defined in the blueprint's
    compute_risk_score() spec (Section 4.1).
    """

    # Risk score threshold above which HITL is triggered
    HITL_THRESHOLD: float = 0.70

    def __init__(
        self,
        queue_dir: str = "data/hitl_queue",
        poll_interval_s: float = 1.0,
        enabled: bool = True,
    ) -> None:
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.poll_interval_s = poll_interval_s
        self.enabled = enabled
        self._test_mode = False
        self._test_outcome = HITLOutcome.APPROVED

    # -----------------------------------------------------------------------
    # Test mode (automated testing only — never in production)
    # -----------------------------------------------------------------------

    def set_test_mode(self, auto_outcome: HITLOutcome = HITLOutcome.APPROVED) -> None:
        """
        Bypass blocking wait for automated tests.
        Never call this in production code.
        """
        self._test_mode = True
        self._test_outcome = auto_outcome

    def disable_test_mode(self) -> None:
        self._test_mode = False

    # -----------------------------------------------------------------------
    # Risk score computation
    # -----------------------------------------------------------------------

    def compute_risk_score(
        self,
        proposal: ActionProposal,
        safety_verdict: SafetyVerdict,
        dqc_flag: str = "high",            # "high" | "moderate" | "low"
        ood_score: float = 0.0,
        ood_threshold: float = 24.0,
        ensemble_disagreement: float = 0.0,
        consecutive_backtracks: int = 0,
        step: int = 1,
        slew_cap: float = 0.10,
        M_min: float = 0.02,
    ) -> float:
        """
        Compute risk score r ∈ [0, 1] for a proposed move.

        Based on blueprint Section 4.1. The score is a weighted sum of
        normalised sub-scores. Values >= HITL_THRESHOLD trigger a HITL gate.

        First move ever (step == 1) always returns r = 1.0.
        """
        # First move — mandatory HITL regardless of other factors
        if step == 1:
            return 1.0

        r = 0.0

        # Safety sub-scores
        min_margin = safety_verdict.min_margin
        if min_margin < 0.05:
            r += 0.50
        elif min_margin < 0.10:
            r += 0.25

        safe_dv = proposal.safe_delta_v or proposal.delta_v
        if safe_dv.l1_norm > slew_cap:
            r += 0.40

        # Data quality sub-scores
        if dqc_flag == "low":
            r += 0.60
        elif dqc_flag == "moderate":
            r += 0.35

        if ood_score > ood_threshold:
            r += 0.40

        # Uncertainty sub-score
        if ensemble_disagreement > 0.30:
            r += 0.35

        # Planning sub-score
        if consecutive_backtracks >= 2:
            r += 0.45

        return min(1.0, r)

    # -----------------------------------------------------------------------
    # Queue management
    # -----------------------------------------------------------------------

    def queue_request(
        self,
        run_id: str,
        step: int,
        stage: TuningStage,
        trigger_reason: str,
        risk_score: float,
        proposal: ActionProposal,
        safety_verdict: SafetyVerdict,
    ) -> HITLEvent:
        """Write a HITL request to the queue and return the HITLEvent."""
        event = HITLEvent(
            run_id=run_id,
            step=step,
            trigger_reason=trigger_reason,
            risk_score=risk_score,
            proposal=proposal,
            safety_verdict=safety_verdict,
            outcome=HITLOutcome.PENDING,
            queued_at=time.time(),
        )

        record = _event_to_dict(event, stage)
        path = self.queue_dir / f"{event.id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        return event

    def await_decision(self, event: HITLEvent) -> HITLEvent:
        """
        Block until a human (or test mode) decides on this event.

        Unlike the hackathon, there is NO timeout auto-approval.
        This method polls indefinitely until the request file is updated.

        In test mode, resolves immediately with the configured outcome.
        """
        if not self.enabled:
            event.outcome = HITLOutcome.APPROVED
            event.decided_at = time.time()
            return event

        if self._test_mode:
            event.outcome = self._test_outcome
            event.decided_at = time.time()
            self._mark_decided(event)
            return event

        path = self.queue_dir / f"{event.id}.json"
        print(f"\n🚨 HITL REQUIRED — Step {event.step}")
        print(f"   Reason: {event.trigger_reason}")
        print(f"   Risk score: {event.risk_score:.2f}")
        print(f"   Request file: {path}")
        print(f"   Waiting for human decision (no timeout — edit the file to proceed)...")
        print(f"   Set 'outcome' to: approved | rejected | modified\n")

        while True:
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    rec = json.load(f)
                outcome_str = rec.get("outcome", "pending")
                if outcome_str != "pending":
                    event.outcome = HITLOutcome(outcome_str)
                    event.decided_at = rec.get("decided_at") or time.time()
                    event.deciding_human = rec.get("deciding_human", "")
                    if outcome_str == "modified" and rec.get("modified_delta_vg1") is not None:
                        event.modified_delta_v = VoltagePoint(
                            vg1=rec["modified_delta_vg1"],
                            vg2=rec["modified_delta_vg2"],
                        )
                    print(f"✓ HITL decision received: {event.outcome.value}")
                    return event
            time.sleep(self.poll_interval_s)

    def get_pending(self) -> List[Dict[str, Any]]:
        """Return all pending requests as raw dicts (for UI polling)."""
        pending = []
        for p in self.queue_dir.glob("*.json"):
            with open(p, encoding="utf-8") as f:
                rec = json.load(f)
            if rec.get("outcome", "pending") == "pending":
                pending.append(rec)
        return pending

    def approve(self, event_id: str, deciding_human: str = "") -> None:
        """Approve a request (called by UI or test code)."""
        self._write_decision(event_id, HITLOutcome.APPROVED, deciding_human)

    def reject(self, event_id: str, deciding_human: str = "") -> None:
        """Reject a request."""
        self._write_decision(event_id, HITLOutcome.REJECTED, deciding_human)

    def modify(
        self,
        event_id: str,
        new_delta_vg1: float,
        new_delta_vg2: float,
        deciding_human: str = "",
    ) -> None:
        """Approve with a modified voltage step."""
        path = self.queue_dir / f"{event_id}.json"
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        rec["outcome"] = HITLOutcome.MODIFIED.value
        rec["decided_at"] = time.time()
        rec["deciding_human"] = deciding_human
        rec["modified_delta_vg1"] = new_delta_vg1
        rec["modified_delta_vg2"] = new_delta_vg2
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _write_decision(
        self,
        event_id: str,
        outcome: HITLOutcome,
        deciding_human: str,
    ) -> None:
        path = self.queue_dir / f"{event_id}.json"
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        rec["outcome"] = outcome.value
        rec["decided_at"] = time.time()
        rec["deciding_human"] = deciding_human
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)

    def _mark_decided(self, event: HITLEvent) -> None:
        """Update the queue file to reflect a test-mode decision."""
        path = self.queue_dir / f"{event.id}.json"
        if not path.exists():
            return
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        rec["outcome"] = event.outcome.value
        rec["decided_at"] = event.decided_at
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _event_to_dict(event: HITLEvent, stage: TuningStage) -> Dict[str, Any]:
    """Serialise a HITLEvent to a JSON-safe dict."""
    proposal = event.proposal
    safe_dv = proposal.safe_delta_v if proposal else None
    dv = proposal.delta_v if proposal else None

    return {
        "id": str(event.id),
        "run_id": event.run_id,
        "step": event.step,
        "stage": stage.name,
        "trigger_reason": event.trigger_reason,
        "risk_score": event.risk_score,
        "outcome": event.outcome.value,
        "queued_at": event.queued_at,
        "decided_at": event.decided_at,
        "deciding_human": event.deciding_human,
        "proposal": {
            "delta_vg1": dv.vg1 if dv else None,
            "delta_vg2": dv.vg2 if dv else None,
            "safe_delta_vg1": safe_dv.vg1 if safe_dv else None,
            "safe_delta_vg2": safe_dv.vg2 if safe_dv else None,
            "clipped": proposal.clipped if proposal else False,
            "clip_warnings": proposal.clip_warnings if proposal else [],
        },
        # Fields the UI writes back
        "modified_delta_vg1": None,
        "modified_delta_vg2": None,
    }
