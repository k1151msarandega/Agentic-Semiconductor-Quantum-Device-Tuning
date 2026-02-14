"""
qdot/core/governance.py
=======================
GovernanceLogger — writes and reads the immutable audit trail.

Every agent action, observation, and plan transition is logged here.
The schema is structured JSON; every record is a Decision from types.py.

Ported from the hackathon's governance_log_decision() function, but:
  - Uses the Decision datatype rather than free-form dicts
  - Adds structured querying (load by run_id, filter by intent)
  - Thread-safe append via file-level locking
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any
from uuid import UUID

from qdot.core.types import Decision, TuningStage


class GovernanceLogger:
    """
    Append-only governance log for a single agent run.

    One logger instance per run. Each call to log() appends a JSON line
    to <log_dir>/<run_id>.jsonl.

    Usage:
        logger = GovernanceLogger(run_id="abc123", log_dir="data/governance")
        logger.log(decision)

        decisions = GovernanceLogger.load(run_id="abc123", log_dir="data/governance")
    """

    def __init__(self, run_id: str, log_dir: str = "data/governance") -> None:
        self.run_id = run_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.log_dir / f"{run_id}.jsonl"
        self._lock = threading.Lock()

    # -----------------------------------------------------------------------
    # Core write operation
    # -----------------------------------------------------------------------

    def log(self, decision: Decision) -> None:
        """Append a Decision to the log. Thread-safe."""
        record = _decision_to_dict(decision)
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

    def log_raw(
        self,
        intent: str,
        stage: TuningStage,
        observation_summary: Optional[Dict[str, Any]] = None,
        action_summary: Optional[Dict[str, Any]] = None,
        rationale: str = "",
        llm_tokens_used: int = 0,
        llm_call_id: Optional[str] = None,
        step: int = 0,
    ) -> Decision:
        """
        Convenience method: create a Decision from raw fields and log it.
        Returns the Decision so callers can attach it to ExperimentState.
        """
        d = Decision(
            run_id=self.run_id,
            step=step,
            timestamp=time.time(),
            intent=intent,
            stage=stage,
            observation_summary=observation_summary or {},
            action_summary=action_summary or {},
            rationale=rationale,
            llm_tokens_used=llm_tokens_used,
            llm_call_id=llm_call_id,
        )
        self.log(d)
        return d

    # -----------------------------------------------------------------------
    # Read / query operations
    # -----------------------------------------------------------------------

    @classmethod
    def load(cls, run_id: str, log_dir: str = "data/governance") -> List[Decision]:
        """Load all decisions for a run. Returns empty list if run not found."""
        path = Path(log_dir) / f"{run_id}.jsonl"
        if not path.exists():
            return []
        decisions = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    decisions.append(_dict_to_decision(json.loads(line)))
        return decisions

    @classmethod
    def filter(
        cls,
        decisions: List[Decision],
        intent: Optional[str] = None,
        stage: Optional[TuningStage] = None,
        step_range: Optional[tuple] = None,
    ) -> List[Decision]:
        """Filter a list of decisions by intent, stage, or step range."""
        result = decisions
        if intent is not None:
            result = [d for d in result if d.intent == intent]
        if stage is not None:
            result = [d for d in result if d.stage == stage]
        if step_range is not None:
            lo, hi = step_range
            result = [d for d in result if lo <= d.step <= hi]
        return result

    # -----------------------------------------------------------------------
    # Summary export
    # -----------------------------------------------------------------------

    @classmethod
    def summarise(cls, decisions: List[Decision]) -> Dict[str, Any]:
        """Produce a summary dict suitable for the final report."""
        if not decisions:
            return {"total": 0}

        intent_counts: Dict[str, int] = {}
        total_tokens = 0
        safety_violations = 0

        for d in decisions:
            intent_counts[d.intent] = intent_counts.get(d.intent, 0) + 1
            total_tokens += d.llm_tokens_used
            if d.action_summary.get("safety_violations"):
                safety_violations += d.action_summary["safety_violations"]

        return {
            "run_id": decisions[0].run_id,
            "total_decisions": len(decisions),
            "intent_breakdown": intent_counts,
            "total_llm_tokens": total_tokens,
            "safety_violations_logged": safety_violations,
            "steps_taken": max(d.step for d in decisions),
            "duration_s": (
                decisions[-1].timestamp - decisions[0].timestamp
                if len(decisions) > 1
                else 0.0
            ),
        }


# ---------------------------------------------------------------------------
# Serialisation helpers (keep JSON-serialisable — no numpy, no UUID objects)
# ---------------------------------------------------------------------------

def _decision_to_dict(d: Decision) -> Dict[str, Any]:
    return {
        "id": str(d.id),
        "run_id": d.run_id,
        "step": d.step,
        "timestamp": d.timestamp,
        "intent": d.intent,
        "stage": d.stage.name,
        "observation_summary": d.observation_summary,
        "action_summary": d.action_summary,
        "rationale": d.rationale,
        "llm_tokens_used": d.llm_tokens_used,
        "llm_call_id": d.llm_call_id,
    }


def _dict_to_decision(rec: Dict[str, Any]) -> Decision:
    return Decision(
        id=UUID(rec["id"]),
        run_id=rec["run_id"],
        step=rec["step"],
        timestamp=rec["timestamp"],
        intent=rec["intent"],
        stage=TuningStage[rec["stage"]],
        observation_summary=rec.get("observation_summary", {}),
        action_summary=rec.get("action_summary", {}),
        rationale=rec.get("rationale", ""),
        llm_tokens_used=rec.get("llm_tokens_used", 0),
        llm_call_id=rec.get("llm_call_id"),
    )
