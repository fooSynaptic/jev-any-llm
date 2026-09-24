"""System Two sketch — escalate hard questions (Phase 2+; not Phase 1 scope).

Minimal stub so callers can wire a router later without inventing a new
package layout. Default: never escalate (System One only).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from jev_any_llm.contract import DecideResponse, Question


@dataclass(frozen=True)
class EscalateDecision:
    """Whether a question should leave the fast closed-set path."""

    escalate: bool
    reason: str = ""
    confidence: float | None = None


class SystemTwoRouter(Protocol):
    """Frozen-backbone head (or heuristic) that flags hard questions."""

    def should_escalate(
        self,
        state: Any,
        question_id: str,
        question: Question,
        system_one: DecideResponse | None = None,
    ) -> EscalateDecision:
        ...


class NeverEscalate:
    """Default Phase 1 behavior: stay on System One for every question."""

    name = "never"

    def should_escalate(
        self,
        state: Any,
        question_id: str,
        question: Question,
        system_one: DecideResponse | None = None,
    ) -> EscalateDecision:
        del state, question_id, question, system_one
        return EscalateDecision(escalate=False, reason="phase1_system_one_only")


class LowConfidenceEscalate:
    """Heuristic sketch: escalate when System One confidence is below a floor."""

    name = "low_confidence"

    def __init__(self, floor: float = 0.55):
        self.floor = float(floor)

    def should_escalate(
        self,
        state: Any,
        question_id: str,
        question: Question,
        system_one: DecideResponse | None = None,
    ) -> EscalateDecision:
        del state, question
        if system_one is None or question_id not in system_one.answers:
            return EscalateDecision(escalate=False, reason="no_system_one_answer")
        answer = system_one.answers[question_id]
        conf = getattr(answer, "confidence", None)
        if conf is None and hasattr(answer, "noul"):
            # Binary margin as a stand-in confidence for Noul answers.
            conf = abs(2.0 * float(answer.noul) - 1.0)
        if conf is None:
            return EscalateDecision(escalate=False, reason="no_confidence")
        if float(conf) < self.floor:
            return EscalateDecision(
                escalate=True,
                reason=f"confidence<{self.floor}",
                confidence=float(conf),
            )
        return EscalateDecision(
            escalate=False, reason="confidence_ok", confidence=float(conf)
        )
