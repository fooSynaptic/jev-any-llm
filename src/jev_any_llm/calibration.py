"""Optional post-hoc temperature profile (default: uncalibrated T=1)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TemperatureProfile:
    """Per-model (optional per-question) temperature for closed-set softmax.

    ``default`` applies when a question id is absent from ``by_question``.
    """

    model: str
    default: float = 1.0
    by_question: dict[str, float] | None = None

    def temperature_for(self, question_id: str | None = None) -> float:
        if question_id and self.by_question and question_id in self.by_question:
            return float(self.by_question[question_id])
        return float(self.default)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "default": self.default,
            "by_question": dict(self.by_question or {}),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TemperatureProfile":
        return cls(
            model=str(payload.get("model") or ""),
            default=float(payload.get("default") or 1.0),
            by_question={
                str(key): float(value)
                for key, value in (payload.get("by_question") or {}).items()
            },
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TemperatureProfile":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def scale_logits(logits: dict[str, float], temperature: float) -> dict[str, float]:
    t = max(float(temperature), 1e-4)
    return {key: value / t for key, value in logits.items()}
