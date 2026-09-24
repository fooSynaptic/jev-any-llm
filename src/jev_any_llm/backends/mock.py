"""Deterministic backend for contract tests (no network / GPU)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from jev_any_llm.backends.base import LogprobResult


@dataclass
class MockBackend:
    """Scores aliases from a scripted map, with optional prompt regex keys.

    ``scores`` maps a key to ``{canonical_alias: logit}``. The key is either
    ``"*"`` (default) or a regex matched against the prompt. Isolation probes
    can plant different scores when a sibling answer string appears in the
    prompt — which must *not* happen on the wrap path.
    """

    name: str = "mock"
    scores: dict[str, dict[str, float]] = field(default_factory=dict)
    force_missing_logprobs: bool = False
    greedy_alias: str | None = None

    def next_token_logprobs(
        self,
        prompt: str,
        *,
        aliases: dict[str, list[str]],
        model: str | None = None,
    ) -> LogprobResult:
        table = self.scores.get("*", {})
        for pattern, values in self.scores.items():
            if pattern == "*":
                continue
            if re.search(pattern, prompt, flags=re.IGNORECASE | re.DOTALL):
                table = values
                break
        if self.force_missing_logprobs:
            winner = self.greedy_alias or next(iter(aliases))
            return LogprobResult(
                alias_logprobs={},
                greedy_token=winner,
                greedy_alias=winner,
                logprobs_missing=True,
                input_tokens=len(prompt.split()),
                output_tokens=1,
            )
        alias_logprobs = {alias: float(table.get(alias, -10.0)) for alias in aliases}
        # Soft pick for greedy display.
        winner = max(alias_logprobs, key=alias_logprobs.get)
        return LogprobResult(
            alias_logprobs=alias_logprobs,
            greedy_token=winner,
            greedy_alias=winner,
            logprobs_missing=False,
            input_tokens=len(prompt.split()),
            output_tokens=1,
            raw={"table": table},
        )

    def prefill_state(self, state_text: str) -> dict:
        """Mock shared prefill — stores the full shared prefix text."""
        return {"prefix": state_text, "prefix_tokens": max(1, len(state_text.split()))}

    def branch_next_token_logprobs(
        self,
        cache: dict,
        question_suffix: str,
        *,
        aliases: dict[str, list[str]],
        strict_single_token: bool = True,
    ) -> LogprobResult:
        del strict_single_token
        prompt = f"{cache.get('prefix', '')}{question_suffix}"
        result = self.next_token_logprobs(prompt, aliases=aliases)
        result.input_tokens = int(cache.get("prefix_tokens", 0)) + len(question_suffix.split())
        return result


def peaked(alias: str, aliases: list[str], margin: float = 3.0) -> dict[str, float]:
    """Helper: one alias dominates by ``margin`` nats."""
    return {name: (margin if name == alias else 0.0) for name in aliases}


def uniform(aliases: list[str]) -> dict[str, float]:
    return {name: 0.0 for name in aliases}
