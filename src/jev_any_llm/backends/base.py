"""Backend protocol: next-token logprobs over a closed alias set."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class LogprobResult:
    """Unified readout for one isolated generate.

    ``alias_logprobs`` maps canonical alias → best logprob among surface variants.
    When the backend cannot return logprobs, ``logprobs_missing`` is True and
    ``greedy_alias`` (if set) is used to build a one-hot distribution.
    """

    alias_logprobs: dict[str, float] = field(default_factory=dict)
    greedy_token: str | None = None
    greedy_alias: str | None = None
    logprobs_missing: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    raw: dict | None = None


class GenerateBackend(Protocol):
    name: str

    def next_token_logprobs(
        self,
        prompt: str,
        *,
        aliases: dict[str, list[str]],
        model: str | None = None,
    ) -> LogprobResult:
        """Score the next token under ``prompt``.

        ``aliases`` maps canonical label (e.g. ``"Yes"``, ``"A"``) to surface
        spellings to try (``["Yes", " Yes", "yes", "YES"]``).
        """
        ...
