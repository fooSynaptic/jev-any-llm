"""Small numeric helpers for closed-set post-process."""

from __future__ import annotations

import math


def softmax(logits: dict[str, float]) -> dict[str, float]:
    if not logits:
        raise ValueError("softmax requires at least one logit")
    peak = max(logits.values())
    exps = {key: math.exp(value - peak) for key, value in logits.items()}
    total = sum(exps.values())
    return {key: value / total for key, value in exps.items()}


def confidence_from_probs(probs: dict[str, float]) -> float:
    """1 - H(p) / log(K). Uniform → 0, one-hot → 1."""
    k = len(probs)
    if k <= 1:
        return 1.0
    entropy = 0.0
    for p in probs.values():
        if p > 0.0:
            entropy -= p * math.log(p)
    return 1.0 - entropy / math.log(k)


def one_hot(keys: list[str], winner: str) -> dict[str, float]:
    if winner not in keys:
        raise ValueError(f"winner {winner!r} not in keys")
    return {key: (1.0 if key == winner else 0.0) for key in keys}
