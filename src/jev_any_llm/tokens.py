"""Strict single-token alias resolution for closed-set readout."""

from __future__ import annotations

from typing import Any, Protocol

from jev_any_llm.errors import JevAnyLlmBackendError


class _Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...


def single_token_id(tokenizer: _Tokenizer, surface: str) -> int | None:
    """Return vocab id if ``surface`` is exactly one token; else None."""
    ids = tokenizer.encode(surface, add_special_tokens=False)
    if len(ids) == 1:
        return int(ids[0])
    return None


def resolve_alias_token_ids(
    tokenizer: _Tokenizer,
    aliases: dict[str, list[str]],
    *,
    strict: bool = True,
) -> dict[str, int]:
    """Map each canonical alias to one vocab id.

    Prefers the first surface that is a single token. Under ``strict=True``,
    raises if a canonical has no single-token surface (no silent truncate).
    """
    resolved: dict[str, int] = {}
    for canonical, surfaces in aliases.items():
        token_id = None
        for surface in surfaces:
            token_id = single_token_id(tokenizer, surface)
            if token_id is not None:
                break
        if token_id is None:
            if strict:
                raise JevAnyLlmBackendError(
                    f"alias {canonical!r} has no single-token surface among {surfaces!r}; "
                    "refusing multi-token truncate"
                )
            continue
        resolved[canonical] = token_id
    if strict and len(resolved) != len(aliases):
        missing = sorted(set(aliases) - set(resolved))
        raise JevAnyLlmBackendError(f"unresolved single-token aliases: {missing}")
    return resolved


def validate_closed_set_tokenizer(
    tokenizer: Any,
    *,
    choice_k: int = 4,
    score_k: int = 5,
) -> dict[str, Any]:
    """Smoke check that Yes/No, A.., and 0.. are single tokens for a tokenizer."""
    from jev_any_llm.wrap import alias_surfaces, bind_aliases
    from jev_any_llm.contract import choice, noul, score

    report: dict[str, Any] = {"ok": True, "sets": {}}
    checks = {
        "noul": bind_aliases(noul("smoke?"))[0],
        "choice": bind_aliases(
            choice(
                "smoke?",
                {f"opt{i}": f"option {i}" for i in range(choice_k)},
            )
        )[0],
        "score": bind_aliases(
            score("smoke?", [f"level {i}" for i in range(score_k)])
        )[0],
    }
    for name, aliases in checks.items():
        try:
            ids = resolve_alias_token_ids(tokenizer, aliases, strict=True)
            report["sets"][name] = {"ok": True, "ids": ids}
        except JevAnyLlmBackendError as exc:
            report["ok"] = False
            report["sets"][name] = {"ok": False, "error": str(exc)}
    # Also record which surfaces worked for letter A (debug).
    report["surfaces_A"] = {
        surface: single_token_id(tokenizer, surface) for surface in alias_surfaces("A")
    }
    return report
