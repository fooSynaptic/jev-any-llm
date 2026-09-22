"""OpenAI-compatible chat/completions backend (vLLM, cloud APIs)."""

from __future__ import annotations

import os
from typing import Any

import httpx

from jev_any_llm.backends.base import LogprobResult
from jev_any_llm.errors import JevAnyLlmBackendError


class OpenAICompatBackend:
    """OpenAI ``/v1/chat/completions`` wire format (the mainstream custom-LLM entry).

    Compatible hosts include OpenAI, Azure OpenAI-style gateways, vLLM, SGLang,
    TGI, DeepSeek, Together, Fireworks, SiliconFlow, and DashScope-compatible
    OpenAI endpoints. Requires ``logprobs=true`` (falls back to one-hot when the
    server omits logprobs).
    """

    name = "openai_compat"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        default_model: str | None = None,
        timeout: float = 120.0,
        extra_headers: dict[str, str] | None = None,
    ):
        env_base = (
            os.environ.get("JEV_ANY_LLM_OPENAI_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        self.base_url = _normalize_base_url(base_url or env_base)
        self.api_key = (
            api_key
            or os.environ.get("JEV_ANY_LLM_OPENAI_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        )
        self.default_model = default_model or os.environ.get("JEV_ANY_LLM_MODEL") or "gpt-4o-mini"
        self.timeout = timeout
        self.extra_headers = dict(extra_headers or {})

    def next_token_logprobs(
        self,
        prompt: str,
        *,
        aliases: dict[str, list[str]],
        model: str | None = None,
    ) -> LogprobResult:
        model_id = model or self.default_model
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers.setdefault("Authorization", f"Bearer {self.api_key}")
        # Ask for top logprobs large enough to cover alias variants.
        body: dict[str, Any] = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": min(20, max(5, len(aliases) * 4)),
        }
        url = f"{self.base_url}/chat/completions"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, headers=headers, json=body)
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPError as exc:
            raise JevAnyLlmBackendError(f"openai_compat request failed: {exc}") from exc

        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        greedy = content[:32] if content else None
        usage = payload.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)

        top = _extract_top_logprobs(choice)
        if greedy is None and top:
            # Prefer the highest-prob token string if content was empty.
            greedy = max(top, key=top.get)

        if not top:
            greedy_alias = _match_alias(greedy, aliases)
            return LogprobResult(
                alias_logprobs={},
                greedy_token=greedy,
                greedy_alias=greedy_alias,
                logprobs_missing=True,
                input_tokens=input_tokens,
                output_tokens=output_tokens or 1,
                raw=payload,
            )

        alias_logprobs = _best_alias_logprobs(top, aliases)
        greedy_alias = _match_alias(greedy, aliases)
        if not alias_logprobs:
            return LogprobResult(
                alias_logprobs={},
                greedy_token=greedy,
                greedy_alias=greedy_alias,
                logprobs_missing=True,
                input_tokens=input_tokens,
                output_tokens=output_tokens or 1,
                raw=payload,
            )
        return LogprobResult(
            alias_logprobs=alias_logprobs,
            greedy_token=greedy,
            greedy_alias=greedy_alias,
            logprobs_missing=False,
            input_tokens=input_tokens,
            output_tokens=output_tokens or 1,
            raw=payload,
        )


def _normalize_base_url(url: str) -> str:
    cleaned = url.rstrip("/")
    if cleaned.endswith("/v1") or cleaned.endswith("/openai/v1"):
        return cleaned
    # Bare host:port (common vLLM launch) → append /v1.
    return f"{cleaned}/v1"


def _extract_top_logprobs(choice: dict[str, Any]) -> dict[str, float]:
    """Parse both chat.completions content[].top_logprobs and legacy shapes."""
    top: dict[str, float] = {}
    logprobs_block = choice.get("logprobs")
    if not isinstance(logprobs_block, dict):
        return top

    content_parts = logprobs_block.get("content") or []
    if content_parts:
        first = content_parts[0] or {}
        for item in first.get("top_logprobs") or []:
            if isinstance(item, dict) and "token" in item:
                top[item["token"]] = float(item.get("logprob", float("-inf")))
            elif isinstance(item, dict):
                # Some servers return {token_str: logprob} maps inside the list.
                for token, value in item.items():
                    if isinstance(value, (int, float)):
                        top[token] = float(value)
        if first.get("token") and first["token"] not in top and "logprob" in first:
            top[first["token"]] = float(first["logprob"])
        return top

    # Legacy completions-style: logprobs.top_logprobs = [{token: lp, ...}, ...]
    legacy = logprobs_block.get("top_logprobs") or []
    if legacy and isinstance(legacy[0], dict):
        for token, value in legacy[0].items():
            if isinstance(value, (int, float)):
                top[token] = float(value)
    return top


def _best_alias_logprobs(top: dict[str, float], aliases: dict[str, list[str]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for canonical, surfaces in aliases.items():
        best = None
        for surface in surfaces:
            if surface in top:
                value = top[surface]
                if best is None or value > best:
                    best = value
        if best is not None:
            out[canonical] = best
    return out


def _match_alias(token: str | None, aliases: dict[str, list[str]]) -> str | None:
    if token is None:
        return None
    stripped = token.strip()
    for canonical, surfaces in aliases.items():
        for surface in surfaces:
            if token == surface or stripped == surface.strip() or stripped.lower() == surface.strip().lower():
                return canonical
    return None
