"""Branched logit readout: shared state prefill, per-question suffix.

Prefix cache on `state` from the Phase 1 wrap path. OpenAI-compat backends
fall back to isolated generates (no portable KV-fork API).
"""

from __future__ import annotations

import copy
from typing import Any, Protocol

from jev_any_llm.backends.base import GenerateBackend, LogprobResult
from jev_any_llm.calibration import TemperatureProfile, scale_logits
from jev_any_llm.contract import (
    DecideRequest,
    DecideResponse,
    Question,
    Usage,
    validate_response,
)
from jev_any_llm.wrap import (
    bind_aliases,
    post_process,
    render_prompt,
    render_state,
)


class BranchCapableBackend(Protocol):
    """Optional capability: shared prefill + branch logits."""

    name: str

    def prefill_state(self, state_text: str) -> Any:
        """Return an opaque cache handle for ``state_text``."""
        ...

    def branch_next_token_logprobs(
        self,
        cache: Any,
        question_suffix: str,
        *,
        aliases: dict[str, list[str]],
        strict_single_token: bool = True,
    ) -> LogprobResult:
        """Continue from ``cache`` with ``question_suffix``; read next-token logprobs."""
        ...


def render_shared_prefix(state: Any) -> str:
    """Shared preamble + State block (prefill once, fork per question)."""
    return (
        "You are a typed predictor. Answer with exactly one alias token "
        "from the closed set.\n"
        f"\nState:\n{render_state(state)}\n\n"
    )


def render_question_suffix(question: Question) -> str:
    """Question block only — continues after ``render_shared_prefix``."""
    full = render_prompt("", question)
    marker = "State:\n\n\nQuestion:"
    if marker in full:
        return "Question:" + full.split(marker, 1)[1]
    idx = full.find("Question:")
    return full[idx:] if idx >= 0 else full


def _fork_cache(cache: Any) -> Any:
    """Clone a backend cache so parallel branches do not mutate each other."""
    if isinstance(cache, dict) and "past_key_values" in cache:
        pkv = cache["past_key_values"]
        if hasattr(pkv, "copy") and callable(pkv.copy):
            forked_pkv = pkv.copy()
        else:
            forked_pkv = copy.deepcopy(pkv)
        return {**cache, "past_key_values": forked_pkv}
    return copy.deepcopy(cache)


def _apply_temperature(
    result: LogprobResult,
    order: list[str],
    temperature: float,
) -> LogprobResult:
    if result.logprobs_missing or not result.alias_logprobs:
        return result
    scaled = scale_logits(
        {alias: result.alias_logprobs.get(alias, -50.0) for alias in order},
        temperature,
    )
    # Convert back to log-space relative probs via log(softmax) is unnecessary —
    # post_process softmaxes alias_logprobs directly, so store scaled logits.
    return LogprobResult(
        alias_logprobs=scaled,
        greedy_token=result.greedy_token,
        greedy_alias=result.greedy_alias,
        logprobs_missing=False,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        raw=result.raw,
    )


def supports_branch(backend: GenerateBackend) -> bool:
    return hasattr(backend, "prefill_state") and hasattr(backend, "branch_next_token_logprobs")


def decide_branched(
    backend: GenerateBackend,
    request: DecideRequest,
    *,
    temperature_profile: TemperatureProfile | None = None,
    strict_single_token: bool = True,
) -> DecideResponse:
    """Shared-state prefill + per-question branch readout when the backend allows.

    Otherwise falls back to isolated ``decide`` (same answers contract).
    """
    if not supports_branch(backend):
        from jev_any_llm.wrap import decide as isolated_decide

        response = isolated_decide(backend, request)
        # Annotate fallback in usage backend string.
        response.usage.backend = f"{response.usage.backend}+isolated_fallback"
        return _maybe_retemp(response, request, backend, temperature_profile)

    prefix = render_shared_prefix(request.state)
    cache = backend.prefill_state(prefix)  # type: ignore[attr-defined]
    answers = {}
    input_tokens = 0
    output_tokens = 0
    missing_count = 0
    for qid, question in request.questions.items():
        aliases, order = bind_aliases(question)
        suffix = render_question_suffix(question)
        branch_cache = _fork_cache(cache)
        result = backend.branch_next_token_logprobs(  # type: ignore[attr-defined]
            branch_cache,
            suffix,
            aliases=aliases,
            strict_single_token=strict_single_token,
        )
        temp = 1.0
        if temperature_profile is not None:
            temp = temperature_profile.temperature_for(qid)
        result = _apply_temperature(result, order, temp)
        answer, missing = post_process(question, result)
        answers[qid] = answer
        input_tokens += result.input_tokens
        output_tokens += result.output_tokens
        if missing:
            missing_count += 1

    response = DecideResponse(
        model=request.model,
        answers=answers,
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            backend=f"{getattr(backend, 'name', 'wrap')}+branched",
            logprobs_missing=missing_count,
        ),
    )
    validate_response(request, response)
    return response


def _maybe_retemp(
    response: DecideResponse,
    request: DecideRequest,
    backend: GenerateBackend,
    temperature_profile: TemperatureProfile | None,
) -> DecideResponse:
    """Isolated path already softmaxed; re-run with temperature if profile set."""
    if temperature_profile is None:
        return response
    from jev_any_llm.wrap import decide_one

    answers = {}
    input_tokens = 0
    output_tokens = 0
    missing_count = 0
    for qid, question in request.questions.items():
        _aliases, order = bind_aliases(question)
        answer, result, missing = decide_one(
            backend, model=request.model, state=request.state, question=question
        )
        del answer
        temp = temperature_profile.temperature_for(qid)
        result = _apply_temperature(result, order, temp)
        answer, missing2 = post_process(question, result)
        answers[qid] = answer
        input_tokens += result.input_tokens
        output_tokens += result.output_tokens
        if missing or missing2:
            missing_count += 1
    out = DecideResponse(
        model=request.model,
        answers=answers,
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            backend=response.usage.backend,
            logprobs_missing=missing_count,
        ),
    )
    validate_response(request, out)
    return out
