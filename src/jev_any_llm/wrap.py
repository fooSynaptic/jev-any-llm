"""Phase 1 wrap: isolated generate + logprob + jev-any-llm post-process."""

from __future__ import annotations

import json
import string
from typing import Any

from jev_any_llm.backends.base import GenerateBackend, LogprobResult
from jev_any_llm.contract import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecideRequest,
    DecideResponse,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
    Usage,
    validate_response,
)
from jev_any_llm.errors import JevAnyLlmBackendError
from jev_any_llm.mathutil import confidence_from_probs, one_hot, softmax


def render_state(state: Any) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def alias_surfaces(canonical: str) -> list[str]:
    """Common tokenizer spellings for a short alias token."""
    variants = [
        canonical,
        f" {canonical}",
        canonical.lower(),
        f" {canonical.lower()}",
        canonical.upper(),
        f" {canonical.upper()}",
    ]
    # Preserve order, drop dupes.
    seen: set[str] = set()
    out: list[str] = []
    for item in variants:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def bind_aliases(question: Question) -> tuple[dict[str, list[str]], list[str]]:
    """Return (canonical→surfaces, ordered canonical ids for the closed set)."""
    if isinstance(question, NoulQuestion):
        order = ["Yes", "No"]
        return {name: alias_surfaces(name) for name in order}, order
    if isinstance(question, ChoiceQuestion):
        letters = list(string.ascii_uppercase)
        if len(question.criteria) > len(letters):
            raise JevAnyLlmBackendError("choice cardinality exceeds A–Z alias budget in wrap v1")
        order = letters[: len(question.criteria)]
        return {name: alias_surfaces(name) for name in order}, order
    if isinstance(question, ScoreQuestion):
        order = [str(i) for i in range(len(question.criteria))]
        return {name: alias_surfaces(name) for name in order}, order
    raise JevAnyLlmBackendError(f"unsupported question type: {type(question)}")


def render_prompt(state: Any, question: Question) -> str:
    state_text = render_state(state)
    lines = [
        "You are a typed predictor. Answer with exactly one alias token from the closed set.",
        "",
        "State:",
        state_text,
        "",
        f"Question: {question.instructions}",
    ]
    if isinstance(question, NoulQuestion):
        true_text = question.true_text or "The condition holds"
        false_text = question.false_text or "The condition does not hold"
        lines.extend(
            [
                "",
                "Closed set:",
                f"- Yes: {true_text}",
                f"- No: {false_text}",
                "",
                "Answer (Yes or No):",
            ]
        )
    elif isinstance(question, ChoiceQuestion):
        lines.append("")
        lines.append("Closed set:")
        for letter, (opt_id, text) in zip(
            string.ascii_uppercase, question.criteria.items(), strict=False
        ):
            lines.append(f"- {letter} ({opt_id}): {text}")
        letters = ", ".join(string.ascii_uppercase[: len(question.criteria)])
        lines.extend(["", f"Answer ({letters}):"])
    elif isinstance(question, ScoreQuestion):
        lines.append("")
        lines.append("Closed set (ordered levels):")
        for i, text in enumerate(question.criteria):
            lines.append(f"- {i}: {text}")
        lines.extend(["", f"Answer (0–{len(question.criteria) - 1}):"])
    else:
        raise JevAnyLlmBackendError("unknown question type")
    return "\n".join(lines)


def _probs_from_result(
    result: LogprobResult, order: list[str]
) -> tuple[dict[str, float], bool]:
    if result.logprobs_missing or not result.alias_logprobs:
        winner = result.greedy_alias or order[0]
        if winner not in order:
            winner = order[0]
        return one_hot(order, winner), True
    logits = {alias: result.alias_logprobs.get(alias, -50.0) for alias in order}
    return softmax(logits), False


def post_process(question: Question, result: LogprobResult) -> tuple[Any, bool]:
    aliases, order = bind_aliases(question)
    del aliases  # surfaces already consumed by the backend call
    probs, missing = _probs_from_result(result, order)

    if isinstance(question, NoulQuestion):
        return NoulAnswer(type="noul", noul=float(probs["Yes"])), missing

    if isinstance(question, ChoiceQuestion):
        opt_ids = list(question.criteria.keys())
        mapped = {opt_ids[i]: probs[order[i]] for i in range(len(opt_ids))}
        winner = max(mapped, key=mapped.get)
        return (
            ChoiceAnswer(
                type="choice",
                choice=winner,
                probabilities=mapped,
                confidence=confidence_from_probs(mapped),
            ),
            missing,
        )

    if isinstance(question, ScoreQuestion):
        legend = {str(i): text for i, text in enumerate(question.criteria)}
        expected = sum(i * probs[str(i)] for i in range(len(question.criteria)))
        return (
            ScoreAnswer(
                type="score",
                score=float(expected),
                legend=legend,
                probabilities={k: float(v) for k, v in probs.items()},
                confidence=confidence_from_probs(probs),
            ),
            missing,
        )

    raise JevAnyLlmBackendError("unknown question type")


def decide_one(
    backend: GenerateBackend,
    *,
    model: str,
    state: Any,
    question: Question,
) -> tuple[Any, LogprobResult, bool]:
    aliases, _order = bind_aliases(question)
    prompt = render_prompt(state, question)
    result = backend.next_token_logprobs(prompt, aliases=aliases, model=model)
    answer, missing = post_process(question, result)
    return answer, result, missing


def decide(backend: GenerateBackend, request: DecideRequest) -> DecideResponse:
    """Run one isolated generate per question, then assemble a typed response."""
    answers = {}
    input_tokens = 0
    output_tokens = 0
    missing_count = 0
    for qid, question in request.questions.items():
        answer, result, missing = decide_one(
            backend, model=request.model, state=request.state, question=question
        )
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
            backend=getattr(backend, "name", "wrap"),
            logprobs_missing=missing_count,
        ),
    )
    validate_response(request, response)
    return response
