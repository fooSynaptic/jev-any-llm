"""Request / response types and validation for decide() / POST /v1/systemone."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

from jev_any_llm.errors import JevAnyLlmSchemaError, JevAnyLlmValidationError

QuestionType = Literal["noul", "choice", "score"]

CHOICE_MIN, CHOICE_MAX = 2, 255
SCORE_MIN, SCORE_MAX = 2, 10


@dataclass(frozen=True)
class NoulQuestion:
    type: Literal["noul"]
    instructions: str
    criteria: dict[str, str] | None = None

    @property
    def true_text(self) -> str | None:
        if not self.criteria:
            return None
        return self.criteria.get("true")

    @property
    def false_text(self) -> str | None:
        if not self.criteria:
            return None
        return self.criteria.get("false")


@dataclass(frozen=True)
class ChoiceQuestion:
    type: Literal["choice"]
    instructions: str
    criteria: dict[str, str]


@dataclass(frozen=True)
class ScoreQuestion:
    type: Literal["score"]
    instructions: str
    criteria: list[str]


Question = NoulQuestion | ChoiceQuestion | ScoreQuestion


@dataclass
class DecideRequest:
    model: str
    state: Any
    questions: dict[str, Question]


@dataclass
class NoulAnswer:
    type: Literal["noul"]
    noul: float

    def to_dict(self) -> dict:
        return {"type": "noul", "noul": self.noul}


@dataclass
class ChoiceAnswer:
    type: Literal["choice"]
    choice: str
    probabilities: dict[str, float]
    confidence: float

    def to_dict(self) -> dict:
        return {
            "type": "choice",
            "choice": self.choice,
            "probabilities": self.probabilities,
            "confidence": self.confidence,
        }


@dataclass
class ScoreAnswer:
    type: Literal["score"]
    score: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float

    def to_dict(self) -> dict:
        return {
            "type": "score",
            "score": self.score,
            "legend": self.legend,
            "probabilities": self.probabilities,
            "confidence": self.confidence,
        }


Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    backend: str = "wrap"
    logprobs_missing: int = 0

    def to_dict(self) -> dict:
        out = {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "backend": self.backend,
        }
        if self.logprobs_missing:
            out["logprobs_missing"] = self.logprobs_missing
        return out


@dataclass
class DecideResponse:
    model: str
    answers: dict[str, Answer]
    usage: Usage = field(default_factory=Usage)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "answers": {k: v.to_dict() for k, v in self.answers.items()},
            "usage": self.usage.to_dict(),
        }


def noul(instructions: str, true: str | None = None, false: str | None = None) -> NoulQuestion:
    criteria = None
    if true is not None or false is not None:
        criteria = {}
        if true is not None:
            criteria["true"] = true
        if false is not None:
            criteria["false"] = false
    return NoulQuestion(type="noul", instructions=instructions, criteria=criteria)


def choice(instructions: str, criteria: dict[str, str]) -> ChoiceQuestion:
    return ChoiceQuestion(type="choice", instructions=instructions, criteria=criteria)


def score(instructions: str, criteria: list[str]) -> ScoreQuestion:
    return ScoreQuestion(type="score", instructions=instructions, criteria=criteria)


def _require_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise JevAnyLlmValidationError(f"{field_name} must be a non-empty string")
    return value


def parse_question(qid: str, raw: Any) -> Question:
    if not isinstance(raw, dict):
        raise JevAnyLlmValidationError(f"question {qid!r} must be an object", details={"id": qid})
    qtype = raw.get("type")
    instructions = _require_str(raw.get("instructions"), f"questions.{qid}.instructions")
    if qtype == "noul":
        criteria = raw.get("criteria")
        if criteria is not None and not isinstance(criteria, dict):
            raise JevAnyLlmValidationError(f"questions.{qid}.criteria must be an object or omitted")
        if criteria:
            for key in criteria:
                if key not in ("true", "false"):
                    raise JevAnyLlmValidationError(
                        f"questions.{qid}.criteria keys must be true/false",
                        details={"id": qid, "key": key},
                    )
                _require_str(criteria[key], f"questions.{qid}.criteria.{key}")
        return NoulQuestion(type="noul", instructions=instructions, criteria=criteria)
    if qtype == "choice":
        criteria = raw.get("criteria")
        if not isinstance(criteria, dict):
            raise JevAnyLlmValidationError(f"questions.{qid}.criteria must be an object of option→text")
        if not (CHOICE_MIN <= len(criteria) <= CHOICE_MAX):
            raise JevAnyLlmValidationError(
                f"choice cardinality must be {CHOICE_MIN}–{CHOICE_MAX}",
                details={"id": qid, "n": len(criteria)},
            )
        cleaned: dict[str, str] = {}
        for opt_id, text in criteria.items():
            cleaned[_require_str(opt_id, f"questions.{qid}.criteria key")] = _require_str(
                text, f"questions.{qid}.criteria[{opt_id}]"
            )
        return ChoiceQuestion(type="choice", instructions=instructions, criteria=cleaned)
    if qtype == "score":
        criteria = raw.get("criteria")
        if not isinstance(criteria, list):
            raise JevAnyLlmValidationError(f"questions.{qid}.criteria must be an ordered list of levels")
        if not (SCORE_MIN <= len(criteria) <= SCORE_MAX):
            raise JevAnyLlmValidationError(
                f"score levels must be {SCORE_MIN}–{SCORE_MAX}",
                details={"id": qid, "n": len(criteria)},
            )
        levels = [_require_str(item, f"questions.{qid}.criteria[{i}]") for i, item in enumerate(criteria)]
        return ScoreQuestion(type="score", instructions=instructions, criteria=levels)
    raise JevAnyLlmValidationError(
        f"questions.{qid}.type must be noul|choice|score",
        details={"id": qid, "type": qtype},
    )


def parse_request(raw: Any) -> DecideRequest:
    if not isinstance(raw, dict):
        raise JevAnyLlmValidationError("request body must be a JSON object")
    model = _require_str(raw.get("model"), "model")
    if "state" not in raw:
        raise JevAnyLlmValidationError("state is required")
    state = raw["state"]
    if not isinstance(state, (str, dict, list)):
        raise JevAnyLlmValidationError("state must be a string, object, or array")
    questions_raw = raw.get("questions")
    if not isinstance(questions_raw, dict) or not questions_raw:
        raise JevAnyLlmValidationError("questions must be a non-empty object")
    questions: dict[str, Question] = {}
    for qid, qbody in questions_raw.items():
        _require_str(qid, "question id")
        questions[qid] = parse_question(qid, qbody)
    return DecideRequest(model=model, state=state, questions=questions)


def validate_answer(qid: str, question: Question, answer: Answer) -> None:
    if question.type == "noul":
        if not isinstance(answer, NoulAnswer):
            raise JevAnyLlmSchemaError(f"answer {qid!r} type mismatch", details={"id": qid})
        if not (0.0 <= answer.noul <= 1.0) or math.isnan(answer.noul):
            raise JevAnyLlmSchemaError(f"answer {qid!r}.noul out of [0,1]", details={"id": qid})
        return
    if question.type == "choice":
        if not isinstance(answer, ChoiceAnswer):
            raise JevAnyLlmSchemaError(f"answer {qid!r} type mismatch", details={"id": qid})
        assert isinstance(question, ChoiceQuestion)
        opts = set(question.criteria)
        if set(answer.probabilities) != opts:
            raise JevAnyLlmSchemaError(
                f"answer {qid!r}.probabilities keys must match criteria",
                details={"id": qid},
            )
        if answer.choice not in opts:
            raise JevAnyLlmSchemaError(f"answer {qid!r}.choice not in criteria", details={"id": qid})
        _check_prob_dist(qid, answer.probabilities)
        _check_unit(qid, "confidence", answer.confidence)
        return
    if question.type == "score":
        if not isinstance(answer, ScoreAnswer):
            raise JevAnyLlmSchemaError(f"answer {qid!r} type mismatch", details={"id": qid})
        assert isinstance(question, ScoreQuestion)
        keys = {str(i) for i in range(len(question.criteria))}
        if set(answer.probabilities) != keys or set(answer.legend) != keys:
            raise JevAnyLlmSchemaError(
                f"answer {qid!r} legend/probabilities must be 0..K-1",
                details={"id": qid},
            )
        for i, text in enumerate(question.criteria):
            if answer.legend[str(i)] != text:
                raise JevAnyLlmSchemaError(f"answer {qid!r}.legend mismatch", details={"id": qid})
        _check_prob_dist(qid, answer.probabilities)
        _check_unit(qid, "confidence", answer.confidence)
        if math.isnan(answer.score):
            raise JevAnyLlmSchemaError(f"answer {qid!r}.score is nan", details={"id": qid})
        return
    raise JevAnyLlmSchemaError(f"unknown question type for {qid!r}")


def validate_response(request: DecideRequest, response: DecideResponse) -> None:
    if set(response.answers) != set(request.questions):
        raise JevAnyLlmSchemaError(
            "answers keys must match questions keys",
            details={
                "expected": sorted(request.questions),
                "got": sorted(response.answers),
            },
        )
    for qid, question in request.questions.items():
        validate_answer(qid, question, response.answers[qid])


def _check_unit(qid: str, name: str, value: float) -> None:
    if not (0.0 <= value <= 1.0) or math.isnan(value):
        raise JevAnyLlmSchemaError(f"answer {qid!r}.{name} out of [0,1]", details={"id": qid})


def _check_prob_dist(qid: str, probs: dict[str, float]) -> None:
    total = 0.0
    for key, value in probs.items():
        if value < -1e-9 or math.isnan(value):
            raise JevAnyLlmSchemaError(
                f"answer {qid!r}.probabilities[{key}] invalid",
                details={"id": qid},
            )
        total += value
    if abs(total - 1.0) > 1e-3:
        raise JevAnyLlmSchemaError(
            f"answer {qid!r}.probabilities must sum to 1",
            details={"id": qid, "sum": total},
        )
