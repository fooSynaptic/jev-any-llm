"""Contract tests for Phase 1 wrap path."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from jev_any_llm import Client, choice, noul, score
from jev_any_llm.backends.mock import MockBackend, peaked
from jev_any_llm.contract import (
    DecideRequest,
    parse_request,
    validate_response,
)
from jev_any_llm.errors import JevAnyLlmValidationError
from jev_any_llm.mathutil import confidence_from_probs, softmax
from jev_any_llm.serve import create_app
from jev_any_llm.wrap import decide, decide_one, render_prompt


STATE = {
    "message": "Charged twice for order A-104. I want a refund today.",
    "order": {"id": "A-104", "charges": [49, 49]},
}


def _standard_questions():
    return {
        "refund": noul(
            "Does `message` ask for money back?",
            true="Explicit refund or chargeback request",
            false="No money-back request",
        ),
        "team": choice(
            "Which team should handle `message`?",
            {
                "billing": "Charges, invoices, refunds",
                "technical": "Bugs, outages, integrations",
                "other": "None of the above",
            },
        ),
        "urgency": score(
            "How time-sensitive is `message`?",
            [
                "No deadline or consequence",
                "Wants a reply this week",
                "Asks for action today or cites ongoing loss",
            ],
        ),
    }


def _scripted_backend() -> MockBackend:
    return MockBackend(
        scores={
            "*": {
                **peaked("Yes", ["Yes", "No"], margin=4.0),
                **peaked("A", ["A", "B", "C"], margin=3.0),
                **peaked("2", ["0", "1", "2"], margin=2.5),
            }
        }
    )


def test_parse_and_validate_round_trip():
    body = {
        "model": "mock",
        "state": STATE,
        "questions": {
            "refund": {
                "type": "noul",
                "instructions": "Does `message` ask for money back?",
                "criteria": {"true": "yes", "false": "no"},
            },
            "team": {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {"billing": "b", "technical": "t", "other": "o"},
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent?",
                "criteria": ["low", "mid", "high"],
            },
        },
    }
    request = parse_request(body)
    backend = _scripted_backend()
    response = decide(backend, request)
    validate_response(request, response)
    assert response.answers["refund"].noul > 0.9
    assert response.answers["team"].choice == "billing"
    assert abs(sum(response.answers["team"].probabilities.values()) - 1.0) < 1e-6
    assert response.answers["urgency"].score > 1.0
    assert set(response.answers["urgency"].legend) == {"0", "1", "2"}


def test_client_decide_helpers():
    client = Client(model="mock", backend=_scripted_backend())
    result = client.decide(STATE, _standard_questions())
    assert result.to_dict()["answers"]["team"]["choice"] == "billing"
    assert 0.0 <= result.answers["urgency"].confidence <= 1.0


def test_schema_rejects_bad_choice_cardinality():
    with pytest.raises(JevAnyLlmValidationError):
        parse_request(
            {
                "model": "mock",
                "state": "x",
                "questions": {
                    "q": {
                        "type": "choice",
                        "instructions": "only one",
                        "criteria": {"a": "alone"},
                    }
                },
            }
        )


def test_multi_question_matches_isolated_singles():
    backend = _scripted_backend()
    questions = _standard_questions()
    multi = decide(
        backend,
        DecideRequest(model="mock", state=STATE, questions=questions),
    )
    for qid, question in questions.items():
        single = decide(
            backend,
            DecideRequest(model="mock", state=STATE, questions={qid: question}),
        )
        multi_ans = multi.answers[qid].to_dict()
        single_ans = single.answers[qid].to_dict()
        assert multi_ans == single_ans


def test_isolation_probe_no_leak_on_wrap_path():
    """Gold for B flips if A's answer is shown. Wrap prompts must not leak A."""
    backend = MockBackend(
        scores={
            "*": peaked("No", ["Yes", "No"], margin=4.0),
            # Contaminated prompt (sibling answer in state/prompt) flips B.
            r"refund=Yes|Answer:\s*Yes": peaked("Yes", ["Yes", "No"], margin=4.0),
        }
    )
    q_a = noul("Is this a refund request?")
    q_b = noul("Should we escalate immediately?")

    clean = decide(
        backend,
        DecideRequest(
            model="mock",
            state=STATE,
            questions={"refund": q_a, "escalate": q_b},
        ),
    )
    assert clean.answers["escalate"].noul < 0.5

    # Contaminated application-style second call: first answer put into state.
    contaminated_state = {**STATE, "prior_answers": {"refund": "Yes"}}
    # Also inject into a prompt-visible string so the mock regex can fire if
    # wrap incorrectly inlined A's answer into B's prompt.
    leaked = decide_one(
        backend,
        model="mock",
        state=contaminated_state,
        question=q_b,
    )[0]
    # State contamination alone must not change B on the wrap path: only the
    # rendered prompt matters, and wrap does not paste sibling answers.
    prompt_b = render_prompt(contaminated_state, q_b)
    assert "Answer: Yes" not in prompt_b
    assert "refund=Yes" not in prompt_b
    # escalate stays No because wrap isolation holds.
    assert leaked.noul < 0.5

    # Positive control: a deliberately leaked prompt *does* flip B.
    leaked_prompt_backend = MockBackend(
        scores={
            "*": peaked("No", ["Yes", "No"], margin=4.0),
            r"sibling answer was Yes": peaked("Yes", ["Yes", "No"], margin=4.0),
        }
    )
    from jev_any_llm.backends.base import LogprobResult
    from jev_any_llm.wrap import bind_aliases, post_process

    aliases, _ = bind_aliases(q_b)
    # Simulate a bad packed prompt that includes the sibling answer text.
    bad = leaked_prompt_backend.next_token_logprobs(
        "Question B. Note: sibling answer was Yes",
        aliases=aliases,
        model="mock",
    )
    flipped, _ = post_process(q_b, bad)
    assert flipped.noul > 0.5


def test_logprobs_missing_fallback_one_hot():
    backend = MockBackend(force_missing_logprobs=True, greedy_alias="B")
    client = Client(model="mock", backend=backend)
    result = client.decide(
        STATE,
        {
            "team": choice(
                "Which team?",
                {"billing": "b", "technical": "t", "other": "o"},
            )
        },
    )
    assert result.answers["team"].choice == "technical"
    assert result.answers["team"].probabilities == {
        "billing": 0.0,
        "technical": 1.0,
        "other": 0.0,
    }
    assert result.usage.logprobs_missing == 1


def test_http_systemone_round_trip():
    app = create_app(Client(model="mock", backend=_scripted_backend()))
    http = TestClient(app)
    models = http.get("/v1/models")
    assert models.status_code == 200
    assert models.json()["data"][0]["id"] == "mock"

    body = {
        "model": "mock",
        "state": STATE,
        "questions": {
            "refund": {
                "type": "noul",
                "instructions": "Does message ask for money back?",
            },
            "team": {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {
                    "billing": "Charges",
                    "technical": "Bugs",
                    "other": "Other",
                },
            },
            "urgency": {
                "type": "score",
                "instructions": "Urgency?",
                "criteria": ["low", "mid", "high"],
            },
        },
    }
    response = http.post("/v1/systemone", json=body)
    assert response.status_code == 200
    payload = response.json()
    assert payload["answers"]["team"]["choice"] == "billing"
    assert "noul" in payload["answers"]["refund"]

    alias = http.post("/v1/decide", json=body)
    assert alias.status_code == 200


def test_http_validation_error():
    app = create_app(Client(model="mock", backend=MockBackend()))
    http = TestClient(app)
    response = http.post("/v1/systemone", json={"model": "mock", "state": {}})
    assert response.status_code == 400
    assert response.json()["error"] == "validation_error"


def test_from_openai_factory_uses_compat_backend():
    client = Client.from_openai(
        model="Qwen/Qwen2.5-7B-Instruct",
        base_url="http://127.0.0.1:8000",
        api_key="EMPTY",
    )
    assert client.backend.name == "openai_compat"
    assert client.backend.base_url.endswith("/v1")
    assert client.backend.default_model == "Qwen/Qwen2.5-7B-Instruct"


def test_base_url_implies_openai_without_backend_kind():
    client = Client(model="my-served-model", base_url="http://llm.example.com:8000/v1")
    assert client.backend.name == "openai_compat"

