"""Branched prefill, single-token aliases, temperature profile, System Two sketch."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from jev_any_llm import Client, choice, noul, score
from jev_any_llm.backends.mock import MockBackend, peaked
from jev_any_llm.branched import decide_branched, render_question_suffix, render_shared_prefix
from jev_any_llm.calibration import TemperatureProfile, scale_logits
from jev_any_llm.contract import DecideRequest
from jev_any_llm.errors import JevAnyLlmBackendError
from jev_any_llm.serve import create_app
from jev_any_llm.system_two import LowConfidenceEscalate, NeverEscalate
from jev_any_llm.tokens import resolve_alias_token_ids, single_token_id, validate_closed_set_tokenizer
from jev_any_llm.wrap import alias_surfaces, bind_aliases


STATE = {"message": "Charged twice. Refund today."}


class _ToyTokenizer:
    """Maps short closed-set surfaces to single fake ids."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        table = {
            "Yes": 1,
            " No": 2,
            "No": 3,
            "A": 10,
            " B": 11,
            "B": 12,
            "C": 13,
            "0": 20,
            "1": 21,
            "2": 22,
            "hello world": 99,
        }
        # Prefer exact match; multi-word → multi-id for strict fail tests.
        if text in table and text != "hello world":
            return [table[text]]
        if text == "hello world":
            return [99, 100]
        # Fallback: one id per character so unknown multi-char fails strict.
        if len(text) == 1 or text.startswith(" ") and len(text) == 2:
            return [hash(text) % 1000 + 50]
        return [ord(c) for c in text] or [0]


def _scripted() -> MockBackend:
    return MockBackend(
        scores={
            "*": {
                **peaked("Yes", ["Yes", "No"], margin=4.0),
                **peaked("A", ["A", "B", "C"], margin=3.0),
                **peaked("2", ["0", "1", "2"], margin=2.5),
            }
        }
    )


def test_single_token_resolve_strict():
    tok = _ToyTokenizer()
    assert single_token_id(tok, "Yes") == 1
    assert single_token_id(tok, "hello world") is None
    aliases = {"Yes": alias_surfaces("Yes"), "No": ["No", " No"]}
    ids = resolve_alias_token_ids(tok, aliases, strict=True)
    assert set(ids) == {"Yes", "No"}
    with pytest.raises(JevAnyLlmBackendError):
        resolve_alias_token_ids(tok, {"X": ["hello world"]}, strict=True)


def test_validate_closed_set_toy_tokenizer():
    report = validate_closed_set_tokenizer(_ToyTokenizer())
    assert report["ok"] is True
    assert report["sets"]["noul"]["ok"] is True


def test_branched_matches_isolated_on_mock():
    backend = _scripted()
    questions = {
        "refund": noul("Refund?"),
        "team": choice("Team?", {"billing": "b", "technical": "t", "other": "o"}),
        "urgency": score("Urgency?", ["low", "mid", "high"]),
    }
    request = DecideRequest(model="mock", state=STATE, questions=questions)
    branched = decide_branched(backend, request)
    isolated = Client(model="mock", backend=backend, mode="isolated").decide(
        STATE, questions
    )
    assert branched.answers["team"].choice == isolated.answers["team"].choice
    assert branched.answers["refund"].noul == pytest.approx(
        isolated.answers["refund"].noul, abs=1e-9
    )
    assert "branched" in branched.usage.backend


def test_client_branched_mode():
    client = Client.from_mock(_scripted(), mode="branched")
    result = client.decide(
        STATE,
        {"refund": noul("Refund?"), "team": choice("T?", {"a": "A", "b": "B", "c": "C"})},
    )
    assert result.answers["refund"].noul > 0.9
    assert result.answers["team"].choice == "a"
    assert "branched" in result.usage.backend


def test_temperature_profile_scales_and_roundtrips(tmp_path: Path):
    profile = TemperatureProfile(model="mock", default=2.0, by_question={"refund": 0.5})
    path = tmp_path / "t.json"
    profile.save(path)
    loaded = TemperatureProfile.load(path)
    assert loaded.temperature_for("refund") == 0.5
    assert loaded.temperature_for("other") == 2.0
    scaled = scale_logits({"Yes": 4.0, "No": 0.0}, 2.0)
    assert scaled["Yes"] == 2.0

    hot = Client(
        model="mock",
        backend=_scripted(),
        mode="branched",
        temperature_profile=TemperatureProfile(model="mock", default=0.5),
    ).decide(STATE, {"refund": noul("Refund?")})
    cool = Client(
        model="mock",
        backend=_scripted(),
        mode="branched",
        temperature_profile=TemperatureProfile(model="mock", default=5.0),
    ).decide(STATE, {"refund": noul("Refund?")})
    # Lower T → sharper → higher |noul - 0.5| margin from 0.5 toward 1.
    assert abs(hot.answers["refund"].noul - 0.5) > abs(cool.answers["refund"].noul - 0.5)


def test_shared_prefix_and_suffix_compose():
    q = noul("Does message ask for money back?")
    prefix = render_shared_prefix(STATE)
    suffix = render_question_suffix(q)
    assert "State:" in prefix
    assert "Question:" in suffix
    assert "State:" not in suffix
    assert "typed predictor" in prefix


def test_system_two_never_and_low_confidence():
    client = Client.from_mock(_scripted())
    result = client.decide(
        STATE,
        {
            "team": choice(
                "Team?",
                {"billing": "b", "technical": "t", "other": "o"},
            )
        },
    )
    never = NeverEscalate()
    assert (
        never.should_escalate(STATE, "team", choice("T?", {"a": "A", "b": "B"}), result).escalate
        is False
    )
    router = LowConfidenceEscalate(floor=0.99)
    decision = router.should_escalate(
        STATE,
        "team",
        choice("T?", {"billing": "b", "technical": "t", "other": "o"}),
        result,
    )
    assert decision.confidence is not None
    weak = LowConfidenceEscalate(floor=0.01)
    assert (
        weak.should_escalate(
            STATE,
            "team",
            choice("T?", {"billing": "b", "technical": "t", "other": "o"}),
            result,
        ).escalate
        is False
    )


def test_playground_and_decide_http():
    app = create_app(Client.from_mock(_scripted(), mode="branched"))
    http = TestClient(app)
    page = http.get("/")
    assert page.status_code == 200
    assert "playground" in page.text.lower() or "jev-any-llm" in page.text
    body = {
        "model": "mock",
        "state": STATE,
        "questions": {
            "refund": {"type": "noul", "instructions": "Refund?"},
        },
    }
    response = http.post("/v1/systemone", json=body)
    assert response.status_code == 200
    assert response.json()["answers"]["refund"]["noul"] > 0.9


def test_example_temperature_profile_exists():
    path = Path(__file__).resolve().parents[1] / "configs" / "temperature_profile.example.json"
    profile = TemperatureProfile.load(path)
    assert profile.model
    assert profile.default == 1.0
