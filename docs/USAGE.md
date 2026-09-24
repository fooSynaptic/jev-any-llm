# Usage

Wire any instruct LLM that can return next-token logprobs into Jev-mode, then call
`decide(state, questions)`.

## Install

```bash
pip install -e '.[dev]'
pytest -q
```

Optional local weights: `pip install -e '.[hf]'`.

## Quick start

```python
from jev_any_llm import Client, noul, choice, score

client = Client.from_openai(
    model="Qwen/Qwen2.5-7B-Instruct",
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY",
)

result = client.decide(
    state={"message": "Charged twice for order A-104. I want a refund today."},
    questions={
        "refund": noul("Does `message` ask for money back?"),
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
    },
)

if result.answers["refund"].noul > 0.8:
    route_billing(result.answers["urgency"].score)
```

## Branched mode (shared prefill)

When the backend can fork a KV cache (local HF / mock), set `mode="branched"` so
`state` is prefaced once and each question only appends its suffix before reading
closed-set logits (no answer-token generation):

```python
client = Client.from_hf("Qwen/Qwen3.5-4B", mode="branched")
# or
client = Client.from_mock(mode="branched")
```

OpenAI-compatible servers have no portable KV-fork API; `mode="branched"` falls
back to isolated generates and tags `usage.backend` with `isolated_fallback`.

Optional temperature artifact (default T=1, uncalibrated):

```python
from jev_any_llm.calibration import TemperatureProfile

client = Client.from_mock(
    mode="branched",
    temperature_profile="configs/temperature_profile.example.json",
)
```

Strict single-token alias check (tokenizer only; three bench models):

```bash
python scripts/smoke_tokenizer_aliases.py
# or: JEV_OPENMODELS_ROOT=/path/to/openmodels python scripts/smoke_tokenizer_aliases.py
```

## Hosted Jev interchange

The request body for `POST /v1/systemone` matches the public Jev System One shape
(`model` + `state` + `questions`). Swap only the transport:

| Field | This wrap | Hosted Jev |
| --- | --- | --- |
| Base URL | your `jev-any-llm-serve` or gateway | hosted `/v1` root |
| Auth | `Authorization: Bearer …` if your gateway needs it | vendor API key |
| `model` | instruct checkpoint / served id | hosted model id |

Answers keep the same `noul` / `choice` / `score` / `probabilities` / `confidence`
fields so application code can point at either endpoint.

Playground: start the server and open `http://127.0.0.1:8080/`.

System Two (escalate sketch, Phase 2+): [SYSTEM_TWO.md](SYSTEM_TWO.md).

Branched accuracy / latency on the three bench models:
[REPORT_branched.md](../experiments/jev_mode_benchmark/REPORT_branched.md).

## Backends (custom LLM entry)

The default custom entry is the **OpenAI-compatible** chat API
(`POST {base_url}/chat/completions` with `logprobs=true`). That wire format is
spoken by vLLM, SGLang, TGI, OpenAI, DeepSeek, Together, Fireworks,
SiliconFlow, DashScope-compatible gateways, and Azure OpenAI-style `base_url`s.

```python
from jev_any_llm import Client

# Local vLLM / SGLang / TGI (OpenAI server mode)
client = Client.from_vllm(
    model="Qwen/Qwen2.5-7B-Instruct",
    base_url="http://127.0.0.1:8000/v1",  # bare host:port also ok → /v1 appended
)

# Hosted OpenAI-compatible endpoint
client = Client.from_openai(
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    api_key="sk-…",
)

# Official OpenAI (or any gateway on OPENAI_BASE_URL)
client = Client.from_openai(model="gpt-4o-mini", api_key="sk-…")

# Local Hugging Face weights
client = Client.from_hf("Qwen/Qwen2.5-7B-Instruct")

# Constructor form — any base_url implies OpenAI-compatible
client = Client(
    model="Qwen/Qwen2.5-7B-Instruct",
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY",
)
```

### Environment

| Variable | Role |
| --- | --- |
| `OPENAI_BASE_URL` / `JEV_ANY_LLM_OPENAI_BASE_URL` | Default chat API root (normalized to end in `/v1`) |
| `OPENAI_API_KEY` / `JEV_ANY_LLM_OPENAI_API_KEY` | Bearer token |
| `JEV_ANY_LLM_MODEL` | Default model id when the constructor omits one |

Servers that omit logprobs get a logged `logprobs=missing` one-hot fallback.

### HTTP server

```bash
jev-any-llm-serve --backend openai --model Qwen/Qwen2.5-7B-Instruct \
  --base-url http://127.0.0.1:8000/v1 --port 8080
```

Exposes `POST /v1/systemone` in front of the same `Client`.

## Typed questions

| Primitive | What is scored | Answer fields |
| --- | --- | --- |
| **Noul** | `Yes` / `No` (or `true`/`false`) | `noul` = P(yes) after 2-way softmax |
| **Choice** | One alias token per option | `choice` = argmax, full `probabilities`, `confidence` |
| **Score** | One alias per ordered level | `score` = Σ i·p(i), `legend`, `probabilities`, `confidence` |

Confidence is a concentration of the same logprob distribution:
`1 − H(p) / log K`.

## Wrap pipeline (Phase 1)

| Step | What happens |
| --- | --- |
| 1. Plug in a model | Instruct checkpoint or `chat/completions` endpoint with logprobs |
| 2. Isolate questions | Each prompt is `state` + that question only |
| 3. Generate + logprob | Map options to short aliases; keep those tokens; softmax |
| 4. Post-process | Fill `noul` / `choice` / `score` / `probabilities` / `confidence` |

Multi-question calls default to **N isolated generates** that share `state` text
only. With `mode="branched"` on HF/mock, `state` is prefaced once and each
question reuses that KV before reading option-token logits.

Application code owns thresholds, routing, and side effects.

Architecture detail, locks, and Phases 2–3 (native heads / single-forward):
[DESIGN.md](../DESIGN.md).
