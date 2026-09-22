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

Multi-question calls are **N isolated generates** that share `state` text only.
Application code owns thresholds, routing, and side effects.

Architecture detail, locks, and Phases 2–3 (native heads / single-forward):
[DESIGN.md](../DESIGN.md).
