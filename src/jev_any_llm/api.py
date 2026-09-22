"""Public Python API: Client, noul, choice, score helpers."""

from __future__ import annotations

from typing import Any

from jev_any_llm.backends.base import GenerateBackend
from jev_any_llm.backends.mock import MockBackend
from jev_any_llm.backends.openai_compat import OpenAICompatBackend
from jev_any_llm.contract import (
    DecideRequest,
    DecideResponse,
    Question,
    choice as choice_fn,
    noul as noul_fn,
    parse_request,
    score as score_fn,
)
from jev_any_llm.wrap import decide as wrap_decide

# Re-export helpers at the package edge.
noul = noul_fn
choice = choice_fn
score = score_fn


class Client:
    """Wire any instruct LLM that can return next-token logprobs.

    Preferred entry for custom / hosted models is the OpenAI-compatible
    chat API (vLLM, SGLang, TGI, OpenAI, DeepSeek, Together, Fireworks,
    DashScope-compatible, Azure OpenAI-style ``base_url``, …)::

        client = Client.from_openai(
            model="Qwen/Qwen2.5-7B-Instruct",
            base_url="http://127.0.0.1:8000/v1",
            api_key="EMPTY",
        )
    """

    def __init__(
        self,
        model: str = "mock",
        *,
        backend: GenerateBackend | None = None,
        backend_kind: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        hf_device: str | None = None,
        timeout: float = 120.0,
        extra_headers: dict[str, str] | None = None,
        default_headers: dict[str, str] | None = None,
    ):
        self.model = model
        headers = extra_headers or default_headers
        if backend is not None:
            self.backend = backend
        else:
            kind = backend_kind or _infer_backend_kind(model, base_url)
            self.backend = _build_backend(
                kind,
                model=model,
                base_url=base_url,
                api_key=api_key,
                hf_device=hf_device,
                timeout=timeout,
                extra_headers=headers,
            )

    @classmethod
    def from_openai(
        cls,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
        extra_headers: dict[str, str] | None = None,
    ) -> "Client":
        """Any OpenAI-compatible ``/v1/chat/completions`` endpoint with logprobs.

        Works with OpenAI, vLLM, SGLang, TGI, DeepSeek, Together, Fireworks,
        SiliconFlow, and most gateways that speak the same JSON schema.
        ``base_url`` should include the ``/v1`` suffix when the server uses it.
        """
        return cls(
            model=model,
            backend_kind="openai",
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            extra_headers=extra_headers,
        )

    @classmethod
    def from_vllm(
        cls,
        model: str,
        *,
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key: str = "EMPTY",
        timeout: float = 120.0,
    ) -> "Client":
        """Local or remote vLLM OpenAI server (default ``localhost:8000``)."""
        return cls.from_openai(
            model, base_url=base_url, api_key=api_key, timeout=timeout
        )

    @classmethod
    def from_hf(
        cls,
        model: str,
        *,
        device: str | None = None,
    ) -> "Client":
        """Local ``transformers`` CausalLM (requires ``pip install 'jev-any-llm[hf]'``)."""
        return cls(model=model, backend_kind="hf", hf_device=device)

    @classmethod
    def from_mock(cls, backend: GenerateBackend | None = None) -> "Client":
        return cls(model="mock", backend=backend or MockBackend())

    def decide(
        self,
        state: Any,
        questions: dict[str, Question],
        *,
        model: str | None = None,
    ) -> DecideResponse:
        request = DecideRequest(
            model=model or self.model,
            state=state,
            questions=questions,
        )
        return wrap_decide(self.backend, request)

    def decide_raw(self, body: dict[str, Any]) -> DecideResponse:
        request = parse_request(body)
        return wrap_decide(self.backend, request)


def _infer_backend_kind(model: str, base_url: str | None) -> str:
    if model == "mock" or model.startswith("mock:"):
        return "mock"
    # Any explicit base_url is treated as OpenAI-compatible — the mainstream
    # wire format for custom instruct endpoints.
    if base_url:
        return "openai"
    if model.startswith("openai:") or model.startswith("gpt-") or model.startswith("o1"):
        return "openai"
    if model.startswith("hf:") or ( "/" in model and not model.startswith("openai/")):
        # Hugging Face hub ids look like org/name; still allow openai/… via prefix.
        if model.startswith("openai/"):
            return "openai"
        return "hf"
    return "openai"


def _build_backend(
    kind: str,
    *,
    model: str,
    base_url: str | None,
    api_key: str | None,
    hf_device: str | None,
    timeout: float = 120.0,
    extra_headers: dict[str, str] | None = None,
) -> GenerateBackend:
    if kind == "mock":
        return MockBackend()
    if kind in ("openai", "openai_compat", "vllm", "sglang", "tgi"):
        cleaned = model.removeprefix("openai:").removeprefix("openai/")
        return OpenAICompatBackend(
            base_url=base_url,
            api_key=api_key,
            default_model=cleaned,
            timeout=timeout,
            extra_headers=extra_headers,
        )
    if kind in ("hf", "hf_local", "huggingface"):
        from jev_any_llm.backends.hf_local import HuggingFaceBackend

        return HuggingFaceBackend(model.removeprefix("hf:"), device=hf_device)
    raise ValueError(f"unknown backend_kind: {kind}")
