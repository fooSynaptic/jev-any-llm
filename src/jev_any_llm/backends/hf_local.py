"""Local Hugging Face CausalLM backend (optional dependency)."""

from __future__ import annotations

import math
from typing import Any

from jev_any_llm.backends.base import LogprobResult
from jev_any_llm.backends.openai_compat import _match_alias
from jev_any_llm.errors import JevAnyLlmBackendError
from jev_any_llm.tokens import resolve_alias_token_ids


class HuggingFaceBackend:
    name = "hf_local"

    def __init__(
        self,
        model_id: str,
        *,
        device: str | None = None,
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise JevAnyLlmBackendError(
                "hf backend requires optional deps: pip install 'jev-any-llm[hf]'"
            ) from exc

        self.torch = torch
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        torch_dtype = getattr(torch, dtype, torch.bfloat16)
        kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "dtype": torch_dtype,
        }
        if device == "cpu":
            kwargs["device_map"] = "cpu"
        else:
            kwargs["device_map"] = "auto" if device is None else device
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()

    def _chat_text(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return prompt

    def _logprobs_from_vector(
        self,
        log_probs,
        aliases: dict[str, list[str]],
        *,
        strict_single_token: bool,
        input_tokens: int,
    ) -> LogprobResult:
        token_ids = resolve_alias_token_ids(
            self.tokenizer, aliases, strict=strict_single_token
        )
        alias_logprobs = {
            canonical: float(log_probs[tid].item())
            for canonical, tid in token_ids.items()
            if math.isfinite(float(log_probs[tid].item()))
        }
        greedy_id = int(log_probs.argmax().item())
        greedy_token = self.tokenizer.decode([greedy_id])
        greedy_alias = _match_alias(greedy_token, aliases)
        if not alias_logprobs:
            return LogprobResult(
                alias_logprobs={},
                greedy_token=greedy_token,
                greedy_alias=greedy_alias,
                logprobs_missing=True,
                input_tokens=input_tokens,
                output_tokens=1,
            )
        return LogprobResult(
            alias_logprobs=alias_logprobs,
            greedy_token=greedy_token,
            greedy_alias=greedy_alias,
            logprobs_missing=False,
            input_tokens=input_tokens,
            output_tokens=1,
        )

    def next_token_logprobs(
        self,
        prompt: str,
        *,
        aliases: dict[str, list[str]],
        model: str | None = None,
        strict_single_token: bool = True,
    ) -> LogprobResult:
        torch = self.torch
        text = self._chat_text(prompt)
        encoded = self.tokenizer(text, return_tensors="pt")
        encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
        input_tokens = int(encoded["input_ids"].shape[-1])
        with torch.inference_mode():
            logits = self.model(**encoded).logits[0, -1, :].float()
            log_probs = torch.log_softmax(logits, dim=-1)
        return self._logprobs_from_vector(
            log_probs,
            aliases,
            strict_single_token=strict_single_token,
            input_tokens=input_tokens,
        )

    def prefill_state(self, state_text: str) -> dict[str, Any]:
        """Shared prefix (system + State) → KV cache; caller forks per branch."""
        torch = self.torch
        encoded = self.tokenizer(state_text, return_tensors="pt")
        encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
        with torch.inference_mode():
            out = self.model(**encoded, use_cache=True)
        return {
            "past_key_values": out.past_key_values,
            "prefix_tokens": int(encoded["input_ids"].shape[-1]),
        }

    def branch_next_token_logprobs(
        self,
        cache: dict[str, Any],
        question_suffix: str,
        *,
        aliases: dict[str, list[str]],
        strict_single_token: bool = True,
    ) -> LogprobResult:
        torch = self.torch
        encoded = self.tokenizer(question_suffix, return_tensors="pt")
        encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
        suffix_tokens = int(encoded["input_ids"].shape[-1])
        with torch.inference_mode():
            out = self.model(
                **encoded,
                past_key_values=cache["past_key_values"],
                use_cache=True,
            )
            log_probs = torch.log_softmax(out.logits[0, -1, :].float(), dim=-1)
        return self._logprobs_from_vector(
            log_probs,
            aliases,
            strict_single_token=strict_single_token,
            input_tokens=int(cache["prefix_tokens"]) + suffix_tokens,
        )
