"""Local Hugging Face CausalLM backend (optional dependency)."""

from __future__ import annotations

import math
from typing import Any

from jev_any_llm.backends.base import LogprobResult
from jev_any_llm.backends.openai_compat import _match_alias
from jev_any_llm.errors import JevAnyLlmBackendError


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

    def next_token_logprobs(
        self,
        prompt: str,
        *,
        aliases: dict[str, list[str]],
        model: str | None = None,
    ) -> LogprobResult:
        torch = self.torch
        messages = [{"role": "user", "content": prompt}]
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = prompt

        encoded = self.tokenizer(text, return_tensors="pt")
        encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
        input_tokens = int(encoded["input_ids"].shape[-1])
        with torch.inference_mode():
            logits = self.model(**encoded).logits[0, -1, :].float()
            log_probs = torch.log_softmax(logits, dim=-1)

        alias_logprobs: dict[str, float] = {}
        for canonical, surfaces in aliases.items():
            best = None
            for surface in surfaces:
                token_ids = self.tokenizer.encode(surface, add_special_tokens=False)
                if len(token_ids) != 1:
                    # Multi-token surfaces: use first token as a proxy (wrap path
                    # prefers single-token aliases; variants stay single-token).
                    if not token_ids:
                        continue
                    token_ids = token_ids[:1]
                value = float(log_probs[token_ids[0]].item())
                if best is None or value > best:
                    best = value
            if best is not None and math.isfinite(best):
                alias_logprobs[canonical] = best

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
