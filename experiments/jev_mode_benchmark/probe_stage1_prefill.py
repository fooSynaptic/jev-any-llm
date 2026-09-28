#!/usr/bin/env python3
"""Stage 1 prefill-only latency for one instruct decoder.

Same AR prompt, batch-1 row, CUDA-event window, and 10 warmups as
benchmark_vanilla.latency_probe. The timed call is one prompt forward with
use_cache=True and no generated tokens.
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import ar_prompt, cuda_time_ms, latency_summary, read_ag_news, render_instruct, stratified_subset, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-class", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--latency-calls", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = stratified_subset(read_ag_news(args.test_csv), args.per_class, seed=20260920)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    prompts = [render_instruct(tokenizer, [ar_prompt(rows[0]["text"])])[0]]
    batch = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    batch = {key: value.cuda(non_blocking=True) for key, value in batch.items()}

    def fn():
        return model(**batch, use_cache=True)

    with torch.inference_mode():
        for _ in range(10):
            fn()
        times = [cuda_time_ms(fn)[1] for _ in range(args.latency_calls)]
    summary = latency_summary(times, 1)
    summary["prompt_tokens"] = int(batch["attention_mask"][0].sum().item())
    payload = {
        "model": args.model,
        "arm": "stage1_prefill_only",
        "prompt": "ar_prompt",
        "prefill_only": summary,
    }
    write_json(args.output, payload)
    print(payload, flush=True)


if __name__ == "__main__":
    main()
