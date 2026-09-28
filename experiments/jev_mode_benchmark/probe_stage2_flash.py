#!/usr/bin/env python3
"""Stage 2 latency gaps on DeepSeek-V4.1-Flash TP8.

Scheme 4 letter-forward and scheme 2 packed quality are already on the
scorecard. This probe times scheme 1 (generate one token) and scheme 3
(codebook forward) at batch 1. Scheme 5 stays unsupported: the official CSA2
kernels have no bidirectional mask switch.
"""

from __future__ import annotations

import argparse

import torch
import torch.distributed as dist

from benchmark_flash_tp8 import is_rank0, last_logits, load_stack, log, to_ids
from common import codebook_prompt, choice_prompt, cuda_time_ms, latency_summary, read_ag_news, stratified_subset, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--inference-dir", required=True)
    parser.add_argument("--encoding-dir", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--latency-calls", type=int, default=24)
    parser.add_argument("--latency-batch", type=int, default=4)
    parser.add_argument("--fwd-batch", type=int, default=8)
    return parser.parse_args()


def time_calls(fn, warmups: int, calls: int) -> dict:
    with torch.inference_mode():
        for _ in range(warmups):
            fn()
            dist.barrier()
        times = []
        for _ in range(calls):
            _, elapsed = cuda_time_ms(fn)
            dist.barrier()
            times.append(elapsed)
    return latency_summary(times, 1)


def main() -> None:
    args = parse_args()
    stack = load_stack(args)
    rows = stratified_subset(read_ag_news(args.test_csv), 1000, seed=20260920)
    text = rows[0]["text"]
    letter_ids = to_ids(stack["tokenizer"], stack["encode_messages"], text, choice_prompt, args.max_length)
    code_ids = to_ids(stack["tokenizer"], stack["encode_messages"], text, codebook_prompt, args.max_length)
    eos = stack["tokenizer"].eos_token_id
    generate = stack["ds_generate"]
    model = stack["model"]

    def generate1():
        return generate(model, [letter_ids], 1, eos)

    latency = {
        "scheme_1_generate1": time_calls(generate1, 6, args.latency_calls),
        "scheme_3_codebook_forward": time_calls(lambda: last_logits(model, code_ids), 6, args.latency_calls),
    }
    latency["scheme_1_generate1"]["prompt_tokens"] = len(letter_ids)
    latency["scheme_3_codebook_forward"]["prompt_tokens"] = len(code_ids)
    payload = {
        "model": "DeepSeek-V4.1-Flash",
        "arm": "stage2_gaps",
        "latency": latency,
        "scheme_5": "unsupported on CSA2 official kernels",
    }
    if is_rank0():
        write_json(args.output, payload)
        log(payload)
    dist.barrier()


if __name__ == "__main__":
    main()
