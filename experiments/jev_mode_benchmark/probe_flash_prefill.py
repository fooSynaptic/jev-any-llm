#!/usr/bin/env python3
"""Stage 1 prefill-only latency on DeepSeek-V4.1-Flash TP8.

Times one start_pos=0 forward of the same AR prompt used by benchmark_flash_tp8.
Warmup and call count match that file's latency_probe (6 warmups, 24 calls).
"""

from __future__ import annotations

import argparse

import torch
import torch.distributed as dist

from benchmark_flash_tp8 import is_rank0, last_logits, load_stack, log, to_ids
from common import ar_prompt, cuda_time_ms, latency_summary, read_ag_news, stratified_subset, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--inference-dir", required=True)
    parser.add_argument("--encoding-dir", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-class", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--latency-calls", type=int, default=24)
    parser.add_argument("--latency-batch", type=int, default=4)
    parser.add_argument("--fwd-batch", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stack = load_stack(args)
    rows = stratified_subset(read_ag_news(args.test_csv), args.per_class, seed=20260920)
    ids = to_ids(stack["tokenizer"], stack["encode_messages"], rows[0]["text"], ar_prompt, args.max_length)

    def fn():
        return last_logits(stack["model"], ids)

    with torch.inference_mode():
        for _ in range(6):
            fn()
            dist.barrier()
        times = []
        for _ in range(args.latency_calls):
            _, elapsed = cuda_time_ms(fn)
            dist.barrier()
            times.append(elapsed)
    summary = latency_summary(times, 1)
    summary["prompt_tokens"] = len(ids)
    payload = {
        "model": "DeepSeek-V4.1-Flash",
        "arm": "stage1_prefill_only",
        "prompt": "ar_prompt",
        "prefill_only": summary,
    }
    if is_rank0():
        write_json(args.output, payload)
        log(payload)
    dist.barrier()


if __name__ == "__main__":
    main()
