#!/usr/bin/env python3
"""Shared-prefill KV-cache scoring for multi-token label words.

Naive scoring re-runs the whole prompt for every candidate. Candidates share
the prompt, so one prefill plus teacher-forced decode of the tails is enough.
This file measures that cost and checks that the scores match the naive arm.
"""

from __future__ import annotations

import argparse

from copy import deepcopy

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark_zero_train import (
    FULL_WORDS,
    left_pad,
    prompt_ids,
    softmax_rows,
    word_prompt,
)
from common import (
    cuda_time_ms,
    latency_summary,
    multiclass_metrics,
    read_ag_news,
    stratified_subset,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-class", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--latency-calls", type=int, default=24)
    parser.add_argument("--agree-n", type=int, default=200)
    parser.add_argument("--only", choices=["all", "quality", "latency"], default="all")
    return parser.parse_args()


def encode_tails(tokenizer, words: list[str]) -> list[list[int]]:
    return [tokenizer.encode(word, add_special_tokens=False) for word in words]


TENSOR_ATTRS = ("keys", "values", "conv_states", "recurrent_states")


def clone_cache(past):
    """Qwen3.5 mixes DynamicLayer and LinearAttentionLayer; deepcopy keeps both."""
    if past is None:
        return None
    return deepcopy(past)


def expand_cache(past, repeats: int):
    """Mutates `past`. Callers prefill a fresh cache and do not reuse it."""
    if repeats == 1 or past is None:
        return past
    for layer in past.layers:
        for attr in TENSOR_ATTRS:
            tensor = getattr(layer, attr, None)
            if isinstance(tensor, torch.Tensor) and tensor.ndim >= 1:
                setattr(layer, attr, tensor.repeat_interleave(repeats, dim=0))
    return past


def decode_step(model, token_ids: torch.Tensor, cache, position: int):
    cache_position = torch.tensor([position], device=token_ids.device, dtype=torch.long)
    return model(
        input_ids=token_ids,
        past_key_values=cache,
        use_cache=True,
        cache_position=cache_position,
    )


def prefill(model, context: list[int], pad_id: int):
    batch = left_pad([context], pad_id)
    out = model(**batch, use_cache=True)
    return out.logits[:, -1, :].float(), out.past_key_values


def naive_scores(model, tokenizer, context: list[int], tails: list[list[int]]) -> np.ndarray:
    totals = np.zeros(len(tails), dtype=np.float64)
    lengths = np.array([len(tail) for tail in tails], dtype=np.float64)
    for index, tail in enumerate(tails):
        batch = left_pad([context + tail], tokenizer.pad_token_id)
        with torch.inference_mode():
            logits = model(**batch, use_cache=False).logits
        window = logits[0, -(len(tail) + 1) : -1, :].float().log_softmax(dim=-1)
        target = torch.tensor(tail, device=window.device)
        totals[index] = window.gather(1, target[:, None]).sum().item()
    return totals / lengths


def kv_sequential_scores(
    model, tokenizer, context: list[int], tails: list[list[int]]
) -> np.ndarray:
    with torch.inference_mode():
        first_logits, past = prefill(model, context, tokenizer.pad_token_id)
        totals = np.zeros(len(tails), dtype=np.float64)
        for index, tail in enumerate(tails):
            cache = clone_cache(past)
            logits = first_logits
            total = 0.0
            for offset, token in enumerate(tail):
                total += torch.log_softmax(logits[0], dim=-1)[token].item()
                step = torch.tensor([[token]], device=logits.device)
                out = decode_step(model, step, cache, len(context) + offset)
                cache = out.past_key_values
                logits = out.logits[:, -1, :].float()
            totals[index] = total / len(tail)
    return totals


def kv_batched_scores(
    model, tokenizer, context: list[int], tails: list[list[int]]
) -> np.ndarray:
    n_cand = len(tails)
    width = max(len(tail) for tail in tails)
    padded = torch.full((n_cand, width), tokenizer.pad_token_id, device="cuda")
    valid = torch.zeros((n_cand, width), dtype=torch.bool, device="cuda")
    for index, tail in enumerate(tails):
        padded[index, : len(tail)] = torch.tensor(tail, device="cuda")
        valid[index, : len(tail)] = True
    lengths = torch.tensor([len(tail) for tail in tails], device="cuda", dtype=torch.float64)

    with torch.inference_mode():
        first_logits, past = prefill(model, context, tokenizer.pad_token_id)
        cache = expand_cache(past, n_cand)
        logits = first_logits.repeat_interleave(n_cand, dim=0)
        totals = torch.zeros(n_cand, device="cuda", dtype=torch.float64)
        for step in range(width):
            tokens = padded[:, step]
            logp = torch.log_softmax(logits.float(), dim=-1)
            picked = logp.gather(1, tokens[:, None]).squeeze(1)
            totals += picked.double() * valid[:, step]
            if step == width - 1:
                break
            out = decode_step(model, tokens[:, None], cache, len(context) + step)
            cache = out.past_key_values
            logits = out.logits[:, -1, :]
    return (totals / lengths).detach().cpu().numpy()


def steady(times: list[float]) -> list[float]:
    return times[2:] if len(times) > 4 else times


def time_fn(fn, calls: int) -> dict:
    with torch.inference_mode():
        for _ in range(6):
            fn()
        times = [cuda_time_ms(fn)[1] for _ in range(calls)]
    return latency_summary(steady(times), 1)


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    rows = stratified_subset(read_ag_news(args.test_csv), args.per_class)
    labels = np.asarray([row["label"] for row in rows])
    contexts = prompt_ids(
        tokenizer, [word_prompt(row["text"]) for row in rows], args.max_length
    )
    tails = encode_tails(tokenizer, FULL_WORDS)
    print("tails", {word: ids for word, ids in zip(FULL_WORDS, tails)}, flush=True)

    probe = contexts[0]
    naive = naive_scores(model, tokenizer, probe, tails)
    seq = kv_sequential_scores(model, tokenizer, probe, tails)
    batched = kv_batched_scores(model, tokenizer, probe, tails)
    print("probe naive", naive, flush=True)
    print("probe sequential", seq, "max abs", float(np.max(np.abs(seq - naive))), flush=True)
    print("probe batched", batched, "max abs", float(np.max(np.abs(batched - naive))), flush=True)

    agree_n = 0 if args.only == "latency" else min(args.agree_n, len(contexts))
    agreement = None
    if agree_n > 0:
        naive_all = np.stack(
            [naive_scores(model, tokenizer, ctx, tails) for ctx in contexts[:agree_n]]
        )
        kv_all = np.stack(
            [kv_batched_scores(model, tokenizer, ctx, tails) for ctx in contexts[:agree_n]]
        )
        naive_pred = naive_all.argmax(axis=1)
        kv_pred = kv_all.argmax(axis=1)
        agreement = {
            "n": agree_n,
            "argmax_agreement": float((naive_pred == kv_pred).mean()),
            "mean_abs_score_diff": float(np.abs(naive_all - kv_all).mean()),
            "max_abs_score_diff": float(np.abs(naive_all - kv_all).max()),
            "naive_accuracy": float((naive_pred == labels[:agree_n]).mean()),
            "kv_accuracy": float((kv_pred == labels[:agree_n]).mean()),
            "naive_metrics": multiclass_metrics(softmax_rows(naive_all), labels[:agree_n]),
            "kv_metrics": multiclass_metrics(softmax_rows(kv_all), labels[:agree_n]),
        }
        print("agreement", agreement, flush=True)

    def run_naive():
        naive_scores(model, tokenizer, probe, tails)

    def run_seq():
        kv_sequential_scores(model, tokenizer, probe, tails)

    def run_batched():
        kv_batched_scores(model, tokenizer, probe, tails)

    def run_prefill_only():
        with torch.inference_mode():
            prefill(model, probe, tokenizer.pad_token_id)

    latency = None
    if args.only != "quality":
        latency = {
            "naive_four_full_forwards": time_fn(run_naive, args.latency_calls),
            "kv_sequential_four_candidates": time_fn(run_seq, args.latency_calls),
            "kv_batched_four_candidates": time_fn(run_batched, args.latency_calls),
            "prefill_only": time_fn(run_prefill_only, args.latency_calls),
            "prompt_tokens": len(probe),
            "candidate_token_counts": [len(tail) for tail in tails],
        }
        print("latency", latency, flush=True)

    payload = {
        "dataset": "AG News",
        "words": FULL_WORDS,
        "agreement": agreement,
        "latency": latency,
        "probe_scores": {
            "naive": naive.tolist(),
            "sequential": seq.tolist(),
            "batched": batched.tolist(),
        },
    }
    write_json(args.out, payload)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
