#!/usr/bin/env python3
"""Vanilla Qwen baseline: autoregressive latency, TTFT, and verbalizer choice."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (
    ALIASES,
    LABELS,
    alias_token_ids,
    ar_prompt,
    choice_prompt,
    closed_probs,
    cuda_time_ms,
    latency_summary,
    multiclass_metrics,
    read_ag_news,
    render_instruct,
    stratified_subset,
    write_json,
)

WORDS = ["World", "Sports", "Business", "Sci"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-class", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--latency-calls", type=int, default=40)
    return parser.parse_args()


def word_prompt(text: str) -> str:
    return (
        "Classify the news article into exactly one topic.\n"
        "World: international affairs, politics, governments, war.\n"
        "Sports: games, teams, athletes, tournaments.\n"
        "Business: companies, markets, finance, economy.\n"
        "Sci/Tech: science, technology, computers, space.\n"
        "Return only the topic name.\n\n"
        f"Article:\n{text}\n\nTopic:"
    )


def tokenize(tokenizer, prompts: list[str], max_length: int) -> dict[str, torch.Tensor]:
    batch = tokenizer(
        render_instruct(tokenizer, prompts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {key: value.cuda(non_blocking=True) for key, value in batch.items()}


def forward_last_logits(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    try:
        return model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1, :]
    except TypeError:
        return model(**batch, use_cache=False).logits[:, -1, :]


def steady(times: list[float]) -> list[float]:
    return times[2:] if len(times) > 4 else times


def verbalizer_scores(
    model,
    tokenizer,
    rows: list[dict],
    prompt_fn,
    token_ids: list[int],
    max_length: int,
    batch_size: int,
) -> tuple[np.ndarray, list[float], float]:
    probabilities = []
    times = []
    hits = 0
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        batch = tokenize(tokenizer, [prompt_fn(row["text"]) for row in chunk], max_length)
        with torch.inference_mode():
            logits, elapsed = cuda_time_ms(lambda: forward_last_logits(model, batch))
        probabilities.append(closed_probs(logits, token_ids).cpu().numpy())
        hits += int((logits.argmax(dim=-1).cpu().numpy()[:, None] == np.asarray(token_ids)).any(axis=1).sum())
        times.append(elapsed)
    return np.concatenate(probabilities), times, hits / len(rows)


def latency_probe(
    model,
    tokenizer,
    rows: list[dict],
    prompt_fn,
    mode: str,
    max_length: int,
    batch_size: int,
    calls: int,
) -> dict:
    prompts = [prompt_fn(row["text"]) for row in rows[:batch_size]]
    batch = tokenize(tokenizer, prompts, max_length)

    if mode == "forward":
        def fn():
            return forward_last_logits(model, batch)
    else:
        new_tokens = int(mode.split(":")[1])

        def fn():
            return model.generate(
                **batch,
                max_new_tokens=new_tokens,
                min_new_tokens=new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

    with torch.inference_mode():
        for _ in range(10):
            fn()
        times = [cuda_time_ms(fn)[1] for _ in range(calls)]
    summary = latency_summary(times, batch_size)
    summary["prompt_tokens"] = int(batch["attention_mask"][0].sum().item())
    return summary


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()

    rows = stratified_subset(read_ag_news(args.test_csv), args.per_class)
    labels = np.asarray([row["label"] for row in rows])

    word_ids = alias_token_ids(tokenizer, WORDS, leading_space=False)
    letter_ids = alias_token_ids(tokenizer, ALIASES, leading_space=False)

    word_probs, word_times, word_hit = verbalizer_scores(
        model, tokenizer, rows, word_prompt, word_ids, args.max_length, args.batch_size
    )
    letter_probs, letter_times, letter_hit = verbalizer_scores(
        model, tokenizer, rows, choice_prompt, letter_ids, args.max_length, args.batch_size
    )

    result = {
        "model": args.model,
        "dataset": "AG News",
        "test": {"n": len(rows), "per_class": args.per_class, "max_length": args.max_length},
        "label_words": dict(zip(WORDS, word_ids)),
        "verbalizer_word_logprob": {
            **multiclass_metrics(word_probs, labels),
            "greedy_token_in_closed_set_rate": word_hit,
            "throughput": latency_summary(steady(word_times), args.batch_size),
        },
        "verbalizer_letter_logprob": {
            **multiclass_metrics(letter_probs, labels),
            "greedy_token_in_closed_set_rate": letter_hit,
            "throughput": latency_summary(steady(letter_times), args.batch_size),
        },
        "verbalizer_agreement": {
            "argmax_agreement": float(
                (word_probs.argmax(axis=1) == letter_probs.argmax(axis=1)).mean()
            )
        },
        "vanilla_autoregressive_latency": {
            "batch_1": {
                "full_answer_16_tokens": latency_probe(
                    model, tokenizer, rows, ar_prompt, "generate:16", args.max_length, 1, args.latency_calls
                ),
                "ttft_first_token": latency_probe(
                    model, tokenizer, rows, ar_prompt, "generate:1", args.max_length, 1, args.latency_calls
                ),
                "word_verbalizer_forward": latency_probe(
                    model, tokenizer, rows, word_prompt, "forward", args.max_length, 1, args.latency_calls
                ),
            },
            f"batch_{args.batch_size}": {
                "full_answer_16_tokens": latency_probe(
                    model,
                    tokenizer,
                    rows,
                    ar_prompt,
                    "generate:16",
                    args.max_length,
                    args.batch_size,
                    max(12, args.latency_calls // 2),
                ),
                "word_verbalizer_forward": latency_probe(
                    model,
                    tokenizer,
                    rows,
                    word_prompt,
                    "forward",
                    args.max_length,
                    args.batch_size,
                    max(12, args.latency_calls // 2),
                ),
            },
        },
    }
    write_json(args.output, result)
    print(Path(args.output).read_text())


if __name__ == "__main__":
    main()
