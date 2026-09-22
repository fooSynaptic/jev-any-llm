#!/usr/bin/env python3
"""Zero-training AG News benchmark for schemes 0, 1, 3, and 4."""

from __future__ import annotations

import argparse
import re
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
    codebook_prompt,
    cuda_time_ms,
    latency_summary,
    multiclass_metrics,
    read_ag_news,
    render_instruct,
    stratified_subset,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-class", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--latency-calls", type=int, default=80)
    return parser.parse_args()


def steady(times: list[float]) -> list[float]:
    return times[2:] if len(times) > 4 else times


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


def evaluate_logits(
    model,
    tokenizer,
    rows: list[dict],
    prompt_fn,
    token_ids: list[int],
    max_length: int,
    batch_size: int,
) -> tuple[np.ndarray, list[float]]:
    all_probs = []
    times = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        batch = tokenize(tokenizer, [prompt_fn(row["text"]) for row in chunk], max_length)
        with torch.inference_mode():
            logits, elapsed = cuda_time_ms(lambda: forward_last_logits(model, batch))
        all_probs.append(closed_probs(logits, token_ids).cpu().numpy())
        times.append(elapsed)
    return np.concatenate(all_probs), times


def generate_one_scores(
    model,
    tokenizer,
    rows: list[dict],
    token_ids: list[int],
    max_length: int,
    batch_size: int,
) -> tuple[np.ndarray, list[float], float]:
    all_probs = []
    times = []
    hits = 0
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        batch = tokenize(tokenizer, [choice_prompt(row["text"]) for row in chunk], max_length)

        def run():
            return model.generate(
                **batch,
                max_new_tokens=1,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        with torch.inference_mode():
            output, elapsed = cuda_time_ms(run)
        score = output.scores[0]
        all_probs.append(closed_probs(score, token_ids).cpu().numpy())
        generated = output.sequences[:, -1].cpu().numpy()
        hits += sum(int(token) in token_ids for token in generated)
        times.append(elapsed)
    return np.concatenate(all_probs), times, hits / len(rows)


def parse_ar_category(text: str) -> int:
    normalized = text.lower()
    patterns = [
        ("sci/tech", 3),
        ("science", 3),
        ("technology", 3),
        ("sports", 1),
        ("business", 2),
        ("world", 0),
    ]
    for needle, label in patterns:
        if needle in normalized:
            return label
    match = re.search(r"\b([abcd])\b", normalized)
    return "abcd".index(match.group(1)) if match else -1


def evaluate_ar(
    model,
    tokenizer,
    rows: list[dict],
    max_length: int,
    batch_size: int,
) -> dict:
    predictions = []
    generated_lengths = []
    times = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        batch = tokenize(tokenizer, [ar_prompt(row["text"]) for row in chunk], max_length)
        input_width = batch["input_ids"].shape[1]

        def run():
            return model.generate(
                **batch,
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        with torch.inference_mode():
            output, elapsed = cuda_time_ms(run)
        suffix = output[:, input_width:]
        texts = tokenizer.batch_decode(suffix, skip_special_tokens=True)
        predictions.extend(parse_ar_category(text) for text in texts)
        generated_lengths.extend((suffix != tokenizer.pad_token_id).sum(dim=1).cpu().tolist())
        times.append(elapsed)
    labels = np.array([row["label"] for row in rows])
    predictions_array = np.asarray(predictions)
    valid = predictions_array >= 0
    return {
        "n": len(rows),
        "accuracy": float((predictions_array == labels).mean()),
        "parse_rate": float(valid.mean()),
        "mean_generated_tokens": float(np.mean(generated_lengths)),
        "throughput": latency_summary(steady(times), min(batch_size, len(rows))),
    }


def latency_probe(
    model,
    tokenizer,
    rows: list[dict],
    mode: str,
    max_length: int,
    batch_size: int,
    calls: int,
) -> dict:
    prompts = [choice_prompt(row["text"]) for row in rows[:batch_size]]
    batch = tokenize(tokenizer, prompts, max_length)

    if mode == "forward":
        fn = lambda: forward_last_logits(model, batch)
    elif mode == "generate1":
        fn = lambda: model.generate(
            **batch,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    else:
        raise ValueError(mode)

    with torch.inference_mode():
        for _ in range(20):
            fn()
        times = [cuda_time_ms(fn)[1] for _ in range(calls)]
    return latency_summary(times, batch_size)


def main() -> None:
    args = parse_args()
    torch.manual_seed(20260920)
    torch.backends.cuda.matmul.allow_tf32 = True

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
    labels = np.array([row["label"] for row in rows])
    alias_ids = alias_token_ids(tokenizer, ALIASES, leading_space=False)

    direct_probs, direct_times = evaluate_logits(
        model,
        tokenizer,
        rows,
        choice_prompt,
        alias_ids,
        args.max_length,
        args.batch_size,
    )
    generated_probs, generated_times, alias_hit_rate = generate_one_scores(
        model,
        tokenizer,
        rows,
        alias_ids,
        args.max_length,
        args.batch_size,
    )

    symbols = list("ABCDEFGHIJKLMNOP")
    code_ids = alias_token_ids(tokenizer, symbols, leading_space=False)
    code_probs, code_times = evaluate_logits(
        model,
        tokenizer,
        rows,
        codebook_prompt,
        code_ids,
        args.max_length,
        args.batch_size,
    )
    class_probs = code_probs.reshape(len(rows), 4, 4).sum(axis=2)
    class_probs /= class_probs.sum(axis=1, keepdims=True)

    result = {
        "model": args.model,
        "dataset": "AG News",
        "test": {
            "n": len(rows),
            "per_class": args.per_class,
            "max_length": args.max_length,
        },
        "alias_tokens": dict(zip(ALIASES, alias_ids)),
        "scheme_0_ar": evaluate_ar(
            model,
            tokenizer,
            rows,
            args.max_length,
            args.batch_size,
        ),
        "scheme_1_generate_logprob": {
            **multiclass_metrics(generated_probs, labels),
            "alias_hit_rate": alias_hit_rate,
            "throughput": latency_summary(steady(generated_times), args.batch_size),
        },
        "scheme_3_multibit_codebook_zero_shot": {
            **multiclass_metrics(class_probs, labels),
            "codebook_size": 16,
            "throughput": latency_summary(steady(code_times), args.batch_size),
        },
        "scheme_4_direct_last_logits": {
            **multiclass_metrics(direct_probs, labels),
            "throughput": latency_summary(steady(direct_times), args.batch_size),
        },
        "scheme_1_vs_4": {
            "max_probability_abs_diff": float(np.abs(generated_probs - direct_probs).max()),
            "mean_probability_abs_diff": float(np.abs(generated_probs - direct_probs).mean()),
            "argmax_agreement": float(
                (generated_probs.argmax(axis=1) == direct_probs.argmax(axis=1)).mean()
            ),
        },
        "latency_gpu_compute": {
            "batch_1": {
                "scheme_1_generate1": latency_probe(
                    model, tokenizer, rows, "generate1", args.max_length, 1, args.latency_calls
                ),
                "scheme_4_forward": latency_probe(
                    model, tokenizer, rows, "forward", args.max_length, 1, args.latency_calls
                ),
            },
            f"batch_{args.batch_size}": {
                "scheme_1_generate1": latency_probe(
                    model,
                    tokenizer,
                    rows,
                    "generate1",
                    args.max_length,
                    args.batch_size,
                    max(20, args.latency_calls // 2),
                ),
                "scheme_4_forward": latency_probe(
                    model,
                    tokenizer,
                    rows,
                    "forward",
                    args.max_length,
                    args.batch_size,
                    max(20, args.latency_calls // 2),
                ),
            },
        },
    }
    write_json(args.output, result)
    print(Path(args.output).read_text())


if __name__ == "__main__":
    main()
