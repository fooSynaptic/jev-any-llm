#!/usr/bin/env python3
"""Scheme 2: shared-prefill packed judgments versus isolated judgments."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor

from common import (
    ALIASES,
    alias_token_ids,
    choice_prompt,
    closed_probs,
    cuda_time_ms,
    latency_summary,
    multiclass_metrics,
    packed_prompt,
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
    parser.add_argument("--latency-calls", type=int, default=60)
    return parser.parse_args()


def steady(times: list[float]) -> list[float]:
    return times[2:] if len(times) > 4 else times


def binary_prompt(text: str, topic: str) -> str:
    return (
        f"Is the news article primarily about {topic}?\n"
        "Y = yes\nN = no\n"
        "Return exactly one code: Y or N.\n\n"
        f"Article:\n{text}\n\nAnswer:"
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


class StepClosedSet(LogitsProcessor):
    def __init__(self, prompt_width: int, allowed_per_step: list[list[int]]):
        self.prompt_width = prompt_width
        self.allowed_per_step = allowed_per_step

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        step = input_ids.shape[1] - self.prompt_width
        allowed = self.allowed_per_step[min(step, len(self.allowed_per_step) - 1)]
        blocked = torch.full_like(scores, -torch.inf)
        blocked[:, allowed] = scores[:, allowed]
        return blocked


def isolated_probs(model, tokenizer, rows, max_length, batch_size, choice_ids, yn_ids):
    prompts = []
    kinds = []
    for row in rows:
        prompts.extend(
            [
                choice_prompt(row["text"]),
                binary_prompt(row["text"], "Sports"),
                binary_prompt(row["text"], "Business"),
                binary_prompt(row["text"], "science or technology"),
            ]
        )
        kinds.extend([0, 1, 1, 1])

    results = [[], [], [], []]
    times = []
    for start in range(0, len(prompts), batch_size):
        chunk_prompts = prompts[start : start + batch_size]
        chunk_kinds = kinds[start : start + batch_size]
        batch = tokenize(tokenizer, chunk_prompts, max_length)
        with torch.inference_mode():
            logits, elapsed = cuda_time_ms(lambda: forward_last_logits(model, batch))
        times.append(elapsed)
        for offset, (row_logits, kind) in enumerate(zip(logits, chunk_kinds)):
            global_index = start + offset
            question_index = global_index % 4
            token_ids = choice_ids if kind == 0 else yn_ids
            results[question_index].append(
                closed_probs(row_logits.unsqueeze(0), token_ids).cpu().numpy()[0]
            )
    return [np.asarray(item) for item in results], times


def packed_probs(model, tokenizer, rows, max_length, batch_size, choice_ids, yn_ids):
    results = [[], [], [], []]
    times = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        batch = tokenize(tokenizer, [packed_prompt(row["text"]) for row in chunk], max_length)
        width = batch["input_ids"].shape[1]
        processor = StepClosedSet(width, [choice_ids, yn_ids, yn_ids, yn_ids])

        def run():
            return model.generate(
                **batch,
                max_new_tokens=4,
                min_new_tokens=4,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                logits_processor=[processor],
                pad_token_id=tokenizer.pad_token_id,
            )

        with torch.inference_mode():
            output, elapsed = cuda_time_ms(run)
        times.append(elapsed)
        for question_index, scores in enumerate(output.scores):
            ids = choice_ids if question_index == 0 else yn_ids
            results[question_index].extend(closed_probs(scores, ids).cpu().numpy())
    return [np.asarray(item) for item in results], times


def latency_one_article(
    model, tokenizer, row, max_length, choice_ids, yn_ids, calls
) -> dict:
    independent_prompts = [
        choice_prompt(row["text"]),
        binary_prompt(row["text"], "Sports"),
        binary_prompt(row["text"], "Business"),
        binary_prompt(row["text"], "science or technology"),
    ]
    single_batches = [tokenize(tokenizer, [prompt], max_length) for prompt in independent_prompts]
    batched = tokenize(tokenizer, independent_prompts, max_length)
    packed = tokenize(tokenizer, [packed_prompt(row["text"])], max_length)
    processor = StepClosedSet(
        packed["input_ids"].shape[1], [choice_ids, yn_ids, yn_ids, yn_ids]
    )

    def independent_sequential():
        return [forward_last_logits(model, batch) for batch in single_batches]

    def independent_batch4():
        return forward_last_logits(model, batched)

    def packed_generate4():
        return model.generate(
            **packed,
            max_new_tokens=4,
            min_new_tokens=4,
            do_sample=False,
            logits_processor=[processor],
            pad_token_id=tokenizer.pad_token_id,
        )

    functions = {
        "isolated_sequential_4_prefills": independent_sequential,
        "isolated_batch4": independent_batch4,
        "packed_1_prefill_plus_4_tokens": packed_generate4,
    }
    output = {}
    with torch.inference_mode():
        for name, fn in functions.items():
            for _ in range(20):
                fn()
            times = [cuda_time_ms(fn)[1] for _ in range(calls)]
            output[name] = latency_summary(times)
    return output


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
    binary_labels = [
        (labels == 1).astype(int),
        (labels == 2).astype(int),
        (labels == 3).astype(int),
    ]
    choice_ids = alias_token_ids(tokenizer, ALIASES, leading_space=False)
    # Order is N, Y so binary gold remains 0/1.
    yn_ids = alias_token_ids(tokenizer, ["N", "Y"], leading_space=False)

    isolated, isolated_times = isolated_probs(
        model,
        tokenizer,
        rows,
        args.max_length,
        args.batch_size,
        choice_ids,
        yn_ids,
    )
    packed, packed_times = packed_probs(
        model,
        tokenizer,
        rows,
        args.max_length,
        args.batch_size,
        choice_ids,
        yn_ids,
    )

    isolated_metrics = {
        "topic": multiclass_metrics(isolated[0], labels),
        "is_sports": multiclass_metrics(isolated[1], binary_labels[0]),
        "is_business": multiclass_metrics(isolated[2], binary_labels[1]),
        "is_scitech": multiclass_metrics(isolated[3], binary_labels[2]),
    }
    packed_metrics = {
        "topic": multiclass_metrics(packed[0], labels),
        "is_sports": multiclass_metrics(packed[1], binary_labels[0]),
        "is_business": multiclass_metrics(packed[2], binary_labels[1]),
        "is_scitech": multiclass_metrics(packed[3], binary_labels[2]),
    }
    l1_shift = [
        float(np.abs(isolated[index] - packed[index]).sum(axis=1).mean())
        for index in range(4)
    ]
    flip_rate = [
        float((isolated[index].argmax(axis=1) != packed[index].argmax(axis=1)).mean())
        for index in range(4)
    ]

    result = {
        "model": args.model,
        "dataset": "AG News",
        "n": len(rows),
        "scheme_2_isolated": {
            "metrics": isolated_metrics,
            "throughput_for_4n_prompts": latency_summary(
                steady(isolated_times), args.batch_size
            ),
        },
        "scheme_2_packed": {
            "metrics": packed_metrics,
            "throughput_for_n_articles": latency_summary(
                steady(packed_times), args.batch_size
            ),
            "mean_l1_probability_shift_vs_isolated": dict(
                zip(["topic", "is_sports", "is_business", "is_scitech"], l1_shift)
            ),
            "argmax_flip_rate_vs_isolated": dict(
                zip(["topic", "is_sports", "is_business", "is_scitech"], flip_rate)
            ),
            "contract_note": "Packed answers are causally dependent and violate Jev isolation.",
        },
        "single_article_gpu_latency": latency_one_article(
            model,
            tokenizer,
            rows[0],
            args.max_length,
            choice_ids,
            yn_ids,
            args.latency_calls,
        ),
    }
    write_json(args.output, result)
    print(Path(args.output).read_text())


if __name__ == "__main__":
    main()
