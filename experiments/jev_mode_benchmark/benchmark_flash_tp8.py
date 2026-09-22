#!/usr/bin/env python3
"""AG News Jev-mode protocol on DeepSeek-V4.1-Flash (official TP8 inference)."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (  # noqa: E402
    ALIASES,
    LABELS,
    alias_token_ids,
    ar_prompt,
    choice_prompt,
    closed_probs,
    codebook_prompt,
    cuda_time_ms,
    fit_temperature,
    latency_summary,
    multiclass_metrics,
    packed_prompt,
    read_ag_news,
    softmax_numpy,
    stratified_disjoint,
    stratified_subset,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--inference-dir", required=True)
    parser.add_argument("--encoding-dir", required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-class", type=int, default=1000)
    parser.add_argument("--train-per-class", type=int, default=2000)
    parser.add_argument("--calibration-per-class", type=int, default=500)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--latency-calls", type=int, default=24)
    parser.add_argument("--latency-batch", type=int, default=4)
    parser.add_argument("--fwd-batch", type=int, default=8)
    parser.add_argument("--layers", default="1,2,4,6,8,12,16,20,24,32,40")
    parser.add_argument("--skip-depth", action="store_true")
    parser.add_argument("--quality-only", action="store_true")
    return parser.parse_args()


def is_rank0() -> bool:
    return int(os.getenv("RANK", "0")) == 0


def log(*parts) -> None:
    if is_rank0():
        print(*parts, flush=True)


def load_stack(args):
    sys.path.insert(0, args.encoding_dir)
    sys.path.insert(0, args.inference_dir)
    from encoding import encode_messages
    from generate import generate as ds_generate
    from model import ModelArgs, Transformer, make_identity_pre_mix
    from safetensors.torch import load_model

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(20260920)

    with open(args.config) as handle:
        model_args = ModelArgs(**json.load(handle))
    model_args.temperature = 0.0
    model_args.max_batch_size = max(4, args.latency_batch, args.fwd_batch)
    model_args.max_seq_len = max(512, args.max_length + 64)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path or args.ckpt_path, trust_remote_code=True
    )
    log("build model")
    with torch.device("cuda"):
        model = Transformer(model_args, tokenizer)
    log("load model")
    load_model(
        model,
        os.path.join(args.ckpt_path, f"model{rank}-mp{world_size}.safetensors"),
    )
    torch.set_default_device("cuda")
    model.temperature = 0.0
    return {
        "model": model,
        "tokenizer": tokenizer,
        "encode_messages": encode_messages,
        "ds_generate": ds_generate,
        "make_identity_pre_mix": make_identity_pre_mix,
        "world_size": world_size,
        "rank": rank,
    }


def to_ids(tokenizer, encode_messages, text: str, prompt_fn, max_length: int) -> list[int]:
    rendered = encode_messages(
        [{"role": "user", "content": prompt_fn(text)}],
        thinking_mode="chat",
    )
    ids = tokenizer.encode(rendered)
    return ids[:max_length]


def as_cuda(ids: list[int]) -> torch.Tensor:
    return torch.tensor([ids], dtype=torch.long, device="cuda")


@torch.inference_mode()
def last_logits(model, ids: list[int]) -> torch.Tensor:
    _, logits, _ = model.forward(as_cuda(ids), 0)
    return logits.float()


@torch.inference_mode()
def last_logits_many(model, batch_ids: list[list[int]]) -> torch.Tensor:
    tokens = torch.tensor(batch_ids, dtype=torch.long, device="cuda")
    _, logits, _ = model.forward(tokens, 0)
    return logits.float()


def binary_prompt(text: str, topic: str) -> str:
    return (
        f"Is the news article primarily about {topic}?\n"
        "Y = yes\nN = no\n"
        "Return exactly one code: Y or N.\n\n"
        f"Article:\n{text}\n\nAnswer:"
    )


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


def native_prompt(text: str) -> str:
    return f"News article:\n{text}"


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


def evaluate_closed(
    stack, rows, prompt_fn, token_ids, max_length, batch_size: int
) -> tuple[np.ndarray, list[float], float]:
    model, tokenizer, encode = stack["model"], stack["tokenizer"], stack["encode_messages"]
    encoded = [to_ids(tokenizer, encode, row["text"], prompt_fn, max_length) for row in rows]
    probs = np.zeros((len(rows), len(token_ids)), dtype=np.float64)
    times = []
    hits = 0
    allowed = set(token_ids)
    by_len: dict[int, list[int]] = {}
    for index, ids in enumerate(encoded):
        by_len.setdefault(len(ids), []).append(index)
    for indices in by_len.values():
        for start in range(0, len(indices), batch_size):
            chunk = indices[start : start + batch_size]
            batch = [encoded[i] for i in chunk]

            def fn(batch=batch):
                return last_logits_many(model, batch)

            logits, elapsed = cuda_time_ms(fn)
            dist.barrier()
            times.append(elapsed)
            if is_rank0() and len(times) % 25 == 0:
                log("closed_fwd_batches", len(times), "ms", round(elapsed, 1))
            closed = closed_probs(logits, token_ids).cpu().numpy()
            greedy = logits.argmax(dim=-1).cpu().numpy()
            for row_offset, row_index in enumerate(chunk):
                probs[row_index] = closed[row_offset]
                hits += int(int(greedy[row_offset]) in allowed)
    return probs, times, hits / len(rows)


def evaluate_ar(stack, rows, max_length, batch_size: int) -> dict:
    model, tokenizer, encode, generate = (
        stack["model"],
        stack["tokenizer"],
        stack["encode_messages"],
        stack["ds_generate"],
    )
    encoded = [to_ids(tokenizer, encode, row["text"], ar_prompt, max_length) for row in rows]
    predictions = [None] * len(rows)
    lengths = [0] * len(rows)
    times = []
    eos = tokenizer.eos_token_id
    by_len: dict[int, list[int]] = {}
    for index, ids in enumerate(encoded):
        by_len.setdefault(len(ids), []).append(index)
    for indices in by_len.values():
        for start in range(0, len(indices), batch_size):
            chunk = indices[start : start + batch_size]
            batch = [encoded[i] for i in chunk]

            def fn(batch=batch):
                return generate(model, batch, 16, eos)

            tokens, elapsed = cuda_time_ms(fn)
            dist.barrier()
            times.append(elapsed)
            for row_offset, row_index in enumerate(chunk):
                suffix = tokens[row_offset]
                predictions[row_index] = parse_ar_category(tokenizer.decode(suffix))
                lengths[row_index] = len(suffix)
    labels = np.asarray([row["label"] for row in rows])
    pred = np.asarray(predictions)
    return {
        "accuracy": float((pred == labels).mean()),
        "parse_rate": float((pred >= 0).mean()),
        "mean_generated_tokens": float(np.mean(lengths)),
        "batch_quality": latency_summary(times[2:] if len(times) > 4 else times, batch_size),
    }


@torch.inference_mode()
def packed_batch(model, batch_ids: list[list[int]], choice_ids, yn_ids) -> list[np.ndarray]:
    tokens = torch.tensor(batch_ids, dtype=torch.long, device="cuda")
    _, logits, _ = model.forward(tokens, 0)
    pos = int(tokens.shape[1])
    steps = []
    for step, allowed in enumerate([choice_ids, yn_ids, yn_ids, yn_ids]):
        steps.append(closed_probs(logits, allowed).cpu().numpy())
        chosen = [allowed[int(prob.argmax())] for prob in steps[-1]]
        nxt = torch.tensor([[x] for x in chosen], dtype=torch.long, device="cuda")
        if step < 3:
            _, logits, _ = model.forward(nxt, pos)
            pos += 1
    return steps


def packed_one(model, ids: list[int], choice_ids, yn_ids):
    steps = packed_batch(model, [ids], choice_ids, yn_ids)
    return [item[0] for item in steps], None


def evaluate_packed(stack, rows, max_length, choice_ids, yn_ids, batch_size: int):
    model, tokenizer, encode = stack["model"], stack["tokenizer"], stack["encode_messages"]
    encoded = [to_ids(tokenizer, encode, row["text"], packed_prompt, max_length) for row in rows]
    stored = [np.zeros((len(rows), 4 if i == 0 else 2), dtype=np.float64) for i in range(4)]
    stored[0] = np.zeros((len(rows), len(choice_ids)))
    stored[1] = np.zeros((len(rows), len(yn_ids)))
    stored[2] = np.zeros((len(rows), len(yn_ids)))
    stored[3] = np.zeros((len(rows), len(yn_ids)))
    times = []
    by_len: dict[int, list[int]] = {}
    for index, ids in enumerate(encoded):
        by_len.setdefault(len(ids), []).append(index)
    for indices in by_len.values():
        for start in range(0, len(indices), batch_size):
            chunk = indices[start : start + batch_size]
            batch = [encoded[i] for i in chunk]

            def fn(batch=batch):
                return packed_batch(model, batch, choice_ids, yn_ids)

            steps, elapsed = cuda_time_ms(fn)
            dist.barrier()
            times.append(elapsed)
            for question, matrix in enumerate(steps):
                for offset, row_index in enumerate(chunk):
                    stored[question][row_index] = matrix[offset]
    return stored, times


def isolated_four(stack, rows, max_length, choice_ids, yn_ids, batch_size: int):
    specs = []
    for row_index, row in enumerate(rows):
        specs.extend(
            [
                (row_index, 0, choice_prompt, choice_ids),
                (row_index, 1, lambda t: binary_prompt(t, "Sports"), yn_ids),
                (row_index, 2, lambda t: binary_prompt(t, "Business"), yn_ids),
                (row_index, 3, lambda t: binary_prompt(t, "science or technology"), yn_ids),
            ]
        )
    model, tokenizer, encode = stack["model"], stack["tokenizer"], stack["encode_messages"]
    encoded = [
        to_ids(tokenizer, encode, rows[row_index]["text"], prompt_fn, max_length)
        for row_index, _, prompt_fn, _ in specs
    ]
    stored = [[None] * len(rows) for _ in range(4)]
    times = []
    by_len: dict[int, list[int]] = {}
    for index, ids in enumerate(encoded):
        by_len.setdefault(len(ids), []).append(index)
    for indices in by_len.values():
        for start in range(0, len(indices), batch_size):
            chunk = indices[start : start + batch_size]
            batch = [encoded[i] for i in chunk]

            def fn(batch=batch):
                return last_logits_many(model, batch)

            logits, elapsed = cuda_time_ms(fn)
            dist.barrier()
            times.append(elapsed)
            for offset, spec_index in enumerate(chunk):
                row_index, question, _, token_ids = specs[spec_index]
                stored[question][row_index] = closed_probs(
                    logits[offset], token_ids
                ).cpu().numpy()
    return [np.stack(item) for item in stored], times


def latency_probe(stack, row, max_length, choice_ids, yn_ids, calls, batch):
    model, tokenizer, encode, generate = (
        stack["model"],
        stack["tokenizer"],
        stack["encode_messages"],
        stack["ds_generate"],
    )
    letter_ids = to_ids(tokenizer, encode, row["text"], choice_prompt, max_length)
    word_ids = to_ids(tokenizer, encode, row["text"], word_prompt, max_length)
    ar_ids = to_ids(tokenizer, encode, row["text"], ar_prompt, max_length)
    packed_ids = to_ids(tokenizer, encode, row["text"], packed_prompt, max_length)
    isolated = [
        to_ids(tokenizer, encode, row["text"], choice_prompt, max_length),
        to_ids(tokenizer, encode, row["text"], lambda t: binary_prompt(t, "Sports"), max_length),
        to_ids(tokenizer, encode, row["text"], lambda t: binary_prompt(t, "Business"), max_length),
        to_ids(tokenizer, encode, row["text"], lambda t: binary_prompt(t, "science or technology"), max_length),
    ]
    eos = tokenizer.eos_token_id
    specs = {
        "vanilla_generate_16": lambda: generate(model, [ar_ids], 16, eos),
        "ttft_generate_1": lambda: generate(model, [ar_ids], 1, eos),
        "letter_forward": lambda: last_logits(model, letter_ids),
        "word_forward": lambda: last_logits(model, word_ids),
        "isolated_sequential_4": lambda: [last_logits(model, item) for item in isolated],
        "packed_prefill_plus_4": lambda: packed_one(model, packed_ids, choice_ids, yn_ids),
    }
    # Same-length copies measure batched prefill of one prompt.
    if batch > 1:
        copies = [letter_ids for _ in range(batch)]

        def batched_generate():
            return generate(model, copies, 1, eos)

        specs[f"letter_generate1_batch_{batch}"] = batched_generate

    output = {"prompt_tokens_letter": len(letter_ids), "prompt_tokens_ar": len(ar_ids)}
    with torch.inference_mode():
        for name, fn in specs.items():
            for _ in range(6):
                fn()
                dist.barrier()
            times = []
            for _ in range(calls):
                _, elapsed = cuda_time_ms(fn)
                dist.barrier()
                times.append(elapsed)
            examples = batch if name.endswith(f"batch_{batch}") else 1
            output[name] = latency_summary(times, examples)
    return output


@torch.inference_mode()
def hidden_by_layer(model, mix, ids, layers: list[int]) -> dict[int, dict[str, np.ndarray]]:
    if isinstance(ids[0], int):
        ids = [ids]
    input_ids = torch.tensor(ids, dtype=torch.long, device="cuda")
    engram_hashes = model.engram_hash(input_ids, 0, None) if model.engram_hash is not None else None
    h = model.embed(input_ids)
    h = h.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
    pre_mix = mix(h, model.hc_mult)
    grabbed = {}
    for i, layer in enumerate(model.layers):
        if layer.engram is not None:
            h = layer.engram(h, engram_hashes[:, :, layer.engram.layer_hash_index, :], None)
        h, pre_mix = layer(h, 0, pre_mix, None)
        depth = i + 1
        if depth in layers:
            pooled = h.mean(dim=2).float()
            last = pooled[:, -1, :].cpu().numpy()
            mean = pooled.mean(dim=1).cpu().numpy()
            grabbed[depth] = {
                "last": last,
                "mean": mean,
            }
    return grabbed


def extract_features(stack, rows, layers, max_length, batch_size: int):
    model, tokenizer, encode, mix = (
        stack["model"],
        stack["tokenizer"],
        stack["encode_messages"],
        stack["make_identity_pre_mix"],
    )
    encoded = [to_ids(tokenizer, encode, row["text"], native_prompt, max_length) for row in rows]
    buckets = {
        k: {"last": np.zeros((len(rows), 5120), dtype=np.float32), "mean": np.zeros((len(rows), 5120), dtype=np.float32)}
        for k in layers
    }
    by_len: dict[int, list[int]] = {}
    for index, ids in enumerate(encoded):
        by_len.setdefault(len(ids), []).append(index)
    for indices in by_len.values():
        for start in range(0, len(indices), batch_size):
            chunk = indices[start : start + batch_size]
            batch = [encoded[i] for i in chunk]
            grabbed = hidden_by_layer(model, mix, batch, layers)
            dist.barrier()
            for k in layers:
                for offset, row_index in enumerate(chunk):
                    buckets[k]["last"][row_index] = grabbed[k]["last"][offset]
                    buckets[k]["mean"][row_index] = grabbed[k]["mean"][offset]
    return buckets


def nan_report(array: np.ndarray) -> dict:
    finite = np.isfinite(array)
    return {
        "n": int(array.shape[0]),
        "nan_frac": float((~finite).any(axis=1).mean()) if array.size else 0.0,
        "nonfinite": int((~finite).sum()),
    }


def train_head(train_x, train_y, calibration_x, calibration_y, test_x, test_y) -> dict:
    report = {
        "train": nan_report(train_x),
        "calibration": nan_report(calibration_x),
        "test": nan_report(test_x),
    }
    train_ok = np.isfinite(train_x).all(axis=1)
    cal_ok = np.isfinite(calibration_x).all(axis=1)
    test_ok = np.isfinite(test_x).all(axis=1)
    if int(train_ok.sum()) < 32 or int(cal_ok.sum()) < 8 or int(test_ok.sum()) < 32:
        return {"skipped": True, "reason": "too_few_finite_rows", **report}
    classifier = LogisticRegression(max_iter=200, C=1.0, solver="lbfgs", random_state=20260920)
    classifier.fit(train_x[train_ok], train_y[train_ok])
    test_logits = classifier.decision_function(test_x[test_ok])
    temperature = fit_temperature(
        classifier.decision_function(calibration_x[cal_ok]), calibration_y[cal_ok]
    )
    calibrated = multiclass_metrics(softmax_numpy(test_logits / temperature), test_y[test_ok])
    uncalibrated = multiclass_metrics(softmax_numpy(test_logits), test_y[test_ok])
    return {
        "skipped": False,
        "accuracy": calibrated["accuracy"],
        "brier": calibrated["brier"],
        "ece_15": calibrated["ece_15"],
        "ece_15_uncalibrated": uncalibrated["ece_15"],
        "temperature": temperature,
        "n_test_finite": int(test_ok.sum()),
        **report,
    }


def latency_by_depth(stack, row, layers, max_length, batch, calls):
    model, tokenizer, encode = stack["model"], stack["tokenizer"], stack["encode_messages"]
    original = model.layers
    ids = to_ids(tokenizer, encode, row["text"], native_prompt, max_length)
    copies = [ids for _ in range(batch)]
    head = torch.nn.Linear(model.embed.weight.shape[-1] if hasattr(model.embed, "weight") else 5120, 4, device="cuda")
    # ParallelEmbedding may not expose weight that way; use hidden size from layer output.
    head = torch.nn.Linear(5120, 4, device="cuda").float()
    head.eval()
    output = {"prompt_tokens": len(ids)}
    try:
        for k in layers:
            model.layers = original[:k]

            def fn():
                hidden = hidden_by_layer(model, stack["make_identity_pre_mix"], ids, [k])
                return head(torch.as_tensor(hidden[k]["last"], device="cuda"))

            with torch.inference_mode():
                for _ in range(4):
                    fn()
                    dist.barrier()
                times = []
                for _ in range(calls):
                    _, elapsed = cuda_time_ms(fn)
                    dist.barrier()
                    times.append(elapsed)
            output[f"layer_{k}"] = latency_summary(times, 1)
            if batch > 1:
                tokens = torch.tensor(copies, dtype=torch.long, device="cuda")

                def batched():
                    return stack["ds_generate"](model, copies, 1, tokenizer.eos_token_id)

                # Keep per-layer latency on a single sequence; batched generate after prune is a
                # different object because generate still expects the full stack.
                del tokens
                del batched
    finally:
        model.layers = original
    return output


def dump(path: str, payload: dict) -> None:
    if is_rank0():
        write_json(path, payload)
        log("wrote", path)


def main() -> None:
    args = parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    stack = load_stack(args)
    tokenizer = stack["tokenizer"]
    letter_ids = alias_token_ids(tokenizer, ALIASES, leading_space=False)
    yn_ids = alias_token_ids(tokenizer, ["Y", "N"], leading_space=False)
    word_ids = alias_token_ids(tokenizer, ["World", "Sports", "Business", "Sci"], leading_space=False)
    code_ids = alias_token_ids(tokenizer, list("ABCDEFGHIJKLMNOP"), leading_space=False)

    test_rows = stratified_subset(read_ag_news(args.test_csv), args.per_class, seed=20260920)
    labels = np.asarray([row["label"] for row in test_rows])
    log("n_test", len(test_rows), "layers", layers, "fwd_batch", args.fwd_batch)
    warmup_ids = to_ids(tokenizer, stack["encode_messages"], test_rows[0]["text"], choice_prompt, args.max_length)
    with torch.inference_mode():
        last_logits(stack["model"], warmup_ids)
        dist.barrier()
        log("warmup done", len(warmup_ids))

    letter_probs, letter_times, letter_hit = evaluate_closed(
        stack, test_rows, choice_prompt, letter_ids, args.max_length, args.fwd_batch
    )
    log("letter acc", float((letter_probs.argmax(1) == labels).mean()))
    word_probs, word_times, word_hit = evaluate_closed(
        stack, test_rows, word_prompt, word_ids, args.max_length, args.fwd_batch
    )
    code_probs, code_times, _ = evaluate_closed(
        stack, test_rows, codebook_prompt, code_ids, args.max_length, args.fwd_batch
    )
    code_class = code_probs.reshape(len(test_rows), 4, 4).sum(axis=2)
    vanilla = evaluate_ar(stack, test_rows, args.max_length, args.fwd_batch)
    isolated, isolated_times = isolated_four(
        stack, test_rows, args.max_length, letter_ids, yn_ids, args.fwd_batch
    )
    packed, packed_times = evaluate_packed(
        stack, test_rows, args.max_length, letter_ids, yn_ids, args.fwd_batch
    )

    result = {
        "model": "DeepSeek-V4.1-Flash",
        "ckpt": args.ckpt_path,
        "dataset": "AG News",
        "thinking_mode": "chat",
        "test": {"n": len(test_rows), "per_class": args.per_class, "max_length": args.max_length},
        "alias_ids": {"letters": letter_ids, "words": word_ids, "yn": yn_ids},
        "vanilla_autoregressive": vanilla,
        "scheme1_and_4_letter_logprob": {
            **multiclass_metrics(letter_probs, labels),
            "greedy_token_in_closed_set_rate": letter_hit,
            "quality_throughput": latency_summary(letter_times[2:] if len(letter_times) > 4 else letter_times, args.fwd_batch),
            "note": "TP8 forward last-logits; generate-1 and direct logits share this readout.",
        },
        "verbalizer_word_logprob": {
            **multiclass_metrics(word_probs, labels),
            "greedy_token_in_closed_set_rate": word_hit,
            "quality_throughput": latency_summary(word_times[2:] if len(word_times) > 4 else word_times, args.fwd_batch),
            "argmax_agreement_with_letters": float(
                (word_probs.argmax(1) == letter_probs.argmax(1)).mean()
            ),
        },
        "scheme3_codebook": {
            **multiclass_metrics(code_class, labels),
            "quality_throughput": latency_summary(code_times[2:] if len(code_times) > 4 else code_times, args.fwd_batch),
        },
        "scheme2_isolated": {
            "topic": multiclass_metrics(isolated[0], labels),
            "flip_vs_packed": [
                float((isolated[i].argmax(1) != packed[i].argmax(1)).mean()) for i in range(4)
            ],
            "quality_throughput": latency_summary(
                isolated_times[2:] if len(isolated_times) > 4 else isolated_times, args.fwd_batch
            ),
        },
        "scheme2_packed": {
            "topic": multiclass_metrics(packed[0], labels),
            "quality_throughput": latency_summary(packed_times[2:] if len(packed_times) > 4 else packed_times, args.fwd_batch),
        },
    }
    dump(args.output.replace(".json", ".quality.json"), result)

    if not args.quality_only:
        result["latency"] = latency_probe(
            stack,
            test_rows[0],
            args.max_length,
            letter_ids,
            yn_ids,
            args.latency_calls,
            args.latency_batch,
        )
        dump(args.output.replace(".json", ".latency.json"), result)

    if not args.skip_depth:
        train_rows, calibration_rows = stratified_disjoint(
            read_ag_news(args.train_csv),
            args.train_per_class,
            args.calibration_per_class,
            seed=20260920,
        )
        layers = [k for k in layers if k <= len(stack["model"].layers)]
        log("depth extract", len(train_rows), len(calibration_rows), len(test_rows))
        try:
            train_f = extract_features(stack, train_rows, layers, args.max_length, args.fwd_batch)
            calibration_f = extract_features(stack, calibration_rows, layers, args.max_length, args.fwd_batch)
            test_f = extract_features(stack, test_rows, layers, args.max_length, args.fwd_batch)
        except Exception as exc:
            result["depth_sweep"] = {"failed": True, "error": repr(exc)}
            dump(args.output.replace(".json", ".depth.json"), result)
            raise
        train_y = np.asarray([row["label"] for row in train_rows])
        calibration_y = np.asarray([row["label"] for row in calibration_rows])
        test_y = np.asarray([row["label"] for row in test_rows])
        per_layer = {}
        for k in layers:
            per_layer[f"layer_{k}"] = {
                pooling: train_head(
                    train_f[k][pooling],
                    train_y,
                    calibration_f[k][pooling],
                    calibration_y,
                    test_f[k][pooling],
                    test_y,
                )
                for pooling in ("last", "mean")
            }
            log("depth", k, per_layer[f"layer_{k}"])
            result["depth_sweep"] = {
                "total_layers": len(stack["model"].layers),
                "causal_mean_and_last": per_layer,
                "bidirectional": "unsupported on CSA2 official kernels",
                "split": {
                    "train_n": len(train_rows),
                    "calibration_n": len(calibration_rows),
                    "test_n": len(test_rows),
                },
            }
            dump(args.output.replace(".json", ".depth.json"), result)
        result["depth_latency"] = latency_by_depth(
            stack, test_rows[0], layers, args.max_length, 1, max(8, args.latency_calls // 2)
        )

    if is_rank0():
        write_json(args.output, result)
        print(json.dumps({k: result[k] for k in result if k != "depth_sweep"}, ensure_ascii=False)[:4000])
    if stack["world_size"] > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
