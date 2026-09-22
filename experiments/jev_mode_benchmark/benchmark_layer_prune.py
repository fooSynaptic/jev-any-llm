#!/usr/bin/env python3
"""Convert Qwen into a non-autoregressive predictor: depth sweep + bidirectional attention."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (
    cuda_time_ms,
    fit_temperature,
    latency_summary,
    multiclass_metrics,
    read_ag_news,
    softmax_numpy,
    stratified_disjoint,
    stratified_subset,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", default="4,8,12,16,24,32")
    parser.add_argument("--train-per-class", type=int, default=2000)
    parser.add_argument("--calibration-per-class", type=int, default=500)
    parser.add_argument("--test-per-class", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--latency-calls", type=int, default=40)
    return parser.parse_args()


def native_prompt(text: str) -> str:
    return f"News article:\n{text}"


def decoder_stack(model) -> torch.nn.Module:
    """Return the module that owns `.layers` for this checkpoint."""
    candidates = [model.model, getattr(model.model, "language_model", None), model]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "layers"):
            return candidate
    raise RuntimeError("no decoder stack with .layers found")


def tokenize(tokenizer, prompts: list[str], max_length: int) -> dict[str, torch.Tensor]:
    batch = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {key: value.cuda(non_blocking=True) for key, value in batch.items()}


def bidirectional_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Every real token attends to every real token, in both directions."""
    batch, length = attention_mask.shape
    keep = attention_mask[:, None, None, :].to(dtype)
    blocked = (1.0 - keep) * torch.finfo(dtype).min
    return blocked.expand(batch, 1, length, length).contiguous()


def forward_hidden(model, batch: dict[str, torch.Tensor], bidirectional: bool) -> tuple:
    inputs = dict(batch)
    if bidirectional:
        inputs["attention_mask"] = bidirectional_mask(batch["attention_mask"], model.dtype)
    return model.model(**inputs, use_cache=False, output_hidden_states=True).hidden_states


def pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, np.ndarray]:
    last = hidden[:, -1, :]
    weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
    mean = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
    return {
        "last": last.float().cpu().numpy(),
        "mean": mean.float().cpu().numpy(),
    }


def extract_features(
    model, tokenizer, rows, layers, max_length, batch_size, bidirectional
) -> dict[int, dict[str, np.ndarray]]:
    buckets: dict[int, dict[str, list]] = {k: {"last": [], "mean": []} for k in layers}
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        batch = tokenize(tokenizer, [native_prompt(row["text"]) for row in chunk], max_length)
        with torch.inference_mode():
            hidden_states = forward_hidden(model, batch, bidirectional)
        for k in layers:
            pooled = pool(hidden_states[k], batch["attention_mask"])
            buckets[k]["last"].append(pooled["last"])
            buckets[k]["mean"].append(pooled["mean"])
    return {
        k: {name: np.concatenate(parts) for name, parts in value.items()}
        for k, value in buckets.items()
    }


def train_head(train_x, train_y, calibration_x, calibration_y, test_x, test_y) -> dict:
    classifier = LogisticRegression(
        max_iter=200, C=1.0, solver="lbfgs", random_state=20260920
    )
    classifier.fit(train_x, train_y)
    test_logits = classifier.decision_function(test_x)
    temperature = fit_temperature(
        classifier.decision_function(calibration_x), calibration_y
    )
    calibrated = multiclass_metrics(softmax_numpy(test_logits / temperature), test_y)
    uncalibrated = multiclass_metrics(softmax_numpy(test_logits), test_y)
    return {
        "accuracy": calibrated["accuracy"],
        "brier": calibrated["brier"],
        "ece_15": calibrated["ece_15"],
        "ece_15_uncalibrated": uncalibrated["ece_15"],
        "temperature": temperature,
    }


def latency_by_depth(model, tokenizer, row, layers, max_length, batch_size, calls) -> dict:
    stack = decoder_stack(model)
    original = stack.layers
    batch = tokenize(tokenizer, [native_prompt(row["text"])] * batch_size, max_length)
    head = torch.nn.Linear(model.config.get_text_config().hidden_size, 4, device="cuda").float()
    head.eval()
    output = {}
    try:
        for k in layers:
            stack.layers = original[:k]

            def fn():
                hidden = model.model(**batch, use_cache=False).last_hidden_state[:, -1, :]
                return head(hidden.float())

            with torch.inference_mode():
                for _ in range(15):
                    fn()
                times = [cuda_time_ms(fn)[1] for _ in range(calls)]
            output[f"layer_{k}"] = latency_summary(times, batch_size)
    finally:
        stack.layers = original
    output["prompt_tokens"] = int(batch["attention_mask"][0].sum().item())
    return output


def mask_sanity(model, tokenizer, rows, max_length) -> dict:
    """Confirm the 4D mask really removes causality instead of being ignored."""
    batch = tokenize(tokenizer, [native_prompt(row["text"]) for row in rows[:4]], max_length)
    with torch.inference_mode():
        causal = forward_hidden(model, batch, False)[-1]
        bidirectional = forward_hidden(model, batch, True)[-1]
    difference = (causal.float() - bidirectional.float()).abs()
    return {
        "mean_abs_hidden_difference": float(difference.mean()),
        "max_abs_hidden_difference": float(difference.max()),
        "causality_removed": bool(difference.max() > 1e-3),
    }


def main() -> None:
    args = parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    ).eval()

    train_rows, calibration_rows = stratified_disjoint(
        read_ag_news(args.train_csv),
        args.train_per_class,
        args.calibration_per_class,
        seed=20260920,
    )
    test_rows = stratified_subset(read_ag_news(args.test_csv), args.test_per_class, seed=20260920)
    train_y = np.asarray([row["label"] for row in train_rows])
    calibration_y = np.asarray([row["label"] for row in calibration_rows])
    test_y = np.asarray([row["label"] for row in test_rows])

    result = {
        "model": args.model,
        "dataset": "AG News",
        "total_layers": len(decoder_stack(model).layers),
        "layers_probed": layers,
        "split": {
            "train_n": len(train_rows),
            "calibration_n": len(calibration_rows),
            "test_n": len(test_rows),
        },
        "mask_sanity": mask_sanity(model, tokenizer, test_rows, args.max_length),
        "depth_sweep": {},
    }

    for mode, bidirectional in [("causal", False), ("bidirectional", True)]:
        train_f = extract_features(
            model, tokenizer, train_rows, layers, args.max_length, args.batch_size, bidirectional
        )
        calibration_f = extract_features(
            model, tokenizer, calibration_rows, layers, args.max_length, args.batch_size, bidirectional
        )
        test_f = extract_features(
            model, tokenizer, test_rows, layers, args.max_length, args.batch_size, bidirectional
        )
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
            print(mode, k, per_layer[f"layer_{k}"], flush=True)
        result["depth_sweep"][mode] = per_layer
        del train_f, calibration_f, test_f

    result["latency"] = {
        "batch_1": latency_by_depth(
            model, tokenizer, test_rows[0], layers, args.max_length, 1, args.latency_calls
        ),
        f"batch_{args.batch_size}": latency_by_depth(
            model,
            tokenizer,
            test_rows[0],
            layers,
            args.max_length,
            args.batch_size,
            max(12, args.latency_calls // 2),
        ),
    }
    write_json(args.output, result)
    print(Path(args.output).read_text())


if __name__ == "__main__":
    main()
