#!/usr/bin/env python3
"""Stage 2 gaps for one dense decoder.

Fills the cells that the closed 4B run already has and 27B still shares:

- scheme 1: batch-1 generate exactly one token on the letter prompt
- scheme 3: batch-1 last-logit forward on the 16-symbol codebook prompt
- scheme 4: batch-1 last-logit forward on the letter prompt
- scheme 5: causal vs bidirectional mean-pool linear head at layer 8

Accuracy for schemes 1–4 is already on the scorecard, so those arms are latency
only. Scheme 5 trains the layer-8 head on the locked 8000/2000/4000 split.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (
    choice_prompt,
    codebook_prompt,
    cuda_time_ms,
    fit_temperature,
    latency_summary,
    multiclass_metrics,
    read_ag_news,
    render_instruct,
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
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--latency-calls", type=int, default=40)
    parser.add_argument("--feature-batch", type=int, default=4)
    parser.add_argument("--exit-layer", type=int, default=8)
    return parser.parse_args()


def chat_batch(tokenizer, prompts: list[str], max_length: int) -> dict[str, torch.Tensor]:
    batch = tokenizer(
        render_instruct(tokenizer, prompts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {key: value.cuda(non_blocking=True) for key, value in batch.items()}


def raw_batch(tokenizer, prompts: list[str], max_length: int) -> dict[str, torch.Tensor]:
    batch = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {key: value.cuda(non_blocking=True) for key, value in batch.items()}


def last_logits(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    try:
        return model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1, :]
    except TypeError:
        return model(**batch, use_cache=False).logits[:, -1, :]


def time_calls(fn, warmups: int, calls: int) -> dict:
    with torch.inference_mode():
        for _ in range(warmups):
            fn()
        times = [cuda_time_ms(fn)[1] for _ in range(calls)]
    return latency_summary(times, 1)


def decoder_stack(model) -> torch.nn.Module:
    candidates = [model.model, getattr(model.model, "language_model", None), model]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "layers"):
            return candidate
    raise RuntimeError("no decoder stack with .layers found")


def bidirectional_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    batch, length = attention_mask.shape
    keep = attention_mask[:, None, None, :].to(dtype)
    blocked = (1.0 - keep) * torch.finfo(dtype).min
    return blocked.expand(batch, 1, length, length).contiguous()


def native_prompt(text: str) -> str:
    return f"News article:\n{text}"


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
    weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
    mean = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
    return mean.float().cpu().numpy()


def extract_layer(model, tokenizer, rows, layer: int, max_length: int, batch_size: int, bidirectional: bool) -> np.ndarray:
    stack = decoder_stack(model)
    original = stack.layers
    stack.layers = original[:layer]
    parts = []
    try:
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            batch = raw_batch(tokenizer, [native_prompt(row["text"]) for row in chunk], max_length)
            inputs = dict(batch)
            if bidirectional:
                inputs["attention_mask"] = bidirectional_mask(batch["attention_mask"], model.dtype)
            with torch.inference_mode():
                hidden = model.model(**inputs, use_cache=False).last_hidden_state
            parts.append(mean_pool(hidden, batch["attention_mask"]))
            if (start // batch_size) % 50 == 0:
                print(f"features bidirectional={bidirectional} {start}/{len(rows)}", flush=True)
    finally:
        stack.layers = original
    return np.concatenate(parts)


def train_head(train_x, train_y, calibration_x, calibration_y, test_x, test_y) -> dict:
    classifier = LogisticRegression(max_iter=200, C=1.0, solver="lbfgs", random_state=20260920)
    classifier.fit(train_x, train_y)
    test_logits = classifier.decision_function(test_x)
    temperature = fit_temperature(classifier.decision_function(calibration_x), calibration_y)
    calibrated = multiclass_metrics(softmax_numpy(test_logits / temperature), test_y)
    return {
        "accuracy": calibrated["accuracy"],
        "brier": calibrated["brier"],
        "ece_15": calibrated["ece_15"],
        "temperature": temperature,
    }


def mask_sanity(model, tokenizer, rows, max_length: int, layer: int) -> dict:
    stack = decoder_stack(model)
    original = stack.layers
    stack.layers = original[:layer]
    batch = raw_batch(tokenizer, [native_prompt(row["text"]) for row in rows[:4]], max_length)
    try:
        with torch.inference_mode():
            causal = model.model(**batch, use_cache=False).last_hidden_state
            bidi_inputs = dict(batch)
            bidi_inputs["attention_mask"] = bidirectional_mask(batch["attention_mask"], model.dtype)
            bidi = model.model(**bidi_inputs, use_cache=False).last_hidden_state
    finally:
        stack.layers = original
    difference = (causal.float() - bidi.float()).abs()
    return {
        "max_abs_hidden_difference": float(difference.max()),
        "mean_abs_hidden_difference": float(difference.mean()),
        "causality_removed": bool(difference.max() > 1e-3),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(20260920)
    test_rows = stratified_subset(read_ag_news(args.test_csv), 1000, seed=20260920)
    train_rows, calibration_rows = stratified_disjoint(
        read_ag_news(args.train_csv), 2000, 500, seed=20260920
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    letter = chat_batch(tokenizer, [choice_prompt(test_rows[0]["text"])], args.max_length)
    code = chat_batch(tokenizer, [codebook_prompt(test_rows[0]["text"])], args.max_length)

    def generate1():
        return model.generate(
            **letter,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    latency = {
        "scheme_1_generate1": time_calls(generate1, 10, args.latency_calls),
        "scheme_4_last_logits": time_calls(lambda: last_logits(model, letter), 10, args.latency_calls),
        "scheme_3_codebook_forward": time_calls(lambda: last_logits(model, code), 10, args.latency_calls),
    }
    for name, summary in latency.items():
        summary["prompt_tokens"] = int(
            (code if name.startswith("scheme_3") else letter)["attention_mask"][0].sum().item()
        )
        print(name, summary, flush=True)

    labels = lambda rows: np.asarray([row["label"] for row in rows])
    causal_x = {
        "train": extract_layer(model, tokenizer, train_rows, args.exit_layer, args.max_length, args.feature_batch, False),
        "calibration": extract_layer(model, tokenizer, calibration_rows, args.exit_layer, args.max_length, args.feature_batch, False),
        "test": extract_layer(model, tokenizer, test_rows, args.exit_layer, args.max_length, args.feature_batch, False),
    }
    bidi_x = {
        "train": extract_layer(model, tokenizer, train_rows, args.exit_layer, args.max_length, args.feature_batch, True),
        "calibration": extract_layer(model, tokenizer, calibration_rows, args.exit_layer, args.max_length, args.feature_batch, True),
        "test": extract_layer(model, tokenizer, test_rows, args.exit_layer, args.max_length, args.feature_batch, True),
    }
    y_train, y_cal, y_test = labels(train_rows), labels(calibration_rows), labels(test_rows)
    causal = train_head(causal_x["train"], y_train, causal_x["calibration"], y_cal, causal_x["test"], y_test)
    bidi = train_head(bidi_x["train"], y_train, bidi_x["calibration"], y_cal, bidi_x["test"], y_test)
    scheme5 = {
        "exit_layer": args.exit_layer,
        "pooling": "mean",
        "causal": causal,
        "bidirectional": bidi,
        "accuracy_delta_pp": (bidi["accuracy"] - causal["accuracy"]) * 100.0,
        "mask_sanity": mask_sanity(model, tokenizer, test_rows, args.max_length, args.exit_layer),
    }
    print("scheme_5", scheme5, flush=True)
    payload = {
        "model": args.model.rstrip("/").rsplit("/", 1)[-1],
        "arm": "stage2_gaps",
        "latency": latency,
        "scheme_5": scheme5,
    }
    write_json(args.output, payload)


if __name__ == "__main__":
    main()
