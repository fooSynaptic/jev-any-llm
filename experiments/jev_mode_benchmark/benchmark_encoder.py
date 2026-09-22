#!/usr/bin/env python3
"""Scheme 5: bidirectional encoder plus a trained Choice head."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoTokenizer

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
    parser.add_argument("--train-per-class", type=int, default=2000)
    parser.add_argument("--calibration-per-class", type=int, default=500)
    parser.add_argument("--test-per-class", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latency-calls", type=int, default=100)
    return parser.parse_args()


def tokenize(tokenizer, texts: list[str], max_length: int) -> dict[str, torch.Tensor]:
    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {key: value.cuda(non_blocking=True) for key, value in batch.items()}


def mean_pool(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    hidden = model(**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def extract(model, tokenizer, rows, max_length, batch_size):
    features = []
    lengths = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        batch = tokenize(tokenizer, [row["text"] for row in chunk], max_length)
        with torch.inference_mode():
            pooled = mean_pool(model, batch)
        features.append(pooled.float().cpu().numpy())
        lengths.extend(batch["attention_mask"].sum(dim=1).cpu().tolist())
    return np.concatenate(features), lengths


def latency_probe(model, tokenizer, row, head, max_length, batch_size, calls):
    batch = tokenize(tokenizer, [row["text"]] * batch_size, max_length)

    def run():
        return head(mean_pool(model, batch).float())

    with torch.inference_mode():
        for _ in range(20):
            run()
        times = [cuda_time_ms(run)[1] for _ in range(calls)]
    result = latency_summary(times, batch_size)
    result["input_tokens"] = int(batch["attention_mask"][0].sum().item())
    return result


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    ).cuda().eval()
    train_rows, calibration_rows = stratified_disjoint(
        read_ag_news(args.train_csv),
        args.train_per_class,
        args.calibration_per_class,
        seed=20260920,
    )
    test_rows = stratified_subset(
        read_ag_news(args.test_csv), args.test_per_class, seed=20260920
    )
    train_y = np.asarray([row["label"] for row in train_rows])
    calibration_y = np.asarray([row["label"] for row in calibration_rows])
    test_y = np.asarray([row["label"] for row in test_rows])
    train_x, train_lengths = extract(
        model, tokenizer, train_rows, args.max_length, args.batch_size
    )
    calibration_x, _ = extract(
        model, tokenizer, calibration_rows, args.max_length, args.batch_size
    )
    test_x, test_lengths = extract(
        model, tokenizer, test_rows, args.max_length, args.batch_size
    )
    classifier = LogisticRegression(
        max_iter=300,
        C=1.0,
        solver="lbfgs",
        random_state=20260920,
    )
    classifier.fit(train_x, train_y)
    raw_logits = classifier.decision_function(test_x)
    calibration_logits = classifier.decision_function(calibration_x)
    raw_probabilities = softmax_numpy(raw_logits)
    temperature = fit_temperature(calibration_logits, calibration_y)
    probabilities = softmax_numpy(raw_logits / temperature)
    head = torch.nn.Linear(train_x.shape[1], 4, device="cuda").float()
    with torch.no_grad():
        head.weight.copy_(torch.from_numpy(classifier.coef_).cuda())
        head.bias.copy_(torch.from_numpy(classifier.intercept_).cuda())
    head.eval()

    result = {
        "model": args.model,
        "dataset": "AG News",
        "scheme_5_bidirectional_encoder_trained_head": {
            **multiclass_metrics(probabilities, test_y),
            "uncalibrated": multiclass_metrics(raw_probabilities, test_y),
            "training": {
                "method": "frozen bidirectional encoder + mean pool + logistic Choice head",
                "train_n": len(train_rows),
                "calibration_n": len(calibration_rows),
                "test_n": len(test_rows),
                "temperature": temperature,
                "feature_dim": train_x.shape[1],
                "train_mean_tokens": float(np.mean(train_lengths)),
                "test_mean_tokens": float(np.mean(test_lengths)),
                "classifier_iterations": classifier.n_iter_.tolist(),
            },
        },
        "gpu_latency": {
            "batch_1": latency_probe(
                model,
                tokenizer,
                test_rows[0],
                head,
                args.max_length,
                1,
                args.latency_calls,
            ),
            f"batch_{args.batch_size}": latency_probe(
                model,
                tokenizer,
                test_rows[0],
                head,
                args.max_length,
                args.batch_size,
                max(30, args.latency_calls // 2),
            ),
        },
    }
    write_json(args.output, result)
    np.savez_compressed(
        str(Path(args.output).with_suffix(".head.npz")),
        coef=classifier.coef_,
        intercept=classifier.intercept_,
        classes=classifier.classes_,
        temperature=temperature,
    )
    print(Path(args.output).read_text())


if __name__ == "__main__":
    main()
