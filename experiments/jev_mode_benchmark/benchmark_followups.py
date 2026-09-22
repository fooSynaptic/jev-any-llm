#!/usr/bin/env python3
"""Follow-up conversions: harder Choice, Noul, Score, MLP head, learned prefix."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (
    as_logits,
    cuda_time_ms,
    fit_temperature,
    knee_from_accuracies,
    latency_summary,
    multiclass_metrics,
    n_classes,
    ordinal_metrics,
    read_ag_news,
    read_labeled_csv,
    softmax_numpy,
    stratified_disjoint,
    stratified_subset,
    write_json,
)


LAYERS = [1, 2, 4, 6, 8, 12, 16, 24, 32]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True, choices=["depth", "mlp", "prefix"])
    parser.add_argument("--task", default="agnews", choices=["agnews", "20newsgroups", "sst2", "sst5"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", default="1,2,4,6,8,12,16,24,32")
    parser.add_argument("--prefix-exits", default="4,8,16,32")
    parser.add_argument("--prefix-mid-exit", type=int, default=0)
    parser.add_argument("--train-per-class", type=int, default=0)
    parser.add_argument("--calibration-per-class", type=int, default=0)
    parser.add_argument("--test-per-class", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--latency-calls", type=int, default=24)
    parser.add_argument("--prefix-length", type=int, default=8)
    parser.add_argument("--micro-batch", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--only", choices=["all", "quality", "latency"], default="all")
    return parser.parse_args()


def native_prompt(text: str, task: str) -> str:
    if task == "sst2":
        return f"Review:\n{text}\nIs the sentiment positive?"
    if task == "sst5":
        return f"Review:\n{text}\nSentiment from 0 (very negative) to 4 (very positive):"
    if task == "20newsgroups":
        return f"Newsgroup post:\n{text}"
    return f"News article:\n{text}"


def load_rows(path: str, task: str) -> list[dict]:
    if task == "agnews":
        return read_ag_news(path)
    return read_labeled_csv(path)


def auto_split(train_rows, test_rows, train_n, cal_n, test_n):
    n = n_classes(train_rows)
    counts = [sum(row["label"] == i for row in train_rows) for i in range(n)]
    test_counts = [sum(row["label"] == i for row in test_rows) for i in range(n)]
    train_n = train_n or max(80, min(counts) - 40)
    cal_n = cal_n or min(40, min(counts) - train_n)
    if train_n + cal_n > min(counts):
        cal_n = max(20, min(counts) // 5)
        train_n = min(counts) - cal_n
    test_n = test_n or min(test_counts)
    train, calibration = stratified_disjoint(train_rows, train_n, cal_n)
    test = stratified_subset(test_rows, test_n)
    return train, calibration, test, {"train_per_class": train_n, "calibration_per_class": cal_n, "test_per_class": test_n}


def decoder_stack(model) -> torch.nn.Module:
    candidates = [getattr(model.model, "language_model", None), model.model, model]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "layers"):
            return candidate
    raise RuntimeError("no decoder stack with .layers found")


def tokenize(tokenizer, prompts, max_length):
    batch = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {key: value.cuda(non_blocking=True) for key, value in batch.items()}


def mean_pool(hidden, attention_mask):
    weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


def extract_mean(model, tokenizer, rows, layers, max_length, batch_size, task):
    buckets = {k: [] for k in layers}
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        batch = tokenize(
            tokenizer,
            [native_prompt(row["text"], task) for row in chunk],
            max_length,
        )
        with torch.inference_mode():
            hidden_states = model.model(
                **batch, use_cache=False, output_hidden_states=True
            ).hidden_states
        for k in layers:
            buckets[k].append(mean_pool(hidden_states[k], batch["attention_mask"]).float().cpu().numpy())
    return {k: np.concatenate(parts) for k, parts in buckets.items()}


def train_linear(train_x, train_y, calibration_x, calibration_y, test_x, test_y, kind: str):
    classifier = LogisticRegression(
        max_iter=300, C=1.0, solver="lbfgs", random_state=20260920
    )
    classifier.fit(train_x, train_y)
    test_logits = as_logits(classifier.decision_function(test_x))
    calibration_logits = as_logits(classifier.decision_function(calibration_x))
    temperature = fit_temperature(calibration_logits, calibration_y)
    probs = softmax_numpy(test_logits / temperature)
    metrics = ordinal_metrics(probs, test_y) if kind == "score" else multiclass_metrics(probs, test_y)
    return {**metrics, "temperature": temperature, "head": "linear"}


class MLPHead(torch.nn.Module):
    def __init__(self, dim: int, n_out: int, hidden: int = 512):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden, n_out),
        )

    def forward(self, features):
        return self.net(features)


def train_mlp(train_x, train_y, calibration_x, calibration_y, test_x, test_y, kind: str):
    device = "cuda"
    n_out = int(max(train_y.max(), test_y.max()) + 1)
    head = MLPHead(train_x.shape[1], n_out).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    x = torch.tensor(train_x, device=device)
    y = torch.tensor(train_y, device=device)
    head.train()
    for _ in range(40):
        perm = torch.randperm(len(x), device=device)
        for start in range(0, len(x), 256):
            index = perm[start : start + 256]
            loss = torch.nn.functional.cross_entropy(head(x[index]), y[index])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    head.eval()
    with torch.inference_mode():
        test_logits = head(torch.tensor(test_x, device=device)).float().cpu().numpy()
        calibration_logits = head(torch.tensor(calibration_x, device=device)).float().cpu().numpy()
    temperature = fit_temperature(calibration_logits, calibration_y)
    probs = softmax_numpy(test_logits / temperature)
    metrics = ordinal_metrics(probs, test_y) if kind == "score" else multiclass_metrics(probs, test_y)
    return {**metrics, "temperature": temperature, "head": "mlp_512"}


def load_model(path):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    ).eval()
    return tokenizer, model


def run_depth(args, kind: str, heads: list[str]):
    tokenizer, model = load_model(args.model)
    train_rows, calibration_rows, test_rows, split = auto_split(
        load_rows(args.train_csv, args.task),
        load_rows(args.test_csv, args.task),
        args.train_per_class,
        args.calibration_per_class,
        args.test_per_class,
    )
    train_y = np.asarray([row["label"] for row in train_rows])
    calibration_y = np.asarray([row["label"] for row in calibration_rows])
    test_y = np.asarray([row["label"] for row in test_rows])
    print("split", args.task, split, "n_classes", int(train_y.max()) + 1, flush=True)

    layers = [int(item) for item in args.layers.split(",") if item.strip()]
    train_f = extract_mean(model, tokenizer, train_rows, layers, args.max_length, args.batch_size, args.task)
    calibration_f = extract_mean(
        model, tokenizer, calibration_rows, layers, args.max_length, args.batch_size, args.task
    )
    test_f = extract_mean(model, tokenizer, test_rows, layers, args.max_length, args.batch_size, args.task)

    result = {
        "task": args.task,
        "kind": kind,
        "split": {**split, "train_n": len(train_rows), "calibration_n": len(calibration_rows), "test_n": len(test_rows)},
        "n_classes": int(train_y.max()) + 1,
        "heads": {},
    }
    for head_name in heads:
        trainer = train_mlp if head_name == "mlp" else train_linear
        per_layer = {}
        accuracies = {}
        for k in layers:
            metrics = trainer(
                train_f[k], train_y, calibration_f[k], calibration_y, test_f[k], test_y, kind
            )
            per_layer[f"layer_{k}"] = metrics
            accuracies[k] = metrics["accuracy"]
            print(args.task, head_name, k, {key: metrics[key] for key in metrics if key in {"accuracy", "mae", "ece_15", "brier"}}, flush=True)
        result["heads"][head_name] = {
            "by_layer": per_layer,
            "knee": knee_from_accuracies(accuracies),
        }
    write_json(args.output, result)
    print(Path(args.output).read_text())


def prefix_forward(model, batch, prefix: torch.nn.Parameter, n_layers: int):
    stack = decoder_stack(model)
    original = stack.layers
    try:
        stack.layers = original[:n_layers]
        token_embeds = stack.embed_tokens(batch["input_ids"])
        expanded = prefix.to(dtype=token_embeds.dtype).unsqueeze(0).expand(token_embeds.size(0), -1, -1)
        inputs_embeds = torch.cat([expanded, token_embeds], dim=1)
        prefix_mask = torch.ones(
            token_embeds.size(0),
            prefix.size(0),
            device=token_embeds.device,
            dtype=batch["attention_mask"].dtype,
        )
        attention_mask = torch.cat([prefix_mask, batch["attention_mask"]], dim=1)
        # Deep exits rely on HF per-layer gradient checkpointing (enabled in
        # train_prefix). Do not wrap the whole stack in one checkpoint: the
        # backward recompute would still materialise all n_layers at once.
        hidden = stack(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        ).last_hidden_state
        return mean_pool(hidden[:, prefix.size(0) :, :], batch["attention_mask"])
    finally:
        stack.layers = original


def train_prefix(model, tokenizer, train_rows, calibration_rows, test_rows, args, n_layers, prefix_length):
    n_out = n_classes(train_rows)
    hidden = model.config.get_text_config().hidden_size
    prefix = torch.nn.Parameter(0.02 * torch.randn(prefix_length, hidden, device="cuda"))
    head = torch.nn.Linear(hidden, n_out, device="cuda")
    opt = torch.optim.AdamW(list(head.parameters()) + [prefix], lr=1e-3, weight_decay=1e-4)
    for param in model.parameters():
        param.requires_grad_(False)
    # Per-layer checkpointing: only prefix/head are trained, but grads still
    # flow through every kept layer back to the prefix tokens.
    use_ckpt = args.only != "latency" and n_layers >= 32
    if use_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        stack = decoder_stack(model)
        if hasattr(stack, "gradient_checkpointing_enable"):
            stack.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        model.train()
        stack.train()
    else:
        model.eval()
    # Timing depends only on prefix_length and exit layer, never on the learned
    # values, so the latency pass reuses the random initialisation.
    try:
        for epoch in range(args.epochs if args.only != "latency" else 0):
            order = np.random.default_rng(20260920 + epoch).permutation(len(train_rows))
            losses = []
            micro = args.micro_batch or args.batch_size
            for start in range(0, len(train_rows), args.batch_size):
                chunk = [train_rows[int(i)] for i in order[start : start + args.batch_size]]
                opt.zero_grad(set_to_none=True)
                total = 0.0
                # Accumulating over micro-batches keeps the effective batch (and so
                # the optimisation) identical while the backward graph stays small
                # enough for deep exits on a 27B backbone.
                for offset in range(0, len(chunk), micro):
                    part = chunk[offset : offset + micro]
                    batch = tokenize(
                        tokenizer,
                        [native_prompt(row["text"], args.task) for row in part],
                        args.max_length,
                    )
                    labels = torch.tensor([row["label"] for row in part], device="cuda")
                    pooled = prefix_forward(model, batch, prefix, n_layers)
                    loss = torch.nn.functional.cross_entropy(head(pooled.float()), labels)
                    (loss * (len(part) / len(chunk))).backward()
                    total += float(loss.detach()) * len(part) / len(chunk)
                opt.step()
                losses.append(total)
            print("prefix-epoch", n_layers, prefix_length, epoch, float(np.mean(losses)), flush=True)
    finally:
        if use_ckpt and hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        model.eval()

    def collect(rows):
        logits = []
        for start in range(0, len(rows), args.batch_size):
            chunk = rows[start : start + args.batch_size]
            batch = tokenize(
                tokenizer,
                [native_prompt(row["text"], args.task) for row in chunk],
                args.max_length,
            )
            with torch.inference_mode():
                pooled = prefix_forward(model, batch, prefix, n_layers)
                logits.append(head(pooled.float()).cpu().numpy())
        return np.concatenate(logits)

    if args.only != "latency":
        calibration_logits = collect(calibration_rows)
        test_logits = collect(test_rows)
        calibration_y = np.asarray([row["label"] for row in calibration_rows])
        test_y = np.asarray([row["label"] for row in test_rows])
        temperature = fit_temperature(calibration_logits, calibration_y)
        metrics = multiclass_metrics(softmax_numpy(test_logits / temperature), test_y)
    else:
        temperature = None
        metrics = {}

    probe_batch = tokenize(
        tokenizer, [native_prompt(test_rows[0]["text"], args.task)], args.max_length
    )

    def latency_fn():
        with torch.inference_mode():
            pooled = prefix_forward(model, probe_batch, prefix, n_layers)
            return head(pooled.float())

    metrics.update(
        {
            "temperature": temperature,
            "prefix_length": prefix_length,
            "exit_layer": n_layers,
        }
    )
    if args.only != "quality":
        with torch.inference_mode():
            for _ in range(10):
                latency_fn()
            times = [cuda_time_ms(latency_fn)[1] for _ in range(args.latency_calls)]
        metrics["latency"] = latency_summary(times, 1)
    return metrics


def run_prefix_job(args):
    tokenizer, model = load_model(args.model)
    train_rows, calibration_rows, test_rows, split = auto_split(
        load_rows(args.train_csv, args.task),
        load_rows(args.test_csv, args.task),
        args.train_per_class or 2000,
        args.calibration_per_class or 500,
        args.test_per_class or 1000,
    )
    result = {"task": args.task, "split": split, "configs": []}
    exits = [int(item) for item in args.prefix_exits.split(",") if item.strip()]
    # prefix_mid_exit < 0 disables the extra length sweep; 0 means "second exit".
    if args.prefix_mid_exit < 0:
        mid = None
    elif args.prefix_mid_exit > 0:
        mid = args.prefix_mid_exit
    else:
        mid = exits[min(1, len(exits) - 1)]
    for n_layers in exits:
        extra = (4, 16) if mid is not None and n_layers == mid else ()
        for prefix_length in (args.prefix_length, *extra):
            metrics = train_prefix(
                model, tokenizer, train_rows, calibration_rows, test_rows, args, n_layers, prefix_length
            )
            print("prefix-result", metrics, flush=True)
            result["configs"].append(metrics)
    write_json(args.output, result)
    print(Path(args.output).read_text())


def main() -> None:
    args = parse_args()
    kind = {"agnews": "choice", "20newsgroups": "choice", "sst2": "noul", "sst5": "score"}[args.task]
    if args.job == "prefix":
        run_prefix_job(args)
        return
    heads = ["linear", "mlp"] if args.job == "mlp" else ["linear"]
    run_depth(args, kind, heads)


if __name__ == "__main__":
    main()
