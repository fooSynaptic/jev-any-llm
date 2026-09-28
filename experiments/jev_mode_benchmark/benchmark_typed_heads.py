#!/usr/bin/env python3
"""Typed heads on one layer-8 mean-pool of the state.

One truncated forward produces the state vector. Three heads then read it:

- Noul: binary logistic regression.
- Choice: low-rank scorer of each option text. Softmax uses only the options
  in the request, so the option count can change per call.
- Score: proportional-odds cumulative logits. One direction, one cut per
  boundary, shared across the ordered levels.

The fixed multiclass logistic on the same vectors is recorded beside Choice
and Score so the new heads have a matched comparison.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (
    LABELS,
    as_logits,
    cuda_time_ms,
    fit_temperature,
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


AG_OPTIONS = [f"Topic:\n{name}" for name in LABELS]
LEVELS = [
    "very negative",
    "negative",
    "neutral",
    "positive",
    "very positive",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--ag-train", required=True)
    parser.add_argument("--ag-test", required=True)
    parser.add_argument("--sst2-train", required=True)
    parser.add_argument("--sst2-test", required=True)
    parser.add_argument("--sst5-train", required=True)
    parser.add_argument("--sst5-test", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exit-layer", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--feature-batch", type=int, default=8)
    parser.add_argument("--latency-calls", type=int, default=40)
    parser.add_argument("--rank", type=int, default=32)
    return parser.parse_args()


def prompt_for(task: str, text: str) -> str:
    if task == "sst2":
        return f"Review:\n{text}\nIs the sentiment positive?"
    if task == "sst5":
        return f"Review:\n{text}\nSentiment from 0 (very negative) to 4 (very positive):"
    return f"News article:\n{text}"


def take_split(train_rows, test_rows, train_n=2000, cal_n=500, test_n=1000):
    n = n_classes(train_rows)
    train_counts = [sum(row["label"] == i for row in train_rows) for i in range(n)]
    test_counts = [sum(row["label"] == i for row in test_rows) for i in range(n)]
    train_n = min(train_n, min(train_counts) - 20)
    cal_n = min(cal_n, min(train_counts) - train_n)
    test_n = min(test_n, min(test_counts))
    train, calibration = stratified_disjoint(train_rows, train_n, cal_n, seed=20260920)
    test = stratified_subset(test_rows, test_n, seed=20260920)
    return train, calibration, test, {
        "train_per_class": train_n,
        "calibration_per_class": cal_n,
        "test_per_class": test_n,
    }


def decoder_stack(model):
    candidates = [model.model, getattr(model.model, "language_model", None), model]
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


def extract(model, tokenizer, rows, task, max_length, batch_size, name):
    parts = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        batch = tokenize(tokenizer, [prompt_for(task, row["text"]) for row in chunk], max_length)
        with torch.inference_mode():
            hidden = model.model(**batch, use_cache=False).last_hidden_state
        parts.append(mean_pool(hidden, batch["attention_mask"]).float().cpu().numpy())
        if start % (batch_size * 25) == 0:
            print(f"features {name} {start}/{len(rows)}", flush=True)
    return np.concatenate(parts)


def extract_texts(model, tokenizer, texts, max_length):
    batch = tokenize(tokenizer, texts, max_length)
    with torch.inference_mode():
        hidden = model.model(**batch, use_cache=False).last_hidden_state
    return mean_pool(hidden, batch["attention_mask"]).float().cpu().numpy()


def standardize(train_x, *others):
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.clip(std, 1e-4, None)
    return tuple((array - mean) / std for array in (train_x, *others)), mean, std


def labels_of(rows):
    return np.asarray([row["label"] for row in rows], dtype=np.int64)


def fit_logistic(train_x, train_y, cal_x, cal_y, test_x, test_y, ordinal=False):
    classifier = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=20260920)
    classifier.fit(train_x, train_y)
    temperature = fit_temperature(as_logits(classifier.decision_function(cal_x)), cal_y)
    probs = softmax_numpy(as_logits(classifier.decision_function(test_x)) / temperature)
    metrics = ordinal_metrics(probs, test_y) if ordinal else multiclass_metrics(probs, test_y)
    metrics["temperature"] = temperature
    return metrics, classifier


def fit_set_scorer(train_x, train_y, cal_x, cal_y, options, rank, steps=400):
    (train_z, option_z, cal_z), mean, std = standardize(train_x, options, cal_x)
    state = torch.tensor(train_z, dtype=torch.float32, device="cuda")
    target = torch.tensor(train_y, dtype=torch.long, device="cuda")
    option = torch.tensor(option_z, dtype=torch.float32, device="cuda")
    cal_state = torch.tensor(cal_z, dtype=torch.float32, device="cuda")
    cal_target = torch.tensor(cal_y, dtype=torch.long, device="cuda")
    left = torch.nn.Linear(state.shape[1], rank, bias=False).cuda()
    right = torch.nn.Linear(option.shape[1], rank, bias=False).cuda()
    torch.nn.init.normal_(left.weight, std=0.02)
    torch.nn.init.normal_(right.weight, std=0.02)
    scale = torch.nn.Parameter(torch.tensor(1.0, device="cuda"))
    params = [*left.parameters(), *right.parameters(), scale]
    optimizer = torch.optim.Adam(params, lr=0.01, weight_decay=1e-2)
    best_loss = float("inf")
    best_state = None
    for step in range(steps):
        optimizer.zero_grad()
        logits = scale * (left(state) @ right(option).T)
        loss = torch.nn.functional.cross_entropy(logits, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        if step % 25 == 0 or step == steps - 1:
            with torch.inference_mode():
                cal_logits = scale * (left(cal_state) @ right(option).T)
                cal_loss = float(torch.nn.functional.cross_entropy(cal_logits, cal_target))
            if cal_loss < best_loss:
                best_loss = cal_loss
                best_state = {
                    "left": left.state_dict(),
                    "right": right.state_dict(),
                    "scale": scale.detach().clone(),
                    "step": step,
                    "train_loss": float(loss),
                }
            if step % 100 == 0 or step == steps - 1:
                print("choice", step, round(float(loss), 4), "cal", round(cal_loss, 4), flush=True)
    left.load_state_dict(best_state["left"])
    right.load_state_dict(best_state["right"])
    with torch.no_grad():
        scale.copy_(best_state["scale"])
    print("choice best", best_state["step"], round(best_loss, 4), flush=True)
    return left, right, scale, mean, std


def set_logits(left, right, scale, features, options, mean, std):
    features = (features - mean) / std
    options = (options - mean) / std
    with torch.inference_mode():
        state = torch.tensor(features, dtype=torch.float32, device="cuda")
        option = torch.tensor(options, dtype=torch.float32, device="cuda")
        return (scale * (left(state) @ right(option).T)).float().cpu().numpy()


def subset_accuracy(logits, labels, k, seed=20260920):
    rng = np.random.default_rng(seed)
    n_options = logits.shape[1]
    correct = 0
    for index, label in enumerate(labels):
        others = [item for item in range(n_options) if item != int(label)]
        chosen = [int(label), *rng.choice(others, size=k - 1, replace=False).tolist()]
        winner = chosen[int(np.argmax(logits[index, chosen]))]
        correct += int(winner == int(label))
    return {"k": k, "n": int(len(labels)), "accuracy": correct / len(labels), "chance": 1.0 / k}


def ordinal_log_probs(score, cuts):
    logs = [-torch.nn.functional.softplus(score - cuts[0])]
    for index in range(cuts.shape[0] - 1):
        upper = score - cuts[index]
        lower = score - cuts[index + 1]
        gap = torch.log(torch.expm1((upper - lower).clamp_min(1e-6)))
        logs.append(lower - torch.nn.functional.softplus(upper) - torch.nn.functional.softplus(lower) + gap)
    logs.append(-torch.nn.functional.softplus(cuts[-1] - score))
    return torch.stack(logs, dim=1)


def fit_cumulative(train_x, train_y, cal_x, cal_y, n_levels, steps=800):
    feature_scale = float(np.clip(train_x.std(), 1e-4, None))
    state = torch.tensor(train_x / feature_scale, dtype=torch.float32, device="cuda")
    target = torch.tensor(train_y, dtype=torch.long, device="cuda")
    cal_state = torch.tensor(cal_x / feature_scale, dtype=torch.float32, device="cuda")
    cal_target = torch.tensor(cal_y, dtype=torch.long, device="cuda")
    direction = torch.nn.Parameter(torch.zeros(state.shape[1], device="cuda"))
    raw_gaps = torch.nn.Parameter(torch.zeros(n_levels - 1, device="cuda"))
    optimizer = torch.optim.Adam([direction, raw_gaps], lr=0.05, weight_decay=1e-4)
    best_loss = float("inf")
    best_state = None

    def class_nll(features, labels, direction_param, gap_param):
        cuts = torch.cumsum(torch.nn.functional.softplus(gap_param) + 1e-3, dim=0)
        log_probs = ordinal_log_probs(features @ direction_param, cuts)
        return -log_probs[torch.arange(len(labels), device=features.device), labels].mean()

    for step in range(steps):
        optimizer.zero_grad()
        loss = class_nll(state, target, direction, raw_gaps)
        loss = loss + 1e-4 * direction.square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([direction, raw_gaps], 1.0)
        optimizer.step()
        if step % 25 == 0 or step == steps - 1:
            with torch.inference_mode():
                cal_loss = float(class_nll(cal_state, cal_target, direction, raw_gaps))
            if cal_loss < best_loss:
                best_loss = cal_loss
                best_state = {
                    "direction": direction.detach().clone(),
                    "raw_gaps": raw_gaps.detach().clone(),
                    "step": step,
                }
            if step % 100 == 0 or step == steps - 1:
                print("score", step, round(float(loss), 4), "cal", round(cal_loss, 4), flush=True)
    print("score best", best_state["step"], round(best_loss, 4), flush=True)
    return best_state["direction"], best_state["raw_gaps"], feature_scale


def cumulative_probs(direction, raw_gaps, features, temperature, feature_scale):
    state = torch.tensor(features / feature_scale, dtype=torch.float32, device="cuda")
    with torch.inference_mode():
        cuts = torch.cumsum(torch.nn.functional.softplus(raw_gaps) + 1e-3, dim=0)
        log_probs = ordinal_log_probs((state @ direction) / temperature, cuts)
        return torch.exp(log_probs).float().cpu().numpy()


def fit_cumulative_temperature(direction, raw_gaps, features, labels, feature_scale):
    temperatures = np.exp(np.linspace(np.log(0.1), np.log(10.0), 81))
    losses = []
    for temperature in temperatures:
        probs = cumulative_probs(direction, raw_gaps, features, float(temperature), feature_scale)
        picked = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
        losses.append(float(-np.log(picked).mean()))
    return float(temperatures[int(np.argmin(losses))])


def time_calls(fn, warmups, calls):
    with torch.inference_mode():
        for _ in range(warmups):
            fn()
        times = [cuda_time_ms(fn)[1] for _ in range(calls)]
    return latency_summary(times, 1)


def main() -> None:
    args = parse_args()
    torch.manual_seed(20260920)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    stack = decoder_stack(model)
    stack.layers = stack.layers[: args.exit_layer]

    ag_train, ag_cal, ag_test, ag_split = take_split(read_ag_news(args.ag_train), read_ag_news(args.ag_test))
    s2_train, s2_cal, s2_test, s2_split = take_split(
        read_labeled_csv(args.sst2_train), read_labeled_csv(args.sst2_test)
    )
    s5_train, s5_cal, s5_test, s5_split = take_split(
        read_labeled_csv(args.sst5_train), read_labeled_csv(args.sst5_test)
    )

    ag_x = {
        "train": extract(model, tokenizer, ag_train, "ag", args.max_length, args.feature_batch, "ag-train"),
        "calibration": extract(model, tokenizer, ag_cal, "ag", args.max_length, args.feature_batch, "ag-cal"),
        "test": extract(model, tokenizer, ag_test, "ag", args.max_length, args.feature_batch, "ag-test"),
    }
    s2_x = {
        "train": extract(model, tokenizer, s2_train, "sst2", args.max_length, args.feature_batch, "sst2-train"),
        "calibration": extract(model, tokenizer, s2_cal, "sst2", args.max_length, args.feature_batch, "sst2-cal"),
        "test": extract(model, tokenizer, s2_test, "sst2", args.max_length, args.feature_batch, "sst2-test"),
    }
    s5_x = {
        "train": extract(model, tokenizer, s5_train, "sst5", args.max_length, args.feature_batch, "sst5-train"),
        "calibration": extract(model, tokenizer, s5_cal, "sst5", args.max_length, args.feature_batch, "sst5-cal"),
        "test": extract(model, tokenizer, s5_test, "sst5", args.max_length, args.feature_batch, "sst5-test"),
    }
    options = extract_texts(model, tokenizer, AG_OPTIONS, args.max_length)
    y_ag = {name: labels_of(rows) for name, rows in (("train", ag_train), ("calibration", ag_cal), ("test", ag_test))}
    y_s2 = {name: labels_of(rows) for name, rows in (("train", s2_train), ("calibration", s2_cal), ("test", s2_test))}
    y_s5 = {name: labels_of(rows) for name, rows in (("train", s5_train), ("calibration", s5_cal), ("test", s5_test))}

    noul, _ = fit_logistic(s2_x["train"], y_s2["train"], s2_x["calibration"], y_s2["calibration"], s2_x["test"], y_s2["test"])
    print("noul", noul, flush=True)
    fixed_choice, choice_clf = fit_logistic(
        ag_x["train"], y_ag["train"], ag_x["calibration"], y_ag["calibration"], ag_x["test"], y_ag["test"]
    )
    left, right, scale, choice_mean, choice_std = fit_set_scorer(
        ag_x["train"], y_ag["train"], ag_x["calibration"], y_ag["calibration"], options, args.rank
    )
    choice_test_logits = set_logits(left, right, scale, ag_x["test"], options, choice_mean, choice_std)
    choice_cal_logits = set_logits(left, right, scale, ag_x["calibration"], options, choice_mean, choice_std)
    choice_temperature = fit_temperature(choice_cal_logits, y_ag["calibration"])
    choice_probs = softmax_numpy(choice_test_logits / choice_temperature)
    choice_full = multiclass_metrics(choice_probs, y_ag["test"])
    choice_full["temperature"] = choice_temperature
    choice_subsets = [subset_accuracy(choice_test_logits, y_ag["test"], k) for k in (2, 3, 4)]
    print("choice", choice_full, choice_subsets, flush=True)

    n_levels = int(y_s5["train"].max()) + 1
    direction, raw_gaps, score_scale = fit_cumulative(
        s5_x["train"], y_s5["train"], s5_x["calibration"], y_s5["calibration"], n_levels
    )
    score_temperature = fit_cumulative_temperature(
        direction, raw_gaps, s5_x["calibration"], y_s5["calibration"], score_scale
    )
    score_probs = cumulative_probs(direction, raw_gaps, s5_x["test"], score_temperature, score_scale)
    score = ordinal_metrics(score_probs, y_s5["test"])
    score["temperature"] = score_temperature
    fixed_score, _ = fit_logistic(
        s5_x["train"], y_s5["train"], s5_x["calibration"], y_s5["calibration"], s5_x["test"], y_s5["test"], ordinal=True
    )
    print("score", score, fixed_score, flush=True)

    sample = tokenize(tokenizer, [prompt_for("ag", ag_test[0]["text"])], args.max_length)
    choice_mean_t = torch.tensor(choice_mean, dtype=torch.float32, device="cuda")
    choice_std_t = torch.tensor(choice_std, dtype=torch.float32, device="cuda")
    score_scale_t = torch.tensor(score_scale, dtype=torch.float32, device="cuda")
    option_z = (options - choice_mean) / choice_std
    option_t = torch.tensor(option_z, dtype=torch.float32, device="cuda")
    noul_clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=20260920)
    noul_clf.fit(s2_x["train"], y_s2["train"])
    noul_coef = torch.tensor(noul_clf.coef_, dtype=torch.float32, device="cuda")

    def one_forward_three_heads():
        hidden = model.model(**sample, use_cache=False).last_hidden_state
        pooled = mean_pool(hidden, sample["attention_mask"]).float()
        noul_logit = pooled @ noul_coef.T
        choice_state = (pooled - choice_mean_t) / choice_std_t
        choice_logit = scale * (left(choice_state) @ right(option_t).T)
        score_state = pooled / score_scale_t
        score_logit = score_state @ direction
        return noul_logit, choice_logit, score_logit

    def three_forwards():
        first = model.model(**sample, use_cache=False).last_hidden_state
        second = model.model(**sample, use_cache=False).last_hidden_state
        third = model.model(**sample, use_cache=False).last_hidden_state
        return first, second, third

    latency = {
        "one_forward_three_heads": time_calls(one_forward_three_heads, 10, args.latency_calls),
        "three_separate_forwards": time_calls(three_forwards, 10, args.latency_calls),
    }
    latency["one_forward_three_heads"]["prompt_tokens"] = int(sample["attention_mask"][0].sum())
    print("latency", latency, flush=True)

    payload = {
        "model": args.model.rstrip("/").rsplit("/", 1)[-1],
        "arm": "typed_heads_layer8",
        "exit_layer": args.exit_layer,
        "pooling": "mean",
        "splits": {"ag_news": ag_split, "sst2": s2_split, "sst5": s5_split},
        "noul": noul,
        "choice": {
            "set_scorer_k4": choice_full,
            "subsets_including_gold": choice_subsets,
            "fixed_logistic": fixed_choice,
        },
        "score": {"cumulative": score, "fixed_logistic": fixed_score, "levels": LEVELS[:n_levels]},
        "latency": latency,
    }
    write_json(args.output, payload)
    print("wrote", args.output, flush=True)


if __name__ == "__main__":
    main()
