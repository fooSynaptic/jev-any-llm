"""Shared utilities for the AG News Jev-mode benchmark."""

from __future__ import annotations

import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

LABELS = ["World", "Sports", "Business", "Sci/Tech"]
ALIASES = ["A", "B", "C", "D"]


def read_ag_news(path: str | Path) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for label, title, description in csv.reader(handle):
            rows.append(
                {
                    "label": int(label) - 1,
                    "text": f"{title.strip()}\n{description.strip()}",
                }
            )
    return rows


def read_labeled_csv(path: str | Path) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({"label": int(row["label"]), "text": row["text"]})
    return rows


def n_classes(rows: list[dict]) -> int:
    return 1 + max(row["label"] for row in rows)


def grouped_by_label(rows: list[dict]) -> dict[int, list[dict]]:
    groups: dict[int, list[dict]] = {i: [] for i in range(n_classes(rows))}
    for row in rows:
        groups[row["label"]].append(row)
    return groups


def stratified_subset(rows: list[dict], per_class: int, seed: int = 20260920) -> list[dict]:
    rng = random.Random(seed)
    groups = grouped_by_label(rows)
    selected = []
    for label, group in groups.items():
        if len(group) < per_class:
            raise ValueError(f"label {label}: requested {per_class}, only {len(group)}")
        selected.extend(rng.sample(group, per_class))
    rng.shuffle(selected)
    return selected


def stratified_disjoint(
    rows: list[dict],
    first_per_class: int,
    second_per_class: int,
    seed: int = 20260920,
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    groups = grouped_by_label(rows)
    first, second = [], []
    for label, group in groups.items():
        rng.shuffle(group)
        required = first_per_class + second_per_class
        if len(group) < required:
            raise ValueError(f"label {label}: requested {required}, only {len(group)}")
        first.extend(group[:first_per_class])
        second.extend(group[first_per_class:required])
    rng.shuffle(first)
    rng.shuffle(second)
    return first, second


def choice_prompt(text: str) -> str:
    return (
        "Classify the news article into exactly one topic.\n"
        "A = World: international affairs, politics, governments, war.\n"
        "B = Sports: games, teams, athletes, tournaments.\n"
        "C = Business: companies, markets, finance, economy.\n"
        "D = Sci/Tech: science, technology, computers, space.\n"
        "Return exactly one code: A, B, C, or D.\n\n"
        f"Article:\n{text}\n\nAnswer:"
    )


def ar_prompt(text: str) -> str:
    return (
        "Classify the news article. Return two lines only:\n"
        "Category: World|Sports|Business|Sci/Tech\n"
        "Confidence: a number from 0 to 1\n\n"
        f"Article:\n{text}\n\nResponse:"
    )


def codebook_prompt(text: str, bins: int = 4) -> str:
    symbols = list("ABCDEFGHIJKLMNOP")
    mids = [0.40, 0.625, 0.825, 0.95]
    lines = []
    for cls, label in enumerate(LABELS):
        for confidence_bin in range(bins):
            symbol = symbols[cls * bins + confidence_bin]
            lines.append(
                f"{symbol} = {label}, confidence about {mids[confidence_bin]:.3f}"
            )
    legend = "\n".join(lines)
    return (
        "Classify the news article and encode both topic and confidence in one code.\n"
        f"{legend}\n"
        "Return exactly one code from A through P.\n\n"
        f"Article:\n{text}\n\nAnswer:"
    )


def packed_prompt(text: str) -> str:
    return (
        "Read the article once and answer four judgments in order.\n"
        "Q1 topic: A=World, B=Sports, C=Business, D=Sci/Tech.\n"
        "Q2 is the topic Sports? Y=yes, N=no.\n"
        "Q3 is the topic Business? Y=yes, N=no.\n"
        "Q4 is the topic Sci/Tech? Y=yes, N=no.\n"
        "Return four codes in order. Each code is chosen independently from its set.\n\n"
        f"Article:\n{text}\n\nAnswers:"
    )


def alias_token_ids(tokenizer, aliases: Iterable[str], leading_space: bool = True) -> list[int]:
    ids = []
    for alias in aliases:
        text = f" {alias}" if leading_space else alias
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f"alias {text!r} is not one token: {token_ids}")
        ids.append(token_ids[0])
    return ids


def render_instruct(tokenizer, prompts: Iterable[str]) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in prompts
    ]


def closed_probs(logits: torch.Tensor, token_ids: list[int]) -> torch.Tensor:
    return torch.softmax(logits[..., token_ids].float(), dim=-1)


def multiclass_metrics(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> dict:
    eps = 1e-12
    pred = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    one_hot = np.eye(probs.shape[1], dtype=np.float64)[labels]
    accuracy = float((pred == labels).mean())
    nll = float(-np.log(np.clip(probs[np.arange(len(labels)), labels], eps, 1)).mean())
    brier = float(np.square(probs - one_hot).sum(axis=1).mean())
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            ece += float(mask.mean()) * abs(
                float((pred[mask] == labels[mask]).mean()) - float(confidence[mask].mean())
            )
    entropy = -(probs * np.log(np.clip(probs, eps, 1))).sum(axis=1)
    entropy_confidence = 1.0 - entropy / math.log(probs.shape[1])
    return {
        "n": int(len(labels)),
        "accuracy": accuracy,
        "nll": nll,
        "brier": brier,
        "ece_15": float(ece),
        "mean_max_probability": float(confidence.mean()),
        "mean_entropy_confidence": float(entropy_confidence.mean()),
    }


def ordinal_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    levels = np.arange(probs.shape[1], dtype=np.float64)
    expected = (probs * levels).sum(axis=1)
    return {
        **multiclass_metrics(probs, labels),
        "mae": float(np.abs(expected - labels).mean()),
        "mean_expected_score": float(expected.mean()),
    }


def as_logits(decision) -> np.ndarray:
    array = np.asarray(decision, dtype=np.float64)
    if array.ndim == 1:
        zeros = np.zeros_like(array)
        return np.stack([zeros, array], axis=1)
    return array


def knee_from_accuracies(by_layer: dict[int, float], slack: float = 0.01) -> dict:
    peak_layer = max(by_layer, key=by_layer.get)
    peak = by_layer[peak_layer]
    within = min(layer for layer, value in by_layer.items() if value >= peak - slack)
    return {
        "peak_layer": int(peak_layer),
        "peak_accuracy": float(peak),
        "first_layer_within_1pp": int(within),
        "first_within_1pp_accuracy": float(by_layer[within]),
    }


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    temperatures = np.exp(np.linspace(math.log(0.1), math.log(10.0), 401))
    losses = []
    for temperature in temperatures:
        probabilities = softmax_numpy(logits / temperature)
        losses.append(
            -np.log(
                np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)
            ).mean()
        )
    return float(temperatures[int(np.argmin(losses))])


def latency_summary(milliseconds: list[float], examples_per_call: int = 1) -> dict:
    values = np.asarray(milliseconds, dtype=np.float64)
    return {
        "calls": int(len(values)),
        "examples_per_call": int(examples_per_call),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "mean_ms": float(values.mean()),
        "qps": float(1000.0 * examples_per_call / values.mean()),
    }


def cuda_time_ms(fn) -> tuple[object, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, float(start.elapsed_time(end))


def wall_time_ms(fn) -> tuple[object, float]:
    start = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - start) * 1000.0


def write_json(path: str | Path, payload: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
