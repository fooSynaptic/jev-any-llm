#!/usr/bin/env python3
"""Zero-training ways to close the gap to the free-text baseline.

Four levers, none of which uses a label:
  A  multi-token label words scored with length-normalised logprob
  B  contextual calibration against content-free inputs
  C  verbalizer ensembles marginalised per class
  D  a short generated rationale before the closed-set readout
"""

from __future__ import annotations

import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (
    ALIASES,
    LABELS,
    alias_token_ids,
    closed_probs,
    cuda_time_ms,
    latency_summary,
    multiclass_metrics,
    read_ag_news,
    render_instruct,
    stratified_subset,
    write_json,
)

# Single-token words, the current 83.15% arm. "Sci/Tech" is truncated to "Sci".
SINGLE_WORDS = ["World", "Sports", "Business", "Sci"]

# A: full label words, scored over every token.
FULL_WORDS = ["World", "Sports", "Business", "Sci/Tech"]
DESCRIPTIVE_WORDS = ["World News", "Sports", "Business", "Science and Technology"]

# C: several surface forms per class, marginalised.
ENSEMBLE = [
    ["World", "International", "Global", "Politics"],
    ["Sports", "Athletics", "Football", "Game"],
    ["Business", "Finance", "Economy", "Market"],
    ["Sci/Tech", "Science", "Technology", "Computer"],
]

CONTENT_FREE = ["N/A", "", "[MASK]"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-class", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--latency-calls", type=int, default=24)
    parser.add_argument("--rationale-tokens", type=int, nargs="*", default=[8, 16, 32])
    parser.add_argument(
        "--only", choices=["all", "rationale", "quality", "latency"], default="all"
    )
    return parser.parse_args()


def word_prompt(text: str) -> str:
    return (
        "Classify the news article into exactly one topic.\n"
        "World: international affairs, politics, governments, war.\n"
        "Sports: games, teams, athletes, tournaments.\n"
        "Business: companies, markets, finance, economy.\n"
        "Sci/Tech: science, technology, computers, space.\n"
        "Return exactly one topic.\n\n"
        f"Article:\n{text}\n\nTopic:"
    )


def rationale_prompt(text: str) -> str:
    return (
        "Classify the news article into exactly one topic: "
        "World, Sports, Business, or Sci/Tech.\n"
        "First state the key evidence in a few words, then the topic.\n\n"
        f"Article:\n{text}\n\nEvidence:"
    )


def prompt_ids(tokenizer, prompts: list[str], max_length: int) -> list[list[int]]:
    rendered = render_instruct(tokenizer, prompts)
    encoded = tokenizer(
        rendered, add_special_tokens=False, truncation=True, max_length=max_length
    )
    return encoded["input_ids"]


def left_pad(sequences: list[list[int]], pad_id: int) -> dict[str, torch.Tensor]:
    width = max(len(seq) for seq in sequences)
    input_ids, mask = [], []
    for seq in sequences:
        gap = width - len(seq)
        input_ids.append([pad_id] * gap + seq)
        mask.append([0] * gap + [1] * len(seq))
    return {
        "input_ids": torch.tensor(input_ids, device="cuda"),
        "attention_mask": torch.tensor(mask, device="cuda"),
    }


def candidate_logprobs(
    model,
    tokenizer,
    contexts: list[list[int]],
    candidate: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (summed logprob, token count) of `candidate` after each context."""
    tail = tokenizer(candidate, add_special_tokens=False)["input_ids"]
    length = len(tail)
    totals = np.zeros(len(contexts), dtype=np.float64)

    for start in range(0, len(contexts), batch_size):
        chunk = contexts[start : start + batch_size]
        batch = left_pad([seq + tail for seq in chunk], tokenizer.pad_token_id)
        with torch.inference_mode():
            logits = model(**batch, use_cache=False).logits
        # Left padding puts the candidate in the final `length` positions.
        window = logits[:, -(length + 1) : -1, :].float().log_softmax(dim=-1)
        target = torch.tensor(tail, device=window.device).view(1, length, 1)
        picked = window.gather(2, target.expand(window.size(0), length, 1))
        totals[start : start + len(chunk)] = picked.squeeze(2).sum(dim=1).cpu().numpy()

    return totals, np.full(len(contexts), length, dtype=np.float64)


def score_words(
    model, tokenizer, contexts: list[list[int]], words: list[str], batch_size: int
) -> dict[str, np.ndarray]:
    summed, lengths = [], []
    for word in words:
        total, count = candidate_logprobs(model, tokenizer, contexts, word, batch_size)
        summed.append(total)
        lengths.append(count)
    summed = np.stack(summed, axis=1)
    lengths = np.stack(lengths, axis=1)
    return {"sum": summed, "normalised": summed / lengths}


def softmax_rows(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def content_free_prior(
    model, tokenizer, words: list[str], max_length: int
) -> np.ndarray:
    contexts = prompt_ids(
        tokenizer, [word_prompt(text) for text in CONTENT_FREE], max_length
    )
    scores = score_words(model, tokenizer, contexts, words, len(CONTENT_FREE))
    prior = softmax_rows(scores["normalised"]).mean(axis=0)
    return prior / prior.sum()


def ensemble_scores(
    model, tokenizer, contexts: list[list[int]], batch_size: int
) -> np.ndarray:
    per_class = []
    for members in ENSEMBLE:
        scores = score_words(model, tokenizer, contexts, members, batch_size)
        # marginalise the surface forms of one class
        per_class.append(torch.logsumexp(torch.tensor(scores["normalised"]), dim=1).numpy())
    return np.stack(per_class, axis=1)


def single_token_probs(
    model, tokenizer, prompts: list[str], token_ids: list[int], max_length: int, batch_size: int
) -> np.ndarray:
    out = []
    contexts = prompt_ids(tokenizer, prompts, max_length)
    for start in range(0, len(contexts), batch_size):
        batch = left_pad(contexts[start : start + batch_size], tokenizer.pad_token_id)
        with torch.inference_mode():
            logits = model(**batch, use_cache=False).logits[:, -1, :]
        out.append(closed_probs(logits, token_ids).cpu().numpy())
    return np.concatenate(out, axis=0)


def rationale_probs(
    model,
    tokenizer,
    rows: list[dict],
    token_ids: list[int],
    new_tokens: int,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    out = []
    tail = tokenizer("\nTopic:", add_special_tokens=False)["input_ids"]
    stops = {tokenizer.eos_token_id, tokenizer.pad_token_id}
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        prompts = [rationale_prompt(row["text"]) for row in chunk]
        contexts = prompt_ids(tokenizer, prompts, max_length)
        batch = left_pad(contexts, tokenizer.pad_token_id)
        with torch.inference_mode():
            # No min_new_tokens: forcing a model that wants to stop produces junk.
            grown = model.generate(
                **batch,
                max_new_tokens=new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        produced = grown[:, batch["input_ids"].size(1) :].tolist()
        rebuilt = []
        for context, extra in zip(contexts, produced):
            kept = []
            for token in extra:
                if token in stops:
                    break
                kept.append(token)
            rebuilt.append(context + kept + tail)
        widened = left_pad(rebuilt, tokenizer.pad_token_id)
        with torch.inference_mode():
            logits = model(**widened, use_cache=False).logits[:, -1, :]
        out.append(closed_probs(logits, token_ids).cpu().numpy())
    return np.concatenate(out, axis=0)


def steady(times: list[float]) -> list[float]:
    return times[2:] if len(times) > 4 else times


def multi_pass_latency(
    model, tokenizer, rows: list[dict], words: list[str], max_length: int, calls: int
) -> dict:
    contexts = prompt_ids(tokenizer, [word_prompt(rows[0]["text"])], max_length)

    def run():
        for word in words:
            tail = tokenizer(word, add_special_tokens=False)["input_ids"]
            batch = left_pad([contexts[0] + tail], tokenizer.pad_token_id)
            model(**batch, use_cache=False)

    with torch.inference_mode():
        for _ in range(6):
            run()
        times = [cuda_time_ms(run)[1] for _ in range(calls)]
    return latency_summary(steady(times), 1)


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    rows = stratified_subset(read_ag_news(args.test_csv), args.per_class)
    labels = np.asarray([row["label"] for row in rows])
    prompts = [word_prompt(row["text"]) for row in rows]
    contexts = prompt_ids(tokenizer, prompts, args.max_length)

    want_quality = args.only in ("all", "rationale", "quality")
    want_latency = args.only in ("all", "latency")
    want_abc = args.only in ("all", "quality")

    result: dict = {
        "dataset": "AG News",
        "test": {"n": len(rows), "per_class": args.per_class},
        "note": "no labels used anywhere in this file",
    }

    # Reference arm: single-token words, reproduces the 83.15% number.
    single_ids = alias_token_ids(tokenizer, SINGLE_WORDS, leading_space=False)
    if want_quality:
        single = single_token_probs(
            model, tokenizer, prompts, single_ids, args.max_length, args.batch_size
        )
        result["reference_single_token"] = multiclass_metrics(single, labels)
        print("single-token", result["reference_single_token"], flush=True)

    # A: multi-token label words.
    for name, words in (
        (("full_words", FULL_WORDS), ("descriptive", DESCRIPTIVE_WORDS))
        if want_abc
        else ()
    ):
        scores = score_words(model, tokenizer, contexts, words, args.batch_size)
        entry = {}
        for variant in ("sum", "normalised"):
            probs = softmax_rows(scores[variant])
            entry[variant] = multiclass_metrics(probs, labels)
            if variant == "normalised":
                prior = content_free_prior(model, tokenizer, words, args.max_length)
                # Both terms are probabilities before the division, as in B below.
                calibrated = softmax_rows(
                    np.log(np.clip(probs, 1e-12, 1)) - np.log(prior)[None, :]
                )
                entry["normalised_contextual_calibration"] = multiclass_metrics(
                    calibrated, labels
                )
                entry["content_free_prior"] = prior.tolist()
        result[f"A_{name}"] = entry
        print(f"A_{name}", entry, flush=True)

    if want_abc:
        # C: verbalizer ensemble, with and without contextual calibration.
        ens = ensemble_scores(model, tokenizer, contexts, args.batch_size)
        result["C_ensemble"] = {"plain": multiclass_metrics(softmax_rows(ens), labels)}
        cf_contexts = prompt_ids(
            tokenizer, [word_prompt(text) for text in CONTENT_FREE], args.max_length
        )
        ens_cf = ensemble_scores(model, tokenizer, cf_contexts, len(CONTENT_FREE))
        ens_prior = softmax_rows(ens_cf).mean(axis=0)
        ens_prior = ens_prior / ens_prior.sum()
        ens_probs = softmax_rows(ens)
        result["C_ensemble"]["contextual_calibration"] = multiclass_metrics(
            softmax_rows(
                np.log(np.clip(ens_probs, 1e-12, 1)) - np.log(ens_prior)[None, :]
            ),
            labels,
        )
        result["C_ensemble"]["content_free_prior"] = ens_prior.tolist()
        print("C_ensemble", result["C_ensemble"], flush=True)

        # B on the plain single-token arm, for a like-for-like comparison.
        cf_single = single_token_probs(
            model,
            tokenizer,
            [word_prompt(text) for text in CONTENT_FREE],
            single_ids,
            args.max_length,
            len(CONTENT_FREE),
        ).mean(axis=0)
        cf_single = cf_single / cf_single.sum()
        result["B_single_token_contextual_calibration"] = multiclass_metrics(
            softmax_rows(
                np.log(np.clip(single, 1e-12, 1)) - np.log(cf_single)[None, :]
            ),
            labels,
        )
        result["B_single_token_contextual_calibration"]["content_free_prior"] = (
            cf_single.tolist()
        )
        print("B_single", result["B_single_token_contextual_calibration"], flush=True)

    # D: short rationale before the readout.
    result["D_rationale"] = {}
    for new_tokens in args.rationale_tokens:
        entry = {}
        if want_quality:
            probs = rationale_probs(
                model, tokenizer, rows, single_ids, new_tokens, args.max_length, args.batch_size
            )
            entry = multiclass_metrics(probs, labels)

        def run(budget=new_tokens):
            return rationale_probs(
                model, tokenizer, rows[:1], single_ids, budget, args.max_length, 1
            )

        if want_latency:
            for _ in range(4):
                run()
            entry["latency_batch_1"] = latency_summary(
                steady([cuda_time_ms(run)[1] for _ in range(args.latency_calls)]), 1
            )
        result["D_rationale"][f"tokens_{new_tokens}"] = entry
        print("D", new_tokens, entry, flush=True)

    # Cost of the multi-pass scoring used by A and C.
    if want_latency:
        result["latency"] = {
            "A_four_passes_batch_1": multi_pass_latency(
                model, tokenizer, rows, FULL_WORDS, args.max_length, args.latency_calls
            ),
            "C_sixteen_passes_batch_1": multi_pass_latency(
                model,
                tokenizer,
                rows,
                [word for members in ENSEMBLE for word in members],
                args.max_length,
                args.latency_calls,
            ),
        }
        print("latency", result["latency"], flush=True)

    write_json(args.out, result)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
