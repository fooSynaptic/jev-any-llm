#!/usr/bin/env python3
"""Shared-prefill branched KV vs isolated wrap — accuracy + latency.

Arms (same closed-set letter/YN aliases, zero training):

1. **isolated** — full prompt per question (state + question), N forwards.
2. **branched** — shared state prefill once; fork KV; per-question suffix logits.
3. **isolated_batch** (latency only) — N independent prompts in one batch.

Branched continues the *same* chat-templated token sequence as isolated by
splitting the rendered string at the shared-prefix / question-suffix boundary
(so quality is an apples-to-apples check, not a second prompt dialect).

Quality lock: AG News Choice A–D (PROTOCOL.md). Multi-Q latency uses the same
four judgments as ``benchmark_packed.py`` (topic + 3 binary), but branched is
KV-fork — not packed into one decode string.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

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


TOPICS_BINARY = [
    ("sports", "Sports"),
    ("business", "Business"),
    ("scitech", "science or technology"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-class", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--latency-calls", type=int, default=40)
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def steady(times: list[float]) -> list[float]:
    return times[2:] if len(times) > 4 else times


def state_block(text: str) -> str:
    return f"State:\nArticle:\n{text}\n\n"


def topic_suffix() -> str:
    return "\n".join(
        [
            "Question: Classify the news article into exactly one topic.",
            "Closed set:",
            "A = World: international affairs, politics, governments, war.",
            "B = Sports: games, teams, athletes, tournaments.",
            "C = Business: companies, markets, finance, economy.",
            "D = Sci/Tech: science, technology, computers, space.",
            "Answer (A, B, C, or D):",
        ]
    )


def binary_suffix(topic: str) -> str:
    return (
        f"Question: Is the news article primarily about {topic}?\n"
        "Closed set:\n"
        "Y = yes\n"
        "N = no\n"
        "Answer (Y or N):"
    )


def full_prompt(text: str, suffix: str) -> str:
    return (
        "You are a typed predictor. Answer with exactly one alias token "
        "from the closed set.\n\n"
        f"{state_block(text)}{suffix}"
    )


def shared_prefix(text: str) -> str:
    return (
        "You are a typed predictor. Answer with exactly one alias token "
        "from the closed set.\n\n"
        f"{state_block(text)}"
    )


def clone_past(past_key_values):
    if hasattr(past_key_values, "copy") and callable(past_key_values.copy):
        try:
            return past_key_values.copy()
        except Exception:
            pass
    return copy.deepcopy(past_key_values)


def last_logits_from_out(out) -> torch.Tensor:
    return out.logits[:, -1, :]


def forward_full(model, input_ids, attention_mask=None):
    kwargs = {"input_ids": input_ids, "use_cache": False}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    try:
        return model(**kwargs, logits_to_keep=1)
    except TypeError:
        return model(**kwargs)


def split_chat_ids(
    tokenizer,
    text: str,
    suffix: str,
    max_length: int,
    device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (prefix_ids, suffix_ids, full_ids) with prefix+suffix == full.

    Splits the chat-rendered string at the first occurrence of ``suffix`` so
    branched KV continues the identical sequence isolated would score.
    """
    full = full_prompt(text, suffix)
    rendered = render_instruct(tokenizer, [full])[0]
    marker = suffix
    idx = rendered.find(marker)
    if idx < 0:
        # Fallback: split on shared raw prefix inside rendered user blob.
        pref = shared_prefix(text)
        idx = rendered.find(pref)
        if idx < 0:
            raise RuntimeError("cannot locate shared prefix inside chat template")
        idx = idx + len(pref)
    prefix_str = rendered[:idx]
    suffix_str = rendered[idx:]
    pref_ids = tokenizer(
        prefix_str,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )["input_ids"].to(device)
    # Remaining suffix may be truncated if prefix ate the budget — keep aligned.
    remain = max(1, max_length - int(pref_ids.shape[-1]))
    suf_ids = tokenizer(
        suffix_str,
        add_special_tokens=False,
        truncation=True,
        max_length=remain,
        return_tensors="pt",
    )["input_ids"].to(device)
    full_ids = torch.cat([pref_ids, suf_ids], dim=-1)
    return pref_ids, suf_ids, full_ids


def tokenize_batch_full(tokenizer, texts_suffixes, max_length, device):
    prompts = [full_prompt(text, suffix) for text, suffix in texts_suffixes]
    rendered = render_instruct(tokenizer, prompts)
    batch = tokenizer(
        rendered,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.inference_mode()
def quality_isolated(model, tokenizer, rows, max_length, batch_size, token_ids, device):
    probs_all = []
    times = []
    suffix = topic_suffix()
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        batch = tokenize_batch_full(
            tokenizer, [(row["text"], suffix) for row in chunk], max_length, device
        )

        def _run():
            return last_logits_from_out(
                forward_full(model, batch["input_ids"], batch.get("attention_mask"))
            )

        logits, ms = cuda_time_ms(_run)
        times.append(ms)
        probs_all.append(closed_probs(logits, token_ids).cpu().numpy())
    return np.concatenate(probs_all, axis=0), times


@torch.inference_mode()
def quality_branched(model, tokenizer, rows, max_length, token_ids, device):
    probs_all = []
    times = []
    suffix = topic_suffix()
    for row in rows:

        def _run():
            pref_ids, suf_ids, _ = split_chat_ids(
                tokenizer, row["text"], suffix, max_length, device
            )
            out_pref = model(input_ids=pref_ids, use_cache=True)
            past = clone_past(out_pref.past_key_values)
            out = model(input_ids=suf_ids, past_key_values=past, use_cache=True)
            return last_logits_from_out(out)

        logits, ms = cuda_time_ms(_run)
        times.append(ms)
        probs_all.append(closed_probs(logits[0], token_ids).cpu().numpy())
    return np.stack(probs_all, axis=0), times


@torch.inference_mode()
def latency_multi_q(
    model,
    tokenizer,
    rows,
    *,
    mode: str,
    max_length: int,
    latency_calls: int,
    choice_ids: list[int],
    yn_ids: list[int],
    device,
) -> dict:
    probe = rows[: max(1, min(len(rows), latency_calls))]
    times = []
    suffixes = [topic_suffix()] + [binary_suffix(t) for _, t in TOPICS_BINARY]

    for row in probe:
        if mode == "isolated":

            def _run(r=row):
                outs = []
                for i, suffix in enumerate(suffixes):
                    batch = tokenize_batch_full(
                        tokenizer, [(r["text"], suffix)], max_length, device
                    )
                    logits = last_logits_from_out(
                        forward_full(
                            model, batch["input_ids"], batch.get("attention_mask")
                        )
                    )
                    ids = choice_ids if i == 0 else yn_ids
                    outs.append(closed_probs(logits[0], ids))
                return outs

        elif mode == "branched":

            def _run(r=row):
                # Prefill longest shared raw prefix once using first question's
                # chat split; other questions re-split (suffix-only after shared
                # state). For fair KV reuse, prefill shared_prefix via the first
                # question's prefix_ids, then for each question run only the
                # suffix tokens from that question's full split — requiring the
                # prefix_ids to match across questions.
                pref0, _, _ = split_chat_ids(
                    tokenizer, r["text"], suffixes[0], max_length, device
                )
                out_pref = model(input_ids=pref0, use_cache=True)
                base_past = out_pref.past_key_values
                outs = []
                for i, suffix in enumerate(suffixes):
                    pref_i, suf_i, _ = split_chat_ids(
                        tokenizer, r["text"], suffix, max_length, device
                    )
                    if pref_i.shape != pref0.shape or not torch.equal(pref_i, pref0):
                        # Chat boundary drifted — fall back to full isolated for this Q.
                        batch = tokenize_batch_full(
                            tokenizer, [(r["text"], suffix)], max_length, device
                        )
                        logits = last_logits_from_out(
                            forward_full(
                                model, batch["input_ids"], batch.get("attention_mask")
                            )
                        )
                    else:
                        past = clone_past(base_past)
                        out = model(
                            input_ids=suf_i, past_key_values=past, use_cache=True
                        )
                        logits = last_logits_from_out(out)
                    ids = choice_ids if i == 0 else yn_ids
                    outs.append(closed_probs(logits[0], ids))
                return outs

        elif mode == "isolated_batch":

            def _run(r=row):
                batch = tokenize_batch_full(
                    tokenizer, [(r["text"], s) for s in suffixes], max_length, device
                )
                logits = last_logits_from_out(
                    forward_full(model, batch["input_ids"], batch.get("attention_mask"))
                )
                return [
                    closed_probs(logits[0], choice_ids),
                    closed_probs(logits[1], yn_ids),
                    closed_probs(logits[2], yn_ids),
                    closed_probs(logits[3], yn_ids),
                ]

        else:
            raise ValueError(mode)

        _, ms = cuda_time_ms(_run)
        times.append(ms)

    return latency_summary(steady(times), examples_per_call=1)


def main() -> int:
    args = parse_args()
    rows = stratified_subset(read_ag_news(args.test_csv), args.per_class)
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map=args.device,
    ).eval()
    device = next(model.parameters()).device

    try:
        choice_ids = alias_token_ids(tokenizer, ALIASES, leading_space=True)
    except ValueError:
        choice_ids = alias_token_ids(tokenizer, ALIASES, leading_space=False)
    try:
        yn_ids = alias_token_ids(tokenizer, ["Y", "N"], leading_space=True)
    except ValueError:
        yn_ids = alias_token_ids(tokenizer, ["Y", "N"], leading_space=False)

    payload = {
        "model": args.model,
        "n": len(rows),
        "per_class": args.per_class,
        "max_length": args.max_length,
        "labels": LABELS,
        "choice_token_ids": choice_ids,
        "yn_token_ids": yn_ids,
        "protocol": "branched_kv_vs_isolated_ag_news",
        "arms": {},
    }

    if not args.skip_quality:
        print("quality isolated…", flush=True)
        iso_probs, iso_times = quality_isolated(
            model, tokenizer, rows, args.max_length, args.batch_size, choice_ids, device
        )
        print("quality branched…", flush=True)
        br_probs, br_times = quality_branched(
            model, tokenizer, rows, args.max_length, choice_ids, device
        )
        payload["arms"]["isolated_choice"] = {
            "metrics": multiclass_metrics(iso_probs, labels),
            "latency_batch": latency_summary(
                steady(iso_times), examples_per_call=args.batch_size
            ),
            "note": f"batched quality forward, batch_size={args.batch_size}",
        }
        payload["arms"]["branched_choice"] = {
            "metrics": multiclass_metrics(br_probs, labels),
            "latency_batch1": latency_summary(steady(br_times), examples_per_call=1),
            "note": "batch-1 prefill+branch; same token sequence as isolated",
        }
        iso_pred = iso_probs.argmax(axis=1)
        br_pred = br_probs.argmax(axis=1)
        payload["arms"]["agreement"] = {
            "argmax_match_rate": float((iso_pred == br_pred).mean()),
            "mean_abs_prob_diff": float(np.abs(iso_probs - br_probs).mean()),
        }
        print(
            "isolated_acc",
            round(payload["arms"]["isolated_choice"]["metrics"]["accuracy"], 4),
            "branched_acc",
            round(payload["arms"]["branched_choice"]["metrics"]["accuracy"], 4),
            "agree",
            round(payload["arms"]["agreement"]["argmax_match_rate"], 4),
            flush=True,
        )

    print("latency multi-Q (4 judgments / article)…", flush=True)
    latency_rows = stratified_subset(rows, min(50, args.per_class), seed=20260921)[
        : args.latency_calls
    ]
    for mode in ("isolated", "branched", "isolated_batch"):
        summary = latency_multi_q(
            model,
            tokenizer,
            latency_rows,
            mode=mode,
            max_length=args.max_length,
            latency_calls=args.latency_calls,
            choice_ids=choice_ids,
            yn_ids=yn_ids,
            device=device,
        )
        payload["arms"][f"latency_4q_{mode}"] = summary
        print(f"  {mode}: p50={summary['p50_ms']:.2f} ms", flush=True)

    # Speedup vs sequential isolated.
    if "latency_4q_isolated" in payload["arms"] and "latency_4q_branched" in payload["arms"]:
        iso_p50 = payload["arms"]["latency_4q_isolated"]["p50_ms"]
        br_p50 = payload["arms"]["latency_4q_branched"]["p50_ms"]
        bat_p50 = payload["arms"]["latency_4q_isolated_batch"]["p50_ms"]
        payload["summary"] = {
            "branched_vs_isolated_speedup": float(iso_p50 / br_p50) if br_p50 else None,
            "batch_vs_isolated_speedup": float(iso_p50 / bat_p50) if bat_p50 else None,
            "branched_vs_batch_ratio": float(br_p50 / bat_p50) if bat_p50 else None,
        }
        if "isolated_choice" in payload["arms"]:
            payload["summary"]["isolated_accuracy"] = payload["arms"]["isolated_choice"][
                "metrics"
            ]["accuracy"]
            payload["summary"]["branched_accuracy"] = payload["arms"]["branched_choice"][
                "metrics"
            ]["accuracy"]
            payload["summary"]["argmax_agreement"] = payload["arms"]["agreement"][
                "argmax_match_rate"
            ]

    write_json(args.output, payload)
    print("wrote", args.output, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
