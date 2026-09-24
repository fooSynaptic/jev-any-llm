#!/usr/bin/env python3
"""Branched KV vs isolated wrap on DeepSeek-V4.1-Flash (official TP8).

Mirrors ``benchmark_branched.py`` arms on the Flash inference stack used by
``benchmark_flash_tp8.py``:

- **isolated** — full prompt, ``forward(..., start_pos=0)``
- **branched** — shared-prefix prefill, snapshot KV buffers, per-question
  suffix at ``start_pos=len(prefix)`` (independent questions)
- **isolated_batch** — N prompts in one ``forward`` (latency arm)

Launch (8 GPUs)::

    torchrun --nproc-per-node 8 benchmark_branched_flash_tp8.py ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (  # noqa: E402
    ALIASES,
    LABELS,
    alias_token_ids,
    closed_probs,
    cuda_time_ms,
    latency_summary,
    multiclass_metrics,
    read_ag_news,
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
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--inference-dir", required=True)
    parser.add_argument("--encoding-dir", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-class", type=int, default=500)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--fwd-batch", type=int, default=4)
    parser.add_argument("--latency-calls", type=int, default=24)
    parser.add_argument("--branched-quality-cap", type=int, default=400,
                        help="Max rows for batch-1 branched quality (cost control)")
    parser.add_argument("--skip-quality", action="store_true")
    return parser.parse_args()


def is_rank0() -> bool:
    return int(os.getenv("RANK", "0")) == 0


def log(*parts) -> None:
    if is_rank0():
        print(*parts, flush=True)


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


def load_stack(args):
    sys.path.insert(0, args.encoding_dir)
    sys.path.insert(0, args.inference_dir)
    from encoding import encode_messages
    from model import ModelArgs, Transformer
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
    model_args.max_batch_size = max(4, args.fwd_batch, 4)
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
        "world_size": world_size,
        "rank": rank,
    }


def render(encode_messages, content: str) -> str:
    return encode_messages(
        [{"role": "user", "content": content}],
        thinking_mode="chat",
    )


def split_ids(tokenizer, encode_messages, text: str, suffix: str, max_length: int):
    full_r = render(encode_messages, full_prompt(text, suffix))
    idx = full_r.find(suffix)
    if idx < 0:
        pref = shared_prefix(text)
        idx = full_r.find(pref)
        if idx < 0:
            raise RuntimeError("cannot locate shared prefix in chat render")
        idx = idx + len(pref)
    pref_ids = tokenizer.encode(full_r[:idx], add_special_tokens=False)[:max_length]
    remain = max(1, max_length - len(pref_ids))
    suf_ids = tokenizer.encode(full_r[idx:], add_special_tokens=False)[:remain]
    full_ids = (pref_ids + suf_ids)[:max_length]
    # Re-derive suffix so pref+suf == full under truncation.
    pref_ids = full_ids[: len(pref_ids)]
    suf_ids = full_ids[len(pref_ids) :]
    if not suf_ids:
        suf_ids = full_ids[-1:]
        pref_ids = full_ids[:-1]
    return pref_ids, suf_ids, full_ids


def as_cuda(ids: list[int]) -> torch.Tensor:
    return torch.tensor([ids], dtype=torch.long, device="cuda")


def pad_batch(batch_ids: list[list[int]], pad_id: int) -> torch.Tensor:
    width = max(len(item) for item in batch_ids)
    rows = []
    for ids in batch_ids:
        rows.append(ids + [pad_id] * (width - len(ids)))
    # Flash stack historically left-pads for batch? flash bench groups equal length.
    return torch.tensor(rows, dtype=torch.long, device="cuda")


CACHE_NAME_HINTS = ("cache", "kv_state", "score_state", "k_cache")


def snapshot_kv(model) -> dict[str, torch.Tensor]:
    snap = {}
    for name, buf in model.named_buffers():
        low = name.lower()
        if any(hint in low for hint in CACHE_NAME_HINTS):
            snap[name] = buf.detach().clone()
    return snap


def restore_kv(model, snap: dict[str, torch.Tensor]) -> None:
    lookup = dict(model.named_buffers())
    for name, tensor in snap.items():
        if name in lookup:
            lookup[name].copy_(tensor)


@torch.inference_mode()
def last_logits(model, ids: list[int], start_pos: int = 0) -> torch.Tensor:
    _, logits, _ = model.forward(as_cuda(ids), start_pos)
    return logits.float()


@torch.inference_mode()
def continue_suffix_logits(model, pref_ids: list[int], suf_ids: list[int]) -> torch.Tensor:
    """Prefill prefix, then decode suffix one token at a time (Flash TP8 requires seqlen=1)."""
    _ = last_logits(model, pref_ids, 0)
    pos = len(pref_ids)
    logits = None
    for token in suf_ids:
        logits = last_logits(model, [token], pos)
        pos += 1
    assert logits is not None
    return logits


@torch.inference_mode()
def last_logits_many(model, batch_ids: list[list[int]]) -> torch.Tensor:
    tokens = torch.tensor(batch_ids, dtype=torch.long, device="cuda")
    _, logits, _ = model.forward(tokens, 0)
    return logits.float()


def barrier():
    if dist.is_initialized():
        dist.barrier()


@torch.inference_mode()
def quality_isolated(stack, rows, max_length, batch_size, token_ids):
    model, tokenizer, encode = stack["model"], stack["tokenizer"], stack["encode_messages"]
    suffix = topic_suffix()
    encoded = []
    for row in rows:
        _, _, full_ids = split_ids(tokenizer, encode, row["text"], suffix, max_length)
        encoded.append(full_ids)
    probs = np.zeros((len(rows), len(token_ids)), dtype=np.float64)
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
            barrier()
            times.append(elapsed)
            closed = closed_probs(logits, token_ids).cpu().numpy()
            for offset, row_index in enumerate(chunk):
                probs[row_index] = closed[offset]
    return probs, times


@torch.inference_mode()
def quality_branched(stack, rows, max_length, token_ids):
    model, tokenizer, encode = stack["model"], stack["tokenizer"], stack["encode_messages"]
    suffix = topic_suffix()
    probs = []
    times = []
    for row in rows:
        pref, suf, _ = split_ids(tokenizer, encode, row["text"], suffix, max_length)

        def fn(pref=pref, suf=suf):
            return continue_suffix_logits(model, pref, suf)

        logits, elapsed = cuda_time_ms(fn)
        barrier()
        times.append(elapsed)
        probs.append(closed_probs(logits[0], token_ids).cpu().numpy())
    return np.stack(probs, axis=0), times


@torch.inference_mode()
def latency_multi_q(stack, rows, mode, max_length, choice_ids, yn_ids, latency_calls):
    model, tokenizer, encode = stack["model"], stack["tokenizer"], stack["encode_messages"]
    probe = rows[: max(1, min(len(rows), latency_calls))]
    suffixes = [topic_suffix()] + [binary_suffix(t) for _, t in TOPICS_BINARY]
    times = []
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    for row in probe:
        if mode == "isolated":

            def fn(r=row):
                outs = []
                for i, suffix in enumerate(suffixes):
                    _, _, full_ids = split_ids(tokenizer, encode, r["text"], suffix, max_length)
                    logits = last_logits(model, full_ids, 0)
                    ids = choice_ids if i == 0 else yn_ids
                    outs.append(closed_probs(logits[0], ids))
                return outs

        elif mode == "branched":

            def fn(r=row):
                # Shared prefix from first question; fork via KV snapshot.
                pref0, _, _ = split_ids(tokenizer, encode, r["text"], suffixes[0], max_length)
                _ = last_logits(model, pref0, 0)
                base = snapshot_kv(model)
                outs = []
                for i, suffix in enumerate(suffixes):
                    pref_i, suf_i, full_i = split_ids(
                        tokenizer, encode, r["text"], suffix, max_length
                    )
                    if pref_i != pref0:
                        logits = last_logits(model, full_i, 0)
                    else:
                        restore_kv(model, base)
                        # Decode suffix token-by-token from cached prefix.
                        pos = len(pref0)
                        logits = None
                        for token in suf_i:
                            logits = last_logits(model, [token], pos)
                            pos += 1
                    ids = choice_ids if i == 0 else yn_ids
                    outs.append(closed_probs(logits[0], ids))
                return outs

        elif mode == "isolated_batch":

            def fn(r=row):
                batch = []
                for suffix in suffixes:
                    _, _, full_ids = split_ids(tokenizer, encode, r["text"], suffix, max_length)
                    batch.append(full_ids)
                # Pad to equal length for one forward.
                width = max(len(x) for x in batch)
                # Left-pad so last token is the answer position.
                padded = [[pad_id] * (width - len(x)) + x for x in batch]
                logits = last_logits_many(model, padded)
                return [
                    closed_probs(logits[0], choice_ids),
                    closed_probs(logits[1], yn_ids),
                    closed_probs(logits[2], yn_ids),
                    closed_probs(logits[3], yn_ids),
                ]

        else:
            raise ValueError(mode)

        _, elapsed = cuda_time_ms(fn)
        barrier()
        times.append(elapsed)

    return latency_summary(steady(times), examples_per_call=1)


def main() -> int:
    args = parse_args()
    stack = load_stack(args)
    rows = stratified_subset(read_ag_news(args.test_csv), args.per_class)
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    tokenizer = stack["tokenizer"]

    try:
        choice_ids = alias_token_ids(tokenizer, ALIASES, leading_space=False)
    except ValueError:
        choice_ids = alias_token_ids(tokenizer, ALIASES, leading_space=True)
    try:
        yn_ids = alias_token_ids(tokenizer, ["Y", "N"], leading_space=False)
    except ValueError:
        yn_ids = alias_token_ids(tokenizer, ["Y", "N"], leading_space=True)

    # Warmup
    warm = split_ids(tokenizer, stack["encode_messages"], rows[0]["text"], topic_suffix(), args.max_length)[2]
    last_logits(stack["model"], warm, 0)
    barrier()

    payload = {
        "model": "DeepSeek-V4.1-Flash",
        "ckpt_path": args.ckpt_path,
        "n": len(rows),
        "per_class": args.per_class,
        "max_length": args.max_length,
        "world_size": stack["world_size"],
        "labels": LABELS,
        "choice_token_ids": choice_ids,
        "yn_token_ids": yn_ids,
        "protocol": "branched_kv_vs_isolated_ag_news_flash_tp8",
        "arms": {},
    }

    if not args.skip_quality:
        log("quality isolated…")
        iso_probs, iso_times = quality_isolated(
            stack, rows, args.max_length, args.fwd_batch, choice_ids
        )
        branched_rows = rows[: min(len(rows), args.branched_quality_cap)]
        branched_labels = labels[: len(branched_rows)]
        log(f"quality branched (n={len(branched_rows)})…")
        br_probs, br_times = quality_branched(
            stack, branched_rows, args.max_length, choice_ids
        )
        # Agreement on the branched subset.
        iso_sub = iso_probs[: len(branched_rows)]
        payload["arms"]["isolated_choice"] = {
            "metrics": multiclass_metrics(iso_probs, labels),
            "latency_batch": latency_summary(
                steady(iso_times), examples_per_call=args.fwd_batch
            ),
        }
        payload["arms"]["branched_choice"] = {
            "metrics": multiclass_metrics(br_probs, branched_labels),
            "n": len(branched_rows),
            "latency_batch1": latency_summary(steady(br_times), examples_per_call=1),
            "note": "batch-1 prefill+suffix; subset if branched_quality_cap < full n",
        }
        payload["arms"]["agreement"] = {
            "n": len(branched_rows),
            "argmax_match_rate": float(
                (iso_sub.argmax(1) == br_probs.argmax(1)).mean()
            ),
            "mean_abs_prob_diff": float(np.abs(iso_sub - br_probs).mean()),
        }
        log(
            "isolated_acc",
            round(payload["arms"]["isolated_choice"]["metrics"]["accuracy"], 4),
            "branched_acc",
            round(payload["arms"]["branched_choice"]["metrics"]["accuracy"], 4),
            "agree",
            round(payload["arms"]["agreement"]["argmax_match_rate"], 4),
        )

    log("latency multi-Q…")
    latency_rows = stratified_subset(rows, min(50, args.per_class), seed=20260921)[
        : args.latency_calls
    ]
    for mode in ("isolated", "branched", "isolated_batch"):
        summary = latency_multi_q(
            stack,
            latency_rows,
            mode,
            args.max_length,
            choice_ids,
            yn_ids,
            args.latency_calls,
        )
        payload["arms"][f"latency_4q_{mode}"] = summary
        log(f"  {mode}: p50={summary['p50_ms']:.2f} ms")

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

    if is_rank0():
        write_json(args.output, payload)
        log("wrote", args.output)
    barrier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
