#!/usr/bin/env python3
"""Layer-8 early exit on DeepSeek-V4.1-Flash TP8, then branch or one forward.

Official continuation writes one token when start_pos > 0, so the branch arm
prefills the shared prefix once and then walks each question suffix token by
token. The one-forward arm pools the eighth layer and applies four linear
readouts. Those readouts are a latency stand-in.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import cuda_time_ms, latency_summary, read_ag_news, stratified_subset, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--inference-dir", required=True)
    parser.add_argument("--encoding-dir", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exit-layer", type=int, default=8)
    parser.add_argument("--latency-calls", type=int, default=8)
    parser.add_argument("--long-calls", type=int, default=4)
    parser.add_argument("--long-target", type=int, default=2048)
    parser.add_argument("--max-seq-len", type=int, default=3072)
    return parser.parse_args()


def is_rank0() -> bool:
    return int(os.getenv("RANK", "0")) == 0


def log(*parts) -> None:
    if is_rank0():
        print(*parts, flush=True)


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


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
        "Closed set:\nY = yes\nN = no\nAnswer (Y or N):"
    )


def suffixes() -> list[str]:
    return [topic_suffix()] + [
        binary_suffix(topic) for topic in ("Sports", "Business", "science or technology")
    ]


def full_prompt(text: str, suffix: str) -> str:
    return (
        "You are a typed predictor. Answer with exactly one alias token "
        "from the closed set.\n\n"
        f"{shared_prefix(text)}{suffix}"
    )


def shared_prefix(text: str) -> str:
    return (
        "You are a typed predictor. Answer with exactly one alias token "
        "from the closed set.\n\n"
        f"State:\nArticle:\n{text}\n\n"
    )


def load_stack(args):
    sys.path.insert(0, args.encoding_dir)
    sys.path.insert(0, args.inference_dir)
    from encoding import encode_messages
    from model import ModelArgs, Transformer, make_identity_pre_mix
    from safetensors.torch import load_model

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(20260920)
    with open(args.config) as handle:
        model_args = ModelArgs(**json.load(handle))
    model_args.temperature = 0.0
    model_args.max_batch_size = 8
    model_args.max_seq_len = args.max_seq_len
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path or args.ckpt_path, trust_remote_code=True)
    log("build model")
    with torch.device("cuda"):
        model = Transformer(model_args, tokenizer)
    log("load model")
    load_model(model, os.path.join(args.ckpt_path, f"model{rank}-mp{world_size}.safetensors"))
    torch.set_default_device("cuda")
    model.layers = model.layers[: args.exit_layer]
    return {
        "model": model,
        "tokenizer": tokenizer,
        "encode_messages": encode_messages,
        "mix": make_identity_pre_mix,
        "rank": rank,
    }


def render(encode_messages, content: str) -> str:
    return encode_messages([{"role": "user", "content": content}], thinking_mode="chat")


def to_ids(tokenizer, encode_messages, content: str) -> list[int]:
    rendered = render(encode_messages, content)
    return tokenizer(rendered, add_special_tokens=False)["input_ids"]


def split_ids(tokenizer, encode_messages, text: str, suffix: str):
    rendered = render(encode_messages, full_prompt(text, suffix))
    idx = rendered.find(suffix)
    if idx < 0:
        raise RuntimeError("cannot locate question suffix")
    pref = tokenizer(rendered[:idx], add_special_tokens=False)["input_ids"]
    suf = tokenizer(rendered[idx:], add_special_tokens=False)["input_ids"]
    return pref, suf


def stretch(tokenizer, encode_messages, text: str, target: int) -> tuple[str, int]:
    built = text
    for _ in range(48):
        count = len(to_ids(tokenizer, encode_messages, shared_prefix(built)))
        if count >= target:
            return built, count
        built = f"{built}\n{text}"
    return built, count


def as_cuda(ids: list[int]) -> torch.Tensor:
    return torch.tensor([ids], dtype=torch.long, device="cuda")


def snapshot_kv(model) -> dict[str, torch.Tensor]:
    snap = {}
    for name, buf in model.named_buffers():
        low = name.lower()
        if any(hint in low for hint in ("cache", "kv_state", "score_state", "k_cache")):
            snap[name] = buf.detach().clone()
    return snap


def restore_kv(model, snap: dict[str, torch.Tensor]) -> None:
    lookup = dict(model.named_buffers())
    for name, tensor in snap.items():
        lookup[name].copy_(tensor)


def pooled(model, mix, batch_ids: list[list[int]]) -> torch.Tensor:
    input_ids = torch.tensor(batch_ids, dtype=torch.long, device="cuda")
    hidden = model.embed(input_ids)
    hidden = hidden.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
    pre_mix = mix(hidden, model.hc_mult)
    for layer in model.layers:
        hidden, pre_mix = layer(hidden, 0, pre_mix, None)
    return hidden.mean(dim=2).float().mean(dim=1)


def time_arm(fn, warmup: int, calls: int, questions: int) -> dict:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
            barrier()
        times = []
        for _ in range(calls):
            _, elapsed = cuda_time_ms(fn)
            barrier()
            times.append(elapsed)
    summary = latency_summary(times, questions)
    summary["questions_per_second_p50"] = float(1000.0 * questions / summary["p50_ms"])
    return summary


def main() -> None:
    args = parse_args()
    stack = load_stack(args)
    model = stack["model"]
    tokenizer = stack["tokenizer"]
    encode = stack["encode_messages"]
    mix = stack["mix"]
    rows = stratified_subset(read_ag_news(args.test_csv), 8, seed=20260921)
    text = rows[0]["text"]
    long_text, long_count = stretch(tokenizer, encode, text, args.long_target)
    short_pref, _ = split_ids(tokenizer, encode, text, suffixes()[0])
    long_pref, _ = split_ids(tokenizer, encode, long_text, suffixes()[0])
    short_sufs = [split_ids(tokenizer, encode, text, suffix)[1] for suffix in suffixes()]
    long_sufs = [split_ids(tokenizer, encode, long_text, suffix)[1] for suffix in suffixes()]
    short_fulls = []
    long_fulls = []
    for suffix in suffixes():
        pref, suf = split_ids(tokenizer, encode, text, suffix)
        short_fulls.append(pref + suf)
        pref_l, suf_l = split_ids(tokenizer, encode, long_text, suffix)
        long_fulls.append(pref_l + suf_l)
    log("short prefix", len(short_pref), "suffixes", [len(item) for item in short_sufs], "long", long_count)
    if len(long_pref) + max(len(item) for item in long_sufs) > args.max_seq_len:
        raise RuntimeError("long sequence exceeds max_seq_len")

    with torch.inference_mode():
        width = int(pooled(model, mix, [short_pref]).shape[-1])
    readouts = [torch.randn(width, size, device="cuda", dtype=torch.float32) * 0.02 for size in (4, 2, 2, 2)]

    def one_forward(pref: list[int], copies: int):
        state = pooled(model, mix, [pref for _ in range(copies)])
        return [state @ readout for readout in readouts]

    def isolated(fulls: list[list[int]]):
        outs = []
        for ids in fulls:
            _, logits, _ = model.forward(as_cuda(ids), 0)
            outs.append(logits)
        return outs

    def isolated_batch(fulls: list[list[int]]):
        width_tok = max(len(item) for item in fulls)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        rows_ids = [[pad_id] * (width_tok - len(item)) + item for item in fulls]
        _, logits, _ = model.forward(torch.tensor(rows_ids, dtype=torch.long, device="cuda"), 0)
        return logits

    def branched(pref: list[int], sufs: list[list[int]]):
        _, _, _ = model.forward(as_cuda(pref), 0)
        base = snapshot_kv(model)
        last = None
        for suf in sufs:
            restore_kv(model, base)
            pos = len(pref)
            for token in suf:
                _, last, _ = model.forward(as_cuda([token]), pos)
                pos += 1
        return last

    log("probe")
    with torch.inference_mode():
        one_forward(short_pref, 1)
        isolated([short_fulls[0]])
    log("probe ok")

    arms = {
        "short_isolated": time_arm(lambda: isolated(short_fulls), 2, args.latency_calls, 4),
        "short_isolated_batch": time_arm(lambda: isolated_batch(short_fulls), 2, args.latency_calls, 4),
        "short_branched": time_arm(lambda: branched(short_pref, short_sufs), 1, args.latency_calls, 4),
        "short_one_forward": time_arm(lambda: one_forward(short_pref, 1), 2, args.latency_calls, 4),
        "short_one_forward_batch8": time_arm(lambda: one_forward(short_pref, 8), 2, args.latency_calls, 32),
        "long_isolated": time_arm(lambda: isolated(long_fulls), 1, args.long_calls, 4),
        "long_isolated_batch": time_arm(lambda: isolated_batch(long_fulls), 1, args.long_calls, 4),
        "long_branched": time_arm(lambda: branched(long_pref, long_sufs), 1, args.long_calls, 4),
        "long_one_forward": time_arm(lambda: one_forward(long_pref, 1), 1, args.long_calls, 4),
    }
    for name, summary in arms.items():
        log(name, round(summary["p50_ms"], 2), "qps", round(summary["questions_per_second_p50"], 2))
    if is_rank0():
        write_json(
            args.output,
            {
                "model": "DeepSeek-V4.1-Flash",
                "arm": "early_exit_then_branch",
                "exit_layer": args.exit_layer,
                "questions_per_article": 4,
                "branch_schedule": "prefix once, then each suffix token by token",
                "short_prefix_tokens": len(short_pref),
                "short_suffix_tokens": [len(item) for item in short_sufs],
                "long_prefix_tokens": long_count,
                "long_suffix_tokens": [len(item) for item in long_sufs],
                "calls": {"short": args.latency_calls, "long": args.long_calls},
                "readouts": "latency stand-in, four linear maps on the pooled prefix",
                "latency": arms,
            },
        )
        print("wrote", args.output, flush=True)
    barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
