#!/usr/bin/env python3
"""Layer-8 early exit, then either a KV branch or one forward for four questions.

The full-depth KV-share card and the layer-8 single-head card are already
closed. This run cuts the stack after layer 8 and times four judgments on one
article:

- isolated: four full prompts, one after another
- isolated batch: those four prompts in one batch
- branched: one prefix prefill, KV replicated, one packed suffix forward
- one forward: one prefix prefill, mean-pool, four linear readouts

Short articles and a shared prefix stretched to at least 2048 tokens are both
timed. The linear readouts are a latency stand-in for trained heads. Quality
of those heads is in REPORT_typed_heads.md.
"""

from __future__ import annotations

import argparse
import copy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import cuda_time_ms, latency_summary, read_ag_news, render_instruct, stratified_subset, write_json


TOPICS_BINARY = [
    ("sports", "Sports"),
    ("business", "Business"),
    ("scitech", "science or technology"),
]


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
        f"{shared_prefix(text)}{suffix}"
    )


def shared_prefix(text: str) -> str:
    return (
        "You are a typed predictor. Answer with exactly one alias token "
        "from the closed set.\n\n"
        f"State:\nArticle:\n{text}\n\n"
    )


def split_chat_ids(tokenizer, text: str, suffix: str, max_length: int, device):
    rendered = render_instruct(tokenizer, [full_prompt(text, suffix)])[0]
    idx = rendered.find(suffix)
    if idx < 0:
        raise RuntimeError("cannot locate question suffix inside chat template")
    prefix_str = rendered[:idx]
    suffix_str = rendered[idx:]
    pref_ids = tokenizer(
        prefix_str,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )["input_ids"].to(device)
    remain = max(1, max_length - int(pref_ids.shape[-1]))
    suf_ids = tokenizer(
        suffix_str,
        add_special_tokens=False,
        truncation=True,
        max_length=remain,
        return_tensors="pt",
    )["input_ids"].to(device)
    return pref_ids, suf_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exit-layer", type=int, default=8)
    parser.add_argument("--latency-calls", type=int, default=40)
    parser.add_argument("--long-calls", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--long-target", type=int, default=2048)
    parser.add_argument("--batch-articles", type=int, default=8)
    return parser.parse_args()


def decoder_stack(model):
    candidates = [model.model, getattr(model.model, "language_model", None), model]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "layers"):
            return candidate
    raise RuntimeError("no decoder stack with .layers found")


def suffixes() -> list[str]:
    return [topic_suffix()] + [binary_suffix(topic) for _, topic in TOPICS_BINARY]


def stretch(tokenizer, text: str, target: int) -> tuple[str, int]:
    built = text
    block = text
    count = 0
    for _ in range(48):
        rendered = render_instruct(tokenizer, [shared_prefix(built)])[0]
        count = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
        if count >= target:
            return built, count
        built = f"{built}\n{block}"
    return built, count


def repeat_cache(past, repeats: int):
    cloned = copy.copy(past)
    cloned.layers = []
    for layer in past.layers:
        new_layer = copy.copy(layer)
        keys = getattr(layer, "keys", None)
        if isinstance(keys, torch.Tensor) and keys.ndim > 0 and keys.numel():
            new_layer.keys = keys.repeat_interleave(repeats, dim=0)
            new_layer.values = layer.values.repeat_interleave(repeats, dim=0)
        if getattr(layer, "is_conv_states_initialized", False) and layer.conv_states is not None:
            new_layer.conv_states = layer.conv_states.repeat_interleave(repeats, dim=0)
        if getattr(layer, "is_recurrent_states_initialized", False) and layer.recurrent_states is not None:
            new_layer.recurrent_states = layer.recurrent_states.repeat_interleave(repeats, dim=0)
        cloned.layers.append(new_layer)
    return cloned


def pack_suffixes(suffixes_ids: list[torch.Tensor]):
    lengths = [int(item.shape[-1]) for item in suffixes_ids]
    width = max(lengths)
    pad = suffixes_ids[0].new_zeros(len(suffixes_ids), width)
    mask = suffixes_ids[0].new_ones(len(suffixes_ids), width)
    for index, (item, length) in enumerate(zip(suffixes_ids, lengths)):
        pad[index, :length] = item[0]
        if length < width:
            mask[index, length:] = 0
    gather = torch.tensor([length - 1 for length in lengths], device=pad.device)
    return pad, mask, gather, lengths


def forward_logits(model, input_ids, attention_mask=None, past=None, keep_last=False):
    kwargs = {"input_ids": input_ids, "use_cache": past is None}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    if past is not None:
        kwargs["past_key_values"] = past
        kwargs["use_cache"] = False
    if keep_last:
        try:
            return model(**kwargs, logits_to_keep=1)
        except TypeError:
            return model(**kwargs)
    return model(**kwargs)


def mean_pool(hidden, attention_mask):
    weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


def prepare(tokenizer, text: str, max_length: int, device):
    pref = None
    sufs = []
    fulls = []
    for suffix in suffixes():
        pref_i, suf_i = split_chat_ids(tokenizer, text, suffix, max_length, device)
        if pref is None:
            pref = pref_i
        elif pref_i.shape != pref.shape or not torch.equal(pref_i, pref):
            raise RuntimeError("question prefixes diverged")
        sufs.append(suf_i)
        fulls.append(torch.cat([pref_i, suf_i], dim=-1))
    return {"prefix": pref, "suffixes": sufs, "fulls": fulls}


def time_arm(fn, warmup: int, calls: int, questions: int) -> dict:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        times = [cuda_time_ms(fn)[1] for _ in range(calls)]
    summary = latency_summary(times, questions)
    summary["questions_per_second_p50"] = float(1000.0 * questions / summary["p50_ms"])
    return summary


def main() -> None:
    args = parse_args()
    torch.manual_seed(20260920)
    rows = stratified_subset(read_ag_news(args.test_csv), 50, seed=20260921)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    stack = decoder_stack(model)
    stack.layers = stack.layers[: args.exit_layer]
    device = next(model.parameters()).device

    short_rows = rows[: args.latency_calls]
    long_rows = rows[: args.long_calls]
    long_texts = []
    long_counts = []
    for row in long_rows:
        text, count = stretch(tokenizer, row["text"], args.long_target)
        long_texts.append(text)
        long_counts.append(count)
    print("long prefix tokens", long_counts, flush=True)

    short_ready = [prepare(tokenizer, row["text"], 4096, device) for row in short_rows]
    long_ready = [prepare(tokenizer, text, 8192, device) for text in long_texts]
    print(
        "short prefix",
        int(short_ready[0]["prefix"].shape[-1]),
        "suffixes",
        [int(item.shape[-1]) for item in short_ready[0]["suffixes"]],
        flush=True,
    )

    def branched(sample):
        prefix = sample["prefix"]
        out = model(input_ids=prefix, use_cache=True)
        cache = repeat_cache(out.past_key_values, len(sample["suffixes"]))
        packed, mask, gather, _ = pack_suffixes(sample["suffixes"])
        logits = forward_logits(model, packed, mask, cache).logits
        return logits[torch.arange(logits.shape[0], device=device), gather]

    def isolated(sample):
        outs = []
        for full in sample["fulls"]:
            outs.append(forward_logits(model, full, keep_last=True).logits[:, -1, :])
        return outs

    def isolated_batch(samples):
        fulls = [full for sample in samples for full in sample["fulls"]]
        width = max(int(item.shape[-1]) for item in fulls)
        pad_id = tokenizer.pad_token_id
        batch = fulls[0].new_full((len(fulls), width), pad_id)
        mask = fulls[0].new_zeros(len(fulls), width)
        for index, item in enumerate(fulls):
            length = int(item.shape[-1])
            batch[index, -length:] = item[0]
            mask[index, -length:] = 1
        return forward_logits(model, batch, mask, keep_last=True).logits[:, -1, :]

    with torch.inference_mode():
        probe = model.model(input_ids=short_ready[0]["prefix"], use_cache=False).last_hidden_state
    hidden = int(probe.shape[-1])
    readouts = [
        torch.randn(hidden, width, device=device, dtype=torch.float32) * 0.02
        for width in (4, 2, 2, 2)
    ]

    def one_forward(samples):
        width = max(int(sample["prefix"].shape[-1]) for sample in samples)
        pad_id = tokenizer.pad_token_id
        batch = samples[0]["prefix"].new_full((len(samples), width), pad_id)
        mask = samples[0]["prefix"].new_zeros(len(samples), width)
        for index, sample in enumerate(samples):
            item = sample["prefix"]
            length = int(item.shape[-1])
            batch[index, -length:] = item[0]
            mask[index, -length:] = 1
        hidden_state = model.model(input_ids=batch, attention_mask=mask, use_cache=False).last_hidden_state
        pooled = mean_pool(hidden_state, mask).float()
        return [pooled @ readout for readout in readouts]

    print("probe branched", flush=True)
    with torch.inference_mode():
        branched(short_ready[0])
        one_forward(short_ready[:1])
    print("probe ok", flush=True)

    batch_n = min(args.batch_articles, len(short_ready))
    arms = {
        "short_isolated": time_arm(lambda: isolated(short_ready[0]), args.warmup, args.latency_calls, 4),
        "short_isolated_batch": time_arm(
            lambda: isolated_batch(short_ready[:1]), args.warmup, args.latency_calls, 4
        ),
        "short_branched": time_arm(lambda: branched(short_ready[0]), args.warmup, args.latency_calls, 4),
        "short_one_forward": time_arm(lambda: one_forward(short_ready[:1]), args.warmup, args.latency_calls, 4),
        "short_one_forward_batch8": time_arm(
            lambda: one_forward(short_ready[:batch_n]), args.warmup, args.latency_calls, 4 * batch_n
        ),
        "long_isolated": time_arm(lambda: isolated(long_ready[0]), 4, args.long_calls, 4),
        "long_isolated_batch": time_arm(lambda: isolated_batch(long_ready[:1]), 4, args.long_calls, 4),
        "long_branched": time_arm(lambda: branched(long_ready[0]), 4, args.long_calls, 4),
        "long_one_forward": time_arm(lambda: one_forward(long_ready[:1]), 4, args.long_calls, 4),
    }
    for name, summary in arms.items():
        print(name, round(summary["p50_ms"], 2), "qps", round(summary["questions_per_second_p50"], 2), flush=True)

    payload = {
        "model": args.model.rstrip("/").rsplit("/", 1)[-1],
        "arm": "early_exit_then_branch",
        "exit_layer": args.exit_layer,
        "questions_per_article": 4,
        "short_prefix_tokens": int(short_ready[0]["prefix"].shape[-1]),
        "short_suffix_tokens": [int(item.shape[-1]) for item in short_ready[0]["suffixes"]],
        "long_prefix_tokens": long_counts,
        "long_suffix_tokens": [int(item.shape[-1]) for item in long_ready[0]["suffixes"]],
        "batch_articles": batch_n,
        "readouts": "latency stand-in, four linear maps on the pooled prefix",
        "latency": arms,
    }
    write_json(args.output, payload)
    print("wrote", args.output, flush=True)


if __name__ == "__main__":
    main()
