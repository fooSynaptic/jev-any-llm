# Branched KV vs isolated wrap — AG News conclusions

> Hardware: Hopper (96GB HBM) · CUDA-event GPU compute · PROTOCOL locks (BF16,
> max_len 256, seed 20260920, `enable_thinking=False`).
>
> Artifacts:
> [`results/branched_qwen35_4b.json`](results/branched_qwen35_4b.json) ·
> [`results/branched_qwen38_27b.json`](results/branched_qwen38_27b.json) ·
> [`results/branched_flash_v41.json`](results/branched_flash_v41.json).
>
> Runners: [`benchmark_branched.py`](benchmark_branched.py) (dense HF) ·
> [`benchmark_branched_flash_tp8.py`](benchmark_branched_flash_tp8.py) (Flash TP8).
> Hub: [REPORT.md](REPORT.md) · [PROTOCOL.md](PROTOCOL.md) · [README.md](README.md).

## Aim

This work measures **KV-share + branch**: one shared prefix is prefilled once,
that KV is replicated across branches, and several already-written candidate
suffixes are prefilled together so each suffix position can be scored. The
locked comparison is that schedule against **N isolated full prompts** and
against **one isolated batch of N**, on Qwen3.5-4B, Qwen3.8-27B, and
DeepSeek-V4.1-Flash, at a short AG News prefix and at a shared prefix of about
2000 tokens.

## KV-share + branch

The schedule is for a set of candidate suffixes that are known before the
forward and that all continue the same prefix.

1. Prefill the shared prefix once and keep its KV.
2. Replicate that KV onto one row per candidate.
3. Prefill the candidate suffixes in one batch. Each position of each suffix
   gets a KV, and the readout scores the candidates.

A common suffix length is packed as one `[branches, suffix_len]` forward. A
longer candidate finishes in a short tail forward after that pack. On the
4-judgment articles here the suffixes are about 60–103 tokens; the packed
forward uses the shared length (60).

Streaming chat stays on autoregressive decode with continuous batching and
paged KV. Out of scope for this schedule: a suffix that is produced by
sampling during the forward.

### Where it applies

The payoff is the prefix work that the other branches no longer repeat. That
payoff shows up when the shared prefix is long and several candidates are
scored together.

| Setting | What is already written | What this schedule saves |
| --- | --- | --- |
| Complex reasoning | One long context, then several candidate traces (tree of thought, best-of-N, search) | The shared context is prefilled once; the traces are scored in one suffix batch |
| Agent calls | One long state (history, tool results, plan), then several next actions or tool sequences | The state KV is reused; each candidate call is a suffix, scored together |

On a short prefix the isolated batch of full prompts is the faster arm. On a
shared prefix of about 2000 tokens, KV-share + branch is the faster arm on all
three decoders below.

## Setup

| Lock | Value |
| --- | --- |
| Questions (latency) | 4 per article: topic A–D + 3 binary Y/N |
| Quality | AG News Choice A–D (letter aliases, zero training) |
| Branched sequence | Same chat-templated token string as isolated (prefix/suffix split) |
| Qwen3.5-4B | n=4000 (1000/class), single-GPU HF |
| Qwen3.8-27B | n=2000 (500/class), single-GPU HF |
| DeepSeek-V4.1-Flash | TP8 official stack; isolated n=2000; branched agree subset n=200 |

## Results

### Accuracy (Choice, letter aliases)

| Model | Isolated acc | Branched acc | Argmax agree | ECE (iso / br) |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-4B | **85.15%** | **85.10%** | **99.80%** | 0.066 / 0.067 |
| Qwen3.8-27B | **84.00%** | **83.95%** | **99.85%** | 0.106 / 0.106 |
| Flash (TP8) | **81.85%** (n=2000) | **88.00%** (n=200) | **88.00%** | ~0.13 / 0.11 |

**Read:** on dense Qwen, branched stays within ±0.05 pp of isolated, with
argmax agreement at 99.8%. On Flash, isolated Choice on the typed
State/Question template is 81.85% (n=2000). Branched on the first 200 rows is
88% accurate and agrees with isolated on 88% of those rows. Flash `start_pos>0`
continues the suffix one token at a time.

### Latency — current schedule (p50, CUDA events)

Four judgments per article. The branched arm is one prefix prefill, a KV
replicate, and one packed suffix forward of the shared suffix length (60),
plus a tail for any longer candidate. Parentheses are speedups versus
sequential isolated (`isolated_p50 / arm_p50`). The first two calls are
dropped when a run has more than four calls.

Short prefix: the native article. Qwen calls: 40 (4B) and 30 (27B). Flash
prefix measured 87 tokens, 16 calls.

| Model | Prefix | Isolated ×4 | Branched | Isolated batch ×4 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-4B | short article | 247 ms | 172 ms (1.44×) | **104 ms (2.37×)** |
| Qwen3.8-27B | short article | 468 ms | 374 ms (1.25×) | **378 ms (1.24×)** |
| Flash (TP8) | 87 | 7.89 s | 4.51 s (1.75×) | **2.98 s (2.65×)** |

Long prefix: the article is repeated until the shared prefix reaches at least
2048 tokens. Measured prefix: 2057 on both Qwen tokenizers, 2082 on Flash.
Suffixes: Qwen 102 / 61 / 60 / 60, Flash 103 / 61 / 60 / 60. Eight calls.

| Model | Prefix | Isolated ×4 | Branched | Isolated batch ×4 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-4B | 2057 | 783 ms | **274 ms (2.86×)** | 706 ms (1.11×) |
| Qwen3.8-27B | 2057 | 3.70 s | **1.13 s (3.28×)** | 3.69 s (1.00×) |
| Flash (TP8) | 2082 | 17.43 s | **7.42 s (2.35×)** | 12.78 s (1.36×) |

**Read:** with a short prefix, isolated batch is the fast arm (4B, Flash) or
tied with branched (27B). With a shared prefix of about 2000 tokens, branched
is the fast arm on all three decoders. On the long Qwen rows, isolated batch
lands on top of sequential isolated: the four copies of the long prefix
dominate the packed GEMM. Flash still gets a batch speedup (1.36×) at that
length, and branched is faster than that batch.

The Flash packed suffix is one forward at `start_pos = len(prefix)` with
sequence length 60. The reference TP8 decode path writes a single token when
`start_pos > 0` (`kv.squeeze(1)`). Walking the suffix one token at a time on
the same 2082-token prefix costs 35.1 s (0.50× versus sequential isolated).
The packed forward is the number in the table above.

### Earlier schedule — per-branch Python cache copy (p50 CUDA ms)

Same four judgments, native short articles, one suffix forward per question
after a Python copy of the prefix cache. Flash walks each suffix token by
token. These rows are the earlier measurement; the tables above are the
current schedule.

| Model | Isolated (seq ×4) | Branched | Isolated batch (×4) |
| --- | ---: | ---: | ---: |
| Qwen3.5-4B | 222.3 | 310.7 (0.72×) | **93.6 (2.37×)** |
| Qwen3.8-27B | 463.4 | 574.3 (0.81×) | **315.7 (1.47×)** |
| Flash (TP8) | 7314 | 40563 (0.18×) | **2850 (2.57×)** |

## Product takeaway

1. KV-share + branch is the schedule for a long shared prefix and several
   already-written candidates: complex reasoning traces, and agent calls that
   score several next actions against one state.
2. On that long-prefix measurement, branched is the fastest arm (2.35×–3.28×
   versus sequential isolated) on Qwen3.5-4B, Qwen3.8-27B, and
   DeepSeek-V4.1-Flash.
3. On a short prefix, isolated batch remains the fast arm.
4. Dense Qwen branched wrap accuracy stays within ±0.05 pp of isolated.

## Reproduce

Dense (env-driven):

```bash
export JEV_MODEL=<Qwen3.5-4B-or-Qwen3.8-27B>
export JEV_DATA=<data-root>          # expects ag_news/test.csv
export JEV_OUT=results
export JEV_TAG=branched_qwen35_4b
export JEV_PER_CLASS=1000
bash run_branched_dense.sh
```

Flash TP8:

```bash
export JEV_FLASH_CKPT=<tp8-shard-dir>
export JEV_FLASH_HF=<hf-tree-with-inference-encoding>
export JEV_DATA=<data-root>
export JEV_OUT=results
export JEV_TAG=branched_flash_v41
bash run_branched_flash_tp8.sh
```
