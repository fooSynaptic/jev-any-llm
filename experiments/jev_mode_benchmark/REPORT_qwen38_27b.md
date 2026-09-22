# Jev-Mode Benchmark — Qwen3.8-27B

Cross-model summary: [REPORT.md](REPORT.md). Dense 4B scorecard:
[Qwen3.5-4B](REPORT_qwen35_4b.md).

## Qwen3.8-27B replica

Same locks as the 4B run (AG News 4,000-row test, 256-token cap, BF16, one
Hopper GPU with 96GB HBM per arm, CUDA-event batch-1 p50). The decoder has
**64** layers. Wall clock for the closed protocol was about **6h50m** on a
single node (parallel quality shards after the serial wrap stages).

Artifact root: [`results/`](results/) JSONs named `formal_*.json`,
`zero_train.json`, `kv_score.json` (model path in each file is Qwen3.8-27B).

### Headline on 27B

**Layer-8 mean pooling is 87.7× faster than this model’s vanilla text answers
and 4.1 points more accurate, on the same frozen weights.** Vanilla free text
takes **1105.12 ms** at batch 1 and scores **87.0%**. Exiting after 8 of 64
layers with a trained linear head takes **12.60 ms** and scores **91.07%**
(ECE 1.81%).

Zero-training closed-set wrap is already close to free text on this checkpoint:
letter aliases **86.95%**, spelled-out class words **86.7%** (argmax agreement
97.8%). Length-normalised multi-token label words reach **88.03%**.

### Wrap quality and latency

| Arm | Accuracy | Batch-1 p50 | Notes |
| --- | ---: | ---: | --- |
| Vanilla free text | **87.0%** | **1105.12 ms** | TTFT 96.50 ms; mean ~13 generated tokens |
| Letter `A`–`D` (scheme 1 / 4) | **86.95%** | **89.26 ms** (word-forward probe) | Same closed-set hit ~99.8%; letter↔word agreement 97.8% |
| Word verbalizer (first-token style) | 86.7% | 89.26 ms | ECE 8.86% |
| Length-norm full label words | **88.03%** | 377.70 ms (4 full passes) | Best zero-train accuracy |
| Same + KV batched scoring | — | **225.38 ms** | vs naive 4× forward 363.68 ms; argmax agree 100% on n=64 |
| 16-symbol codebook | 23.98% | — | Rejected (near chance) |
| Isolated topic (of four) | 86.95% | 358.95 ms (4 sequential) | |
| Packed four answers | 85.7% | 293.56 ms | Topic flip vs isolated 5.7%; binary heads drift hard |

Packed generation is slightly faster than four sequential prefills and close to
an independent batch of four (284.88 ms), and it still couples answers — same
rejection as on 4B.

### Depth sweep (compiled linear head)

Frozen backbone, mean vs last-token pooling, temperature on the disjoint
calibration split. Layers probed: 1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64.

| Exit layer | Mean Acc | Mean ECE | Last Acc | Batch-1 p50 | Speedup vs vanilla |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | **91.07%** | 1.81% | 89.03% | **12.60 ms** | **87.7×** |
| 16 | 90.65% | 1.50% | 86.25% | 24.01 ms | 46.0× |
| 24 | 91.17% | 1.35% | 86.83% | 35.76 ms | 30.9× |
| 32 | 91.00% | 1.16% | 86.45% | 47.08 ms | 23.5× |
| 56 | **91.22%** | 1.49% | 84.20% | 81.39 ms | 13.6× |
| 64 | 91.20% | 1.04% | 85.67% | 92.98 ms | 11.9× |

Mean pooling stays near **91%** from layer 8 through the full stack. The 4B
“peak then fall” pattern is weaker here; **early exit still buys the latency**
(12.60 ms vs 93 ms at full depth) while holding quality. Last-token pooling
trails mean pooling at every listed depth.

MLP head (AG News, larger train split): peak **94.64%** at layer 24; first
layer within 1 pp of peak is **layer 8 at 94.21%**. Linear head peaks at layer
32 (**93.75%**), within 1 pp from layer 24.

Cross-task linear depth (same layer grid):

| Task | Peak layer | Peak Acc | Layer-8 Acc |
| --- | ---: | ---: | ---: |
| 20 Newsgroups (20-way) | 64 | 76.26% | 74.42% |
| SST-2 (Noul) | 40 | 94.74% | 87.03% |
| SST-5 (Score) | 32 | 56.04% | 47.55% |

Harder Choice and SST keep wanting deeper exits than AG News layer 8 — same
qualitative message as on 4B.

### Soft prefix

| Exit | Prefix len | Accuracy | Batch-1 p50 | Train per class |
| ---: | ---: | ---: | ---: | --- |
| 8 | 8 | 91.30% | 11.96 ms | 2000 |
| 16 | 4 / 8 / 16 | 91.10–91.58% | ≈22.7 ms | 2000 |
| 32 | 8 | **91.98%** | 44.57 ms | 2000 |
| 64 | 8 | 89.48% | 87.78 ms | **500** |

**Caveat — exit 64 training size.** Full-depth prefix training (2000/class,
effective batch 32) OOMs on one 96GB GPU even at micro-batch 1: gradients still
flow through all 64 layers back to the prefix tokens. Checkpointing plus
micro-batching makes a full 2000/class run feasible but too slow for the
closing wall clock, so exit 64 used **500/class** (calibration 200, test still
1000/class). Accuracy for that row is **not** apples-to-apples with the other
prefix exits. Latency for exit 64 is a full forward and is comparable.

### Zero-training and KV (27B)

| Arm | Accuracy | ECE | Batch-1 p50 |
| --- | ---: | ---: | ---: |
| Single-token class words | 86.90% | 9.27% | (same order as word forward ~89 ms) |
| Length-norm full words | **88.03%** | **5.89%** | 377.70 ms (4 passes) |
| + contextual calibration | 84.60% | 10.20% | — |
| Descriptive words + CC | 84.70% | 6.48% | — |
| Verbalizer ensemble | 88.08% | 5.88% | 1509.45 ms (16 passes) |
| Short rationale (8/16/32 tok) | 87.8–88.0% | — | 666–995 ms |
| KV batched 4 candidates | (agree 100% w/ naive) | — | **225.38 ms** |

Length normalisation helps; contextual calibration and long multi-pass
ensembles do not improve the operating point enough to justify their cost.
Shared-prefill KV batching keeps decisions identical to four full forwards and
cuts that scoring path from 364 ms to 225 ms.

### What this replica supports

| Claim | 27B measurement |
| --- | --- |
| Compiled early-exit + mean pooling | Layer 8: 91.07%, 1.81% ECE, **87.7×** vs own vanilla |
| Exit depth as latency lever | Full-depth mean ~91.2% at 93 ms; layer 8 holds 91.07% at 12.6 ms |
| Zero-training word/letter wrap | Already ~87% (near free text); length-norm words 88.03% |
| KV shared-prefill multi-candidate scoring | 100% argmax agree; 225 ms vs 364 ms naive |
| Soft prefix | Best listed exit 32 at 91.98% / 44.6 ms; exit 64 accuracy under-trained |

## Recommendation (this decoder)

1. **Zero-training wrap:** letter/word closed-set already ~87% (near free text);
   length-norm full words **88.03%**; prefer KV-batched multi-token scoring
   (225 ms vs 364 ms naive).
2. **Compiled predictor:** layer-8 mean linear head **91.07% / 12.60 ms /
   87.7×** vs this model’s vanilla. Deeper mean exits stay near 91% at much
   higher latency — early exit is still the speed lever. Soft-prefix exit 64
   accuracy used a reduced train split (see caveat above).

Hub: [REPORT.md](REPORT.md).
