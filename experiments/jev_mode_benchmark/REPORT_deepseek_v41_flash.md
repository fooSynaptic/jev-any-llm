# Jev-Mode Benchmark — DeepSeek-V4.1-Flash

Cross-model summary: [REPORT.md](REPORT.md). Primary dense scorecard:
[Qwen3.5-4B](REPORT_qwen35_4b.md). Run log: [NOTES_flash_v41.md](NOTES_flash_v41.md).

## DeepSeek-V4.1-Flash replica

Same AG News protocol (seed `20260920`, 4,000 / 8,000 / 2,000 splits, 256-token
cap) on frozen **DeepSeek-V4.1-Flash**, run through the official CED / CSA2
tensor-parallel stack (8 ranks, FP8 / FP4 experts, `thinking_mode=chat`).
Absolute milliseconds are MoE+TP8 scale. Speedups below are relative to this
model’s own vanilla arm. Sanitized artifact:
[`results/flash_v41_summary.json`](results/flash_v41_summary.json). Run log:
[`NOTES_flash_v41.md`](NOTES_flash_v41.md).

### Headline

**Flash exited at layer 8 with mean pooling scores 91.62% at 298.92 ms
(14.9× its own vanilla text baseline).** Vanilla free-text answers take
4439.71 ms and score 64.63%. The compiled early-exit path again beats both
quality and latency on the same frozen weights.

Zero-training closed-set wrap stays weak on this checkpoint: letter aliases
score 34.43% and spelled-out class words score 47.53%, with greedy tokens
inside the closed set only ~28% of the time. For Flash, the working default
is the labelled compiled head.

### Wrap quality and latency

| Arm | Accuracy | Batch-1 p50 | Notes |
| --- | ---: | ---: | --- |
| Vanilla free text | **64.63%** | **4439.71 ms** | parse rate 84.88%; mean 10.44 output tokens |
| Letter `A`–`D` closed set | 34.43% | 1915.96 ms | greedy-in-set 27.9%; ECE 27.2% |
| Word verbalizer | 47.53% | 1848.99 ms | greedy-in-set 28.2%; letter↔word argmax agreement 38.6% |
| 16-symbol codebook | 24.85% | — | near chance (quality throughput p50 4262 ms / batch 8) |
| Isolated topic (of four) | 32.80% | 7098.54 ms (4 sequential) | |
| Packed four answers | 26.05% | 2775.59 ms | flip vs isolated 0.37–0.80 |

Letter forward is within 4% of TTFT (1846.07 ms): almost all wrap cost is one
prefill. Relative to vanilla text, that is **2.32×**. Packed four-answer
generation is **2.56×** faster than four isolated prefills and still couples
answers (same rejection as on Qwen).

NLL / Brier on several wrap arms are undefined (`null` in the JSON) because
some closed-set probability rows are non-finite under FP8 readout; accuracy and
ECE above use the rows that remain.

### Depth sweep (compiled head)

Frozen backbone, linear 4-way head, temperature fit on the disjoint 2,000-row
split. Layers probed: 1, 2, 4, 6, 8, 12, 16, 20, 24, 32, 40. Bidirectional-mask
control is unsupported on CSA2 kernels and was skipped. Rows with non-finite
features are dropped before `LogisticRegression` (test nan fraction rises with
depth; ~3% at layer 8 mean, ~13% at layer 40 mean).

| Exit layer | Mean Acc | Mean ECE | Last Acc | Batch-1 p50 | Speedup vs vanilla |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 89.54% | 2.28% | 83.80% | 53.20 ms | 83.5× |
| 2 | 90.68% | 1.87% | 84.74% | 92.20 ms | 48.2× |
| 4 | 91.23% | 2.06% | 88.04% | 171.81 ms | 25.8× |
| 6 | 91.00% | 1.13% | 87.38% | 237.09 ms | 18.7× |
| **8** | **91.62%** | **1.41%** | **88.91%** | **298.92 ms** | **14.9×** |
| 12 | 91.57% | 1.23% | 87.95% | 424.16 ms | 10.5× |
| 16 | 90.82% | 1.52% | 83.93% | 584.65 ms | 7.6× |
| 20 (CED boundary) | 89.36% | 1.57% | 83.51% | 728.82 ms | 6.1× |
| 24 | 73.55% | 5.15% | 81.86% | 878.39 ms | 5.1× |
| 32 | 51.89% | 22.39% | 85.53% | 1169.48 ms | 3.8× |
| 40 | 62.51% | 24.62% | 87.79% | 1499.62 ms | 3.0× |

Three findings match the Qwen conversion, at CED depth:

1. **Mean pooling beats last-token** on every shallow exit (about +5–6 points
   through layer 12).
2. **Accuracy peaks near a quarter of depth** (layer 8 / 40) and falls as the
   stack continues into generation-oriented layers.
3. **Past the CED encoder boundary (layer 20), mean pooling collapses** while
   last-token readout stays in the mid-80s. The conversion should exit before
   that cliff, or switch pooling if a deeper exit is required.

### What this replica supports

| Claim | Flash measurement |
| --- | --- |
| Compiled early-exit + mean pooling | Layer 8: 91.62%, 1.41% ECE, 14.9× vs own vanilla |
| Exit near ~1/4 depth | Peak at 8 of 40 layers |
| Zero-training letter/word wrap as universal default | Letter 34.43%, word 47.53%; closed-set hit ~28% |
| Packed multi-answer generation | Faster than sequential isolates, lower topic accuracy, high flip |

## Recommendation (this decoder)

Use the **compiled layer-8 + mean pooling** head as the default on Flash
(91.62%, 298.92 ms, **14.9×** vs this model’s vanilla). Zero-training letter/word
wrap stays weak here (34–48%); tune a verbalizer that lands tokens in the closed
set before relying on the universal adapter tier.

Hub: [REPORT.md](REPORT.md). Artifact: [`results/flash_v41_summary.json`](results/flash_v41_summary.json).
