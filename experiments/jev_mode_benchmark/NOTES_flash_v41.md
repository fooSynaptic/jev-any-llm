# DeepSeek-V4.1-Flash replica

## Aim

Repeat the AG News Jev-mode protocol (4,000 balanced test rows, wrap
readout, packed isolation, depth-pruned linear head) on frozen
DeepSeek-V4.1-Flash with the official 8-way tensor-parallel inference
stack, on Hopper GPUs with 96GB HBM.

The Qwen3.5-4B numbers in [REPORT_qwen35_4b.md](REPORT_qwen35_4b.md) remain
the primary dense scorecard. Cross-model hub: [REPORT.md](REPORT.md). This
note records the Flash replica: first abort, then the completed run.

## Status

| Stage | Outcome |
| --- | --- |
| 4-row smoke | Completed (`smoke.json`) |
| Convert HF → TP8 shards | Completed (~501 GB extra copy on the shared FS) |
| Full 4,000-row quality + depth | **Completed 2026-09-21** — [`results/flash_v41_summary.json`](results/flash_v41_summary.json) |

Narrative scorecard: [REPORT_deepseek_v41_flash.md](REPORT_deepseek_v41_flash.md).
Hub: [REPORT.md](REPORT.md).

## First attempt (aborted)

Aborted 2026-09-20 21:18 with no `full.json`. Logged letter closed-set
accuracy **0.34** before the crash. Depth extract finished
(8,000 / 2,000 / 4,000), then `LogisticRegression.fit` raised
`ValueError: Input X contains NaN`.

Concurrent shared-FS fill (TP8 convert copy) was a delivery risk; the
exception that killed workers was the NaN features. See the previous
abort write-up in git history if needed.

## Second attempt (shipped)

Runner changes before re-run:

1. Filter non-finite rows per depth / pooling before `LogisticRegression`.
2. Dump `.quality.json` / `.latency.json` / `.depth.json` incrementally.
3. Shared FS headroom restored before convert.

Headline from the closed run (same AG News splits and seed as Qwen):

| Arm | Accuracy | Batch-1 p50 |
| --- | ---: | ---: |
| Vanilla free text | 64.63% | 4439.7 ms |
| Letter closed-set | 34.43% | 1916.0 ms |
| Word closed-set | 47.53% | 1849.0 ms |
| **Layer 8 + mean pooling** | **91.62%** | **298.9 ms** |

Exit layer **8 / 40** is again the accuracy peak for mean pooling.
Absolute milliseconds are TP8 MoE scale; speedups below are relative to
this model’s own vanilla arm (**14.9×** at layer 8).
