# Jev-Mode Inference Benchmark

## Aim

This work seeks to measure how much latency **jev-any-llm** can remove when an
instruct LLM is used as a typed predictor (`Choice` / `Noul` / `Score`), and
which conversion keeps AG News classification quality on frozen weights.

The locked comparison is wrap-time generate+logprob versus a depth-pruned
single forward pass with a trained typed head. The same protocol runs on three
decoders; each has its own scorecard.

| Decoder | Report |
| --- | --- |
| **Qwen3.5-4B** (primary dense) | [REPORT_qwen35_4b.md](REPORT_qwen35_4b.md) |
| **Qwen3.8-27B** | [REPORT_qwen38_27b.md](REPORT_qwen38_27b.md) |
| **DeepSeek-V4.1-Flash** (TP8 MoE) | [REPORT_deepseek_v41_flash.md](REPORT_deepseek_v41_flash.md) |

Protocol locks, stages, and commands: [PROTOCOL.md](PROTOCOL.md).

## Merged conclusion: how fast jev-any-llm can go while keeping quality

**Operating definition of “keep performance” on AG News:** match or beat that
model’s own vanilla free-text accuracy, or stay within ~1 pp of the best
zero-training closed-set arm when labels are unavailable. Speedups are always
**relative to that model’s own vanilla text baseline** (absolute ms are not
comparable across TP8 MoE vs single-GPU dense).

### Tier A — Compiled early-exit (labels required)

Freeze the backbone, mean-pool the prompt, train a linear Choice head, exit
early. This is the path that **raises accuracy and cuts latency** on every
decoder measured.

| Decoder | Vanilla Acc / batch-1 | jev-any-llm layer-8 mean Acc / batch-1 | Speedup | Δ Acc |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-4B | 87.05% / 511.05 ms | **91.73%** / **12.04 ms** | **42.5×** | **+4.7 pp** |
| Qwen3.8-27B | 87.0% / 1105.12 ms | **91.07%** / **12.60 ms** | **87.7×** | **+4.1 pp** |
| DeepSeek-V4.1-Flash | 64.63% / 4439.71 ms | **91.62%** / **298.92 ms** | **14.9×** | **+27.0 pp** |

Shared design:

- **Mean pooling** over last-token pooling.
- **Exit near ~1/4 of depth** on AG News Choice (layer 8 of 32 / 64 / 40).
- Causal mask kept; bidirectional prefill buys ≤0.4 pp on Qwen and is unused on Flash CSA2.

Task note: the knee **moves**. On Qwen3.5-4B, 20-way Choice peaks at layer 24;
SST-2 / SST-5 peak at layer 16. Production jev-any-llm should probe exit depth per
typed head, then ship the shallowest layer within 1 pp of that peak.

### Tier B — Zero-training wrap (no labels)

Closed-set readout over option tokens, isolated prompts, optional
length-normalised multi-token verbalizer + shared-prefill KV scoring.

| Decoder | Best zero-train Acc | Batch-1 (best arm) | vs vanilla Acc | vs vanilla speed |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-4B | **84.17%** (length-norm words + KV) | 110.03 ms | −2.88 pp | **4.65×** |
| Qwen3.5-4B (cheap) | 83.17% (single-token words) | 44.51 ms | −3.88 pp | **11.5×** |
| Qwen3.8-27B | **88.03%** (length-norm words) | 225–378 ms (KV / 4-pass) | **+1.0 pp** vs text | ~3–5× |
| Qwen3.8-27B (cheap) | ~86.9% (letter/word) | ~89 ms | ≈ match text | **~12×** |
| DeepSeek-V4.1-Flash | 47.53% (word) / 34.43% (letter) | ~1.8–1.9 s | far below text | ~2.3× |

**Read:** on Qwen dense checkpoints, wrap alone is already a large latency win;
on 27B it also matches free-text accuracy. On Flash, wrap quality is weak until
the verbalizer lands tokens in the closed set — compiled Tier A is the reliable
default there.

Rejected on all three (quality and/or contract): packed multi-answer generation
behind one prefill; arbitrary multi-bit codebooks zero-shot; contextual
calibration / long rationales as accuracy fixes on Qwen.

### What “jev-any-llm can accelerate to” means in practice

| Goal | Recommended jev-any-llm path | Typical outcome on AG News |
| --- | --- | --- |
| Max speed **and** beat vanilla accuracy | Tier A, layer-8 mean head | **15×–88×** vs that model’s vanilla; **+4–27 pp** |
| No labels, stay near vanilla accuracy | Tier B word / length-norm verbalizer | Dense Qwen: **~5–12×**, within a few pp (27B matches/beats text) |
| No labels, cheapest ms | Tier B single-token closed set | Dense Qwen: **~11–12×**, a few pp under text |

Absolute batch-1 times for the compiled layer-8 arm: **~12 ms** on single-GPU
Qwen (4B and 27B) and **~299 ms** on Flash TP8 — different stacks, same
relative recipe.

## Per-model headlines

1. **[Qwen3.5-4B](REPORT_qwen35_4b.md)** — Primary dense scorecard, six-route
   verdicts, follow-ups (task knee, MLP, prefix), zero-train sweep. Layer 8 =
   42.5× / 91.73%.
2. **[Qwen3.8-27B](REPORT_qwen38_27b.md)** — Same protocol at 64 layers. Layer 8
   = 87.7× / 91.07%; zero-train already ~87–88%; KV batched scoring 225 ms.
3. **[DeepSeek-V4.1-Flash](REPORT_deepseek_v41_flash.md)** — TP8 MoE replica.
   Layer 8 = 14.9× / 91.62%; wrap letter/word weak (34–48%).

## Shared-prefill follow-up

After the primary scorecards, the same three decoders were measured on
**KV-share + branch**: one shared prefix prefilled once, KV replicated across
branches, then one batched prefill of the already-written candidate suffixes.
Full tables: [REPORT_branched.md](REPORT_branched.md).

| Model | Isolated Acc | Branched Acc | Agree | Long-prefix latency winner |
| --- | ---: | ---: | ---: | --- |
| Qwen3.5-4B | 85.15% | 85.10% | 99.8% | **branched** (2.86× vs seq, prefix 2057) |
| Qwen3.8-27B | 84.00% | 83.95% | 99.9% | **branched** (3.28× vs seq, prefix 2057) |
| Flash (TP8) | 81.85% | 88%† | 88%† | **branched** (2.35× vs seq, prefix 2082) |

† Flash branched quality is a 200-row subset.

**Read:** on dense Qwen, branched matches isolated accuracy. With a shared
prefix of about 2000 tokens, branched is the fastest of the three arms on all
three decoders. On a short AG News prefix, isolated batch is the fast arm.

## Recommendation

Ship **two production tiers**:

1. **Universal adapter (zero training):** spelled-out label words, isolated
   batching; length-normalised multi-token + shared-prefill KV when ECE matters.
   Tune the verbalizer per checkpoint. Skip contextual calibration, large
   ensembles, and generated rationales on the Qwen measurements. On Flash, treat
   this tier as experimental until closed-set hit rate is high.
2. **Compiled predictor (labels):** freeze the LLM, mean-pool, linear typed
   head, **probe exit depth**. Default AG News Choice cut is **layer 8** on all
   three decoders measured — that is the point that jointly maximises (or nearly
   maximises) accuracy and multiplies throughput against each model’s own
   vanilla text path.

## Artifacts

| Path | Role |
| --- | --- |
| [PROTOCOL.md](PROTOCOL.md) | Locks, stages, commands |
| [REPORT_qwen35_4b.md](REPORT_qwen35_4b.md) | Full 4B narrative |
| [REPORT_qwen38_27b.md](REPORT_qwen38_27b.md) | Full 27B narrative |
| [REPORT_deepseek_v41_flash.md](REPORT_deepseek_v41_flash.md) | Full Flash narrative |
| [REPORT_branched.md](REPORT_branched.md) | Branched KV vs isolated follow-up |
| [NOTES_flash_v41.md](NOTES_flash_v41.md) | Flash run log |
| [`results/`](results/) | JSON measurements (`summary.json`, `formal_*.json`, `branched_*.json`, …) |
| [`figures/`](figures/) | Depth / results / zero-train / protocol SVGs |
