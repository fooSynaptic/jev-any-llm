# Early exit, then four questions

## Aim

This work seeks to measure what happens when a stack that already stops at
layer 8 is then asked for four judgments on one article. Two schedules are
timed. One prefills the shared prefix, replicates that cache, and scores the
four question suffixes. The other prefills the prefix once, pools it, and
applies four linear readouts.

The full-depth KV-share card and the single-question layer-8 head are already
closed. This run is the combination. The four readouts are a latency stand-in;
trained-head quality on Qwen3.5-4B is in
[REPORT_typed_heads.md](REPORT_typed_heads.md).

Registering either schedule behind the decide call is out of scope here.

## Setup

Four judgments per article: one four-way topic question and three yes/no
questions. Seed 20260920. CUDA events.

| Model | Short prefix | Long prefix | Calls (short / long) | Branch continuation |
| --- | ---: | ---: | --- | --- |
| Qwen3.5-4B | 86 tokens | 2003 tokens | 40 / 8 | One packed suffix forward |
| Qwen3.8-27B | 86 tokens | 2003 tokens | 40 / 8 | One packed suffix forward |
| DeepSeek-V4.1-Flash | 104 tokens | 2073 tokens | 8 / 4 | Each suffix token, one at a time |

On Flash the official continuation writes a single token once the prefix is
already in the cache, so the branch arm walks the suffix. The Qwen arms pack
the four suffixes into one forward. A batch of eight copies of the short prefix
is timed only for the one-forward arm.

## Results

p50, four questions, one article. Parentheses are questions per second at that
p50.

### Short prefix

| Schedule | Qwen3.5-4B | Qwen3.8-27B | Flash |
| --- | ---: | ---: | ---: |
| Four isolated forwards | 49.89 ms (80) | 47.83 ms (84) | 1589.62 ms (2.5) |
| One batch of four prompts | 16.75 ms (239) | 41.79 ms (96) | 594.08 ms (6.7) |
| Prefix, then branch | 29.98 ms (133) | 43.93 ms (91) | 8585.55 ms (0.47) |
| One forward, four readouts | **11.70 ms (342)** | **11.27 ms (355)** | **338.63 ms (12)** |
| One forward, eight copies | 19.48 ms (1643) | 51.03 ms (627) | 540.04 ms (59) |

### Long prefix, about 2000 tokens

| Schedule | Qwen3.5-4B | Qwen3.8-27B | Flash |
| --- | ---: | ---: | ---: |
| Four isolated forwards | 157.06 ms | 444.81 ms | 3213.74 ms |
| One batch of four prompts | 156.36 ms | 445.60 ms | 2291.67 ms |
| Prefix, then branch | 74.24 ms | 179.42 ms | 9069.57 ms |
| One forward, four readouts | **38.48 ms** | **109.59 ms** | **703.49 ms** |

On a short prefix the one-forward arm is the fast schedule on all three
decoders. The packed Qwen branch sits behind a single batch of four prompts,
which is the same ordering as the full-depth short-prefix card. On Flash the
token-by-token branch is slower than four separate prefills.

On a shared prefix of about 2000 tokens, one forward is again the fast arm:
38.48 ms, 109.59 ms, and 703.49 ms. Against the already published full-depth
branch at that length (274 ms, 1.13 s, 7.42 s), those three numbers are 7.1×,
10.3×, and 10.5×. The Qwen packed branch at layer 8 is the next arm (74.24 ms
and 179.42 ms). The Flash token walk stays the slow arm at 9.07 s.

Eight copies of the short prefix, still one forward and four readouts, move
the question rate to 1643, 627, and 59 questions per second.

JSON:
[results/early_branch_qwen35_4b.json](results/early_branch_qwen35_4b.json) ·
[results/early_branch_qwen38_27b.json](results/early_branch_qwen38_27b.json) ·
[results/early_branch_flash_v41.json](results/early_branch_flash_v41.json).

## Names

| Name | Means |
| --- | --- |
| One forward | The first eight layers run once on the shared prefix. Four linear maps read that pooled state. |
| Branch | The prefix is prefilled once. Each question suffix continues from that cache. Qwen packs the suffixes. Flash walks them token by token. |
| Isolated | Each of the four full prompts runs on its own. |
| Eight copies | The same short prefix is repeated eight times in one forward, so the batch scores 32 questions. |
