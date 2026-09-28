# Typed heads on one state

## Aim

This work seeks to measure three dedicated heads on one shared state from
Qwen3.5-4B, stopped after layer 8. The state is the mean of the prompt tokens.
One head answers yes or no. One scores a set of options whose size can change
with the request. One places the text on an ordered five-point scale, with a
single direction and ordered cutpoints (cumulative logits).

The published layer-8 Choice figure reads the eighth hidden state of a full
forward: 91.73% accuracy, 12.04 ms, ECE 1.14%. This run executes only the first
eight layers, then fits the heads on that truncated state. The control for each
head is a temperature-scaled logistic regression on the same vectors.

Qwen3.8-27B, DeepSeek-V4.1-Flash, and registering these heads behind the decide
call are out of scope here.

## Setup

Frozen Qwen3.5-4B in bfloat16, one GPU, seed 20260920, prompts capped at 256
tokens. Each head is fit on its own labelled task. A scalar temperature is fit
on the calibration split and applied to the test split.

| Task | Head | Train / calibration / test per class | Test rows |
| --- | --- | --- | ---: |
| SST-2 | Yes / no | 2000 / 500 / 428 | 856 |
| AG News | Choice | 2000 / 500 / 1000 | 4000 |
| SST-5 | Score | 1188 / 20 / 278 | 1390 |

Choice encodes each option with the same eight-layer stack, then a rank-32
bilinear score between the article state and the option state. The softmax
covers only the options in that request. The two-option and three-option rows
keep the gold label and draw the remaining options at random, so a uniform
guess scores one over K.

Score uses one shared direction and four ordered cutpoints. The probability of
each level is the gap between successive cumulative sigmoids. The retained
weights are the step with the lowest calibration loss.

Latency uses CUDA events: 10 warmup calls, then 40 measured calls, batch size
1, on the first AG News test prompt (73 tokens). One arm runs the eight-layer
stack once and applies all three heads. The other arm runs that same stack
three times. Head arithmetic is included in the single-forward number.

## Results

| Head | Task | Accuracy | Calibration | Same-state logistic |
| --- | --- | ---: | --- | --- |
| Yes / no | SST-2 | 88.43% | Brier 0.168, ECE 2.27% | This row is that logistic |
| Choice, 4 options | AG News | 88.58% | ECE 4.25%, Brier 0.187 | 89.40%, ECE 1.51%, Brier 0.161 |
| Choice, 2 options | AG News | 95.03% | Chance 50% | Gold kept in the pair |
| Choice, 3 options | AG News | 91.83% | Chance 33.3% | Gold kept in the triple |
| Score, cumulative | SST-5 | 44.53% | MAE 0.715, ECE 3.94% | 44.17%, MAE 0.732, ECE 2.99% |

| Schedule | p50 | p95 |
| --- | ---: | ---: |
| One forward, three heads | 11.84 ms | 12.03 ms |
| Three separate forwards | 34.66 ms | 35.10 ms |

One forward is 2.93× the three-forward schedule. 11.84 ms sits next to the
published layer-8 forward of 12.04 ms. The extra cost of the three heads is
inside that gap.

On these truncated vectors, the variable-size Choice head is 0.8 percentage
points under the fixed four-way logistic, and both two-option and three-option
subsets stay well above chance. The cumulative scale matches the unconstrained
five-way logistic on accuracy and on mean absolute error. Its calibration error
is 3.94%, against 2.99% for that logistic.

The closed depth sweep reads layer 8 from a full forward and records SST-2 at
89.12% and SST-5 at 51.28% (MAE 0.641). Those figures stay as recorded. The
logistic column in the table is fit on this run's truncated states.

The decide call still reads next-token probabilities over a closed alias set.
These heads are trained and timed in this report. Wiring them into that call is
a later step.

JSON: [results/typed_heads_qwen35_4b.json](results/typed_heads_qwen35_4b.json).

## Names

| Name | Means |
| --- | --- |
| Yes / no | Binary logistic on the SST-2 state. The reported probability is the chance the review is positive. |
| Choice | Rank-32 score of the article against each option text. Softmax runs over the options named in the request. |
| Score | Cumulative logits on SST-5. One direction, ordered cuts, five levels from very negative to very positive. |
| Same-state logistic | Temperature-scaled logistic regression fit on the identical truncated vectors, with a fixed number of classes. |
| One forward | The eight-layer stack runs once. All three heads read that pooled state. |
