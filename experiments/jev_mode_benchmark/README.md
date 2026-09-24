# Jev-mode benchmark

Cross-model AG News conversion scorecards for **jev-any-llm**.

## Aim

Measure wrap-time generate+logprob versus compiled early-exit heads, then
measure **branched** shared-prefill readout, on frozen instruct
decoders at small scale.

## Entry map

| Doc | Role |
| --- | --- |
| [REPORT.md](REPORT.md) | Cross-model hub (Tier A / Tier B / branched follow-up) |
| [PROTOCOL.md](PROTOCOL.md) | Locks, stages, commands |
| [REPORT_branched.md](REPORT_branched.md) | Branched KV vs isolated (+ batch) |
| [REPORT_qwen35_4b.md](REPORT_qwen35_4b.md) | Qwen3.5-4B scorecard |
| [REPORT_qwen38_27b.md](REPORT_qwen38_27b.md) | Qwen3.8-27B scorecard |
| [REPORT_deepseek_v41_flash.md](REPORT_deepseek_v41_flash.md) | Flash TP8 scorecard |
| [NOTES_flash_v41.md](NOTES_flash_v41.md) | Flash run log |

## Runners

| Script | What it measures |
| --- | --- |
| `benchmark_branched.py` | Dense HF: isolated / branched / isolated_batch |
| `benchmark_branched_flash_tp8.py` | Flash TP8: same arms |
| `run_branched_dense.sh` | Env-driven dense launch |
| `run_branched_flash_tp8.sh` | Env-driven Flash TP8 launch |
| `benchmark_zero_shot.py` / `benchmark_packed.py` / … | Original PROTOCOL stages |

## Branched artifacts

| Path | Model |
| --- | --- |
| [`results/branched_qwen35_4b.json`](results/branched_qwen35_4b.json) | Qwen3.5-4B |
| [`results/branched_qwen38_27b.json`](results/branched_qwen38_27b.json) | Qwen3.8-27B |
| [`results/branched_flash_v41.json`](results/branched_flash_v41.json) | DeepSeek-V4.1-Flash |

Library unit tests (mock / toy tokenizer): `pytest` from the repo root
(`tests/test_branched.py`, `tests/test_contract.py`).
Tokenizer smoke for the three bench models:
`python scripts/smoke_tokenizer_aliases.py`.
