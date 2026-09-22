"""Fold the latency-only runs back into the quality-only runs.

Splitting a stage into a parallel quality pass and a serial latency pass keeps
the timings clean, but leaves two half-populated JSON files per stage. This
rebuilds the single-file layout the renderers and the 4B run already use.
"""

import argparse
import json
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def merge_prefix(quality: dict, latency: dict) -> dict:
    timings = {
        (cfg["exit_layer"], cfg["prefix_length"]): cfg["latency"]
        for cfg in latency["configs"]
    }
    for cfg in quality["configs"]:
        key = (cfg["exit_layer"], cfg["prefix_length"])
        if key not in timings:
            raise KeyError(f"no latency measured for exit {key[0]} prefix {key[1]}")
        cfg["latency"] = timings[key]
    return quality


def merge_zero_train(quality: dict, latency: dict) -> dict:
    for name, entry in latency.get("D_rationale", {}).items():
        if "latency_batch_1" in entry:
            quality["D_rationale"][name]["latency_batch_1"] = entry["latency_batch_1"]
    quality["latency"] = latency["latency"]
    return quality


def merge_kv(quality: dict, latency: dict) -> dict:
    quality["latency"] = latency["latency"]
    return quality


STAGES = (
    ("formal_prefix", merge_prefix),
    ("zero_train", merge_zero_train),
    ("kv_score", merge_kv),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    root = Path(args.results)

    for stem, merge in STAGES:
        quality = load(root / f"{stem}_quality.json")
        latency = load(root / f"{stem}_latency.json")
        if quality is None or latency is None:
            print(f"skip {stem}: quality={quality is not None} latency={latency is not None}")
            continue
        merged = merge(quality, latency)
        out = root / f"{stem}.json"
        out.write_text(json.dumps(merged, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
