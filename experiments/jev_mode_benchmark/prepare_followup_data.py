#!/usr/bin/env python3
"""Build SST-2, SST-5, and 20 Newsgroups CSVs for the follow-up sweeps."""

from __future__ import annotations

import argparse
import csv
import re
import zipfile
from collections import Counter
from pathlib import Path
from tarfile import TarFile


NEWSGROUPS = [
    "alt.atheism",
    "comp.graphics",
    "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware",
    "comp.windows.x",
    "misc.forsale",
    "rec.autos",
    "rec.motorcycles",
    "rec.sport.baseball",
    "rec.sport.hockey",
    "sci.crypt",
    "sci.electronics",
    "sci.med",
    "sci.space",
    "soc.religion.christian",
    "talk.politics.guns",
    "talk.politics.mideast",
    "talk.politics.misc",
    "talk.religion.misc",
]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "text"])
        writer.writeheader()
        writer.writerows(rows)


def summarize(name: str, rows: list[dict]) -> None:
    counts = Counter(row["label"] for row in rows)
    print(name, "n=", len(rows), "classes=", dict(sorted(counts.items())))


def read_tsv(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            text = (row.get("sentence") or row.get("text") or "").strip()
            if not text:
                continue
            rows.append({"label": int(row["label"]), "text": text})
    return rows


def build_sst2(sst2_zip: Path, out_dir: Path) -> None:
    with zipfile.ZipFile(sst2_zip) as archive:
        archive.extractall(out_dir / "_raw")
    root = out_dir / "_raw" / "SST-2"
    train = read_tsv(root / "train.tsv")
    dev = read_tsv(root / "dev.tsv")
    write_csv(out_dir / "sst2" / "train.csv", train)
    write_csv(out_dir / "sst2" / "test.csv", dev)
    summarize("sst2-train", train)
    summarize("sst2-test", dev)


def build_sst5(sstb_zip: Path, out_dir: Path) -> None:
    with zipfile.ZipFile(sstb_zip) as archive:
        archive.extractall(out_dir / "_raw")
    root = out_dir / "_raw" / "stanfordSentimentTreebank"
    sentences = {}
    with (root / "datasetSentences.txt").open(encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            idx, text = line.rstrip("\n").split("\t", 1)
            sentences[int(idx)] = text.replace("-LRB-", "(").replace("-RRB-", ")")
    splits = {}
    with (root / "datasetSplit.txt").open(encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            idx, split = line.strip().split(",")
            splits[int(idx)] = int(split)
    dictionary = {}
    with (root / "dictionary.txt").open(encoding="utf-8") as handle:
        for line in handle:
            phrase, phrase_id = line.rstrip("\n").rsplit("|", 1)
            dictionary[phrase] = int(phrase_id)
    scores = {}
    with (root / "sentiment_labels.txt").open(encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            phrase_id, score = line.strip().split("|")
            scores[int(phrase_id)] = float(score)

    buckets = {1: [], 2: [], 3: []}
    missing = 0
    for idx, text in sentences.items():
        phrase_id = dictionary.get(text)
        if phrase_id is None:
            missing += 1
            continue
        score = scores[phrase_id]
        if score <= 0.2:
            label = 0
        elif score <= 0.4:
            label = 1
        elif score <= 0.6:
            label = 2
        elif score <= 0.8:
            label = 3
        else:
            label = 4
        buckets[splits[idx]].append({"label": label, "text": text})
    write_csv(out_dir / "sst5" / "train.csv", buckets[1] + buckets[3])
    write_csv(out_dir / "sst5" / "test.csv", buckets[2])
    summarize("sst5-train", buckets[1] + buckets[3])
    summarize("sst5-test", buckets[2])
    print("sst5-unmatched-sentences", missing)


def clean_newsgroup(raw: str) -> str:
    _, _, body = raw.partition("\n\n")
    text = body or raw
    text = re.sub(r"^>.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:4000]


def build_20ng(tarball: Path, out_dir: Path) -> None:
    label_of = {name: index for index, name in enumerate(NEWSGROUPS)}
    buckets = {"20news-bydate-train": [], "20news-bydate-test": []}
    with TarFile.open(tarball, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            parts = Path(member.name).parts
            if len(parts) < 3:
                continue
            split, group = parts[0], parts[1]
            if split not in buckets or group not in label_of:
                continue
            payload = archive.extractfile(member).read()
            text = clean_newsgroup(payload.decode("latin-1"))
            if len(text) < 20:
                continue
            buckets[split].append({"label": label_of[group], "text": text})
    write_csv(out_dir / "20newsgroups" / "train.csv", buckets["20news-bydate-train"])
    write_csv(out_dir / "20newsgroups" / "test.csv", buckets["20news-bydate-test"])
    summarize("20ng-train", buckets["20news-bydate-train"])
    summarize("20ng-test", buckets["20news-bydate-test"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst2-zip", required=True)
    parser.add_argument("--sstb-zip", required=True)
    parser.add_argument("--ng-tar", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out)
    build_sst2(Path(args.sst2_zip), out_dir)
    build_sst5(Path(args.sstb_zip), out_dir)
    build_20ng(Path(args.ng_tar), out_dir)


if __name__ == "__main__":
    main()
