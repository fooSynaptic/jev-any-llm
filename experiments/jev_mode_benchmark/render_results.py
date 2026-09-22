#!/usr/bin/env python3
"""Render the measured AG News accuracy and batch-1 latency summary."""

from __future__ import annotations

import math
from pathlib import Path

OUT = Path(__file__).resolve().parent / "figures" / "results.svg"

VANILLA = "#0f172a"
WRAP = "#2563eb"
WORD = "#0891b2"


def esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


accuracy = [
    ("Vanilla AR text answer", 87.05, VANILLA),
    ("Zero-shot letter logits", 80.53, WRAP),
    ("Zero-shot word logits", 83.15, WORD),
    ("Qwen L8 mean pooling", 91.73, "#16a34a"),
]
latency = [
    ("Vanilla AR, 13 tokens", 511.05, VANILLA),
    ("Zero-shot last logits", 44.51, WRAP),
    ("Qwen L8 mean pooling", 12.04, "#16a34a"),
]

lines: list[str] = []
L = lines.append
L('<?xml version="1.0" encoding="UTF-8"?>')
L('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 460"')
L('     font-family="Helvetica Neue,Arial,sans-serif">')
L("<defs><style>")
L(".title{font-size:20px;font-weight:700;fill:#1e293b}")
L(".subtitle{font-size:12px;fill:#64748b}")
L(".panel-title{font-size:14px;font-weight:700;fill:#1e293b}")
L(".label{font-size:11.5px;fill:#475569}")
L(".value{font-size:11.5px;font-weight:700;fill:#1e293b}")
L(".axis{font-size:10.5px;fill:#64748b}")
L(".note{font-size:10.5px;fill:#64748b}")
L("</style></defs>")
L('<rect width="100%" height="100%" rx="12" fill="#f8fafc" stroke="#e2e8f0"/>')
L('<text x="40" y="38" class="title">AG News: measured quality and latency</text>')
L(
    '<text x="40" y="60" class="subtitle">Qwen3.5-4B'
    ' \u00b7 4,000 test rows \u00b7 GPU compute time after warmup</text>'
)

PANEL_Y = 82
PANEL_H = 324
LOG_MIN, LOG_MAX = 1.0, 1000.0


def log_fraction(value: float) -> float:
    span = math.log10(LOG_MAX) - math.log10(LOG_MIN)
    return (math.log10(max(value, LOG_MIN)) - math.log10(LOG_MIN)) / span


panels = [
    (40, 500, "Accuracy (%, higher is better)", accuracy, "%", False),
    (560, 500, "Batch-1 p50 latency (ms, log scale, lower is better)", latency, " ms", True),
]
for x, w, title, data, suffix, logarithmic in panels:
    L(f'<rect x="{x}" y="{PANEL_Y}" width="{w}" height="{PANEL_H}" rx="10" fill="#fff" stroke="#e2e8f0"/>')
    L(f'<text x="{x + 20}" y="{PANEL_Y + 32}" class="panel-title">{esc(title)}</text>')
    chart_x = x + 190
    chart_w = w - 228
    start_y = PANEL_Y + 70
    row_h = 46
    for index, (label, value, color) in enumerate(data):
        yy = start_y + index * row_h
        fraction = log_fraction(value) if logarithmic else value / 100.0
        L(f'<text x="{x + 20}" y="{yy + 14}" class="label">{esc(label)}</text>')
        L(f'<rect x="{chart_x}" y="{yy}" width="{chart_w}" height="20" rx="5" fill="#f1f5f9"/>')
        L(f'<rect x="{chart_x}" y="{yy}" width="{chart_w * fraction:.2f}" height="20" rx="5" fill="{color}"/>')
        L(f'<text x="{chart_x + chart_w + 10}" y="{yy + 14}" class="value">{value:.2f}{suffix}</text>')

    axis_y = PANEL_Y + PANEL_H - 56
    L(f'<line x1="{chart_x}" y1="{axis_y}" x2="{chart_x + chart_w}" y2="{axis_y}" stroke="#cbd5e1"/>')
    ticks = [1, 10, 100, 1000] if logarithmic else [0, 20, 40, 60, 80, 100]
    for tick in ticks:
        fraction = log_fraction(tick) if logarithmic else tick / 100.0
        xx = chart_x + chart_w * fraction
        L(f'<line x1="{xx:.2f}" y1="{axis_y - 4}" x2="{xx:.2f}" y2="{axis_y + 4}" stroke="#94a3b8"/>')
        L(f'<text x="{xx:.2f}" y="{axis_y + 20}" text-anchor="middle" class="axis">{tick}</text>')

L(
    f'<text x="60" y="{PANEL_Y + PANEL_H - 18}" class="note">'
    "Words = spelled-out class names; letters = A\u2013D aliases.</text>"
)
L(
    f'<text x="580" y="{PANEL_Y + PANEL_H - 18}" class="note">'
    "L8 is the compiled predictor: frozen Qwen, mean pooling, trained 4-way head.</text>"
)
L("</svg>")
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {OUT}")
