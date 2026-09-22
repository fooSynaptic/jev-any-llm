#!/usr/bin/env python3
"""Render the Qwen depth-pruning sweep: accuracy by layer and the accuracy/latency frontier."""

from __future__ import annotations

import math
from pathlib import Path

OUT = Path(__file__).resolve().parent / "figures" / "depth_sweep.svg"

LAYERS = [1, 2, 4, 6, 8, 12, 16, 24, 32]
CAUSAL_MEAN = [86.575, 88.025, 89.85, 90.90, 91.725, 91.60, 91.25, 91.10, 89.55]
BIDI_MEAN = [86.525, 87.925, 90.25, 90.825, 91.45, 90.975, 89.875, 90.375, 88.45]
CAUSAL_LAST = [69.05, 80.25, 86.10, 86.40, 85.375, 83.90, 81.925, 84.325, 84.025]
LATENCY = [2.3876, 3.9416, 6.4161, 9.5503, 12.0388, 17.6068, 23.1265, 34.3481, 45.5133]
VANILLA_ACC = 87.05
VANILLA_MS = 511.05

FRONTIER = [
    ("Qwen L8 mean", 12.0388, 91.725, "#16a34a", True, 4),
    ("Qwen L4 mean", 6.4161, 89.85, "#16a34a", False, 4),
    ("Qwen L2 mean", 3.9416, 88.025, "#16a34a", False, -6),
    ("Qwen L32 mean", 45.5133, 89.55, "#16a34a", False, -12),
    ("Zero-shot last logits", 44.51, 80.53, "#2563eb", False, 16),
    ("Vanilla AR", 511.05, 87.05, "#0f172a", False, 4),
]

lines: list[str] = []
L = lines.append


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


L('<?xml version="1.0" encoding="UTF-8"?>')
L('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1140 520"')
L('     font-family="Helvetica Neue,Arial,sans-serif">')
L("<defs><style>")
L(".title{font-size:20px;font-weight:700;fill:#1e293b}")
L(".subtitle{font-size:12px;fill:#64748b}")
L(".panel-title{font-size:14px;font-weight:700;fill:#1e293b}")
L(".axis{font-size:10.5px;fill:#64748b}")
L(".axis-title{font-size:11.5px;fill:#475569}")
L(".legend{font-size:11.5px;fill:#334155}")
L(".point{font-size:10.5px;fill:#334155}")
L(".hl{font-size:11px;font-weight:700;fill:#15803d}")
L("</style></defs>")
L('<rect width="100%" height="100%" rx="12" fill="#f8fafc" stroke="#e2e8f0"/>')
L('<text x="40" y="36" class="title">Pruning Qwen depth: same weights, no generation</text>')
L(
    '<text x="40" y="57" class="subtitle">AG News \u00b7 4,000 test rows \u00b7 trained linear head on'
    " 8,000 rows \u00b7 temperature calibrated</text>"
)

# ---------------- panel 1: accuracy by depth ----------------
P1X, P1Y, P1W, P1H = 40, 74, 520, 412
L(f'<rect x="{P1X}" y="{P1Y}" width="{P1W}" height="{P1H}" rx="10" fill="#fff" stroke="#e2e8f0"/>')
L(f'<text x="{P1X + 20}" y="{P1Y + 28}" class="panel-title">Accuracy by exit layer</text>')

plot_x, plot_y = P1X + 56, P1Y + 48
plot_w, plot_h = P1W - 96, P1H - 130
acc_min, acc_max = 66.0, 94.0


def layer_x(layer: int) -> float:
    span = math.log2(32)
    return plot_x + plot_w * math.log2(layer) / span


def acc_y(value: float) -> float:
    return plot_y + plot_h * (1 - (value - acc_min) / (acc_max - acc_min))


for tick in range(70, 95, 5):
    yy = acc_y(tick)
    L(f'<line x1="{plot_x}" y1="{yy:.1f}" x2="{plot_x + plot_w}" y2="{yy:.1f}" stroke="#f1f5f9"/>')
    L(f'<text x="{plot_x - 10}" y="{yy + 4:.1f}" text-anchor="end" class="axis">{tick}%</text>')
for layer in LAYERS:
    xx = layer_x(layer)
    L(f'<text x="{xx:.1f}" y="{plot_y + plot_h + 20}" text-anchor="middle" class="axis">{layer}</text>')
L(
    f'<text x="{plot_x + plot_w / 2:.1f}" y="{plot_y + plot_h + 40}" text-anchor="middle"'
    ' class="axis-title">Exit layer (of 32)</text>'
)
L(f'<line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" y2="{plot_y + plot_h}" stroke="#cbd5e1"/>')

vanilla_y = acc_y(VANILLA_ACC)
L(
    f'<line x1="{plot_x}" y1="{vanilla_y:.1f}" x2="{plot_x + plot_w}" y2="{vanilla_y:.1f}"'
    ' stroke="#0f172a" stroke-width="1.5" stroke-dasharray="6 4"/>'
)
L(f'<text x="{plot_x + plot_w - 4}" y="{vanilla_y - 7:.1f}" text-anchor="end" class="axis">vanilla AR 87.05%</text>')

series = [
    ("causal, mean pooling", CAUSAL_MEAN, "#16a34a"),
    ("bidirectional, mean pooling", BIDI_MEAN, "#0891b2"),
    ("causal, last token", CAUSAL_LAST, "#f97316"),
]
for name, values, color in series:
    points = " ".join(f"{layer_x(l):.1f},{acc_y(v):.1f}" for l, v in zip(LAYERS, values))
    L(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.4"/>')
    for layer, value in zip(LAYERS, values):
        L(f'<circle cx="{layer_x(layer):.1f}" cy="{acc_y(value):.1f}" r="3.4" fill="{color}"/>')

best_x, best_y = layer_x(8), acc_y(91.725)
L(f'<circle cx="{best_x:.1f}" cy="{best_y:.1f}" r="7" fill="none" stroke="#15803d" stroke-width="2"/>')
L(f'<text x="{best_x + 12:.1f}" y="{best_y - 10:.1f}" class="hl">L8 = 91.73%</text>')

legend_y = P1Y + P1H - 44
for index, (name, _, color) in enumerate(series):
    yy = legend_y + index * 15
    L(f'<rect x="{P1X + 24}" y="{yy - 8}" width="18" height="4" rx="2" fill="{color}"/>')
    L(f'<text x="{P1X + 48}" y="{yy - 3}" class="legend">{esc(name)}</text>')

# ---------------- panel 2: accuracy vs latency frontier ----------------
P2X, P2Y, P2W, P2H = 580, 74, 520, 412
L(f'<rect x="{P2X}" y="{P2Y}" width="{P2W}" height="{P2H}" rx="10" fill="#fff" stroke="#e2e8f0"/>')
L(f'<text x="{P2X + 20}" y="{P2Y + 28}" class="panel-title">Accuracy vs batch-1 latency</text>')

f_x, f_y = P2X + 56, P2Y + 48
f_w, f_h = P2W - 96, P2H - 130
LOG_MIN, LOG_MAX = 2.0, 600.0


def lat_x(value: float) -> float:
    span = math.log10(LOG_MAX) - math.log10(LOG_MIN)
    return f_x + f_w * (math.log10(value) - math.log10(LOG_MIN)) / span


def f_acc_y(value: float) -> float:
    return f_y + f_h * (1 - (value - 78.0) / (94.0 - 78.0))


for tick in range(80, 95, 5):
    yy = f_acc_y(tick)
    L(f'<line x1="{f_x}" y1="{yy:.1f}" x2="{f_x + f_w}" y2="{yy:.1f}" stroke="#f1f5f9"/>')
    L(f'<text x="{f_x - 10}" y="{yy + 4:.1f}" text-anchor="end" class="axis">{tick}%</text>')
for tick in [2, 5, 10, 50, 100, 500]:
    xx = lat_x(tick)
    L(f'<line x1="{xx:.1f}" y1="{f_y + f_h}" x2="{xx:.1f}" y2="{f_y + f_h + 5}" stroke="#94a3b8"/>')
    L(f'<text x="{xx:.1f}" y="{f_y + f_h + 20}" text-anchor="middle" class="axis">{tick}</text>')
L(
    f'<text x="{f_x + f_w / 2:.1f}" y="{f_y + f_h + 40}" text-anchor="middle"'
    ' class="axis-title">batch-1 p50 latency (ms, log scale)</text>'
)
L(f'<line x1="{f_x}" y1="{f_y + f_h}" x2="{f_x + f_w}" y2="{f_y + f_h}" stroke="#cbd5e1"/>')

pruned = [(lat_x(l), f_acc_y(a)) for l, a in zip(LATENCY, CAUSAL_MEAN)]
L(
    '<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pruned) + '"'
    ' fill="none" stroke="#16a34a" stroke-width="2" stroke-opacity="0.45"/>'
)

for name, ms, acc, color, highlight, dy in FRONTIER:
    xx, yy = lat_x(ms), f_acc_y(acc)
    radius = 8 if highlight else 5.5
    L(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="{radius}" fill="{color}"/>')
    anchor = "end" if ms > 100 else "start"
    offset = -12 if ms > 100 else 12
    L(f'<text x="{xx + offset:.1f}" y="{yy + dy:.1f}" text-anchor="{anchor}" class="point">{esc(name)}</text>')

L(
    f'<text x="{P2X + 24}" y="{P2Y + P2H - 30}" class="legend">'
    "Green line: the same Qwen weights read at layers 1\u219232. Up and to the left is better.</text>"
)
L(
    f'<text x="{P2X + 24}" y="{P2Y + P2H - 14}" class="legend">'
    "Layer 8 beats vanilla by 4.7 points while running 42.5\u00d7 faster.</text>"
)

L("</svg>")
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {OUT}")
