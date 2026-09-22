#!/usr/bin/env python3
"""Render README figure: accuracy by exit layer (Qwen3.5-4B AG News)."""

from __future__ import annotations

import math
from pathlib import Path

OUT = Path(__file__).resolve().parent / "accuracy_by_exit_layer.svg"

LAYERS = [1, 2, 4, 6, 8, 12, 16, 24, 32]
CAUSAL_MEAN = [86.575, 88.025, 89.85, 90.90, 91.725, 91.60, 91.25, 91.10, 89.55]
BIDI_MEAN = [86.525, 87.925, 90.25, 90.825, 91.45, 90.975, 89.875, 90.375, 88.45]
CAUSAL_LAST = [69.05, 80.25, 86.10, 86.40, 85.375, 83.90, 81.925, 84.325, 84.025]
VANILLA_ACC = 87.05

W, H = 720, 480
PLOT_X, PLOT_Y = 72, 88
PLOT_W, PLOT_H = 580, 300
ACC_MIN, ACC_MAX = 66.0, 96.0


def layer_x(layer: int) -> float:
    return PLOT_X + PLOT_W * math.log2(layer) / math.log2(32)


def acc_y(value: float) -> float:
    return PLOT_Y + PLOT_H * (1 - (value - ACC_MIN) / (ACC_MAX - ACC_MIN))


def main() -> None:
    lines: list[str] = []
    L = lines.append
    L('<?xml version="1.0" encoding="UTF-8"?>')
    L(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"')
    L('     font-family="Helvetica Neue,Arial,sans-serif">')
    L("<defs><style>")
    L(".title{font-size:18px;font-weight:700;fill:#1e293b}")
    L(".axis{font-size:11px;fill:#64748b}")
    L(".axis-title{font-size:12px;fill:#475569}")
    L(".legend{font-size:12px;fill:#334155}")
    L(".hl{font-size:12px;font-weight:700;fill:#15803d}")
    L(".base{font-size:11px;fill:#0f172a}")
    L("</style></defs>")
    L('<rect width="100%" height="100%" rx="12" fill="#f8fafc" stroke="#e2e8f0"/>')
    L('<text x="36" y="36" class="title">Accuracy by exit layer</text>')

    for tick in range(70, 97, 10):
        yy = acc_y(tick)
        L(
            f'<line x1="{PLOT_X}" y1="{yy:.1f}" x2="{PLOT_X + PLOT_W}" y2="{yy:.1f}"'
            ' stroke="#e2e8f0"/>'
        )
        L(
            f'<text x="{PLOT_X - 10}" y="{yy + 4:.1f}" text-anchor="end"'
            f' class="axis">{tick}%</text>'
        )
    for layer in LAYERS:
        xx = layer_x(layer)
        L(
            f'<text x="{xx:.1f}" y="{PLOT_Y + PLOT_H + 22}" text-anchor="middle"'
            f' class="axis">{layer}</text>'
        )
    L(
        f'<line x1="{PLOT_X}" y1="{PLOT_Y + PLOT_H}" x2="{PLOT_X + PLOT_W}"'
        f' y2="{PLOT_Y + PLOT_H}" stroke="#cbd5e1"/>'
    )
    L(
        f'<line x1="{PLOT_X}" y1="{PLOT_Y}" x2="{PLOT_X}" y2="{PLOT_Y + PLOT_H}"'
        ' stroke="#cbd5e1"/>'
    )
    L(
        f'<text x="{PLOT_X + PLOT_W / 2:.1f}" y="{PLOT_Y + PLOT_H + 44}"'
        ' text-anchor="middle" class="axis-title">Exit layer (of 32)</text>'
    )
    mid_y = PLOT_Y + PLOT_H / 2
    L(
        f'<text x="22" y="{mid_y:.1f}" text-anchor="middle" class="axis-title"'
        f' transform="rotate(-90 22 {mid_y:.1f})">Accuracy</text>'
    )

    vy = acc_y(VANILLA_ACC)
    L(
        f'<line x1="{PLOT_X}" y1="{vy:.1f}" x2="{PLOT_X + PLOT_W}" y2="{vy:.1f}"'
        ' stroke="#0f172a" stroke-width="1.5" stroke-dasharray="6 4"/>'
    )
    L(
        f'<text x="{PLOT_X + PLOT_W - 4}" y="{vy - 8:.1f}" text-anchor="end"'
        ' class="base">vanilla AR 87.05%</text>'
    )

    series = [
        ("Causal + mean pooling", CAUSAL_MEAN, "#16a34a"),
        ("Bidirectional + mean pooling", BIDI_MEAN, "#0891b2"),
        ("Causal + last token", CAUSAL_LAST, "#f97316"),
    ]
    for _name, values, color in series:
        pts = " ".join(
            f"{layer_x(layer):.1f},{acc_y(value):.1f}"
            for layer, value in zip(LAYERS, values)
        )
        L(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for layer, value in zip(LAYERS, values):
            L(
                f'<circle cx="{layer_x(layer):.1f}" cy="{acc_y(value):.1f}"'
                f' r="3.6" fill="{color}"/>'
            )

    best_x, best_y = layer_x(8), acc_y(91.725)
    L(
        f'<circle cx="{best_x:.1f}" cy="{best_y:.1f}" r="7.5" fill="none"'
        ' stroke="#15803d" stroke-width="2"/>'
    )
    L(f'<text x="{best_x + 12:.1f}" y="{best_y - 10:.1f}" class="hl">L8 = 91.73%</text>')

    lx, ly = PLOT_X + 8, PLOT_Y + PLOT_H + 72
    for index, (name, _values, color) in enumerate(series):
        x = lx + index * 210
        L(f'<rect x="{x}" y="{ly - 8}" width="22" height="4" rx="2" fill="{color}"/>')
        L(f'<text x="{x + 28}" y="{ly - 3}" class="legend">{name}</text>')

    L("</svg>")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
