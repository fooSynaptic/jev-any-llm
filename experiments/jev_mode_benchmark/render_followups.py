#!/usr/bin/env python3
"""Render follow-up depth sweeps: task transfer, Noul/Score, MLP, prefix."""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parent / "figures" / "followups.svg"

LAYERS = [1, 2, 4, 6, 8, 12, 16, 24, 32]
AG = [86.58, 88.03, 89.85, 90.90, 91.73, 91.60, 91.25, 91.10, 89.55]
NG = [37.70, 49.80, 65.00, 68.40, 72.25, 71.60, 69.65, 74.10, 73.22]
SST2 = [74.88, 81.50, 84.00, 86.38, 89.12, 92.75, 93.12, 91.75, 91.38]
SST5 = [35.92, 40.56, 46.32, 50.40, 51.28, 53.44, 54.64, 50.48, 47.92]
LINEAR = [86.67, 88.02, 89.85, 90.92, 91.65, 91.57, 91.40, 91.47, 89.60]
MLP = [89.18, 88.67, 89.55, 88.58, 90.15, 90.08, 90.10, 91.27, 91.83]


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def series_points(values: list[float], x0: float, y0: float, w: float, h: float, lo: float, hi: float) -> str:
    pts = []
    for index, value in enumerate(values):
        xx = x0 + w * index / (len(values) - 1)
        yy = y0 + h * (1 - (value - lo) / (hi - lo))
        pts.append(f"{xx:.1f},{yy:.1f}")
    return " ".join(pts)


lines: list[str] = []
L = lines.append
L('<?xml version="1.0" encoding="UTF-8"?>')
L('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1140 560"')
L('     font-family="Helvetica Neue,Arial,sans-serif">')
L("<defs><style>")
L(".title{font-size:20px;font-weight:700;fill:#1e293b}")
L(".subtitle{font-size:12px;fill:#64748b}")
L(".panel-title{font-size:13.5px;font-weight:700;fill:#1e293b}")
L(".axis{font-size:10.5px;fill:#64748b}")
L(".legend{font-size:11.5px;fill:#334155}")
L("</style></defs>")
L('<rect width="100%" height="100%" rx="12" fill="#f8fafc" stroke="#e2e8f0"/>')
L('<text x="40" y="36" class="title">Follow-ups: task, type, MLP, prefix</text>')
L('<text x="40" y="56" class="subtitle">Frozen Qwen3.5-4B · mean pooling · temperature-calibrated linear head unless marked</text>')

# panel 1: four tasks
P1X, P1Y, P1W, P1H = 40, 72, 540, 450
L(f'<rect x="{P1X}" y="{P1Y}" width="{P1W}" height="{P1H}" rx="10" fill="#fff" stroke="#e2e8f0"/>')
L(f'<text x="{P1X + 20}" y="{P1Y + 28}" class="panel-title">Accuracy by exit layer, four tasks</text>')
plot_x, plot_y, plot_w, plot_h = P1X + 52, P1Y + 48, P1W - 80, P1H - 180
for tick in (40, 60, 80, 100):
    yy = plot_y + plot_h * (1 - (tick - 30) / 70)
    L(f'<line x1="{plot_x}" y1="{yy:.1f}" x2="{plot_x + plot_w}" y2="{yy:.1f}" stroke="#f1f5f9"/>')
    L(f'<text x="{plot_x - 8}" y="{yy + 4:.1f}" text-anchor="end" class="axis">{tick}</text>')
for index, layer in enumerate(LAYERS):
    xx = plot_x + plot_w * index / (len(LAYERS) - 1)
    L(f'<text x="{xx:.1f}" y="{plot_y + plot_h + 18}" text-anchor="middle" class="axis">{layer}</text>')
L(f'<line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" y2="{plot_y + plot_h}" stroke="#cbd5e1"/>')
colors = [
    ("AG News Choice", AG, "#16a34a"),
    ("20 Newsgroups Choice", NG, "#2563eb"),
    ("SST-2 Noul", SST2, "#7c3aed"),
    ("SST-5 Score", SST5, "#f59e0b"),
]
for name, values, color in colors:
    pts = series_points(values, plot_x, plot_y, plot_w, plot_h, 30, 100)
    L(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.4"/>')
L(
    f'<text x="{plot_x + plot_w / 2:.1f}" y="{plot_y + plot_h + 40}" text-anchor="middle"'
    ' class="axis">Exit layer (of 32)</text>'
)
legend_y = P1Y + P1H - 76
for index, (name, _, color) in enumerate(colors):
    yy = legend_y + index * 18
    L(f'<rect x="{P1X + 24}" y="{yy - 8}" width="16" height="4" rx="2" fill="{color}"/>')
    L(f'<text x="{P1X + 46}" y="{yy - 3}" class="legend">{esc(name)}</text>')

# panel 2: mlp vs linear + prefix notes as bars at L8
P2X, P2Y, P2W, P2H = 600, 72, 500, 450
L(f'<rect x="{P2X}" y="{P2Y}" width="{P2W}" height="{P2H}" rx="10" fill="#fff" stroke="#e2e8f0"/>')
L(f'<text x="{P2X + 20}" y="{P2Y + 28}" class="panel-title">AG News: linear vs MLP vs prefix at L8</text>')
plot2_x, plot2_y, plot2_w, plot2_h = P2X + 52, P2Y + 48, P2W - 80, 160
for tick in (86, 88, 90, 92):
    yy = plot2_y + plot2_h * (1 - (tick - 86) / 7)
    L(f'<line x1="{plot2_x}" y1="{yy:.1f}" x2="{plot2_x + plot2_w}" y2="{yy:.1f}" stroke="#f1f5f9"/>')
    L(f'<text x="{plot2_x - 8}" y="{yy + 4:.1f}" text-anchor="end" class="axis">{tick}</text>')
for index, layer in enumerate(LAYERS):
    xx = plot2_x + plot2_w * index / (len(LAYERS) - 1)
    L(f'<text x="{xx:.1f}" y="{plot2_y + plot2_h + 18}" text-anchor="middle" class="axis">{layer}</text>')
L(f'<line x1="{plot2_x}" y1="{plot2_y + plot2_h}" x2="{plot2_x + plot2_w}" y2="{plot2_y + plot2_h}" stroke="#cbd5e1"/>')
L(f'<polyline points="{series_points(LINEAR, plot2_x, plot2_y, plot2_w, plot2_h, 86, 93)}" fill="none" stroke="#16a34a" stroke-width="2.4"/>')
L(f'<polyline points="{series_points(MLP, plot2_x, plot2_y, plot2_w, plot2_h, 86, 93)}" fill="none" stroke="#7c3aed" stroke-width="2.4"/>')
L(
    f'<text x="{plot2_x + plot2_w / 2:.1f}" y="{plot2_y + plot2_h + 40}" text-anchor="middle"'
    ' class="axis">Exit layer (of 32)</text>'
)
L(f'<rect x="{P2X + 24}" y="{P2Y + 266}" width="16" height="4" rx="2" fill="#16a34a"/>')
L(f'<text x="{P2X + 46}" y="{P2Y + 271}" class="legend">linear (peak L8 = 91.65%)</text>')
L(f'<rect x="{P2X + 24}" y="{P2Y + 288}" width="16" height="4" rx="2" fill="#7c3aed"/>')
L(f'<text x="{P2X + 46}" y="{P2Y + 293}" class="legend">MLP-512 (peak L32 = 91.83%)</text>')

# prefix table as text rows
L(f'<text x="{P2X + 24}" y="{P2Y + 324}" class="panel-title">Learned prefix vs linear mean, AG News</text>')
rows = [
    ("L4 linear / prefix-8", "89.85% / 90.63%"),
    ("L8 linear / prefix-4 / 8 / 16", "91.65% / 90.45% / 90.38% / 90.70%"),
    ("L16 linear / prefix-8", "91.40% / 90.00%"),
    ("L32 linear / prefix-8", "89.60% / 90.33%"),
]
for index, (left, right) in enumerate(rows):
    yy = P2Y + 350 + index * 20
    L(f'<text x="{P2X + 24}" y="{yy}" class="legend">{esc(left)}</text>')
    L(f'<text x="{P2X + P2W - 24}" y="{yy}" text-anchor="end" class="legend">{esc(right)}</text>')

L("</svg>")
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {OUT}")
