#!/usr/bin/env python3
"""Render the zero-training arms: accuracy and calibration side by side."""

from pathlib import Path

OUT = Path(__file__).parent / "figures" / "zero_train.svg"

# (label, accuracy %, ECE % or None, kind)
ARMS = [
    ("Compiled layer-8 head (needs labels)", 91.73, 1.14, "trained"),
    ("Vanilla free text", 87.05, None, "reference"),
    ("Verbalizer ensemble, 16 forms", 84.25, 3.88, "zero"),
    ("Full label words, length-normalised", 84.17, 3.55, "best"),
    ("Single-token words", 83.17, 11.79, "zero"),
    ("Rationale, 8 tokens", 82.10, 11.12, "zero"),
    ("Contextual calibration", 77.80, 14.39, "bad"),
    ("Rationale, 32 tokens", 76.48, 16.32, "bad"),
    ("Descriptive label words", 59.15, 32.52, "bad"),
]

FILL = {
    "trained": "#7c3aed",
    "reference": "#334155",
    "best": "#16a34a",
    "zero": "#60a5fa",
    "bad": "#f59e0b",
}

W, H = 1140, 470
P1X, P2X = 360, 760
TOP = 92
ROW = 34
BAR1_W, BAR2_W = 330, 300
ACC_MIN, ACC_MAX = 55.0, 95.0
ECE_MAX = 34.0

lines: list[str] = []
L = lines.append


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


L('<?xml version="1.0" encoding="UTF-8"?>')
L(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"')
L('     font-family="Helvetica Neue,Arial,sans-serif">')
L("<defs><style>")
L(".title{font-size:20px;font-weight:700;fill:#1e293b}")
L(".subtitle{font-size:12px;fill:#64748b}")
L(".col{font-size:13px;font-weight:700;fill:#1e293b}")
L(".arm{font-size:11.5px;fill:#334155}")
L(".val{font-size:11px;fill:#475569}")
L(".axis{font-size:10px;fill:#94a3b8}")
L("</style></defs>")
L(f'<rect width="{W}" height="{H}" fill="#f8fafc"/>')
L('<text x="40" y="40" class="title">Zero-training arms on AG News</text>')
L(
    '<text x="40" y="62" class="subtitle">4,000 balanced rows, frozen Qwen3.5-4B. '
    "No labels except the top bar, shown for scale.</text>"
)
L(f'<rect x="40" y="{TOP - 16}" width="{W - 80}" height="{H - TOP - 20}" rx="10" '
  'fill="#fff" stroke="#e2e8f0"/>')

L(f'<text x="{P1X}" y="{TOP + 6}" class="col">Accuracy</text>')
L(f'<text x="{P2X}" y="{TOP + 6}" class="col">Calibration error (ECE)</text>')

# vanilla reference line on the accuracy panel
ref_x = P1X + BAR1_W * (87.05 - ACC_MIN) / (ACC_MAX - ACC_MIN)
bottom = TOP + 24 + ROW * len(ARMS) - 10
L(f'<line x1="{ref_x:.1f}" y1="{TOP + 14}" x2="{ref_x:.1f}" y2="{bottom}" '
  'stroke="#94a3b8" stroke-width="1" stroke-dasharray="4 3"/>')
L(f'<text x="{ref_x + 5:.1f}" y="{TOP + 22}" class="axis">free text 87.05%</text>')

for index, (label, accuracy, ece, kind) in enumerate(ARMS):
    y = TOP + 40 + index * ROW
    colour = FILL[kind]
    L(f'<text x="64" y="{y + 4}" class="arm">{esc(label)}</text>')

    width = BAR1_W * (accuracy - ACC_MIN) / (ACC_MAX - ACC_MIN)
    L(f'<rect x="{P1X}" y="{y - 9}" width="{width:.1f}" height="14" rx="3" fill="{colour}"/>')
    L(f'<text x="{P1X + width + 8:.1f}" y="{y + 3}" class="val">{accuracy:.2f}%</text>')

    if ece is None:
        L(f'<text x="{P2X}" y="{y + 3}" class="axis">no probability</text>')
        continue
    ece_w = BAR2_W * ece / ECE_MAX
    L(f'<rect x="{P2X}" y="{y - 9}" width="{ece_w:.1f}" height="14" rx="3" '
      f'fill="{colour}" opacity="0.55"/>')
    L(f'<text x="{P2X + ece_w + 8:.1f}" y="{y + 3}" class="val">{ece:.2f}%</text>')

L(f'<text x="64" y="{H - 34}" class="axis">'
  "Green is the best zero-training arm. Shorter calibration bars are better.</text>")

L("</svg>")
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {OUT}")
