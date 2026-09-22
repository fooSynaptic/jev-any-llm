#!/usr/bin/env python3
"""Render the measurement-protocol flowchart (English labels)."""

from pathlib import Path

OUT = Path(__file__).parent / "figures" / "protocol.svg"


def L(lines: list[str], text: str) -> None:
    lines.append(text)


def card(
    lines: list[str],
    x: float,
    y: float,
    w: float,
    h: float,
    badge: str,
    title: str,
    body: str,
    fill: str,
    stroke: str,
    badge_fill: str,
) -> None:
    L(lines, f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    L(lines, f'<rect x="{x}" y="{y}" width="{w}" height="26" rx="10" fill="{badge_fill}"/>')
    L(lines, f'<rect x="{x}" y="{y + 14}" width="{w}" height="12" fill="{badge_fill}"/>')
    L(lines, f'<text x="{x + w / 2:.1f}" y="{y + 18}" text-anchor="middle" class="badge-text">{badge}</text>')
    L(lines, f'<text x="{x + 14}" y="{y + 46}" class="node-title">{title}</text>')
    for index, line in enumerate(body.split("\n")):
        L(lines, f'<text x="{x + 14}" y="{y + 66 + index * 16}" class="node-text">{line}</text>')


def arrow(lines: list[str], x1: float, y1: float, x2: float, y2: float) -> None:
    L(
        lines,
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#64748b" '
        'stroke-width="1.6" marker-end="url(#arrow-gray)"/>',
    )


def main() -> None:
    lines: list[str] = []
    L(lines, '<?xml version="1.0" encoding="UTF-8"?>')
    L(lines, '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1140 760"')
    L(lines, '     font-family="Helvetica Neue,Arial,sans-serif">')
    L(lines, "<defs><style>")
    L(lines, ".title{font-size:20px;font-weight:700;fill:#1e293b}")
    L(lines, ".subtitle{font-size:12px;fill:#64748b}")
    L(lines, ".badge-text{font-size:11px;font-weight:600;fill:#fff}")
    L(lines, ".node-title{font-size:13.5px;font-weight:700;fill:#1e293b}")
    L(lines, ".node-text{font-size:11.5px;fill:#475569}")
    L(lines, "</style>")
    L(lines, '<marker id="arrow-gray" viewBox="0 0 10 8" refX="9" refY="4"')
    L(lines, '        markerWidth="8" markerHeight="6" orient="auto-start-reverse">')
    L(lines, '<path d="M0,0 L10,4 L0,8 Z" fill="#64748b"/>')
    L(lines, "</marker></defs>")
    L(lines, '<rect width="100%" height="100%" rx="12" fill="#f8fafc" stroke="#e2e8f0"/>')
    L(lines, '<text x="40" y="40" class="title">Jev-mode conversion protocol</text>')
    L(lines, '<text x="40" y="62" class="subtitle">Frozen Qwen3.5-4B · AG News locks · wrap schemes then compiled early-exit</text>')

    card(
        lines, 40, 88, 250, 118, "LOCKS", "What stays fixed",
        "Qwen3.5-4B BF16, thinking off\n4k / 8k / 2k stratified splits\nCUDA-event GPU time, 15-bin ECE",
        "#fff", "#94a3b8", "#334155",
    )
    card(
        lines, 320, 88, 250, 118, "DATA", "Classification packs",
        "AG News Choice-4 (main)\n20 Newsgroups Choice-20\nSST-2 Noul, SST-5 Score",
        "#fff", "#93c5fd", "#2563eb",
    )
    card(
        lines, 600, 88, 250, 118, "BASELINE", "Vanilla autoregressive",
        "Free-text Category + Confidence\n511.05 ms, 87.05%, 13.24 tok\nTTFT 49.01 ms",
        "#fff", "#c4b5fd", "#7c3aed",
    )
    card(
        lines, 880, 88, 220, 118, "READOUT", "Closed-set math",
        "Softmax over option tokens\nChoice / Noul / Score fields\nIsolation: one prompt / question",
        "#fff", "#86efac", "#16a34a",
    )

    arrow(lines, 165, 206, 165, 236)
    arrow(lines, 445, 206, 445, 236)
    arrow(lines, 725, 206, 725, 236)
    arrow(lines, 990, 206, 990, 236)

    card(
        lines, 40, 246, 520, 150, "STAGE A", "Zero-training wrap (schemes 1-5)",
        "1 generate+logprob  80.40% / 49.31 ms\n2 packed questions  rejected (60.33%, coupled)\n3 multi-bit codebook  rejected (22.95%)\n4 last-token logits  80.53% / 44.51 ms\n5 bidirectional mask  &lt;0.4 pp vs causal",
        "#fff", "#93c5fd", "#2563eb",
    )
    card(
        lines, 600, 246, 500, 150, "STAGE B", "Compiled early-exit (scheme 6)",
        "Freeze backbone, mean-pool prompt\nLinear Choice head on 8k labels\nTemperature on 2k held-out rows\nSweep layers 1-32; AG News knee = L8\n91.73% / 12.04 ms / ECE 1.14%",
        "#fff", "#86efac", "#16a34a",
    )

    arrow(lines, 300, 396, 300, 426)
    arrow(lines, 850, 396, 850, 426)

    card(
        lines, 40, 436, 520, 150, "STAGE C", "Zero-train gap closing",
        "Length-normalised full label words\n84.17% / ECE 3.55%\nShared-prefill KV batch 110.03 ms\nSkip calibration, ensemble, rationale\nVerbalizer wording is a lock",
        "#fff", "#93c5fd", "#2563eb",
    )
    card(
        lines, 600, 436, 500, 150, "STAGE D", "Task / type / head follow-ups",
        "20NG peak L24, SST Noul/Score L16\nMean pooling beats last token\nMLP-512 moves the peak to L32\nSoft prefix loses at the L8 knee\nPick exit layer per task on a probe",
        "#fff", "#86efac", "#16a34a",
    )

    arrow(lines, 300, 586, 560, 626)
    arrow(lines, 850, 586, 580, 626)

    L(lines, '<rect x="280" y="626" width="580" height="100" rx="10" fill="#fff" stroke="#334155" stroke-width="1.5"/>')
    L(lines, '<rect x="280" y="626" width="580" height="26" rx="10" fill="#334155"/>')
    L(lines, '<rect x="280" y="640" width="580" height="12" fill="#334155"/>')
    L(lines, '<text x="570" y="644" text-anchor="middle" class="badge-text">SHIP</text>')
    L(lines, '<text x="300" y="672" class="node-title">Two production tiers from this protocol</text>')
    L(lines, '<text x="300" y="692" class="node-text">Wrap: spelled-out labels, isolated batching, optional KV multi-token score.</text>')
    L(lines, '<text x="300" y="708" class="node-text">Compiled: mean-pool, linear head, per-task exit layer on a labelled probe.</text>')

    L(lines, "</svg>")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
