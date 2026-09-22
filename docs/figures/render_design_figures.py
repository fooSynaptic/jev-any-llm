#!/usr/bin/env python3
"""Render DESIGN figures for jev-any-llm (English labels)."""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parent


def t(s: str) -> str:
    raw = (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    return "".join(f"&#x{ord(c):X};" if ord(c) > 127 else c for c in raw)


lines: list[str] = []


def L(s: str) -> None:
    lines.append(s)


def reset() -> None:
    lines.clear()


def write(name: str) -> None:
    dest = OUT / name
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", dest)


def defs() -> None:
    L("<defs>")
    L("  <style>")
    L("    .title { font-size: 20px; font-weight: 700; fill: #1a1a2e; }")
    L("    .subtitle { font-size: 12px; fill: #64748b; }")
    L("    .badge-text { font-size: 11px; font-weight: 600; fill: #fff; }")
    L("    .node-title { font-size: 13px; font-weight: 700; fill: #1e293b; }")
    L("    .node-text { font-size: 11.5px; fill: #475569; }")
    L("    .node-text-bold { font-size: 11.5px; font-weight: 600; fill: #1e293b; }")
    L("    .mono { font-size: 11.5px; fill: #334155; }")
    L("    .inv { font-size: 12.5px; font-weight: 600; fill: #1d4ed8; }")
    L("    .legend-text { font-size: 11px; fill: #475569; }")
    L("  </style>")
    L('  <marker id="arrow-blue" viewBox="0 0 10 8" refX="9" refY="4"')
    L('          markerWidth="8" markerHeight="6" orient="auto-start-reverse">')
    L('    <path d="M0,0 L10,4 L0,8 Z" fill="#3b82f6"/>')
    L("  </marker>")
    L('  <marker id="arrow-gray" viewBox="0 0 10 8" refX="9" refY="4"')
    L('          markerWidth="8" markerHeight="6" orient="auto-start-reverse">')
    L('    <path d="M0,0 L10,4 L0,8 Z" fill="#64748b"/>')
    L("  </marker>")
    L("</defs>")


def badge(x: float, y: float, w: float, fill: str, label: str) -> None:
    L(f'<rect x="{x}" y="{y}" width="{w}" height="24" rx="12" fill="{fill}"/>')
    L(
        f'<text x="{x + w / 2}" y="{y + 16}" text-anchor="middle" class="badge-text">{t(label)}</text>'
    )


def inner_card(
    x: float,
    y: float,
    w: float,
    h: float,
    band: str,
    stroke: str,
    title_fill: str,
    title: str,
    body: list[str],
    *,
    mono: bool = False,
) -> None:
    L(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="#fff" stroke="{stroke}"/>')
    L(f'<rect x="{x}" y="{y}" width="{w}" height="26" rx="8" fill="{band}"/>')
    L(f'<rect x="{x}" y="{y + 18}" width="{w}" height="8" fill="{band}"/>')
    L(
        f'<text x="{x + w / 2}" y="{y + 19}" text-anchor="middle" class="node-title" fill="{title_fill}">{t(title)}</text>'
    )
    cls = "mono" if mono else "node-text"
    family = ' font-family="Consolas,monospace"' if mono else ""
    for i, line in enumerate(body):
        L(
            f'<text x="{x + 12}" y="{y + 48 + i * 16}" class="{cls}"{family}>{t(line)}</text>'
        )


def architecture() -> None:
    reset()
    W, H = 1100, 800
    L('<?xml version="1.0" encoding="UTF-8"?>')
    L(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"')
    L('     font-family="Helvetica Neue,Arial,sans-serif">')
    defs()
    L('<rect width="100%" height="100%" rx="12" fill="#f8fafc" stroke="#e2e8f0"/>')
    L(f'<text x="40" y="34" class="title">{t("Wrap: any instruct LLM into Jev-mode")}</text>')
    L(
        f'<text x="40" y="54" class="subtitle">{t("generate + option logprobs  ·  library post-processes Choice / Noul / Score  ·  the model predicts aliases")}</text>'
    )

    L('<rect x="40" y="70" width="1020" height="132" rx="10" fill="#fff" stroke="#93c5fd" stroke-width="1.5"/>')
    badge(56, 84, 108, "#2563eb", "REQUEST")
    L(f'<text x="176" y="101" class="node-text">{t("Client.decide(state, questions)  ·  POST /v1/systemone")}</text>')

    inner_card(
        56, 116, 232, 70, "#dbeafe", "#93c5fd", "#1d4ed8", "state",
        ["string / object / list"],
    )
    inner_card(
        304, 116, 232, 70, "#dcfce7", "#86efac", "#15803d", "Noul",
        ["Yes / No logprobs"],
    )
    inner_card(
        552, 116, 232, 70, "#fef3c7", "#fcd34d", "#b45309", "Choice",
        ["one alias per option"],
    )
    inner_card(
        800, 116, 244, 70, "#f5f3ff", "#c4b5fd", "#6d28d9", "Score",
        ["one alias per level"],
    )

    L('<line x1="550" y1="202" x2="550" y2="222" stroke="#3b82f6" stroke-width="2" marker-end="url(#arrow-blue)"/>')

    L('<rect x="40" y="228" width="1020" height="44" rx="10" fill="#eff6ff" stroke="#93c5fd"/>')
    L(
        f'<text x="550" y="256" text-anchor="middle" class="inv">{t("Isolation: one generate per question. Prompt = state + that question only.")}</text>'
    )

    L('<line x1="550" y1="272" x2="550" y2="292" stroke="#3b82f6" stroke-width="2" marker-end="url(#arrow-blue)"/>')

    L('<rect x="40" y="298" width="1020" height="220" rx="10" fill="#fff" stroke="#d97706" stroke-width="1.5"/>')
    badge(56, 312, 132, "#d97706", "WRAP PATH")
    L(f'<text x="200" y="329" class="node-text">{t("Phase 1  ·  contract on any LLM")}</text>')

    inner_card(
        56, 348, 232, 152, "#dbeafe", "#93c5fd", "#1d4ed8", "1  Any instruct LLM",
        ["Qwen / Llama / Mistral", "or OpenAI-compatible API", "Needs logprobs on generate"],
    )
    inner_card(
        304, 348, 232, 152, "#fef3c7", "#fcd34d", "#b45309", "2  Prompt aliases",
        ["A / B / C for Choice", "Yes / No for Noul", "0 .. K-1 for Score"],
    )
    inner_card(
        552, 348, 232, 152, "#dcfce7", "#86efac", "#15803d", "3  Generate + logprob",
        ["Usually one token", "Read alias logprobs", "Softmax the closed set"],
    )
    inner_card(
        800, 348, 244, 152, "#f5f3ff", "#c4b5fd", "#6d28d9", "4  library post-process",
        ["Map aliases to option ids", "Fill noul / choice / score", "Confidence from entropy"],
    )
    L('<line x1="288" y1="424" x2="300" y2="424" stroke="#3b82f6" stroke-width="2" marker-end="url(#arrow-blue)"/>')
    L('<line x1="536" y1="424" x2="548" y2="424" stroke="#3b82f6" stroke-width="2" marker-end="url(#arrow-blue)"/>')
    L('<line x1="784" y1="424" x2="796" y2="424" stroke="#3b82f6" stroke-width="2" marker-end="url(#arrow-blue)"/>')

    L('<line x1="550" y1="518" x2="550" y2="538" stroke="#3b82f6" stroke-width="2" marker-end="url(#arrow-blue)"/>')

    L('<rect x="40" y="544" width="1020" height="150" rx="10" fill="#fff" stroke="#64748b" stroke-width="1.5"/>')
    badge(56, 558, 108, "#334155", "ANSWERS")
    L(f'<text x="176" y="575" class="node-text">{t("Jev-shaped JSON assembled by the library. Confidence is logprob concentration.")}</text>')

    inner_card(
        56, 590, 320, 88, "#dcfce7", "#86efac", "#15803d", "noul",
        ["noul: 0.93"],
        mono=True,
    )
    inner_card(
        392, 590, 320, 88, "#fef3c7", "#fcd34d", "#b45309", "choice",
        ["choice: billing   confidence: 0.61", "probabilities: {billing: 0.84, ...}"],
        mono=True,
    )
    inner_card(
        728, 590, 316, 88, "#f5f3ff", "#c4b5fd", "#6d28d9", "score",
        ["score: 1.30   confidence: 0.54", "probabilities: {0: 0.00, 1: 0.70, 2: 0.30}"],
        mono=True,
    )

    L('<rect x="40" y="708" width="1020" height="68" rx="10" fill="#ecfdf5" stroke="#16a34a" stroke-width="1.5"/>')
    badge(56, 722, 148, "#15803d", "LATENCY PATH")
    L(
        f'<text x="220" y="739" class="node-text">{t("Phases 2-3, in scope. Wrap costs N generates. Native heads score every question in one forward pass.")}</text>'
    )

    L("</svg>")
    write("architecture.svg")


def characteristics() -> None:
    reset()
    W, H = 1100, 560
    L('<?xml version="1.0" encoding="UTF-8"?>')
    L(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"')
    L('     font-family="Helvetica Neue,Arial,sans-serif">')
    defs()
    L('<rect width="100%" height="100%" rx="12" fill="#f8fafc" stroke="#e2e8f0"/>')
    L(f'<text x="40" y="36" class="title">{t("Wrap locks")}</text>')
    L(
        f'<text x="40" y="56" class="subtitle">{t("The product is the wrap. Any instruct LLM in; Jev-mode answers out.")}</text>'
    )

    cards = [
        (40, 78, "#2563eb", "01", "Any LLM in",
         ["Point any instruct model at the contract.", "Local weights or an HTTP API.", "The caller keeps their model."]),
        (400, 78, "#0f766e", "02", "Generate + logprob",
         ["Usually a one-token generate.", "Read alias-token logprobs.", "Softmax over the closed set."]),
        (760, 78, "#d97706", "03", "library post-process",
         ["The library builds the Jev JSON.", "noul / choice / score fields.", "Confidence from logprob entropy."]),
        (40, 318, "#7c3aed", "04", "Isolated questions",
         ["One generate per question.", "Prompt is state + that question.", "Isolation is a measured probe."]),
        (400, 318, "#be185d", "05", "Typed primitives",
         ["Choice, Noul, or Score only.", "Answer space = caller schema.", "Aliases map back to option ids."]),
        (760, 318, "#334155", "06", "Code owns the branch",
         ["Model supplies probabilities.", "Application thresholds and routes.", "Risk tolerance lives in your code."]),
    ]

    for x, y, fill, num, title, body in cards:
        L(f'<rect x="{x}" y="{y}" width="300" height="216" rx="10" fill="#fff" stroke="#e2e8f0" stroke-width="1.5"/>')
        L(f'<rect x="{x}" y="{y}" width="300" height="44" rx="10" fill="{fill}"/>')
        L(f'<rect x="{x}" y="{y + 28}" width="300" height="16" fill="{fill}"/>')
        L(f'<text x="{x + 18}" y="{y + 28}" class="badge-text">{t(num)}</text>')
        L(f'<text x="{x + 52}" y="{y + 28}" class="badge-text">{t(title)}</text>')
        for i, line in enumerate(body):
            yy = y + 78 + i * 36
            if i:
                L(
                    f'<rect x="{x + 16}" y="{yy - 18}" width="268" height="1" fill="#f1f5f9"/>'
                )
            L(f'<text x="{x + 20}" y="{yy}" class="node-text">{t(line)}</text>')

    L("</svg>")
    write("characteristics.svg")


def field_survey() -> None:
    reset()
    W, H = 1100, 520
    L('<?xml version="1.0" encoding="UTF-8"?>')
    L(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"')
    L('     font-family="Helvetica Neue,Arial,sans-serif">')
    defs()
    L('<rect width="100%" height="100%" rx="12" fill="#f8fafc" stroke="#e2e8f0"/>')
    L(f'<text x="40" y="34" class="title">{t("Field split: generate, judge, execute")}</text>')
    L(
        f'<text x="40" y="54" class="subtitle">{t("15 public Jev builds  ·  closed options + confidence  ·  code owns the branch")}</text>'
    )

    L('<rect x="40" y="70" width="1020" height="108" rx="10" fill="#fff" stroke="#93c5fd" stroke-width="1.5"/>')
    inner_card(
        56, 86, 312, 76, "#dbeafe", "#93c5fd", "#1d4ed8", "LLM generates",
        ["Plan, write, fill parameters"],
    )
    inner_card(
        394, 86, 312, 76, "#fef3c7", "#fcd34d", "#b45309", "Jev-mode judges",
        ["Noul / Choice / Score + confidence"],
    )
    inner_card(
        732, 86, 312, 76, "#dcfce7", "#86efac", "#15803d", "Code executes",
        ["Click, skip, keep, escalate, SQL"],
    )
    L('<line x1="368" y1="124" x2="390" y2="124" stroke="#3b82f6" stroke-width="2" marker-end="url(#arrow-blue)"/>')
    L('<line x1="706" y1="124" x2="728" y2="124" stroke="#3b82f6" stroke-width="2" marker-end="url(#arrow-blue)"/>')

    clusters = [
        (40, 198, "#2563eb", "Browser", "Choice over next action", "DOM / a11y tree as state"),
        (256, 198, "#0f766e", "Harness", "Noul on skill / tool / keep", "Semantic if in the loop"),
        (472, 198, "#d97706", "Batch", "Classify then escalate", "Confidence is the router"),
        (688, 198, "#7c3aed", "Content", "Score many axes at once", "Full set, independent Qs"),
        (904, 198, "#be185d", "SQL", "Predicate as a function", "Exact filters stay in code"),
    ]
    for x, y, fill, title, line1, line2 in clusters:
        L(f'<rect x="{x}" y="{y}" width="196" height="168" rx="10" fill="#fff" stroke="#e2e8f0" stroke-width="1.5"/>')
        L(f'<rect x="{x}" y="{y}" width="196" height="40" rx="10" fill="{fill}"/>')
        L(f'<rect x="{x}" y="{y + 24}" width="196" height="16" fill="{fill}"/>')
        L(f'<text x="{x + 98}" y="{y + 26}" text-anchor="middle" class="badge-text">{t(title)}</text>')
        L(f'<text x="{x + 12}" y="{y + 78}" class="node-text-bold">{t(line1)}</text>')
        L(f'<rect x="{x + 12}" y="{y + 96}" width="172" height="1" fill="#f1f5f9"/>')
        L(f'<text x="{x + 12}" y="{y + 122}" class="node-text">{t(line2)}</text>')

    L('<rect x="40" y="388" width="1020" height="100" rx="10" fill="#ecfdf5" stroke="#16a34a" stroke-width="1.5"/>')
    badge(56, 404, 132, "#15803d", "KEV FIT")
    L(
        f'<text x="204" y="421" class="node-text">{t("Wrap: any instruct LLM speaks this split. Latency: many independent questions want one forward pass.")}</text>'
    )
    L(
        f'<text x="56" y="456" class="node-text">{t("Isolation lock matches 61 scores on one draft. Confidence lock matches cascade: judge first, upgrade the low-confidence tail.")}</text>'
    )

    L("</svg>")
    write("field-survey.svg")


if __name__ == "__main__":
    architecture()
    characteristics()
    field_survey()
