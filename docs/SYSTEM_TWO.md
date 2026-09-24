# System Two sketch (Phase 2+)

Phase 1 stays on **System One**: closed-set next-token logits (isolated or
branched). System Two is an optional escalate path for hard questions — not
required for the wrap API.

## Intent

1. Run System One (`decide` / `decide_branched`) as usual.
2. A small router marks questions that look unreliable (low confidence, high
   entropy, out-of-domain state, …).
3. Escalated questions call a slower path (longer CoT, tool use, or a larger
   model) and overwrite only those answers.

## Minimal API (this repo)

```python
from jev_any_llm.system_two import NeverEscalate, LowConfidenceEscalate

router = LowConfidenceEscalate(floor=0.55)
# After System One:
# for qid, q in questions.items():
#     if router.should_escalate(state, qid, q, system_one=result).escalate:
#         ... slow path ...
```

`NeverEscalate` is the Phase 1 default (no behavior change).

## Non-goals for Phase 1

- No trained escalate head ships with the library.
- No automatic second-pass generation in `Client.decide`.
- Hosted Jev / this wrap remain interchangeable on the System One body.

Full design context: [DESIGN.md](../DESIGN.md) (Phase 1 System One; Phases 2–3 latency).
