"""Routing evals: does the graph take the right path and pause where it must?

Offline by default (RuleBasedLLM). Pass --anthropic to evaluate Claude's classification and
planning on the same scenarios. Non-zero exit on any failure, so it can gate CI.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from triage_graph.graph import build_graph  # noqa: E402
from triage_graph.llm import RuleBasedLLM  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anthropic", action="store_true")
    args = ap.parse_args()
    if args.anthropic:
        from triage_graph.llm import AnthropicLLM

        llm = AnthropicLLM()
    else:
        llm = RuleBasedLLM()

    scenarios = json.loads((ROOT / "evals" / "scenarios.json").read_text(encoding="utf-8"))
    graph = build_graph(llm)
    failures = 0
    for sc in scenarios:
        cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
        s = graph.invoke({"ticket_id": sc["id"], "text": sc["text"], "audit": []}, cfg)
        paused = "__interrupt__" in s
        probs = []
        if s["plan"].action != sc["expect_action"]:
            probs.append(f"action {s['plan'].action!r} != {sc['expect_action']!r}")
        if paused != sc["expect_pause"]:
            probs.append(f"pause={paused} expected {sc['expect_pause']}")
        if "expect_flags" in sc and sorted(s.get("risk_flags", [])) != sorted(sc["expect_flags"]):
            probs.append(f"flags {s.get('risk_flags')} != {sc['expect_flags']}")
        if "expect_priority" in sc and s["classification"].priority != sc["expect_priority"]:
            probs.append(f"priority {s['classification'].priority!r}")
        if "expect_category" in sc and s["classification"].category != sc["expect_category"]:
            probs.append(f"category {s['classification'].category!r}")
        status = "PASS" if not probs else "FAIL"
        failures += bool(probs)
        print(f"{status:4} {sc['id']:24} {'; '.join(probs)}")
    print(f"\n{len(scenarios) - failures}/{len(scenarios)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
