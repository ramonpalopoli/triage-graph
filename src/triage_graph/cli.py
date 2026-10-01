"""CLI: run a ticket through the graph, pause for approval, resume.

triage-graph "Fui cobrado em dobro, R$ 320, quero reembolso" --offline
triage-graph "Fui cobrado em dobro, R$ 320, quero reembolso"            # uses Claude
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid

from langgraph.types import Command

from triage_graph.graph import build_graph
from triage_graph.llm import RuleBasedLLM


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("text", help="ticket text")
    ap.add_argument("--offline", action="store_true", help="rule-based LLM, no network")
    ap.add_argument("--approve", choices=["yes", "no"], help="auto-decide instead of prompting")
    args = ap.parse_args()

    if args.offline:
        llm = RuleBasedLLM()
    else:
        from triage_graph.llm import AnthropicLLM

        llm = AnthropicLLM()

    graph = build_graph(llm)
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    state = graph.invoke({"ticket_id": "tic_demo", "text": args.text, "audit": []}, cfg)

    if "__interrupt__" in state:
        payload = state["__interrupt__"][0].value
        print("\n⏸  Approval required:")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if args.approve:
            approved = args.approve == "yes"
        else:
            approved = input("Approve? [y/N] ").strip().lower() == "y"
        state = graph.invoke(Command(resume={"approved": approved, "note": "cli"}), cfg)

    print(f"\nOutcome: {state['outcome']}")
    print("Audit trail:")
    for ev in state["audit"]:
        print("  -", json.dumps({k: v for k, v in ev.items() if k != "ts"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
