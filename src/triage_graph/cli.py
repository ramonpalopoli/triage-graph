"""CLI: run a ticket through the graph, pause for approval, resume.

triage-graph "Fui cobrado em dobro, R$ 320, quero reembolso" --offline
triage-graph "Fui cobrado em dobro, R$ 320, quero reembolso"            # uses Claude

Durable pause across processes (SQLite checkpointer):
triage-graph "Cobrado em dobro R$ 320, reembolso" --offline \\
    --checkpoint sqlite:///data/triage.db --thread-id t-42 --defer
triage-graph --resume t-42 --approve yes --offline --checkpoint sqlite:///data/triage.db
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid

from langgraph.types import Command

from triage_graph.checkpoint import open_checkpointer
from triage_graph.graph import build_graph
from triage_graph.llm import RuleBasedLLM


def _print_result(state: dict) -> None:
    print(f"\nOutcome: {state['outcome']}")
    print("Audit trail:")
    for ev in state["audit"]:
        print("  -", json.dumps({k: v for k, v in ev.items() if k != "ts"}, ensure_ascii=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", help="ticket text (omit with --resume)")
    ap.add_argument("--offline", action="store_true", help="rule-based LLM, no network")
    ap.add_argument("--approve", choices=["yes", "no"], help="auto-decide instead of prompting")
    ap.add_argument(
        "--checkpoint",
        help="memory | sqlite:///path.db | postgresql://... (default: $TRIAGE_CHECKPOINT_URL)",
    )
    ap.add_argument("--thread-id", help="thread id to use (default: random)")
    ap.add_argument("--defer", action="store_true", help="stop at the approval gate and exit")
    ap.add_argument("--resume", metavar="THREAD_ID", help="resume a paused thread")
    args = ap.parse_args()
    if not args.text and not args.resume:
        ap.error("ticket text is required unless --resume is given")

    if args.offline:
        llm = RuleBasedLLM()
    else:
        from triage_graph.llm import AnthropicLLM

        llm = AnthropicLLM()

    with open_checkpointer(args.checkpoint) as saver:
        graph = build_graph(llm, checkpointer=saver)
        thread_id = args.resume or args.thread_id or str(uuid.uuid4())
        cfg = {"configurable": {"thread_id": thread_id}}

        if args.resume:
            snap = graph.get_state(cfg)
            if not snap.interrupts:
                print(f"thread {thread_id!r} is not waiting for approval", file=sys.stderr)
                return 2
            payload = snap.interrupts[0].value
            state = None
        else:
            state = graph.invoke({"ticket_id": "tic_demo", "text": args.text, "audit": []}, cfg)
            payload = state["__interrupt__"][0].value if "__interrupt__" in state else None

        if payload is not None:
            print(f"\n⏸  Approval required (thread {thread_id}):")
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            if args.defer and not args.resume:
                print(f"\nDeferred. Resume with: triage-graph --resume {thread_id} ...")
                return 0
            if args.approve:
                approved = args.approve == "yes"
            else:
                approved = input("Approve? [y/N] ").strip().lower() == "y"
            state = graph.invoke(Command(resume={"approved": approved, "note": "cli"}), cfg)

        _print_result(state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
