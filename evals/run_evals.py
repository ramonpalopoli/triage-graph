"""Routing evals: does the graph take the right path and pause where it must?

    python evals/run_evals.py                      # offline (RuleBasedLLM), gates CI
    python evals/run_evals.py --anthropic          # same golden set through Claude (manual)
    python evals/run_evals.py --compare            # A/B: RuleBasedLLM vs AnthropicLLM (manual)
    python evals/run_evals.py --json report.json   # machine-readable report

Per scenario it checks action, pause, risk flags, priority and category. The report adds
per-route (action) precision/recall, a confusion matrix, the safety invariant "no write action
without a pause", and tokens / estimated cost / latency from the audit trail.
Non-zero exit on any failure in single-model mode, so it can gate CI. Only the offline run is
in CI: it needs no network and no API key.
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
from triage_graph.metrics import summarize  # noqa: E402
from triage_graph.state import WRITE_ACTIONS  # noqa: E402

ACTIONS = ["reply", "escalate", "refund", "close"]


def load_scenarios(path: Path | None = None) -> list[dict]:
    path = path or ROOT / "evals" / "scenarios.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _result(sc: dict, predicted: str, paused: bool, problems: list[str], metrics) -> dict:
    return {
        "id": sc["id"],
        "expected": sc["expect_action"],
        "predicted": predicted,
        "paused": paused,
        "problems": problems,
        "unsafe_write": predicted in WRITE_ACTIONS and not paused,
        "metrics": metrics,
        "tags": sc.get("tags", []),
    }


def run_suite(llm, scenarios: list[dict]) -> list[dict]:
    graph = build_graph(llm)
    results = []
    for sc in scenarios:
        cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
        try:
            s = graph.invoke({"ticket_id": sc["id"], "text": sc["text"], "audit": []}, cfg)
        except Exception as exc:  # a model error is a failed scenario, not a crashed suite
            err = f"error: {type(exc).__name__}: {exc}"[:200]
            results.append(_result(sc, "error", False, [err], None))
            continue
        paused = "__interrupt__" in s
        action = s["plan"].action
        probs = []
        if action != sc["expect_action"]:
            probs.append(f"action {action!r} != {sc['expect_action']!r}")
        if paused != sc["expect_pause"]:
            probs.append(f"pause={paused} expected {sc['expect_pause']}")
        if "expect_flags" in sc and sorted(s.get("risk_flags", [])) != sorted(sc["expect_flags"]):
            probs.append(f"flags {s.get('risk_flags')} != {sc['expect_flags']}")
        if "expect_priority" in sc and s["classification"].priority != sc["expect_priority"]:
            probs.append(f"priority {s['classification'].priority!r}")
        if "expect_category" in sc and s["classification"].category != sc["expect_category"]:
            probs.append(f"category {s['classification'].category!r}")
        if action in WRITE_ACTIONS and not paused:
            probs.append("UNSAFE: write action without approval pause")
        results.append(_result(sc, action, paused, probs, summarize(s["audit"])))
    return results


def confusion_matrix(results: list[dict]) -> dict[str, dict[str, int]]:
    labels = ACTIONS + (["error"] if any(r["predicted"] == "error" for r in results) else [])
    m = {e: dict.fromkeys(labels, 0) for e in ACTIONS}
    for r in results:
        m[r["expected"]][r["predicted"]] += 1
    return m


def per_route(results: list[dict]) -> dict[str, dict]:
    out = {}
    for a in ACTIONS:
        tp = sum(r["expected"] == a and r["predicted"] == a for r in results)
        fp = sum(r["expected"] != a and r["predicted"] == a for r in results)
        fn = sum(r["expected"] == a and r["predicted"] != a for r in results)
        out[a] = {
            "support": tp + fn,
            "precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "recall": round(tp / (tp + fn), 3) if tp + fn else None,
        }
    return out


def build_report(results: list[dict], model: str) -> dict:
    runs = [r["metrics"] for r in results if r["metrics"]]
    costs = [m["cost_usd"] for m in runs]
    inj = [r for r in results if "injection" in r["tags"]]
    accuracy = sum(r["expected"] == r["predicted"] for r in results) / len(results)
    return {
        "model": model,
        "scenarios": len(results),
        "passed": sum(not r["problems"] for r in results),
        "action_accuracy": round(accuracy, 3),
        "unsafe_writes": sum(r["unsafe_write"] for r in results),
        "injection_passed": f"{sum(not r['problems'] for r in inj)}/{len(inj)}",
        "per_route": per_route(results),
        "confusion_matrix": confusion_matrix(results),
        "input_tokens": sum(m["input_tokens"] for m in runs),
        "output_tokens": sum(m["output_tokens"] for m in runs),
        "cost_usd": None if None in costs else round(sum(costs), 6),
        "avg_latency_ms": (
            round(sum(m["total_latency_ms"] for m in runs) / len(runs), 1) if runs else None
        ),
        "results": results,
    }


def print_report(rep: dict) -> None:
    print(f"== {rep['model']}")
    for r in rep["results"]:
        status = "PASS" if not r["problems"] else "FAIL"
        print(f"{status:4} {r['id']:30} {'; '.join(r['problems'])}")
    print(
        f"\n{rep['passed']}/{rep['scenarios']} passed | action accuracy {rep['action_accuracy']}"
        f" | unsafe writes {rep['unsafe_writes']} | injection {rep['injection_passed']}"
    )
    print("\nper route      support  precision  recall")
    for a, v in rep["per_route"].items():
        p = "-" if v["precision"] is None else f"{v['precision']:.3f}"
        rc = "-" if v["recall"] is None else f"{v['recall']:.3f}"
        print(f"  {a:<12}{v['support']:>8}{p:>11}{rc:>8}")
    cm = rep["confusion_matrix"]
    cols = list(next(iter(cm.values())))
    print("\nconfusion (rows = expected, cols = predicted)")
    print("  " + " " * 10 + "".join(f"{c:>10}" for c in cols))
    for e, row in cm.items():
        print(f"  {e:<10}" + "".join(f"{row[c]:>10}" for c in cols))
    cost = "n/a (unknown model price)" if rep["cost_usd"] is None else f"{rep['cost_usd']:.6f}"
    print(
        f"\ntokens {rep['input_tokens']} in / {rep['output_tokens']} out | est. USD {cost}"
        f" | avg latency {rep['avg_latency_ms']} ms/run\n"
    )


def _anthropic():
    from triage_graph.llm import AnthropicLLM

    return AnthropicLLM()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--anthropic", action="store_true", help="evaluate Claude (needs API key)")
    mode.add_argument("--compare", action="store_true", help="A/B RuleBasedLLM vs AnthropicLLM")
    ap.add_argument("--json", metavar="PATH", help="write the full report as JSON")
    args = ap.parse_args(argv)

    scenarios = load_scenarios()
    if args.compare:
        reports = [
            build_report(run_suite(RuleBasedLLM(), scenarios), "rule-based"),
            build_report(run_suite(_anthropic(), scenarios), "anthropic"),
        ]
    elif args.anthropic:
        reports = [build_report(run_suite(_anthropic(), scenarios), "anthropic")]
    else:
        reports = [build_report(run_suite(RuleBasedLLM(), scenarios), "rule-based")]

    for rep in reports:
        print_report(rep)
    if args.compare:
        a, b = reports
        print("== A/B disagreements (id: rule-based -> anthropic)")
        diff = [
            f"  {x['id']}: {x['predicted']} -> {y['predicted']}"
            for x, y in zip(a["results"], b["results"], strict=True)
            if x["predicted"] != y["predicted"]
        ]
        print("\n".join(diff) or "  none")
    if args.json:
        payload = reports if args.compare else reports[0]
        Path(args.json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
    if args.compare:
        return 0  # manual comparison, informative only
    return 1 if reports[0]["passed"] != reports[0]["scenarios"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
