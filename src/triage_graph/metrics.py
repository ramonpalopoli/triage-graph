"""Per-node observability: latency, tokens and estimated cost, carried in the audit trail.

Every node event in `state["audit"]` gets `latency_ms`; the two LLM nodes (`classify`, `plan`)
also get an `llm` block with model, token counts and estimated USD cost. `summarize()` turns an
audit trail into a report, so the numbers survive checkpoints and process restarts with the run.

Costs are *estimates* from a static price table (USD per million tokens, Anthropic first-party
list prices). Unknown models report `cost_usd = None` instead of a wrong number. The offline
RuleBasedLLM reports zero tokens and zero cost.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# model id -> (input $/MTok, output $/MTok). Cache reads bill at 0.1x input, 5-minute cache
# writes at 1.25x input. Update when prices change; keep unknown models out rather than guess.
PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.00, 5.00),
}
CACHE_READ_MULT = 0.10
CACHE_WRITE_MULT = 1.25


@dataclass
class Usage:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float | None = 0.0

    def as_event(self) -> dict:
        return asdict(self)


def estimate_cost(u: Usage) -> float | None:
    price = PRICING_USD_PER_MTOK.get(u.model)
    if price is None:
        return None
    pin, pout = price
    cost = (
        u.input_tokens * pin
        + u.output_tokens * pout
        + u.cache_read_tokens * pin * CACHE_READ_MULT
        + u.cache_write_tokens * pin * CACHE_WRITE_MULT
    ) / 1_000_000
    return round(cost, 6)


def summarize(audit: list[dict]) -> dict:
    """Aggregate an audit trail into per-node latency and run totals."""
    nodes: dict[str, dict] = {}
    tokens_in = tokens_out = 0
    cost = 0.0
    unknown: set[str] = set()
    for ev in audit:
        node = ev.get("node", "?")
        row = nodes.setdefault(node, {"calls": 0, "latency_ms": 0.0, "tokens": 0, "cost_usd": 0.0})
        row["calls"] += 1
        row["latency_ms"] = round(row["latency_ms"] + float(ev.get("latency_ms", 0.0)), 3)
        llm = ev.get("llm")
        if llm:
            t_in, t_out = int(llm.get("input_tokens", 0)), int(llm.get("output_tokens", 0))
            tokens_in += t_in
            tokens_out += t_out
            row["tokens"] += t_in + t_out
            if llm.get("cost_usd") is None:
                unknown.add(str(llm.get("model")))
            else:
                row["cost_usd"] = round(row["cost_usd"] + llm["cost_usd"], 6)
                cost += llm["cost_usd"]
    return {
        "nodes": nodes,
        "total_latency_ms": round(sum(r["latency_ms"] for r in nodes.values()), 3),
        "input_tokens": tokens_in,
        "output_tokens": tokens_out,
        "cost_usd": None if unknown else round(cost, 6),
        "cost_unknown_models": sorted(unknown),
    }


def format_report(summary: dict) -> str:
    lines = ["Run metrics", f"  {'node':<12}{'calls':>6}{'latency ms':>12}{'tokens':>9}{'USD':>11}"]
    for node, r in summary["nodes"].items():
        lines.append(
            f"  {node:<12}{r['calls']:>6}{r['latency_ms']:>12.1f}{r['tokens']:>9}"
            f"{r['cost_usd']:>11.6f}"
        )
    cost = summary["cost_usd"]
    cost_s = (
        f"{cost:.6f}"
        if cost is not None
        else f"unknown ({', '.join(summary['cost_unknown_models'])})"
    )
    lines.append(
        f"  total: {summary['total_latency_ms']:.1f} ms, "
        f"{summary['input_tokens']} in / {summary['output_tokens']} out tokens, USD {cost_s}"
    )
    return "\n".join(lines)
