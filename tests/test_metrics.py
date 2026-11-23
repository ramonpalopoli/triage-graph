"""Observability: per-node latency, tokens and estimated cost in the audit trail (offline)."""

from __future__ import annotations

import time
import uuid
from types import SimpleNamespace

import pytest
from langgraph.types import Command

from triage_graph.graph import build_graph
from triage_graph.llm import AnthropicLLM, RuleBasedLLM
from triage_graph.metrics import Usage, estimate_cost, format_report, summarize
from triage_graph.state import Classification

REFUND = "Fui cobrado em dobro, R$ 320,00, quero reembolso"


class MeteredFake(RuleBasedLLM):
    """RuleBasedLLM decisions, but reporting usage the way a real model adapter would."""

    def __init__(self, model: str) -> None:
        self.model = model

    def complete_with_usage(self, schema, prompt):
        u = Usage(model=self.model, input_tokens=1000, output_tokens=200)
        u.cost_usd = estimate_cost(u)
        return self.complete(schema, prompt), u


def _run(llm, text=REFUND, wait_s: float = 0.0):
    graph = build_graph(llm)
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    s = graph.invoke({"ticket_id": "m1", "text": text, "audit": []}, cfg)
    if "__interrupt__" in s:
        time.sleep(wait_s)
        s = graph.invoke(Command(resume={"approved": True, "note": "ok"}), cfg)
    return s


def test_every_node_event_has_latency_and_rule_based_costs_zero():
    s = _run(RuleBasedLLM())
    assert all(isinstance(e["latency_ms"], float) and e["latency_ms"] >= 0 for e in s["audit"])
    llm_events = [e for e in s["audit"] if "llm" in e]
    assert [e["node"] for e in llm_events] == ["classify", "plan"]
    assert all(e["llm"]["input_tokens"] == 0 and e["llm"]["cost_usd"] == 0 for e in llm_events)
    m = summarize(s["audit"])
    assert m["cost_usd"] == 0.0 and m["input_tokens"] == 0
    assert set(m["nodes"]) == {"classify", "retrieve", "plan", "risk_check", "approve", "execute"}


def test_tokens_and_cost_are_aggregated_for_a_metered_model():
    m = summarize(_run(MeteredFake("claude-sonnet-4-5"))["audit"])
    # 2 LLM calls x (1000 in x $3 + 200 out x $15) / 1M = 2 x 0.006
    assert m["input_tokens"] == 2000 and m["output_tokens"] == 400
    assert m["cost_usd"] == pytest.approx(0.012)
    assert m["nodes"]["plan"]["tokens"] == 1200
    assert "USD 0.012000" in format_report(m)


def test_unknown_model_reports_unknown_cost_instead_of_guessing():
    m = summarize(_run(MeteredFake("some-new-model"))["audit"])
    assert m["cost_usd"] is None
    assert m["cost_unknown_models"] == ["some-new-model"]
    assert "unknown (some-new-model)" in format_report(m)


def test_approval_latency_excludes_human_wait_time():
    s = _run(RuleBasedLLM(), wait_s=0.3)
    approve = next(e for e in s["audit"] if e["node"] == "approve")
    assert approve["latency_ms"] < 200


def test_cache_tokens_are_priced():
    u = Usage("claude-opus-5-5", input_tokens=0, cache_read_tokens=1_000_000)
    assert estimate_cost(u) == pytest.approx(0.40)  # 0.1 x $4


def test_anthropic_adapter_reads_usage_from_response_without_network():
    resp = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="tool_use",
                input={"category": "billing", "priority": "medium", "rationale": "duplicada"},
            )
        ],
        usage=SimpleNamespace(
            input_tokens=500,
            output_tokens=100,
            cache_read_input_tokens=None,
            cache_creation_input_tokens=0,
        ),
    )
    calls = []
    fake_client = SimpleNamespace(
        messages=SimpleNamespace(create=lambda **kw: calls.append(kw) or resp)
    )
    llm = object.__new__(AnthropicLLM)  # skip __init__: no SDK import, no API key, no network
    llm._client, llm.model = fake_client, "claude-sonnet-4-5"

    c, usage = llm.complete_with_usage(Classification, "Ticket: cobrado em dobro")
    assert c.category == "billing"
    assert (usage.input_tokens, usage.output_tokens) == (500, 100)
    assert usage.cost_usd == pytest.approx((500 * 3 + 100 * 15) / 1e6)
    assert calls[0]["tool_choice"]["name"] == "emit_classification"
