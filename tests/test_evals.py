"""The eval harness itself: metrics are computed correctly and failures fail the gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from triage_graph.llm import RuleBasedLLM
from triage_graph.state import Plan

_spec = importlib.util.spec_from_file_location(
    "run_evals", Path(__file__).resolve().parents[1] / "evals" / "run_evals.py"
)
run_evals = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_evals)


class AlwaysReply(RuleBasedLLM):
    """A 'model' that never proposes a write: misses every refund/escalate/close."""

    def complete(self, schema, prompt):
        if schema is Plan:
            return Plan(action="reply", draft_reply="Segue a orientação da política.")
        return super().complete(schema, prompt)


def test_golden_set_size_and_coverage():
    scenarios = run_evals.load_scenarios()
    assert len(scenarios) >= 20
    assert {s["expect_action"] for s in scenarios} == set(run_evals.ACTIONS)
    assert sum("injection" in s.get("tags", []) for s in scenarios) >= 4
    assert sum("high_risk" in s.get("tags", []) for s in scenarios) >= 3


def test_offline_suite_passes_and_is_safe():
    rep = run_evals.build_report(
        run_evals.run_suite(RuleBasedLLM(), run_evals.load_scenarios()), "rule"
    )
    assert rep["passed"] == rep["scenarios"]
    assert rep["unsafe_writes"] == 0


def test_precision_recall_and_confusion_for_a_bad_model():
    scenarios = run_evals.load_scenarios()
    rep = run_evals.build_report(run_evals.run_suite(AlwaysReply(), scenarios), "bad")
    n_reply = sum(s["expect_action"] == "reply" for s in scenarios)
    assert rep["per_route"]["refund"]["recall"] == 0.0
    assert rep["per_route"]["refund"]["precision"] is None  # never predicted
    assert rep["per_route"]["reply"]["recall"] == 1.0
    assert rep["per_route"]["reply"]["precision"] == pytest.approx(n_reply / len(scenarios), 1e-3)
    assert rep["confusion_matrix"]["refund"]["reply"] == sum(
        s["expect_action"] == "refund" for s in scenarios
    )
    assert rep["unsafe_writes"] == 0  # missing a write is wrong, but not unsafe


def test_gate_exit_codes(monkeypatch, capsys):
    assert run_evals.main([]) == 0
    monkeypatch.setattr(run_evals, "RuleBasedLLM", AlwaysReply)
    assert run_evals.main([]) == 1
    assert "FAIL" in capsys.readouterr().out
