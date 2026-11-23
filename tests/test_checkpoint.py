"""Durable HITL pause: pause at `approve`, then resume from a *different process*."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import uuid

import pytest
from langgraph.types import Command
from pydantic import BaseModel

from triage_graph.checkpoint import make_serde, open_checkpointer, redact
from triage_graph.graph import build_graph
from triage_graph.llm import RuleBasedLLM
from triage_graph.state import Plan

REFUND = "Fui cobrado em dobro, R$ 320,00, quero reembolso"

RESUME_SCRIPT = textwrap.dedent(
    """
    import json, sys
    from langgraph.types import Command
    from triage_graph.checkpoint import open_checkpointer
    from triage_graph.graph import build_graph
    from triage_graph.llm import RuleBasedLLM

    url, thread_id = sys.argv[1], sys.argv[2]
    with open_checkpointer(url) as saver:
        graph = build_graph(RuleBasedLLM(), checkpointer=saver)
        cfg = {"configurable": {"thread_id": thread_id}}
        snap = graph.get_state(cfg)
        pending = [i.value for i in snap.interrupts]
        state = graph.invoke(Command(resume={"approved": True, "note": "other process"}), cfg)
        print(json.dumps({
            "pending": pending,
            "outcome": state["outcome"],
            "nodes": [e["node"] for e in state["audit"]],
            "plan_type": type(state["plan"]).__name__,
        }))
    """
)


def _pause(url: str, thread_id: str) -> dict:
    with open_checkpointer(url) as saver:
        graph = build_graph(RuleBasedLLM(), checkpointer=saver)
        cfg = {"configurable": {"thread_id": thread_id}}
        state = graph.invoke({"ticket_id": "t-sql", "text": REFUND, "audit": []}, cfg)
        assert "__interrupt__" in state
        return state["__interrupt__"][0].value


def test_sqlite_pause_survives_process_restart(tmp_path):
    pytest.importorskip("langgraph.checkpoint.sqlite")
    url = f"sqlite:///{(tmp_path / 'triage.db').as_posix()}"
    thread_id = f"t-{uuid.uuid4()}"
    payload = _pause(url, thread_id)
    assert payload["proposed_action"] == "refund"

    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    out = subprocess.run(  # noqa: S603 - fixed interpreter and script, test-only
        [sys.executable, "-c", RESUME_SCRIPT, url, thread_id],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=120,
        check=True,
    )
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result["pending"][0]["refund_amount_brl"] == 320.0
    assert result["outcome"] == "refund executed"
    assert result["nodes"][-2:] == ["approve", "execute"]
    assert result["plan_type"] == "Plan"  # Pydantic model revived through the allow-list


def test_sqlite_resume_in_same_process_after_reopen(tmp_path):
    pytest.importorskip("langgraph.checkpoint.sqlite")
    url = f"sqlite:///{(tmp_path / 'x' / 'triage.db').as_posix()}"
    _pause(url, "reopen")
    with open_checkpointer(url) as saver:
        graph = build_graph(RuleBasedLLM(), checkpointer=saver)
        cfg = {"configurable": {"thread_id": "reopen"}}
        s = graph.invoke(Command(resume={"approved": False, "note": "no"}), cfg)
    assert "REFUSED" in s["outcome"]


@pytest.mark.skipif(
    not os.environ.get("TRIAGE_TEST_POSTGRES_URL"),
    reason="set TRIAGE_TEST_POSTGRES_URL to run against a real Postgres",
)
def test_postgres_pause_and_resume():
    pytest.importorskip("langgraph.checkpoint.postgres")
    url = os.environ["TRIAGE_TEST_POSTGRES_URL"]
    thread_id = f"pg-{uuid.uuid4()}"
    _pause(url, thread_id)
    with open_checkpointer(url) as saver:
        graph = build_graph(RuleBasedLLM(), checkpointer=saver)
        cfg = {"configurable": {"thread_id": thread_id}}
        s = graph.invoke(Command(resume={"approved": True, "note": "pg"}), cfg)
    assert s["outcome"] == "refund executed"


def test_unknown_scheme_is_rejected_without_leaking_credentials():
    with pytest.raises(ValueError) as exc:
        with open_checkpointer("redis://user:secret@host/0"):
            pass
    assert "secret" not in str(exc.value)
    assert redact("postgresql://u:pw@db:5432/x") == "postgresql://***@db:5432/x"


class _NotAllowed(BaseModel):
    x: int = 1


def test_serde_revives_only_allow_listed_models():
    serde = make_serde()
    plan = Plan(action="reply", draft_reply="0123456789")
    assert isinstance(serde.loads_typed(serde.dumps_typed(plan)), Plan)
    revived = serde.loads_typed(serde.dumps_typed(_NotAllowed()))
    assert not isinstance(revived, _NotAllowed)  # blocked: comes back as plain data
    assert revived == {"x": 1}
