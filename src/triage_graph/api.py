"""Human-approval API: start a triage, list runs waiting for a human, approve or reject.

    uvicorn --factory triage_graph.api:create_app       (pip install -e ".[api,sqlite]")

Environment:
    TRIAGE_API_TOKEN       bearer token (>= 24 chars). Unset or short → API refuses (503).
    TRIAGE_CHECKPOINT_URL  memory | sqlite:///path.db | postgresql://...  (default: memory)
    TRIAGE_LLM             rule (default, offline) | anthropic
    TRIAGE_AUDIT_LOG       optional JSONL file for API audit events

Security notes:
- Fail-secure: without a configured token every protected route answers 503, never "open".
- The token is compared in constant time (hmac.compare_digest) and never logged.
- Thread ids are generated server-side; path ids are validated against a strict pattern.
- A per-thread lock plus a re-check of the pending interrupt prevents double resumes within one
  process. Run one worker per checkpoint store, or put a DB-level lock in front, if you scale out.
- One shared token identifies a *client*, not a person. `approver` is recorded as declared; in
  production put SSO / per-user tokens in front and derive it from the identity.
"""

# No `from __future__ import annotations`: FastAPI must resolve the closure-local
# Annotated dependency aliases (Client, ThreadId) at route-definition time.

import hmac
import json
import logging
import os
import threading
import uuid
from collections import deque
from contextlib import ExitStack, asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi import Path as PathParam
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langgraph.types import Command
from pydantic import BaseModel, Field, StrictBool

from triage_graph.checkpoint import open_checkpointer
from triage_graph.graph import build_graph
from triage_graph.metrics import summarize

MIN_TOKEN_LEN = 24
THREAD_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
log = logging.getLogger("triage_graph.api")


# --------------------------------------------------------------------------- audit


class AuditLog:
    """Append-only API audit trail: in-memory ring (for inspection/tests) + optional JSONL file."""

    def __init__(self, path: str | None = None, keep: int = 1000) -> None:
        self.records: deque[dict] = deque(maxlen=keep)
        self._path = Path(path) if path else None
        self._lock = threading.Lock()
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **data: Any) -> None:
        rec = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **data}
        with self._lock:
            self.records.append(rec)
            if self._path:
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        log.info("audit %s", json.dumps(rec, ensure_ascii=False, default=str))


# --------------------------------------------------------------------------- schemas


class TriageRequest(BaseModel):
    ticket_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$")
    text: str = Field(min_length=1, max_length=5000)


class Decision(BaseModel):
    approved: StrictBool  # "yes", 1, "true" are rejected: only a JSON boolean counts
    approver: str = Field(min_length=1, max_length=64, pattern=r"^[\w.@+-]+$")
    note: str = Field(default="", max_length=300)


class RunView(BaseModel):
    thread_id: str
    status: str  # pending_approval | done
    ticket_id: str | None = None
    approval: dict | None = None
    outcome: str | None = None
    risk_flags: list[str] = []
    audit: list[dict] = []
    metrics: dict | None = None  # per-node latency, tokens, estimated cost


# --------------------------------------------------------------------------- helpers


def _default_llm():
    kind = os.environ.get("TRIAGE_LLM", "rule").strip().lower()
    if kind == "anthropic":
        from triage_graph.llm import AnthropicLLM

        return AnthropicLLM()
    if kind != "rule":
        raise RuntimeError(f"TRIAGE_LLM must be 'rule' or 'anthropic', got {kind!r}")
    from triage_graph.llm import RuleBasedLLM

    return RuleBasedLLM()


def _view(graph, thread_id: str) -> RunView:
    snap = graph.get_state({"configurable": {"thread_id": thread_id}})
    values = snap.values or {}
    pending = [i.value for i in snap.interrupts]
    return RunView(
        thread_id=thread_id,
        status="pending_approval" if pending else "done",
        ticket_id=values.get("ticket_id"),
        approval=pending[0] if pending else None,
        outcome=values.get("outcome"),
        risk_flags=values.get("risk_flags", []),
        audit=values.get("audit", []),
        metrics=summarize(values.get("audit", [])),
    )


def _known(graph, thread_id: str) -> bool:
    snap = graph.get_state({"configurable": {"thread_id": thread_id}})
    return bool(snap.values)


# --------------------------------------------------------------------------- app


def create_app(
    *,
    graph=None,
    checkpointer=None,
    token: str | None = None,
    audit: AuditLog | None = None,
) -> FastAPI:
    """Build the API. Tests inject `graph`/`checkpointer`; production reads the environment."""
    configured = token if token is not None else os.environ.get("TRIAGE_API_TOKEN", "")
    api_token = configured if len(configured) >= MIN_TOKEN_LEN else None
    audit_log = audit or AuditLog(os.environ.get("TRIAGE_AUDIT_LOG"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        with ExitStack() as stack:
            if graph is None:
                saver = checkpointer or stack.enter_context(open_checkpointer())
                app.state.graph = build_graph(_default_llm(), checkpointer=saver)
                app.state.saver = saver
            else:
                app.state.graph = graph
                app.state.saver = checkpointer or graph.checkpointer
            if api_token is None:
                log.error("TRIAGE_API_TOKEN missing or shorter than %d chars", MIN_TOKEN_LEN)
            yield

    app = FastAPI(title="triage-graph approvals", version="0.2.0", lifespan=lifespan)
    app.state.audit = audit_log
    locks: dict[str, threading.Lock] = {}
    locks_guard = threading.Lock()
    bearer = HTTPBearer(auto_error=False)

    def lock_for(thread_id: str) -> threading.Lock:
        with locks_guard:
            return locks.setdefault(thread_id, threading.Lock())

    def require_token(
        request: Request,
        creds: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    ) -> str:
        client = request.client.host if request.client else "?"
        if api_token is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "approval API not configured")
        given = creds.credentials if creds and creds.scheme.lower() == "bearer" else ""
        if not hmac.compare_digest(given.encode(), api_token.encode()):
            audit_log.write("auth_failed", client=client, path=request.url.path)
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                "invalid or missing bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return client

    Client = Annotated[str, Depends(require_token)]
    ThreadId = Annotated[str, PathParam(pattern=THREAD_ID_PATTERN)]

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    @app.post("/triage", status_code=status.HTTP_201_CREATED, response_model=RunView)
    def start_triage(req: TriageRequest, request: Request, client: Client) -> RunView:
        g = request.app.state.graph
        thread_id = uuid.uuid4().hex
        cfg = {"configurable": {"thread_id": thread_id}}
        try:
            g.invoke({"ticket_id": req.ticket_id, "text": req.text, "audit": []}, cfg)
        except Exception as exc:  # LLM/network/validation failure: no write happened
            audit_log.write(
                "triage_failed",
                client=client,
                thread_id=thread_id,
                ticket_id=req.ticket_id,
                error=type(exc).__name__,
            )
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, "triage failed") from None
        view = _view(g, thread_id)
        audit_log.write(
            "triage_started",
            client=client,
            thread_id=thread_id,
            ticket_id=req.ticket_id,
            status=view.status,
            proposed_action=(view.approval or {}).get("proposed_action"),
        )
        return view

    @app.get("/approvals", response_model=list[RunView])
    def list_pending(request: Request, client: Client, limit: int = 50) -> list[RunView]:
        g, saver = request.app.state.graph, request.app.state.saver
        limit = max(1, min(limit, 200))
        # Collect ids first: SqliteSaver holds its (non-reentrant) lock while list() iterates,
        # so calling get_state inside the loop would deadlock. O(checkpoints) scan: fine for a
        # demo; a production deployment would keep a pending-approvals index table.
        thread_ids = list(
            dict.fromkeys(t.config["configurable"]["thread_id"] for t in saver.list(None))
        )
        out: list[RunView] = []
        for tid in thread_ids:
            view = _view(g, tid)
            if view.status == "pending_approval":
                out.append(view.model_copy(update={"audit": []}))
                if len(out) >= limit:
                    break
        return out

    @app.get("/threads/{thread_id}", response_model=RunView)
    def get_thread(thread_id: ThreadId, request: Request, client: Client) -> RunView:
        g = request.app.state.graph
        if not _known(g, thread_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown thread")
        return _view(g, thread_id)

    @app.post("/threads/{thread_id}/decision", response_model=RunView)
    def decide(thread_id: ThreadId, body: Decision, request: Request, client: Client) -> RunView:
        g = request.app.state.graph
        with lock_for(thread_id):
            if not _known(g, thread_id):
                raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown thread")
            before = _view(g, thread_id)
            if before.status != "pending_approval":
                audit_log.write(
                    "decision_rejected",
                    client=client,
                    thread_id=thread_id,
                    approver=body.approver,
                    reason="not pending",
                )
                raise HTTPException(status.HTTP_409_CONFLICT, "thread is not waiting for approval")
            note = f"{body.approver}: {body.note}"[:300]
            cfg = {"configurable": {"thread_id": thread_id}}
            g.invoke(Command(resume={"approved": body.approved, "note": note}), cfg)
            after = _view(g, thread_id)
        audit_log.write(
            "decision",
            client=client,
            thread_id=thread_id,
            ticket_id=after.ticket_id,
            approver=body.approver,
            approved=body.approved,
            proposed_action=(before.approval or {}).get("proposed_action"),
            risk_flags=before.risk_flags,
            outcome=after.outcome,
        )
        return after

    return app
