"""Checkpointer selection: in-memory, SQLite or Postgres, chosen by a URL.

    memory                          → MemorySaver (default; lost when the process exits)
    sqlite:///path/to/triage.db     → SqliteSaver   (pip install -e ".[sqlite]")
    postgresql://user:pw@host/db    → PostgresSaver (pip install -e ".[postgres]")

The URL comes from the caller or from `TRIAGE_CHECKPOINT_URL`. Every backend shares the same
restricted serializer: only our own Pydantic models may be revived from a checkpoint
(deny-by-default deserialization, no pickle fallback). A persistent backend is what makes the
HITL pause survive a restart: `approve` can be resumed by a different process.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

ENV_VAR = "TRIAGE_CHECKPOINT_URL"

ALLOWED_MODULES = [
    ("triage_graph.state", "Classification"),
    ("triage_graph.state", "Plan"),
]


def make_serde() -> JsonPlusSerializer:
    """Serializer that revives only the allow-listed Pydantic models. Never pickles."""
    return JsonPlusSerializer(pickle_fallback=False, allowed_msgpack_modules=ALLOWED_MODULES)


def _sqlite_path(url: str) -> str:
    # sqlite:///relative.db, sqlite:////abs/path.db, sqlite:///C:/abs/path.db, sqlite:///:memory:
    path = url[len("sqlite://") :]
    if path.startswith("/"):
        path = path[1:]
    if not path:
        raise ValueError("sqlite URL needs a file path, e.g. sqlite:///triage.db")
    return path


def redact(url: str) -> str:
    """URL safe to log: drops credentials."""
    if "@" in url and "://" in url:
        scheme, rest = url.split("://", 1)
        return f"{scheme}://***@{rest.rsplit('@', 1)[1]}"
    return url


@contextmanager
def open_checkpointer(url: str | None = None) -> Iterator[BaseCheckpointSaver]:
    """Yield a checkpointer for `url` (or $TRIAGE_CHECKPOINT_URL) and close it afterwards."""
    url = (url or os.environ.get(ENV_VAR) or "memory").strip()
    serde = make_serde()

    if url == "memory":
        yield MemorySaver(serde=serde)
        return

    if url.startswith("sqlite:"):
        import sqlite3

        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:  # pragma: no cover - depends on installed extras
            raise RuntimeError('SQLite checkpointer needs: pip install -e ".[sqlite]"') from exc
        path = _sqlite_path(url)
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: the API serves requests from a thread pool; SqliteSaver
        # serializes access with its own lock.
        conn = sqlite3.connect(path, check_same_thread=False)
        try:
            saver = SqliteSaver(conn, serde=serde)
            saver.setup()
            yield saver
        finally:
            conn.close()
        return

    if url.startswith(("postgres://", "postgresql://")):
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            from psycopg import Connection
            from psycopg.rows import dict_row
        except ImportError as exc:  # pragma: no cover - depends on installed extras
            raise RuntimeError('Postgres checkpointer needs: pip install -e ".[postgres]"') from exc
        with Connection.connect(
            url, autocommit=True, prepare_threshold=0, row_factory=dict_row
        ) as c:
            saver = PostgresSaver(c, serde=serde)
            saver.setup()
            yield saver
        return

    raise ValueError(f"unsupported checkpoint URL: {redact(url)!r}")
