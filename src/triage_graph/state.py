"""Graph state and the typed decisions each node produces.

Everything the model emits is validated by Pydantic before it influences routing. The LLM is
treated as an untrusted component: it proposes, the graph disposes.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

Category = Literal["billing", "incident", "access", "commercial", "other"]
Priority = Literal["low", "medium", "high", "urgent"]
Action = Literal["reply", "escalate", "refund", "close"]

# Actions that change the world (or money) need a human. Replies do not.
WRITE_ACTIONS: frozenset[str] = frozenset({"escalate", "refund", "close"})


class Classification(BaseModel):
    category: Category
    priority: Priority
    rationale: str = Field(min_length=3, max_length=300)


class Plan(BaseModel):
    action: Action
    draft_reply: str = Field(min_length=10, max_length=1200)
    refund_amount_brl: float | None = Field(default=None, ge=0, le=5000)
    cites: list[str] = Field(default_factory=list, description="Knowledge sources used")


class TriageState(TypedDict, total=False):
    ticket_id: str
    text: str
    classification: Classification
    passages: list[dict]
    plan: Plan
    risk_flags: list[str]
    approved: bool | None
    approver_note: str
    outcome: str
    audit: Annotated[list[dict], operator.add]  # append-only decision trail
