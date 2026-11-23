"""LLM adapters. The graph only depends on `StructuredLLM.complete(schema, prompt) -> model`.

`RuleBasedLLM` is a deterministic stand-in used by tests, evals and the CLI `--offline` mode:
it makes the whole workflow (routing, retrieval, risk checks, HITL, audit) runnable and
testable with zero network. `AnthropicLLM` uses Claude with tool use to get a JSON object that
matches the Pydantic schema exactly.
"""

from __future__ import annotations

import json
import re
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from triage_graph.metrics import Usage, estimate_cost
from triage_graph.state import Classification, Plan

T = TypeVar("T", bound=BaseModel)


class StructuredLLM(Protocol):
    def complete(self, schema: type[T], prompt: str) -> T: ...


def complete_with_usage(llm: StructuredLLM, schema: type[T], prompt: str) -> tuple[T, Usage]:
    """Call the adapter and return (result, usage). Adapters that do not report usage
    (e.g. third-party stand-ins) count as zero tokens / zero cost."""
    fn = getattr(llm, "complete_with_usage", None)
    if callable(fn):
        return fn(schema, prompt)
    return llm.complete(schema, prompt), Usage(model=type(llm).__name__)


class RuleBasedLLM:
    """Keyword heuristics standing in for a model. Deterministic; good enough to exercise paths."""

    @staticmethod
    def _ticket_text(prompt: str) -> str:
        """Only the ticket matters to the heuristics; policy passages must not trigger actions."""
        low = prompt.lower()
        start = low.find("ticket:")
        body = low[start + len("ticket:") :] if start >= 0 else low
        end = body.find("\n\npolicies:")
        return body[:end] if end >= 0 else body

    def complete_with_usage(self, schema: type[T], prompt: str) -> tuple[T, Usage]:
        return self.complete(schema, prompt), Usage(model="rule-based")  # no tokens, no cost

    def complete(self, schema: type[T], prompt: str) -> T:
        text = self._ticket_text(prompt)
        if schema is Classification:
            return self._classify(text)  # type: ignore[return-value]
        if schema is Plan:
            return self._plan(text)  # type: ignore[return-value]
        raise TypeError(f"unsupported schema {schema.__name__}")

    @staticmethod
    def _classify(text: str) -> Classification:
        if any(w in text for w in ("cobrad", "cobran", "reembols", "repasse", "fatura")):
            cat = "billing"
        elif any(w in text for w in ("terminal", "portão", "portao", "fora do ar", "não funciona")):
            cat = "incident"
        elif any(w in text for w in ("senha", "acesso", "login", "mfa")):
            cat = "access"
        elif any(w in text for w in ("desconto", "proposta", "contrato", "renova")):
            cat = "commercial"
        else:
            cat = "other"
        if "evento" in text and cat == "incident":
            pri = "urgent"
        elif cat == "incident" or "urgente" in text:
            pri = "high"
        elif cat in ("billing", "access"):
            pri = "medium"
        else:
            pri = "low"
        return Classification(category=cat, priority=pri, rationale=f"keywords → {cat}/{pri}")

    @staticmethod
    def _plan(text: str) -> Plan:
        amount = None
        m = re.search(r"r\$\s?([\d.]+(?:,\d{2})?)", text)
        if m:
            amount = float(m.group(1).replace(".", "").replace(",", "."))
        if "reembols" in text or "cobrado em dobro" in text or "duplicad" in text:
            return Plan(
                action="refund",
                draft_reply="Identificamos a cobrança duplicada e vamos processar o reembolso.",
                refund_amount_brl=amount,
                cites=["sop_reembolso.md"],
            )
        if "portão" in text or "portao" in text or ("terminal" in text and "evento" in text):
            return Plan(
                action="escalate",
                draft_reply="Acionamos o suporte N2 em caráter urgente e ativamos o modo offline.",
                cites=["sop_incidentes.md"],
            )
        if "fechar" in text or "resolvido" in text:
            return Plan(action="close", draft_reply="Encerramos o chamado conforme solicitado.")
        return Plan(
            action="reply",
            draft_reply="Segue a orientação com base na nossa política interna.",
            cites=[],
        )


class AnthropicLLM:
    """Claude via tool use: the schema becomes the tool's input_schema, so the output is JSON
    that already matches the Pydantic model. Requires `pip install -e ".[anthropic]"`."""

    def __init__(self, model: str = "claude-sonnet-4-5", api_key: str | None = None) -> None:
        import anthropic  # optional dependency

        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.model = model

    def complete(self, schema: type[T], prompt: str) -> T:
        return self.complete_with_usage(schema, prompt)[0]

    def complete_with_usage(self, schema: type[T], prompt: str) -> tuple[T, Usage]:
        tool = {
            "name": f"emit_{schema.__name__.lower()}",
            "description": f"Return a {schema.__name__} object.",
            "input_schema": schema.model_json_schema(),
        }
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=800,
            tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
            messages=[{"role": "user", "content": prompt}],
        )
        usage = self._usage(resp)
        block = next(b for b in resp.content if b.type == "tool_use")
        try:
            return schema.model_validate(block.input), usage
        except ValidationError as exc:
            raise ValueError(f"model output failed validation: {json.dumps(exc.errors())}") from exc

    def _usage(self, resp: object) -> Usage:
        u = getattr(resp, "usage", None)

        def n(field: str) -> int:
            return int(getattr(u, field, 0) or 0)

        usage = Usage(
            model=self.model,
            input_tokens=n("input_tokens"),
            output_tokens=n("output_tokens"),
            cache_read_tokens=n("cache_read_input_tokens"),
            cache_write_tokens=n("cache_creation_input_tokens"),
        )
        usage.cost_usd = estimate_cost(usage)
        return usage
