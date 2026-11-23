"""LLM adapters. The graph only depends on `StructuredLLM.complete(schema, prompt) -> model`.

`RuleBasedLLM` is a deterministic stand-in used by tests, evals and the CLI `--offline` mode:
it makes the whole workflow (routing, retrieval, risk checks, HITL, audit) runnable and
testable with zero network. `AnthropicLLM` uses Claude with structured outputs
(`output_config.format`) to get a JSON object for the Pydantic schema, validated by Pydantic.
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


DEFAULT_MODEL = "claude-sonnet-5-5"
# Server-side refusal fallback (beta): if the model declines for policy reasons, the API
# re-runs the same request on the model's default fallback inside the same call.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicLLM:
    """Claude with structured outputs: the Pydantic schema becomes `output_config.format`, so the
    reply is constrained to JSON for that schema, and Pydantic still validates it (constraints
    the API cannot enforce, such as string lengths and numeric bounds, are checked here).
    Requires `pip install -e ".[anthropic]"`.

    Current models (Claude Sonnet 5.5 included) reject forced `tool_choice`, so the old
    "single forced tool" trick is gone. Thinking is adaptive (on by default and not
    disable-able on Sonnet 5.5); `effort` keeps it short for these small typed decisions and
    `max_tokens` leaves room for thinking plus the JSON.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        *,
        effort: str | None = "low",
        max_tokens: int = 4096,
        refusal_fallback: bool = True,
        client: object | None = None,
    ) -> None:
        import anthropic  # optional dependency

        self._transform_schema = anthropic.transform_schema
        if client is None:
            client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self._client = client
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.refusal_fallback = refusal_fallback

    def complete(self, schema: type[T], prompt: str) -> T:
        return self.complete_with_usage(schema, prompt)[0]

    def complete_with_usage(self, schema: type[T], prompt: str) -> tuple[T, Usage]:
        output_config: dict = {
            "format": {"type": "json_schema", "schema": self._transform_schema(schema)}
        }
        if self.effort:
            output_config["effort"] = self.effort
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "output_config": output_config,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.refusal_fallback:
            kwargs.update(betas=[FALLBACK_BETA], fallbacks="default")
            resp = self._client.beta.messages.create(**kwargs)
        else:
            resp = self._client.messages.create(**kwargs)
        usage = self._usage(resp)

        stop = getattr(resp, "stop_reason", None)
        if stop == "refusal":
            details = getattr(resp, "stop_details", None)
            category = getattr(details, "category", None)
            raise ValueError(f"model declined the request (refusal, category={category})")
        if stop == "max_tokens":
            raise ValueError("model output was cut off by max_tokens; raise max_tokens")

        text = _final_text(resp.content)
        if not text:
            raise ValueError("model returned no JSON text block")
        try:
            return schema.model_validate_json(text), usage
        except ValidationError as exc:
            raise ValueError(f"model output failed validation: {json.dumps(exc.errors())}") from exc

    def _usage(self, resp: object) -> Usage:
        u = getattr(resp, "usage", None)

        def n(field: str) -> int:
            return int(getattr(u, field, 0) or 0)

        # After a server-side fallback the response names the model that actually served it.
        served_by = getattr(resp, "model", None)
        usage = Usage(
            model=served_by if isinstance(served_by, str) and served_by else self.model,
            input_tokens=n("input_tokens"),
            output_tokens=n("output_tokens"),
            cache_read_tokens=n("cache_read_input_tokens"),
            cache_write_tokens=n("cache_creation_input_tokens"),
        )
        usage.cost_usd = estimate_cost(usage)
        return usage


def _final_text(content: list) -> str:
    """Text of the answer, read by block type: thinking blocks (possibly empty) may come first,
    and after a server-side fallback only the blocks after the last `fallback` marker belong to
    the model that produced the final answer."""
    blocks = list(content)
    last_fallback = max(
        (i for i, b in enumerate(blocks) if getattr(b, "type", "") == "fallback"), default=-1
    )
    return "".join(
        getattr(b, "text", "")
        for b in blocks[last_fallback + 1 :]
        if getattr(b, "type", "") == "text"
    ).strip()
