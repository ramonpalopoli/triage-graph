"""AnthropicLLM against a fake client shaped like Claude Sonnet 5.5 responses (offline).

No network and no API key: the client is injected. Responses mimic the current model: an
adaptive-thinking block (empty text under the default `display: "omitted"`) before the JSON
text block, `stop_reason` values including `refusal`, and `fallback` markers from server-side
refusal fallback.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("anthropic")  # only for anthropic.transform_schema; nothing is sent

from triage_graph.llm import DEFAULT_MODEL, FALLBACK_BETA, AnthropicLLM  # noqa: E402
from triage_graph.state import Classification, Plan  # noqa: E402

CLASSIFICATION = {"category": "billing", "priority": "medium", "rationale": "cobrança duplicada"}


def thinking(text: str = "") -> SimpleNamespace:
    return SimpleNamespace(type="thinking", thinking=text, signature="sig-opaque")


def text(payload: dict | str) -> SimpleNamespace:
    body = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return SimpleNamespace(type="text", text=body)


def fallback(src: str, dst: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="fallback", from_=SimpleNamespace(model=src), to=SimpleNamespace(model=dst)
    )


def message(*blocks, stop_reason="end_turn", model=DEFAULT_MODEL, stop_details=None, **usage):
    u = {
        "input_tokens": 500,
        "output_tokens": 100,
        "cache_read_input_tokens": None,
        "cache_creation_input_tokens": 0,
    }
    u.update(usage)
    return SimpleNamespace(
        model=model,
        content=list(blocks),
        stop_reason=stop_reason,
        stop_details=stop_details,
        usage=SimpleNamespace(**u),
    )


class FakeClient:
    """Records request kwargs on both `messages` and `beta.messages`."""

    def __init__(self, resp) -> None:
        self.calls: list[tuple[str, dict]] = []

        def recorder(kind):
            return SimpleNamespace(create=lambda **kw: self.calls.append((kind, kw)) or resp)

        self.messages = recorder("messages")
        self.beta = SimpleNamespace(messages=recorder("beta"))


def adapter(resp, **kw) -> tuple[AnthropicLLM, FakeClient]:
    client = FakeClient(resp)
    return AnthropicLLM(client=client, **kw), client


def test_defaults_to_sonnet_5_5():
    assert DEFAULT_MODEL == "claude-sonnet-5-5"
    llm, _ = adapter(message(text(CLASSIFICATION)))
    assert llm.model == "claude-sonnet-5-5"


def test_request_uses_structured_outputs_not_forced_tool_choice():
    llm, client = adapter(message(thinking(), text(CLASSIFICATION)))
    llm.complete(Classification, "Ticket: cobrado em dobro")

    ((kind, req),) = client.calls
    assert kind == "beta"  # server-side fallback lives on the beta endpoint
    assert "tool_choice" not in req and "tools" not in req
    assert "thinking" not in req  # adaptive by default; `disabled` would be a 400 on Sonnet 5.5
    assert "temperature" not in req
    assert req["model"] == "claude-sonnet-5-5"
    assert req["max_tokens"] >= 1024  # thinking counts toward max_tokens
    fmt = req["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"]["additionalProperties"] is False
    assert set(fmt["schema"]["required"]) == {"category", "priority", "rationale"}
    assert req["output_config"]["effort"] == "low"
    assert req["betas"] == [FALLBACK_BETA]
    assert req["fallbacks"] == "default"


def test_parses_json_after_an_empty_thinking_block():
    llm, _ = adapter(message(thinking(""), text(CLASSIFICATION)))
    c, usage = llm.complete_with_usage(Classification, "Ticket: cobrado em dobro")
    assert c == Classification(**CLASSIFICATION)
    assert (usage.input_tokens, usage.output_tokens) == (500, 100)
    assert usage.model == "claude-sonnet-5-5"
    assert usage.cost_usd == pytest.approx((500 * 2 + 100 * 10) / 1e6)


def test_plan_with_optional_fields():
    plan = {
        "action": "refund",
        "draft_reply": "Vamos processar o reembolso da cobrança duplicada.",
        "refund_amount_brl": 320.0,
        "cites": ["sop_reembolso.md"],
    }
    llm, client = adapter(message(thinking(), text(plan)))
    p = llm.complete(Plan, "Ticket: reembolso R$ 320,00")
    assert p.action == "refund" and p.refund_amount_brl == 320.0
    schema = client.calls[0][1]["output_config"]["format"]["schema"]
    assert schema["additionalProperties"] is False


def test_pydantic_still_enforces_constraints_the_api_cannot():
    # maxLength/minimum are stripped from the schema sent to the API; Pydantic re-checks them.
    bad = {"action": "refund", "draft_reply": "curto", "refund_amount_brl": 99999}
    llm, _ = adapter(message(text(bad)))
    with pytest.raises(ValueError, match="failed validation"):
        llm.complete(Plan, "Ticket: reembolso")


def test_refusal_is_an_error_not_a_parse_attempt():
    details = SimpleNamespace(type="refusal", category="cyber", explanation="declined")
    llm, _ = adapter(message(stop_reason="refusal", stop_details=details))
    with pytest.raises(ValueError, match="refusal, category=cyber"):
        llm.complete(Classification, "Ticket: ...")


def test_truncated_output_is_reported():
    llm, _ = adapter(message(thinking(), text('{"category": "bil'), stop_reason="max_tokens"))
    with pytest.raises(ValueError, match="max_tokens"):
        llm.complete(Classification, "Ticket: ...")


def test_fallback_answer_is_read_after_the_marker_and_priced_as_the_serving_model():
    resp = message(
        fallback("claude-sonnet-5-5", "claude-sonnet-5"),
        thinking(),
        text(CLASSIFICATION),
        model="claude-sonnet-5",
    )
    llm, _ = adapter(resp)
    c, usage = llm.complete_with_usage(Classification, "Ticket: cobrado em dobro")
    assert c.category == "billing"
    assert usage.model == "claude-sonnet-5"


def test_fallback_can_be_turned_off_for_platforms_without_it():
    llm, client = adapter(message(text(CLASSIFICATION)), refusal_fallback=False, effort=None)
    llm.complete(Classification, "Ticket: cobrado em dobro")
    ((kind, req),) = client.calls
    assert kind == "messages"
    assert "betas" not in req and "fallbacks" not in req
    assert "effort" not in req["output_config"]


def test_no_text_block_is_an_error():
    llm, _ = adapter(message(thinking()))
    with pytest.raises(ValueError, match="no JSON text"):
        llm.complete(Classification, "Ticket: ...")
