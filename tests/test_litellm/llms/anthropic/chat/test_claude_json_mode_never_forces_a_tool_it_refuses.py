"""JSON mode on the newer Claude models: native structured output, never a forced tool it refuses.

THE DEFECT, measured in h2oGPTe CI (2026-09-29): Claude Opus 5.5 thinks on every request, and a
request that forces a tool -- `tool_choice` of type `tool` -- is refused with
`400 tool_choice: type "tool" and "any" are not supported for this model`. JSON mode on this fork
chose native structured output from a hard-coded name list that stopped at opus-4-7 / sonnet-4-6,
so every newer Claude fell back to the forced JSON tool: 81 refusals in one smoke run, every
JSON-mode sub-call (query rewrite, classification) returning `{}` after two retries.

Backported from upstream: the choice is the model map's `supports_native_structured_output`
(#33235), the entries for Opus 5.5 / Opus 5 / Fable 5.1 / Mythos 5 with their flags, and the guard
that never forces a tool on a model mapped `supports_forced_tool_use: false`.
"""

import litellm
import pytest

from litellm.llms.anthropic.chat.transformation import AnthropicConfig

SCHEMA_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "rewrite",
        "schema": {
            "type": "object",
            "properties": {"new_question": {"type": "string"}},
            "required": ["new_question"],
        },
    },
}


def _map(model):
    return AnthropicConfig().map_openai_params(
        non_default_params={"response_format": SCHEMA_FORMAT},
        optional_params={},
        model=model,
        drop_params=False,
    )


@pytest.mark.parametrize(
    "model",
    [
        "claude-opus-5-5",
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-fable-5",
        "claude-fable-5-1",
        "claude-mythos-5",
        "claude-haiku-4-5",
        "claude-haiku-4-5-20251001",
    ],
)
def test_newer_claude_uses_NATIVE_structured_output_and_forces_nothing(model):
    out = _map(model)
    assert out["output_format"]["type"] == "json_schema"
    assert "tool_choice" not in out
    assert out.get("json_mode") is True


@pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-fable-5-1"])
def test_the_models_that_refuse_forced_tool_use_are_mapped_so(model):
    assert AnthropicConfig.forced_tool_use_unsupported(model, "anthropic") is True


def test_the_GUARD_alone_a_model_without_native_output_that_refuses_forced_tools(monkeypatch):
    """No native structured output AND no forced tool use: the JSON tool is offered, not forced."""
    model = "claude-test-no-forced-tool"
    monkeypatch.setitem(
        litellm.model_cost,
        model,
        {"litellm_provider": "anthropic", "mode": "chat", "supports_forced_tool_use": False},
    )
    litellm.get_model_info.cache_clear()
    try:
        out = _map(model)
    finally:
        litellm.get_model_info.cache_clear()
    assert "output_format" not in out
    assert any(t.get("name") == "json_tool_call" for t in out.get("tools", []))
    assert "tool_choice" not in out


def test_a_model_with_NEITHER_flag_keeps_the_forced_json_tool(monkeypatch):
    """Unchanged for everything else: no native output, forced tool use allowed -> forced tool."""
    model = "claude-test-legacy"
    monkeypatch.setitem(litellm.model_cost, model, {"litellm_provider": "anthropic", "mode": "chat"})
    litellm.get_model_info.cache_clear()
    try:
        out = _map(model)
    finally:
        litellm.get_model_info.cache_clear()
    assert out["tool_choice"] == {"name": "json_tool_call", "type": "tool"}
    assert AnthropicConfig.forced_tool_use_unsupported(model, "anthropic") is False


TOOLS = [
    {
        "type": "function",
        "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}},
    }
]


def _tool_choice(model, choice, drop_params):
    return AnthropicConfig().map_openai_params(
        non_default_params={"tools": TOOLS, "tool_choice": choice},
        optional_params={},
        model=model,
        drop_params=drop_params,
    )


@pytest.mark.parametrize("choice", ["required", {"type": "function", "function": {"name": "lookup"}}])
def test_a_CALLERS_forced_tool_choice_is_downgraded_with_drop_params(choice):
    out = _tool_choice("claude-opus-5-5", choice, drop_params=True)
    assert out["tool_choice"]["type"] == "auto" and "name" not in out["tool_choice"]


def test_a_CALLERS_forced_tool_choice_without_drop_params_fails_before_sending(monkeypatch):
    monkeypatch.setattr(litellm, "drop_params", False)
    with pytest.raises(litellm.utils.UnsupportedParamsError, match="forced tool use"):
        _tool_choice("claude-opus-5-5", "required", drop_params=False)


def test_forced_tool_choice_is_UNCHANGED_where_the_model_supports_it():
    out = _tool_choice("claude-sonnet-4-5", "required", drop_params=False)
    assert out["tool_choice"]["type"] == "any"
    out = _tool_choice("claude-opus-5-5", "auto", drop_params=False)
    assert out["tool_choice"]["type"] == "auto"
