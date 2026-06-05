"""Tests for the h2o AnthropicParamsFilterHook param-stripping behaviour.

Focus: Anthropic-only parameters (notably ``output_config``, the adaptive-thinking
effort control that LiteLLM maps ``reasoning_effort`` onto for Claude 4.5/4.6/4.7 /
Opus 4.5) must only ever be forwarded to actual Anthropic / Bedrock-Claude models.
When such a param leaks to an OpenAI/Azure deployment, Azure rejects the entire
request with::

    AzureException - Unrecognized request argument supplied: output_config

so the hook strips these params for non-Anthropic models and preserves them for
Anthropic ones.
"""

import pytest

from litellm.integrations.h2o.litellm_anthropic_params_filter_hook import (
    AnthropicParamsFilterHook,
)


@pytest.fixture
def hook():
    return AnthropicParamsFilterHook()


async def _run(hook, data):
    return await hook.async_pre_call_hook(
        user_api_key_dict={}, cache=None, data=data, call_type="completion"
    )


# --- output_config is stripped for non-Anthropic models -------------------


@pytest.mark.parametrize(
    "model",
    [
        "gpt-4.1",
        "gpt-41_v2025-01-01_GLOBAL",
        "azure/gpt-4.1",
        "gpt-4o-mini",
    ],
)
async def test_output_config_stripped_for_non_anthropic_top_level(hook, model):
    data = {"model": model, "output_config": {"effort": "high"}, "messages": []}
    out = await _run(hook, data)
    assert "output_config" not in out, (
        f"output_config must be stripped for non-Anthropic model {model}"
    )


async def test_output_config_stripped_in_extra_body(hook):
    data = {
        "model": "gpt-4.1",
        "extra_body": {"output_config": {"effort": "low"}},
        "messages": [],
    }
    out = await _run(hook, data)
    assert "output_config" not in out.get("extra_body", {})


async def test_output_config_stripped_in_litellm_params_extra_body(hook):
    data = {
        "model": "gpt-4.1",
        "litellm_params": {"extra_body": {"output_config": {"effort": "medium"}}},
        "messages": [],
    }
    out = await _run(hook, data)
    assert "output_config" not in out["litellm_params"]["extra_body"]


# --- output_config is preserved for Anthropic / Bedrock-Claude models ------


@pytest.mark.parametrize(
    "model",
    [
        "anthropic/claude-sonnet-4-5",
        "claude-opus-4-5",
        "bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0",
    ],
)
async def test_output_config_preserved_for_anthropic(hook, model):
    cfg = {"effort": "high"}
    data = {"model": model, "output_config": cfg, "messages": []}
    out = await _run(hook, data)
    assert out.get("output_config") == cfg, (
        f"output_config must be preserved for Anthropic model {model}"
    )


# --- existing Anthropic-only params remain filtered (regression guard) -----


async def test_thinking_and_context_management_still_stripped(hook):
    data = {
        "model": "gpt-4.1",
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "context_management": {"edits": []},
        "enable_caching": True,
        "messages": [],
    }
    out = await _run(hook, data)
    assert "thinking" not in out
    assert "context_management" not in out
    assert "enable_caching" not in out


async def test_non_anthropic_params_are_untouched(hook):
    """The hook must not strip ordinary params like temperature/max_tokens/tools."""
    data = {
        "model": "gpt-4.1",
        "temperature": 0.5,
        "max_tokens": 256,
        "tools": [{"type": "function", "function": {"name": "f"}}],
        "output_config": {"effort": "high"},
        "messages": [],
    }
    out = await _run(hook, data)
    assert out["temperature"] == 0.5
    assert out["max_tokens"] == 256
    assert out["tools"]
    assert "output_config" not in out
