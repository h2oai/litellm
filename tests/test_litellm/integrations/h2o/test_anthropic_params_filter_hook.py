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


# --- Newer Claude reject temperature/top_p/top_k entirely (Sonnet 5 etc.) ---
#
# Fable 5 / Mythos, Opus 4.7+, Sonnet 5 return HTTP 400 if any sampling param is
# present ("`temperature` is deprecated for this model.") and reject the legacy
# enabled/budget_tokens thinking API. The hook must strip all three params and
# convert thinking to adaptive — like OpenAI o1 / gpt-5 reasoning models.

NO_SAMPLING_MODELS = [
    "claude-sonnet-5",
    "anthropic/claude-sonnet-5",
    "claude-sonnet-5-latest",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-fable-5",
    "claude-mythos-5",
    "bedrock/anthropic.claude-sonnet-5",
]


@pytest.mark.parametrize("model", NO_SAMPLING_MODELS)
async def test_all_sampling_params_stripped_top_level(hook, model):
    data = {
        "model": model,
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 5,
        "messages": [],
    }
    out = await _run(hook, data)
    assert "temperature" not in out, model
    assert "top_p" not in out, model
    assert "top_k" not in out, model


async def test_sampling_params_stripped_in_sub_dicts(hook):
    data = {
        "model": "claude-sonnet-5",
        "messages": [],
        "extra_body": {"temperature": 0.3, "top_k": 10},
        "litellm_params": {
            "temperature": 0.7,
            "extra_body": {"top_p": 0.8},
        },
    }
    out = await _run(hook, data)
    assert "temperature" not in out["extra_body"]
    assert "top_k" not in out["extra_body"]
    assert "temperature" not in out["litellm_params"]
    assert "top_p" not in out["litellm_params"]["extra_body"]


async def test_thinking_enabled_converted_to_adaptive(hook):
    data = {
        "model": "claude-sonnet-5",
        "messages": [],
        "thinking": {"type": "enabled", "budget_tokens": 4096},
    }
    out = await _run(hook, data)
    assert out["thinking"] == {"type": "adaptive"}


async def test_temperature_not_forced_to_one_with_thinking(hook):
    # The old behavior forces temperature=1 when thinking is enabled; that 400s
    # on these models, so temperature must be absent, not 1.
    data = {
        "model": "claude-sonnet-5",
        "messages": [],
        "temperature": 0.2,
        "thinking": {"type": "enabled", "budget_tokens": 2048},
    }
    out = await _run(hook, data)
    assert "temperature" not in out
    assert out["thinking"] == {"type": "adaptive"}


async def test_older_claude_4x_keeps_temperature(hook):
    # Regression guard: older 4.x must NOT be treated as no-sampling — they
    # accept temperature (only temperature+top_p together is rejected).
    for model in ("claude-sonnet-4-6", "claude-opus-4-6"):
        data = {"model": model, "temperature": 0.5, "messages": []}
        out = await _run(hook, data)
        assert out.get("temperature") == 0.5, model


# --- adaptive conversion: budget->effort, output_config preserved, max_tokens ---


async def test_budget_tokens_mapped_to_effort(hook):
    # Dropping budget_tokens should preserve reasoning depth via output_config.effort.
    for budget, expected in [(20000, "high"), (8192, "medium"), (1024, "low")]:
        data = {
            "model": "claude-sonnet-5",
            "messages": [],
            "thinking": {"type": "enabled", "budget_tokens": budget},
        }
        out = await _run(hook, data)
        assert out["thinking"] == {"type": "adaptive"}, budget
        assert out.get("output_config") == {"effort": expected}, (budget, out.get("output_config"))


async def test_existing_output_config_not_overwritten(hook):
    data = {
        "model": "claude-sonnet-5",
        "messages": [],
        "thinking": {"type": "enabled", "budget_tokens": 20000},
        "output_config": {"effort": "low"},
    }
    out = await _run(hook, data)
    assert out["output_config"] == {"effort": "low"}  # user's choice preserved


async def test_output_config_preserved_for_no_sampling_model(hook):
    # output_config (adaptive effort) must NOT be stripped for Anthropic models.
    data = {"model": "claude-sonnet-5", "output_config": {"effort": "high"}, "messages": []}
    out = await _run(hook, data)
    assert out.get("output_config") == {"effort": "high"}


async def test_max_tokens_not_inflated_after_adaptive_conversion(hook):
    # Load-bearing ordering: _ensure_max_tokens_for_thinking must be a no-op once
    # thinking is adaptive (no budget_tokens), so max_tokens stays as the caller set it.
    data = {
        "model": "claude-sonnet-5",
        "messages": [],
        "max_tokens": 1000,
        "thinking": {"type": "enabled", "budget_tokens": 4096},
    }
    out = await _run(hook, data)
    assert out["max_tokens"] == 1000
    assert out["thinking"] == {"type": "adaptive"}


@pytest.mark.parametrize("budget", [None, 0, -1, "abc", {}])
async def test_degenerate_budget_tokens_no_output_config(hook, budget):
    # Degenerate budgets must still convert to adaptive but add NO output_config.
    data = {
        "model": "claude-sonnet-5",
        "messages": [],
        "thinking": {"type": "enabled", "budget_tokens": budget},
    }
    out = await _run(hook, data)
    assert out["thinking"] == {"type": "adaptive"}
    assert "output_config" not in out


async def test_missing_budget_tokens_no_output_config(hook):
    data = {"model": "claude-sonnet-5", "messages": [], "thinking": {"type": "enabled"}}
    out = await _run(hook, data)
    assert out["thinking"] == {"type": "adaptive"}
    assert "output_config" not in out


async def test_effort_lands_in_correct_subdict(hook):
    # thinking in extra_body -> effort in extra_body, not top-level data.
    data = {
        "model": "claude-sonnet-5",
        "messages": [],
        "extra_body": {"thinking": {"type": "enabled", "budget_tokens": 20000}},
    }
    out = await _run(hook, data)
    assert out["extra_body"]["thinking"] == {"type": "adaptive"}
    assert out["extra_body"].get("output_config") == {"effort": "high"}
    assert "output_config" not in out  # not written to top-level


# --- the repairs must not depend on the public alias ----------------------
#
# Every repair in this hook decides by model NAME. At async_pre_call_hook time
# that name is the proxy's public alias (the model_lock display name), which need
# not resemble the deployment behind it. Measured against a live proxy -- same
# deployment, same credentials, same request, only the alias differing:
#
#     model_name: claude-sonnet-4-5     -> 200, top_p stripped
#     model_name: anthropic-sonnet-4-5  -> 400 `temperature` and `top_p` cannot
#                                             both be specified
#
# so async_pre_call_deployment_hook re-runs the sequence against the resolved
# deployment. These tests pin that, and pin that the alias-time dispatch is still
# there (the two cover each other).


async def _deployment(hook, kwargs, call_type="acompletion"):
    return await hook.async_pre_call_deployment_hook(kwargs, call_type)


@pytest.mark.parametrize(
    "alias",
    [
        "anthropic-sonnet-4-5",   # no "claude" anywhere in it
        "sonnet-4-5",
        "fast-chat",              # a vanity name, the worst case
        "gpt-4o",                 # actively misleading about its provider
    ],
)
async def test_alias_time_dispatch_cannot_see_the_provider(hook, alias):
    """Documents WHY the deployment hook is needed, at the alias dispatch."""
    data = {"model": alias, "temperature": 0.2, "top_p": 0.9}
    result = await _run(hook, data)
    # Nothing to key off: the alias does not look Anthropic, so top_p survives
    # here and would reach Anthropic as the rejected pair.
    assert result["top_p"] == 0.9
    assert result["temperature"] == 0.2


@pytest.mark.parametrize(
    "model",
    [
        "anthropic/claude-sonnet-4-5-20250929",
        "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    ],
)
async def test_deployment_dispatch_strips_top_p_whatever_the_alias_was(hook, model):
    """The resolved deployment IS visible, so the pair gets repaired."""
    kwargs = {"model": model, "temperature": 0.2, "top_p": 0.9}
    result = await _deployment(hook, kwargs)
    assert result is not None, "the pair must be repaired at the deployment"
    assert "top_p" not in result
    assert result["temperature"] == 0.2


async def test_deployment_dispatch_returns_none_when_nothing_needed_repair(hook):
    """None tells litellm to keep the original kwargs -- do not hand back a copy
    for every request, and do not touch a request that is already fine."""
    kwargs = {"model": "anthropic/claude-sonnet-4-5-20250929", "temperature": 0.2}
    assert await _deployment(hook, kwargs) is None


async def test_deployment_dispatch_leaves_non_anthropic_sampling_alone(hook):
    """temperature+top_p together is legal everywhere else and must survive."""
    kwargs = {"model": "openai/gpt-4o-mini", "temperature": 0.2, "top_p": 0.9}
    result = await _deployment(hook, kwargs)
    if result is not None:          # other repairs may no-op-copy
        assert result["top_p"] == 0.9
        assert result["temperature"] == 0.2


async def test_deployment_dispatch_skips_text_completion(hook):
    """/v1/completions has no sampling semantics worth rewriting."""
    kwargs = {"model": "anthropic/claude-sonnet-4-5-20250929",
              "temperature": 0.2, "top_p": 0.9}
    assert await _deployment(hook, kwargs, call_type="atext_completion") is None


async def test_deployment_dispatch_does_not_mutate_the_caller_kwargs(hook):
    """litellm chains the returned kwargs; mutating the input in place as well
    would make the chaining order matter."""
    kwargs = {"model": "anthropic/claude-sonnet-4-5-20250929",
              "temperature": 0.2, "top_p": 0.9}
    await _deployment(hook, kwargs)
    assert kwargs["top_p"] == 0.9, "input must be left alone"


async def test_deployment_dispatch_survives_a_broken_repair(hook, monkeypatch):
    """A bug in a repair must not fail every request to the deployment."""
    def boom(*_args, **_kwargs):
        raise RuntimeError("repair is broken")

    monkeypatch.setattr(hook, "_remove_top_p_for_anthropic", boom)
    kwargs = {"model": "anthropic/claude-sonnet-4-5-20250929", "top_p": 0.9}
    assert await _deployment(hook, kwargs) is None
