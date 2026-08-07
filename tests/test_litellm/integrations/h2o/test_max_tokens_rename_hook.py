"""Tests for the h2o MaxTokensRenameHook.

Focus: Azure 2025+ deployments reject ``max_tokens``, so ``launch_litellm.py``
configures them with ``max_completion_tokens`` plus
``additional_drop_params: ["max_tokens"]``. That drop stops the Azure 400 but
silently DISCARDS the caller's limit, so a client asking for 50 output tokens
receives the deployment ceiling instead (measured: 50 requested, 1666
returned). This hook renames the field before the drop applies.

The hook is deployment-level on purpose. A group-level predicate would break
MIXED model groups, which our generated config really contains: ``agent_auto``
spans anthropic, azure, bedrock, gemini, mistral and openrouter under one
model_name. See test_mixed_model_group_* below.
"""

import pytest

from litellm.integrations.h2o.litellm_max_tokens_rename_hook import (
    MaxTokensRenameHook,
)


@pytest.fixture
def hook():
    return MaxTokensRenameHook()


# kwargs as they arrive at async_pre_call_deployment_hook, i.e. AFTER the router
# merged the selected deployment's litellm_params. Values verified against a
# live proxy running a mixed agent_auto group.
AZURE_KWARGS = {
    "model": "azure/gpt-4o-mini",
    "max_tokens": 50,
    "max_completion_tokens": 16384,
    "additional_drop_params": ["max_tokens"],
}
BEDROCK_KWARGS = {
    "model": "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "max_tokens": 50,
    "max_completion_tokens": None,
    "additional_drop_params": None,
}


async def _run(hook, kwargs):
    return await hook.async_pre_call_deployment_hook(kwargs, "acompletion")


@pytest.mark.asyncio
async def test_renames_on_a_completion_tokens_deployment(hook):
    """The regression this hook exists for: the caller's 50 must survive as
    max_completion_tokens instead of being dropped."""
    out = await _run(hook, dict(AZURE_KWARGS))
    assert out is not None
    assert "max_tokens" not in out
    assert out["max_completion_tokens"] == 50


@pytest.mark.asyncio
async def test_leaves_a_max_tokens_native_deployment_untouched(hook):
    """Anthropic/Bedrock/vLLM accept max_tokens; renaming there breaks them."""
    out = await _run(hook, dict(BEDROCK_KWARGS))
    assert out is None, "must not modify a max_tokens-native deployment"


@pytest.mark.asyncio
async def test_mixed_model_group_azure_member_is_renamed(hook):
    """Both members below share one model_name (agent_auto). The azure member
    must be renamed..."""
    out = await _run(hook, dict(AZURE_KWARGS, model="azure/gpt-4o-mini"))
    assert out["max_completion_tokens"] == 50
    assert "max_tokens" not in out


@pytest.mark.asyncio
async def test_mixed_model_group_native_member_keeps_max_tokens(hook):
    """...and the bedrock member of the SAME group must keep max_tokens. A
    group-level predicate returned true for both and corrupted this one."""
    out = await _run(hook, dict(BEDROCK_KWARGS))
    assert out is None
    # and the caller's kwargs are genuinely unchanged
    kwargs = dict(BEDROCK_KWARGS)
    await _run(hook, kwargs)
    assert kwargs["max_tokens"] == 50
    assert kwargs.get("max_completion_tokens") is None


@pytest.mark.asyncio
async def test_detects_completion_tokens_via_drop_params_alone(hook):
    """A deployment signalling intent only through additional_drop_params (no
    configured ceiling) still gets the rename."""
    out = await _run(hook, {"model": "azure/gpt-4o-mini", "max_tokens": 128,
                            "additional_drop_params": ["max_tokens"]})
    assert out["max_completion_tokens"] == 128
    assert "max_tokens" not in out


@pytest.mark.asyncio
async def test_never_raises_a_configured_ceiling(hook):
    """A deployment ceiling tighter than the request must win."""
    out = await _run(hook, {"model": "azure/gpt-4o-mini", "max_tokens": 99999,
                            "max_completion_tokens": 4096,
                            "additional_drop_params": ["max_tokens"]})
    assert out["max_completion_tokens"] == 4096


@pytest.mark.parametrize("bad", [0, -1, None, "50", True])
@pytest.mark.asyncio
async def test_ignores_non_positive_int_max_tokens(hook, bad):
    """0/null/string/bool must never become a max_completion_tokens that
    truncates every response."""
    out = await _run(hook, dict(AZURE_KWARGS, max_tokens=bad))
    assert out is None


@pytest.mark.asyncio
async def test_no_max_tokens_is_a_noop(hook):
    out = await _run(hook, {"model": "azure/gpt-4o-mini",
                            "max_completion_tokens": 16384,
                            "additional_drop_params": ["max_tokens"]})
    assert out is None


@pytest.mark.asyncio
async def test_plain_deployment_with_no_params_is_a_noop(hook):
    """A direct litellm.completion call with no deployment params attached."""
    out = await _run(hook, {"model": "gpt-4o-mini", "max_tokens": 50})
    assert out is None


@pytest.mark.asyncio
async def test_malformed_drop_params_never_propagates(hook):
    """A non-list additional_drop_params must not fail the request."""
    out = await _run(hook, {"model": "azure/gpt-4o-mini", "max_tokens": 50,
                            "additional_drop_params": "max_tokens"})
    assert out is None


@pytest.mark.asyncio
async def test_other_kwargs_are_preserved(hook):
    """The hook returns the FULL kwargs dict; the dispatcher replaces kwargs
    wholesale with whatever comes back, so dropping a key here would drop it
    from the upstream request."""
    kwargs = dict(AZURE_KWARGS, messages=[{"role": "user", "content": "hi"}],
                  temperature=0.5, stream=True)
    out = await _run(hook, kwargs)
    assert out["messages"] == [{"role": "user", "content": "hi"}]
    assert out["temperature"] == 0.5
    assert out["stream"] is True
    assert out["model"] == "azure/gpt-4o-mini"
