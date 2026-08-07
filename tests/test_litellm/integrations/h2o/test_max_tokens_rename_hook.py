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
    """A non-list additional_drop_params must not raise. On a non-Azure model
    it is the only possible signal, so the request is left alone."""
    out = await _run(hook, {"model": "openrouter/some-model", "max_tokens": 50,
                            "additional_drop_params": "max_tokens"})
    assert out is None


@pytest.mark.asyncio
async def test_azure_route_with_a_ceiling_triggers_the_rename(hook):
    """Reasoning Azure entries get max_completion_tokens but are exempt from
    the drop, so route + ceiling has to be recognised without one."""
    out = await _run(hook, {"model": "azure/gpt-5-mini", "max_tokens": 50,
                            "max_completion_tokens": 16384})
    assert out["max_completion_tokens"] == 50
    assert "max_tokens" not in out


@pytest.mark.asyncio
async def test_azure_deployment_that_wants_max_tokens_is_untouched(hook):
    """An Azure entry with `use_max_completion_tokens: false`, which is how an
    older api_version deployment is expressed, generates max_tokens as the
    ceiling with no max_completion_tokens and no drop. Whether Azure wants
    max_tokens or max_completion_tokens is a function of api_version, not of
    being Azure, so treating the route alone as sufficient would override a
    deliberate operator choice on exactly these deployments."""
    out = await _run(hook, {"model": "azure/gpt-4o-mini", "max_tokens": 50,
                            "api_version": "2024-02-01"})
    assert out is None


@pytest.mark.asyncio
async def test_azure_route_without_a_ceiling_or_drop_is_untouched(hook):
    """Same rule stated directly on the predicate: route alone is not enough."""
    out = await _run(hook, {"model": "azure/gpt-4o-mini", "max_tokens": 50})
    assert out is None


@pytest.mark.asyncio
async def test_non_azure_ceiling_alone_does_not_trigger_the_rename(hook):
    """convert_model_to_litellm_config sets max_completion_tokens for EVERY
    reasoning model, Azure or not, and only the Azure branch adds the drop.
    Renaming on a non-Azure provider would strip max_tokens and leave the
    request with no ceiling at all, which is worse than the bug being fixed."""
    out = await _run(hook, {"model": "openrouter/grok-4", "max_tokens": 50,
                            "max_completion_tokens": 16384})
    assert out is None


@pytest.mark.asyncio
async def test_native_deployment_with_caller_supplied_ceiling_is_untouched(hook):
    """A caller sending BOTH fields to a max_tokens-native deployment must not
    have max_tokens stripped."""
    out = await _run(hook, dict(BEDROCK_KWARGS, max_completion_tokens=4096))
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


# ---------------------------------------------------------------------------
# Integration: prove litellm ITSELF dispatches the hook on a real completion.
#
# The tests above call async_pre_call_deployment_hook directly, which cannot
# catch the hook being registered but never invoked. The dispatch does not live
# in router.py or proxy/utils.py (a static trace of those two misses it); it is
# in the @client decorator that wraps litellm.completion / litellm.acompletion,
# at litellm/utils.py async_pre_call_deployment_hook -> wrapper_async. Since the
# router calls litellm.acompletion(**input_kwargs), every routed chat completion
# passes through it.
#
# These drive a real litellm.acompletion and assert the rewrite reached the
# provider-bound kwargs, so if upstream ever moves or drops that dispatch the
# suite fails instead of the hook going quietly inert.
# ---------------------------------------------------------------------------


async def _drive_acompletion(**overrides):
    """Run a real litellm.acompletion with the rename hook plus a recorder.

    The recorder is registered AFTER the rename hook: the dispatcher feeds each
    callback the accumulated kwargs, so the recorder observes what the rename
    hook actually produced inside litellm's own pipeline.

    NB the recorder subclasses CustomLogger directly. CustomLogger defines
    async_pre_call_deployment_hook as a no-op, so a mixin ordered after it in
    the MRO is silently shadowed and the recorder never records.
    """
    import litellm
    from litellm.integrations.custom_logger import CustomLogger

    class Recorder(CustomLogger):
        def __init__(self):
            super().__init__()
            self.seen = None

        async def async_pre_call_deployment_hook(self, kwargs, call_type):
            self.seen = dict(kwargs)
            return None

    recorder = Recorder()
    previous = litellm.callbacks
    litellm.callbacks = [MaxTokensRenameHook(), recorder]
    try:
        params = dict(
            model="azure/gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            api_key="dummy",
            api_version="2025-04-01-preview",
            api_base="https://example.openai.azure.com",
            mock_response="ok",
        )
        params.update(overrides)
        await litellm.acompletion(**params)
    finally:
        litellm.callbacks = previous
    return recorder.seen


@pytest.mark.asyncio
async def test_litellm_dispatches_the_hook_and_applies_the_rename():
    seen = await _drive_acompletion(
        max_tokens=50,
        max_completion_tokens=16384,
        additional_drop_params=["max_tokens"],
    )
    assert seen is not None, (
        "litellm never dispatched async_pre_call_deployment_hook; the hook "
        "would be registered but inert on the chat completion path"
    )
    assert "max_tokens" not in seen, seen
    assert seen["max_completion_tokens"] == 50, seen


@pytest.mark.asyncio
async def test_litellm_dispatch_leaves_a_max_tokens_native_deployment_alone():
    """Same real pipeline, a deployment with neither signal: the request must
    reach the provider with max_tokens intact."""
    seen = await _drive_acompletion(
        model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        max_tokens=50,
    )
    assert seen is not None
    assert seen["max_tokens"] == 50, seen
    assert seen.get("max_completion_tokens") is None, seen


@pytest.mark.asyncio
async def test_anthropic_messages_survives_the_rename_hook():
    """litellm.anthropic_messages backs /v1/messages and bridges Azure, and it
    declares max_tokens as a REQUIRED parameter. The deployment-hook dispatch
    is not chat-specific, so popping max_tokens there made litellm's own
    @client wrapper raise
    `TypeError: anthropic_messages() missing 1 required positional argument`
    on the next `await original_function(...)`. That is outside this hook's
    try/except, and the traceback never names the hook.
    """
    import litellm

    previous = litellm.callbacks
    litellm.callbacks = [MaxTokensRenameHook()]
    try:
        await litellm.anthropic_messages(
            model="azure/gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=50,
            max_completion_tokens=16384,        # merged in by the router
            additional_drop_params=["max_tokens"],
            api_key="dummy",
            api_version="2025-04-01-preview",
            api_base="https://example.openai.azure.com",
            mock_response="ok",
        )
    finally:
        litellm.callbacks = previous


@pytest.mark.asyncio
async def test_non_chat_call_types_are_skipped():
    """Same guard at the unit level, for the other entrypoints that reach this
    dispatch: /v1/completions has no max_completion_tokens, and embeddings
    have no notion of one."""
    hook = MaxTokensRenameHook()
    for call_type in ("anthropic_messages", "atext_completion", "aembedding", None):
        out = await hook.async_pre_call_deployment_hook(dict(AZURE_KWARGS), call_type)
        assert out is None, f"{call_type} must not be rewritten"
