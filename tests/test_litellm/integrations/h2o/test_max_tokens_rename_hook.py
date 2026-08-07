"""Tests for the h2o MaxTokensRenameHook.

Focus: Azure 2025+ deployments reject ``max_tokens``, so
``launch_litellm.py`` configures them with ``max_completion_tokens`` plus
``additional_drop_params: ["max_tokens"]``. That drop stops the Azure 400 but
silently DISCARDS the caller's limit, so a client asking for 50 output tokens
receives the deployment ceiling instead (measured: 50 requested, 1666
returned). This hook renames the field before the drop applies so the caller's
intent survives, while leaving ``max_tokens``-native deployments untouched.
"""

import pytest

from litellm.integrations.h2o.litellm_max_tokens_rename_hook import (
    MaxTokensRenameHook,
)


@pytest.fixture
def hook():
    return MaxTokensRenameHook()


class _FakeRouter:
    def __init__(self, model_list):
        self.model_list = model_list


def _install_router(monkeypatch, model_list):
    """Point the hook's ``llm_router`` lookup at a fake deployment table."""
    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(proxy_server, "llm_router", _FakeRouter(model_list), raising=False)


AZURE_DEPLOYMENT = [{
    "model_name": "gpt-4o-mini",
    "litellm_params": {
        "model": "azure/gpt-4o-mini",
        "max_completion_tokens": 16384,
        "additional_drop_params": ["max_tokens"],
    },
}]

BEDROCK_DEPLOYMENT = [{
    "model_name": "claude-opus-4-6",
    "litellm_params": {
        "model": "bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0",
        "max_tokens": 64000,
    },
}]


async def _run(hook, data):
    return await hook.async_pre_call_hook(
        user_api_key_dict=None, cache=None, data=data, call_type="completion"
    )


@pytest.mark.asyncio
async def test_renames_max_tokens_on_completion_tokens_deployment(hook, monkeypatch):
    """The regression this hook exists for: the caller's 50 must survive as
    max_completion_tokens instead of being dropped."""
    _install_router(monkeypatch, AZURE_DEPLOYMENT)
    out = await _run(hook, {"model": "gpt-4o-mini", "max_tokens": 50})
    assert "max_tokens" not in out
    assert out["max_completion_tokens"] == 50


@pytest.mark.asyncio
async def test_leaves_max_tokens_native_deployment_untouched(hook, monkeypatch):
    """Anthropic/Bedrock/vLLM deployments accept max_tokens; renaming there
    would break them."""
    _install_router(monkeypatch, BEDROCK_DEPLOYMENT)
    out = await _run(hook, {"model": "claude-opus-4-6", "max_tokens": 50})
    assert out["max_tokens"] == 50
    assert "max_completion_tokens" not in out


@pytest.mark.asyncio
async def test_detects_completion_tokens_via_drop_params_alone(hook, monkeypatch):
    """A deployment that only signals intent through additional_drop_params
    (no configured ceiling) still gets the rename."""
    _install_router(monkeypatch, [{
        "model_name": "gpt-4o-mini",
        "litellm_params": {"model": "azure/gpt-4o-mini",
                           "additional_drop_params": ["max_tokens"]},
    }])
    out = await _run(hook, {"model": "gpt-4o-mini", "max_tokens": 128})
    assert out == {"model": "gpt-4o-mini", "max_completion_tokens": 128}


@pytest.mark.asyncio
async def test_keeps_the_tighter_value_when_caller_sends_both(hook, monkeypatch):
    """A caller sending both fields already expressed the tighter intent; the
    rename must not raise their ceiling."""
    _install_router(monkeypatch, AZURE_DEPLOYMENT)
    out = await _run(hook, {"model": "gpt-4o-mini", "max_tokens": 50,
                            "max_completion_tokens": 4096})
    assert "max_tokens" not in out
    assert out["max_completion_tokens"] == 50

    out = await _run(hook, {"model": "gpt-4o-mini", "max_tokens": 4096,
                            "max_completion_tokens": 50})
    assert out["max_completion_tokens"] == 50


@pytest.mark.parametrize("bad", [0, -1, None, "50", True])
@pytest.mark.asyncio
async def test_ignores_non_positive_int_max_tokens(hook, monkeypatch, bad):
    """0/null/string/bool must never become a max_completion_tokens that
    truncates every response."""
    _install_router(monkeypatch, AZURE_DEPLOYMENT)
    out = await _run(hook, {"model": "gpt-4o-mini", "max_tokens": bad})
    assert "max_completion_tokens" not in out
    assert out["max_tokens"] is bad


@pytest.mark.asyncio
async def test_no_max_tokens_is_a_noop(hook, monkeypatch):
    _install_router(monkeypatch, AZURE_DEPLOYMENT)
    out = await _run(hook, {"model": "gpt-4o-mini", "messages": []})
    assert out == {"model": "gpt-4o-mini", "messages": []}


@pytest.mark.asyncio
async def test_unknown_model_is_a_noop(hook, monkeypatch):
    """A model with no matching deployment must not be rewritten."""
    _install_router(monkeypatch, AZURE_DEPLOYMENT)
    out = await _run(hook, {"model": "some-other-model", "max_tokens": 50})
    assert out["max_tokens"] == 50
    assert "max_completion_tokens" not in out


@pytest.mark.asyncio
async def test_router_unavailable_is_a_noop(hook, monkeypatch):
    """Outside the proxy context there is no router; leave the request alone
    rather than guessing."""
    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(proxy_server, "llm_router", None, raising=False)
    out = await _run(hook, {"model": "gpt-4o-mini", "max_tokens": 50})
    assert out["max_tokens"] == 50


@pytest.mark.asyncio
async def test_router_error_never_propagates(hook, monkeypatch):
    """A malformed deployment table must not fail the request."""
    class _Exploding:
        @property
        def model_list(self):
            raise RuntimeError("boom")

    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(proxy_server, "llm_router", _Exploding(), raising=False)
    out = await _run(hook, {"model": "gpt-4o-mini", "max_tokens": 50})
    assert out["max_tokens"] == 50
