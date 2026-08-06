"""Tests for litellm.integrations.h2o.litellm_oauth_auth_hook.

The load-bearing assertions here are the NEGATIVE ones: a model that does not
declare `h2o_oauth` must come out of this hook completely untouched, no matter
what state the router is in. The hook runs on every request through the proxy,
so a regression there affects every model in every deployment.
"""

import os
import sys
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.integrations.h2o.litellm_oauth_auth_hook import OAuthAuthHook  # noqa: E402
from litellm.proxy_auth.async_oauth2 import AsyncOAuth2ClientCredential  # noqa: E402
from litellm.proxy_auth.credentials import AccessToken  # noqa: E402

OAUTH_BLOCK = {
    "token_url": "https://idp.example.com/token",
    "client_id": "gw-client",
    "client_secret": "shhh",
}


def _deployment(
    model_name: str,
    *,
    oauth: Optional[Dict[str, Any]] = None,
    litellm_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    model_info: Dict[str, Any] = {"id": f"{model_name}-id"}
    if oauth is not None:
        model_info["h2o_oauth"] = oauth
    return {
        "model_name": model_name,
        "litellm_params": {"model": f"openai/{model_name}", "api_key": "static-key", **(litellm_params or {})},
        "model_info": model_info,
    }


def _router(deployments: List[Dict[str, Any]]) -> MagicMock:
    router = MagicMock()
    router.model_list = deployments
    return router


def _patch_router(router: Any):
    module = MagicMock()
    module.llm_router = router
    return patch.dict(sys.modules, {"litellm.proxy.proxy_server": module})


def _fake_token(token: str = "minted-token") -> AccessToken:
    return AccessToken(token=token, expires_on=2**31)


async def _run(hook: OAuthAuthHook, data: Dict[str, Any]) -> Dict[str, Any]:
    return await hook.async_pre_call_hook(
        user_api_key_dict=MagicMock(), cache=MagicMock(), data=data, call_type="completion"
    )


# --------------------------------------------------------------------------
# Opt-out: unconfigured models must be untouched
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_model_without_oauth_config_is_untouched():
    hook = OAuthAuthHook()
    data = {"model": "plain-model", "messages": [{"role": "user", "content": "hi"}]}
    with _patch_router(_router([_deployment("plain-model")])):
        out = await _run(hook, data)
    assert out == {"model": "plain-model", "messages": [{"role": "user", "content": "hi"}]}
    assert "extra_headers" not in out


@pytest.mark.asyncio
async def test_unknown_model_is_untouched():
    hook = OAuthAuthHook()
    data = {"model": "never-registered"}
    with _patch_router(_router([_deployment("something-else")])):
        out = await _run(hook, data)
    assert out == {"model": "never-registered"}


@pytest.mark.asyncio
async def test_missing_router_is_untouched():
    hook = OAuthAuthHook()
    data = {"model": "any"}
    with _patch_router(None):
        out = await _run(hook, data)
    assert out == {"model": "any"}


@pytest.mark.asyncio
async def test_no_model_in_request_is_untouched():
    hook = OAuthAuthHook()
    assert await _run(hook, {}) == {}


@pytest.mark.asyncio
async def test_corrupt_router_state_cannot_break_unconfigured_traffic():
    """A router that raises on access must not fail requests."""
    hook = OAuthAuthHook()
    broken = MagicMock()
    type(broken).model_list = property(lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
    data = {"model": "plain-model"}
    with _patch_router(broken):
        out = await _run(hook, data)
    assert out == {"model": "plain-model"}


@pytest.mark.asyncio
async def test_malformed_model_info_is_untouched():
    hook = OAuthAuthHook()
    deployment = _deployment("weird")
    deployment["model_info"] = "not-a-dict"
    with _patch_router(_router([deployment])):
        out = await _run(hook, {"model": "weird"})
    assert out == {"model": "weird"}


# --------------------------------------------------------------------------
# Opt-in: header injection
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_configured_model_gets_bearer_header():
    hook = OAuthAuthHook()
    with _patch_router(_router([_deployment("gw-model", oauth=OAUTH_BLOCK)])):
        with patch.object(AsyncOAuth2ClientCredential, "get_token", return_value=_fake_token()):
            out = await _run(hook, {"model": "gw-model", "messages": []})
    assert out["extra_headers"]["Authorization"] == "Bearer minted-token"


@pytest.mark.asyncio
async def test_custom_header_name_and_scheme():
    hook = OAuthAuthHook()
    oauth = {**OAUTH_BLOCK, "header_name": "X-Gateway-Token", "header_scheme": ""}
    with _patch_router(_router([_deployment("gw-model", oauth=oauth)])):
        with patch.object(AsyncOAuth2ClientCredential, "get_token", return_value=_fake_token("raw-token")):
            out = await _run(hook, {"model": "gw-model"})
    assert out["extra_headers"]["X-Gateway-Token"] == "raw-token"
    assert "Authorization" not in out["extra_headers"]


@pytest.mark.asyncio
async def test_static_deployment_headers_are_preserved():
    """Regression: request kwargs REPLACE litellm_params, so a naive
    data['extra_headers'] = {...} silently drops gateway routing headers."""
    hook = OAuthAuthHook()
    deployment = _deployment(
        "gw-model",
        oauth=OAUTH_BLOCK,
        litellm_params={"extra_headers": {"x-gateway-route": "route-123"}},
    )
    with _patch_router(_router([deployment])):
        with patch.object(AsyncOAuth2ClientCredential, "get_token", return_value=_fake_token()):
            out = await _run(hook, {"model": "gw-model"})
    assert out["extra_headers"]["x-gateway-route"] == "route-123"
    assert out["extra_headers"]["Authorization"] == "Bearer minted-token"


@pytest.mark.asyncio
async def test_request_headers_are_preserved_but_auth_wins():
    hook = OAuthAuthHook()
    deployment = _deployment(
        "gw-model", oauth=OAUTH_BLOCK, litellm_params={"extra_headers": {"a": "from-deployment"}}
    )
    data = {"model": "gw-model", "extra_headers": {"b": "from-request", "Authorization": "Bearer stale"}}
    with _patch_router(_router([deployment])):
        with patch.object(AsyncOAuth2ClientCredential, "get_token", return_value=_fake_token()):
            out = await _run(hook, data)
    assert out["extra_headers"]["a"] == "from-deployment"
    assert out["extra_headers"]["b"] == "from-request"
    assert out["extra_headers"]["Authorization"] == "Bearer minted-token"


@pytest.mark.asyncio
async def test_api_key_is_never_modified():
    """Injecting a rotating api_key would make the router upsert a new
    deployment per token; the hook must only ever touch headers."""
    hook = OAuthAuthHook()
    data = {"model": "gw-model"}
    with _patch_router(_router([_deployment("gw-model", oauth=OAUTH_BLOCK)])):
        with patch.object(AsyncOAuth2ClientCredential, "get_token", return_value=_fake_token()):
            out = await _run(hook, data)
    assert "api_key" not in out
    assert "api_base" not in out
    assert "base_url" not in out


@pytest.mark.asyncio
async def test_credential_is_reused_across_requests():
    hook = OAuthAuthHook()
    with _patch_router(_router([_deployment("gw-model", oauth=OAUTH_BLOCK)])):
        with patch.object(AsyncOAuth2ClientCredential, "get_token", return_value=_fake_token()):
            await _run(hook, {"model": "gw-model"})
            await _run(hook, {"model": "gw-model"})
    assert len(hook.registry) == 1


# --------------------------------------------------------------------------
# Opt-in failure policy: fail closed with an actionable message
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalid_config_fails_closed_naming_the_model():
    hook = OAuthAuthHook()
    with _patch_router(_router([_deployment("gw-model", oauth={"client_id": "no-token-url"})])):
        with pytest.raises(Exception) as exc:
            await _run(hook, {"model": "gw-model"})
    message = str(getattr(exc.value, "detail", exc.value))
    assert "gw-model" in message
    assert "token_url" in message, "the most fundamental missing field is reported first"


@pytest.mark.asyncio
async def test_missing_credential_is_reported_once_the_basics_are_present():
    hook = OAuthAuthHook()
    oauth = {"token_url": "https://idp.example.com/token", "client_id": "c"}
    with _patch_router(_router([_deployment("gw-model", oauth=oauth)])):
        with pytest.raises(Exception) as exc:
            await _run(hook, {"model": "gw-model"})
    message = str(getattr(exc.value, "detail", exc.value))
    assert "client_secret is required" in message


@pytest.mark.asyncio
async def test_token_failure_fails_closed_and_does_not_leak_secrets():
    from litellm.proxy_auth.async_oauth2 import OAuth2TokenError

    hook = OAuthAuthHook()
    with _patch_router(_router([_deployment("gw-model", oauth=OAUTH_BLOCK)])):
        with patch.object(
            AsyncOAuth2ClientCredential,
            "get_token",
            side_effect=OAuth2TokenError("token endpoint returned 401: invalid_client"),
        ):
            with pytest.raises(Exception) as exc:
                await _run(hook, {"model": "gw-model"})
    message = str(getattr(exc.value, "detail", exc.value))
    assert "gw-model" in message
    assert "401" in message
    assert "shhh" not in message


@pytest.mark.asyncio
async def test_one_broken_oauth_model_does_not_affect_a_healthy_plain_model():
    hook = OAuthAuthHook()
    deployments = [
        _deployment("broken-gw", oauth={"client_id": "missing-token-url"}),
        _deployment("plain-model"),
    ]
    with _patch_router(_router(deployments)):
        out = await _run(hook, {"model": "plain-model"})
        assert out == {"model": "plain-model"}
        with pytest.raises(Exception):
            await _run(hook, {"model": "broken-gw"})
