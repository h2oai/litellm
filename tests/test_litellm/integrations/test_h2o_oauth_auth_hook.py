"""Tests for litellm.integrations.h2o.litellm_oauth_auth_hook.

The load-bearing assertions here are the NEGATIVE ones: a model that does not
declare `h2o_oauth` must come out of this hook completely untouched, no matter
what state the router is in. The hook runs on every request through the proxy,
so a regression there affects every model in every deployment.
"""

import json
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


# The hook runs at async_pre_call_deployment_hook, i.e. AFTER the router has
# picked a deployment, so there is no router to look up: litellm hands it the
# selected deployment's model_info directly. These helpers therefore SIMULATE the
# selection -- `_router(...)` records the group and `_run` resolves the entry the
# router would have handed over -- rather than mocking litellm.proxy.proxy_server.
_GROUP: List[Dict[str, Any]] = []


def _router(deployments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return deployments


class _patch_router:
    """Record the deployment group, and expose it as the proxy router.

    The router is still needed -- not to CHOOSE a deployment (litellm has already
    done that by this dispatch point) but to read the chosen deployment's static
    extra_headers back, which litellm drops when the request supplies its own. The
    lookup is by model_info["id"], so it cannot resolve to a sibling.
    """

    def __init__(self, deployments: Any):
        self._deployments = deployments if isinstance(deployments, list) else []
        self._patch = None

    def __enter__(self):
        _GROUP.clear()
        if self._deployments:
            _GROUP.extend(self._deployments)
        module = MagicMock()
        router = MagicMock()
        router.model_list = list(self._deployments)
        module.llm_router = router
        self._patch = patch.dict(sys.modules, {"litellm.proxy.proxy_server": module})
        self._patch.start()
        return self

    def __exit__(self, *exc):
        _GROUP.clear()
        if self._patch is not None:
            self._patch.stop()
        return False


def _fake_token(token: str = "minted-token") -> AccessToken:
    return AccessToken(token=token, expires_on=2**31)


def _selected(model: str) -> Optional[Dict[str, Any]]:
    for deployment in _GROUP:
        if deployment.get("model_name") == model:
            return deployment
    return None


async def _run(
    hook: OAuthAuthHook,
    data: Dict[str, Any],
    deployment: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Drive the deployment hook the way litellm drives it.

    kwargs at that point are the request merged with the SELECTED deployment's
    litellm_params, plus its model_info. A None return means "unchanged", so the
    original kwargs are handed back for the caller's assertions.
    """
    if deployment is None:
        deployment = _selected(data.get("model", ""))
    kwargs: Dict[str, Any] = dict(data)
    injected = set()
    if deployment is not None:
        kwargs["model_info"] = deployment.get("model_info")
        injected.add("model_info")
        for key, value in (deployment.get("litellm_params") or {}).items():
            if key == "model" or key in kwargs:
                continue
            kwargs[key] = value
            injected.add(key)
    before = dict(kwargs)
    result = await hook.async_pre_call_deployment_hook(kwargs, "acompletion")
    out = dict(kwargs if result is None else result)
    # Report the HOOK's effect only. The keys above are what litellm itself merges
    # in before this dispatch point, not something the hook did, so leaving them
    # in would make every "untouched" assertion fail for the wrong reason. A key
    # the hook actually changed (e.g. extra_headers seeded from the deployment)
    # keeps its new value and stays visible.
    for key in injected:
        if key in out and out[key] == before.get(key):
            del out[key]
    return out


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


# --------------------------------------------------------------------------
# The token must reach ONLY the deployment whose config minted it.
#
# The hook used to run at async_pre_call_hook, before the router chose a
# deployment, so it read the FIRST deployment matching the model GROUP name and
# wrote the header into a request the router could send anywhere in that group.
# Measured with two loopback servers and simple-shuffle: gateway A's live bearer
# went to deployment B, which had no OAuth config, a different vendor and its own
# api_key -- credential exfiltration to a third party, plus B broke because the
# injected Authorization overrode its key.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sibling_deployment_without_oauth_gets_no_token():
    """The mixed-group case that leaked. Same model_name, two deployments."""
    with_oauth = _deployment("grp", oauth=OAUTH_BLOCK)
    with_oauth["model_info"]["id"] = "A"
    without = _deployment("grp")
    without["model_info"]["id"] = "B"
    hook = OAuthAuthHook()
    data = {"model": "grp", "messages": [{"role": "user", "content": "hi"}]}

    with _patch_router(_router([with_oauth, without])):
        with patch.object(AsyncOAuth2ClientCredential, "get_token", return_value=_fake_token()):
            # The router selected B -- the one with no OAuth config.
            out = await _run(hook, data, deployment=without)
    assert "extra_headers" not in out, (
        f"a deployment with no h2o_oauth must receive no token, got {out}"
    )

    with _patch_router(_router([with_oauth, without])):
        with patch.object(AsyncOAuth2ClientCredential, "get_token", return_value=_fake_token()):
            out = await _run(hook, data, deployment=with_oauth)
    assert out["extra_headers"]["Authorization"] == "Bearer minted-token"


@pytest.mark.asyncio
async def test_two_gateways_in_one_group_each_get_their_own_token():
    """Neither deployment may be served the other's credential."""
    a = _deployment("grp", oauth={**OAUTH_BLOCK, "client_id": "client-a"})
    a["model_info"]["id"] = "A"
    b = _deployment("grp", oauth={**OAUTH_BLOCK, "client_id": "client-b"})
    b["model_info"]["id"] = "B"
    hook = OAuthAuthHook()
    seen = {}

    async def fake_get_token(self, *, force_refresh: bool = False):
        seen["client_id"] = self.config.client_id
        return _fake_token(f"token-for-{self.config.client_id}")

    data = {"model": "grp", "messages": [{"role": "user", "content": "hi"}]}
    for deployment, expected in ((a, "client-a"), (b, "client-b")):
        with _patch_router(_router([a, b])):
            with patch.object(AsyncOAuth2ClientCredential, "get_token", fake_get_token):
                out = await _run(hook, data, deployment=deployment)
        assert seen["client_id"] == expected
        assert out["extra_headers"]["Authorization"] == f"Bearer token-for-{expected}"


@pytest.mark.asyncio
async def test_the_hook_does_not_mutate_the_caller_kwargs():
    """litellm chains each callback's returned kwargs, so in-place mutation as
    well as returning a copy would make the chaining order matter."""
    deployment = _deployment("gw-model", oauth=OAUTH_BLOCK)
    hook = OAuthAuthHook()
    kwargs = {"model": "gw-model", "model_info": deployment["model_info"]}
    snapshot = dict(kwargs)
    with _patch_router(_router([deployment])):
        with patch.object(AsyncOAuth2ClientCredential, "get_token", return_value=_fake_token()):
            result = await hook.async_pre_call_deployment_hook(kwargs, "acompletion")
    assert result is not None and "extra_headers" in result
    assert kwargs == snapshot, "the input kwargs must be left alone"


@pytest.mark.asyncio
async def test_unconfigured_model_returns_none_not_a_copy():
    """None is litellm's "unchanged" contract; returning a copy for every request
    would make the hook look like it modified something."""
    deployment = _deployment("plain-model")
    hook = OAuthAuthHook()
    with _patch_router(_router([deployment])):
        result = await hook.async_pre_call_deployment_hook(
            {"model": "plain-model", "model_info": deployment["model_info"]}, "acompletion"
        )
    assert result is None


# --------------------------------------------------------------------------
# Secrets must not survive into anything that gets logged, and an auth failure
# must not be routed around.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_oauth_block_is_stripped_from_model_info():
    """model_info IS a litellm param, so it reaches litellm_params and the router
    metadata -- both handed to every logging callback. Measured before the fix at
    .litellm_params.model_info.h2o_oauth.client_secret. OAuth2Config accepts a
    literal secret, so an operator who inlines one had it logged per request."""
    deployment = _deployment("gw-model", oauth={**OAUTH_BLOCK, "client_secret": "literal-secret"})
    hook = OAuthAuthHook()
    with _patch_router(_router([deployment])):
        with patch.object(AsyncOAuth2ClientCredential, "get_token", return_value=_fake_token()):
            result = await hook.async_pre_call_deployment_hook(
                {"model": "openai/gw-model", "model_info": deployment["model_info"]},
                "acompletion",
            )
    assert result is not None
    assert "h2o_oauth" not in (result.get("model_info") or {}), (
        f"the oauth block must not survive into logged kwargs: {result.get('model_info')}"
    )
    # The deployment id must survive -- it is what identifies the deployment.
    assert (result.get("model_info") or {}).get("id")
    assert "literal-secret" not in json.dumps(result, default=str)


@pytest.mark.asyncio
async def test_an_oauth_failure_is_not_retryable_or_fallbackable():
    """At the deployment dispatch point the raise happens INSIDE the
    per-deployment call, so router retries and fallbacks apply. Measured with an
    IdP returning 503 and a fallback group configured: the client got a 200 from a
    DIFFERENT vendor and the operator got no signal. litellm.AuthenticationError
    is in litellm's non-retryable set."""
    import litellm

    deployment = _deployment("gw-model", oauth=OAUTH_BLOCK)
    hook = OAuthAuthHook()

    async def boom(self, *, force_refresh: bool = False):
        raise OAuth2TokenError("token endpoint returned 503")

    with _patch_router(_router([deployment])):
        with patch.object(AsyncOAuth2ClientCredential, "get_token", boom):
            with pytest.raises(litellm.exceptions.AuthenticationError):
                await hook.async_pre_call_deployment_hook(
                    {"model": "openai/gw-model", "model_info": deployment["model_info"]},
                    "acompletion",
                )


@pytest.mark.asyncio
async def test_duplicate_deployment_ids_restore_no_static_headers():
    """The router honours an explicitly configured model_info.id and does not
    reject duplicates, so "an id is unambiguous" holds only for router-generated
    ids. Returning the first match handed one deployment's static headers -- which
    routinely carry gateway subscription keys -- to another."""
    a = _deployment("grp", oauth=OAUTH_BLOCK, litellm_params={"extra_headers": {"X-Which": "A"}})
    a["model_info"]["id"] = "same-id"
    b = _deployment("grp", litellm_params={"extra_headers": {"X-Which": "B"}})
    b["model_info"]["id"] = "same-id"
    hook = OAuthAuthHook()
    with _patch_router(_router([a, b])):
        with patch.object(AsyncOAuth2ClientCredential, "get_token", return_value=_fake_token()):
            result = await hook.async_pre_call_deployment_hook(
                {"model": "openai/grp", "model_info": a["model_info"]}, "acompletion"
            )
    assert result is not None
    assert "X-Which" not in result["extra_headers"], (
        f"ambiguous id must restore nothing, got {result['extra_headers']}"
    )
    assert result["extra_headers"]["Authorization"] == "Bearer minted-token"


@pytest.mark.asyncio
async def test_the_error_names_the_group_the_operator_configured():
    """kwargs["model"] here is the UPSTREAM model string, which appears nowhere
    near the operator's h2o_oauth block."""
    import litellm

    deployment = _deployment("public-gw-name", oauth={"token_url": "https://idp/token"})
    deployment["model_info"]["id"] = "dep-7"
    hook = OAuthAuthHook()
    with _patch_router(_router([deployment])):
        with pytest.raises(litellm.exceptions.AuthenticationError) as excinfo:
            await hook.async_pre_call_deployment_hook(
                {
                    "model": "openai/upstream-model-id",
                    "model_info": deployment["model_info"],
                    "metadata": {"model_group": "public-gw-name"},
                },
                "acompletion",
            )
    message = str(excinfo.value)
    assert "public-gw-name" in message and "dep-7" in message, message


@pytest.mark.asyncio
async def test_the_secret_is_stripped_even_when_metadata_aliases_model_info():
    """The router stores ONE model_info object in two places.

    Router._update_kwargs_with_deployment takes a single per-request
    deployment["model_info"].copy() and puts that same object in BOTH
    kwargs["model_info"] and metadata["model_info"]. Replacing only our key left
    the secret reachable through metadata (and through the caching handler's
    captured request_kwargs) -- measured on streaming and non-streaming. Popping in
    place is what closes every alias, so this test shares the object exactly as the
    router does.
    """
    deployment = _deployment("gw-model", oauth={**OAUTH_BLOCK, "client_secret": "literal-secret"})
    shared_model_info = deployment["model_info"]
    hook = OAuthAuthHook()
    kwargs = {
        "model": "openai/gw-model",
        "model_info": shared_model_info,
        "metadata": {"model_info": shared_model_info, "model_group": "gw-model"},
    }
    with _patch_router(_router([deployment])):
        with patch.object(AsyncOAuth2ClientCredential, "get_token", return_value=_fake_token()):
            result = await hook.async_pre_call_deployment_hook(kwargs, "acompletion")
    assert result is not None
    assert "literal-secret" not in json.dumps(result, default=str)
    # The alias must be clean too, not just our copy.
    assert "h2o_oauth" not in result["metadata"]["model_info"]
    assert "h2o_oauth" not in shared_model_info
    # Still usable for the next request: the router's own model_list entry holds a
    # different dict, so this only clears the per-request copy.
    assert shared_model_info.get("id")
