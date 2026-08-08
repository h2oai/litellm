#!/usr/bin/env python3
"""
LiteLLM Pre-Call Hook: per-model OAuth2 bearer authentication for upstream gateways.

WHY THIS EXISTS
---------------
`litellm_params.api_key` is a STATIC credential. Gateways fronted by an OIDC
layer issue short-lived JWTs instead, so a static key cannot be configured at
all -- the token has to be minted per deployment and refreshed before it
expires.

LiteLLM already ships an outbound OAuth2 helper (`litellm.proxy_auth`), but it
is applied through a module GLOBAL, so its Authorization header lands on every
completion and embedding regardless of which deployment served the request. In a
proxy serving many providers at once that is not usable: one gateway's bearer
would overwrite the credentials of every other model. See
`litellm/proxy_auth/async_oauth2.py` for the full comparison.

This hook makes the same capability per-model.

DISPATCH POINT: AFTER DEPLOYMENT SELECTION
------------------------------------------
It hooks `async_pre_call_deployment_hook`, NOT `async_pre_call_hook`. That is a
correctness requirement, not a preference. `async_pre_call_hook` runs BEFORE the
router picks a deployment, so the only thing it can key off is the model GROUP
name -- it had to look up the *first* deployment whose `model_name` matched, read
that one's `h2o_oauth`, and write the header into a request that the router might
then send anywhere in the group. Measured with two loopback servers and
simple-shuffle: gateway A's live bearer was sent to deployment B, which had no
OAuth config, a different vendor and its own api_key. That is credential
exfiltration to a third party, and it simultaneously broke B by overriding its
api_key. Router `fallbacks:` reuse the same kwargs, so the token followed
fallbacks to other providers too.

`async_pre_call_deployment_hook` runs after selection and before param mapping,
and its kwargs carry the SELECTED deployment's `model_info` (verified: the hook
sees `model_info == {"id": "dep-1", ..., "h2o_oauth": {...}}` for exactly the
deployment the router chose). So the token can only ever reach the deployment
whose configuration minted it, and the router lookup is gone entirely.

WHAT IT MODIFIES
----------------
Exactly one thing: `data["extra_headers"]`, and only for models that declare
`model_info.h2o_oauth`. The minted token is written to the configured header
(default `Authorization: Bearer <token>`), which takes precedence over the
deployment's `api_key` because request kwargs override deployment
`litellm_params` in `Router._acompletion`.

`extra_headers` is used rather than a dynamic `api_key` on purpose: `api_key`,
`api_base` and `base_url` are litellm's `clientside_credential_keys`, so
injecting a rotating `api_key` makes the router `upsert_deployment` a NEW
deployment for every distinct token (measured: 3 tokens -> 4 deployments), which
leaks router state and cost-map registrations for the life of the process.
Headers reuse the cached client and leave the deployment list untouched.

HEADER MERGE ORDER
------------------
Request kwargs REPLACE deployment `litellm_params` wholesale -- they are not
merged -- so writing `data["extra_headers"]` naively would silently drop any
static `extra_headers` configured on the deployment (gateways commonly require
routing or tenant headers there). This hook therefore merges, lowest precedence
first:

    deployment litellm_params.extra_headers  ->  request extra_headers  ->  auth header

FAILURE POLICY
--------------
This hook deliberately does NOT follow the swallow-everything convention of the
other h2o hooks, because a missing credential cannot be recovered downstream: it
would surface as an opaque upstream 401. Instead:

  * models WITHOUT `h2o_oauth`: nothing can raise. The opt-out check happens
    before any work, and even that lookup is guarded -- an unconfigured model is
    returned untouched no matter what state the router is in.
  * models WITH `h2o_oauth`: configuration and token errors are raised with an
    actionable message, so the operator sees "token endpoint returned 401" and
    not a generic gateway rejection.
"""

import os
from typing import Any, Dict, Optional

from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy_auth.async_oauth2 import (
    CredentialRegistry,
    OAuth2Config,
    OAuth2ConfigError,
    OAuth2TokenError,
)

verbose = os.getenv("H2OGPT_VERBOSE", "0") == "1"
verbose_full = os.getenv("H2OGPT_VERBOSE_FULL", "0") == "1"

CONFIG_KEY = "h2o_oauth"


def _reject(message: str) -> None:
    """Fail the request with the clearest error the caller can see.

    Uses fastapi's HTTPException when running under the proxy so the caller gets
    a 502 with the message; falls back to OAuth2TokenError outside the proxy
    (unit tests, SDK use) where fastapi may not be importable.
    """
    try:
        from fastapi import HTTPException

        raise HTTPException(status_code=502, detail=message)
    except ImportError:
        raise OAuth2TokenError(message) from None


class OAuthAuthHook(CustomLogger):
    """Injects a freshly minted OAuth2 bearer token per model, per request."""

    def __init__(self) -> None:
        super().__init__()
        self.registry = CredentialRegistry()
        if verbose or verbose_full:
            print("🔐 OAuthAuthHook: Initialized", flush=True)

    @staticmethod
    def _raw_config(model_info: Any) -> Optional[Any]:
        """Read the OAuth block out of the selected deployment's model_info.

        model_info is used rather than litellm_params because litellm_params is
        merged into the outbound request kwargs -- config placed there can be
        forwarded to the provider as extra_body. model_info is deployment
        metadata and never leaves the proxy (verified: an upstream request for a
        model carrying model_info.h2o_oauth contains only `model` and
        `messages`).
        """
        if not isinstance(model_info, dict):
            return None
        return model_info.get(CONFIG_KEY)

    @staticmethod
    def _static_deployment_headers(model_info: Any) -> Dict[str, Any]:
        """The selected deployment's own `litellm_params.extra_headers`.

        Needed because litellm makes request kwargs REPLACE deployment
        litellm_params wholesale rather than merging them: measured at this
        dispatch point, a deployment configured with
        ``extra_headers: {"a": ...}`` whose request also sends
        ``extra_headers: {"b": ...}`` arrives as ``{"b": ...}`` only. Gateways
        commonly put routing or tenant headers there, so they must be restored as
        the lowest-precedence layer.

        Looked up by the SELECTED deployment's ``model_info["id"]``, never by
        model_name. That is the whole point of running after selection: an id is
        unambiguous, whereas a name can match several deployments and picking the
        first is what sent one gateway's bearer to another deployment.

        ``metadata["deployment"]`` cannot be used -- it is the id string, not the
        deployment dict.
        """
        if not isinstance(model_info, dict):
            return {}
        deployment_id = model_info.get("id")
        if not deployment_id:
            return {}
        try:
            from litellm.proxy.proxy_server import llm_router
        except Exception:
            return {}
        if llm_router is None:
            return {}
        try:
            for deployment in llm_router.model_list or []:
                info = deployment.get("model_info") or {}
                if info.get("id") != deployment_id:
                    continue
                headers = (deployment.get("litellm_params") or {}).get("extra_headers")
                return dict(headers) if isinstance(headers, dict) else {}
        except Exception:
            return {}
        return {}

    async def async_pre_call_deployment_hook(
        self, kwargs: Dict[str, Any], call_type: Any
    ) -> Optional[dict]:
        """Mint and inject the bearer for the deployment the router SELECTED.

        Returns modified kwargs, or None to leave the request untouched (which is
        what litellm expects for "nothing to do" -- see
        utils.py:async_pre_call_deployment_hook, which chains each callback's
        returned kwargs into the next).
        """
        # ---- opt-out path: must never raise, must stay cheap -----------------
        try:
            raw_config = self._raw_config(kwargs.get("model_info"))
            if not raw_config:
                return None
        except Exception as e:
            # A model with no OAuth config must never be affected by a fault in
            # this hook, so this branch cannot escalate.
            verbose_logger.debug("h2o_oauth: skipping (lookup failed): %s", e)
            return None

        model = kwargs.get("model", "")

        # ---- opted-in path: fail closed with an actionable message -----------
        try:
            config = OAuth2Config.from_dict(raw_config)
        except OAuth2ConfigError as e:
            _reject(f"model '{model}' has an invalid h2o_oauth configuration: {e}")
            return None  # unreachable; keeps type checkers happy

        try:
            credential = self.registry.get(config)
            token = await credential.get_token()
        except (OAuth2ConfigError, OAuth2TokenError) as e:
            _reject(f"could not obtain an OAuth2 token for model '{model}': {e}")
            return None  # unreachable
        except Exception as e:
            _reject(
                f"unexpected error obtaining an OAuth2 token for model '{model}': {type(e).__name__}: {e}"
            )
            return None  # unreachable

        # Merge lowest precedence first. At this dispatch point the deployment's
        # own litellm_params have already been merged into kwargs, so
        # kwargs["extra_headers"] already carries any static routing/tenant
        # headers the deployment configured -- they must not be dropped.
        # Lowest precedence first:
        #   deployment litellm_params.extra_headers -> request extra_headers -> auth
        headers: Dict[str, Any] = self._static_deployment_headers(kwargs.get("model_info"))
        existing = kwargs.get("extra_headers")
        if isinstance(existing, dict):
            headers.update(existing)

        header_name, header_value = credential.auth_header(token)
        headers[header_name] = header_value

        modified = dict(kwargs)
        modified["extra_headers"] = headers

        if verbose_full:
            print(
                f"🔐 OAuthAuthHook: {model} -> {header_name} set from {config.token_url}",
                flush=True,
            )
        return modified


oauth_auth_hook = OAuthAuthHook()
