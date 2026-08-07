#!/usr/bin/env python3
"""
LiteLLM Deployment-Level Hook: rename client `max_tokens` to
`max_completion_tokens` on deployments that only accept the latter.

WHY THIS EXISTS
---------------
Azure 2025+ API versions reject a request carrying both `max_tokens` and
`max_completion_tokens`:

    AzureException BadRequestError - Setting 'max_tokens' and
    'max_completion_tokens' at the same time is not supported.

To avoid that, `launch_litellm.py:convert_model_to_litellm_config` configures
those deployments with `max_completion_tokens` plus
`additional_drop_params: ["max_tokens"]`. The drop prevents the 400, but it
DISCARDS the caller's requested output limit instead of honoring it, so the
deployment ceiling applies. Measured through the h2oGPTe OpenAI-compatible
API: a 50-token request returned 1666 (h2oai/h2ogpte#11992).

This hook moves the caller's value onto the field the deployment accepts, so
the limit is honored and only one of the two fields still reaches Azure. The
drop stays as the fallback for images whose litellm predates this hook.

WHY A DEPLOYMENT-LEVEL HOOK AND NOT `async_pre_call_hook`
---------------------------------------------------------
`async_pre_call_hook` runs BEFORE the router picks a concrete deployment, so
it can only ask "does *some* deployment under this model_name use completion
tokens?". That is wrong for a MIXED model group. Our generated config contains
one: `agent_auto` fans out across anthropic, azure, bedrock, gemini, mistral
and openrouter under a single model_name, and a request to it lands on
different members run to run. Rewriting up front would send
`max_completion_tokens` to the Bedrock and Anthropic members, which expect
`max_tokens`.

`async_pre_call_deployment_hook` runs AFTER deployment selection and BEFORE
the request is sent, and its `kwargs` carry the SELECTED deployment's merged
litellm_params. Verified against a live mixed group: the azure member arrives
with `max_completion_tokens=16384, additional_drop_params=['max_tokens']`
while the bedrock member arrives with both unset, so the two are cleanly
distinguishable and each is handled on its own terms. No router lookup is
needed, which also removes the previous dependency on `llm_router` being
reachable.

SCOPE
-----
Fires only when BOTH hold for the deployment actually selected:

  * the request carries a positive integer `max_tokens`, and
  * that deployment expects `max_completion_tokens` (it has one configured,
    or `max_tokens` in its `additional_drop_params`).

Deployments that natively accept `max_tokens` (Anthropic, Bedrock, vLLM,
non-2025 Azure) are left untouched, including when they share a model group
with an Azure deployment.

Two scope limits worth knowing:

  * ASYNC PATH ONLY. `async_pre_call_deployment_hook` is dispatched by the
    @client decorator's ASYNC wrapper (litellm/utils.py, in wrapper_async).
    The sync wrapper does not dispatch it, so a direct sync
    `litellm.completion()` or `Router._completion()` bypasses this hook. That
    is fine for how the hook is deployed: it is registered only in the proxy
    config, and the proxy maps /chat/completions to `acompletion`
    (proxy/route_llm_request.py), so all proxied traffic takes the async path.
    Code embedding litellm in-process and calling sync `completion()` would
    not get the rename.

  * The predicate reads the MERGED kwargs, which cannot distinguish a
    deployment-configured `max_completion_tokens` from a caller-supplied one.
    A request sending BOTH `max_tokens` and `max_completion_tokens` to a
    max_tokens-native deployment is therefore rewritten. The reported case,
    and everything h2oGPTe core emits, sends `max_tokens` alone. Keying only
    on `additional_drop_params` would remove this edge, at the cost of not
    firing for an Azure deployment configured with a ceiling but no drop.

INTERACTION WITH THE CAP HOOK
-----------------------------
`litellm_max_tokens_cap_hook` clips both fields down to the deployment
ceiling. The two are order-independent: clipping then renaming, and renaming
then clipping, produce the same value. Taking the minimum below means a
configured deployment ceiling is never raised by the rename.

NEVER PROPAGATES
----------------
Any unexpected error is swallowed and the request proceeds unmodified.
Failing a request because a rename could not be derived would be worse than
letting the pre-existing behaviour stand.
"""

import os
from typing import Any, Dict, Optional

from litellm.integrations.custom_logger import CustomLogger

verbose = os.getenv('H2OGPT_VERBOSE', '0') == '1'
verbose_full = os.getenv('H2OGPT_VERBOSE_FULL', '0') == '1'


class MaxTokensRenameHook(CustomLogger):
    """Move client `max_tokens` onto `max_completion_tokens` for the selected
    deployment when that deployment only accepts the latter, so the caller's
    limit is honored instead of silently dropped."""

    def __init__(self):
        super().__init__()
        self.enabled = True
        if verbose or verbose_full:
            print("MaxTokensRenameHook: Initialized", flush=True)

    @staticmethod
    def _positive_int(value: Any) -> Optional[int]:
        """Return `value` as a positive int, or None if it isn't one.

        Guards the `max_tokens: 0` / `null` / string shapes a client can put on
        the wire, which must not become a `max_completion_tokens` that
        truncates every response to nothing. `bool` is rejected explicitly
        because it is an int subclass in Python.
        """
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        return value if value > 0 else None

    @staticmethod
    def _deployment_uses_completion_tokens(kwargs: Dict[str, Any]) -> bool:
        """True when the SELECTED deployment expects `max_completion_tokens`.

        Read straight off the merged kwargs rather than the router, so a mixed
        model group is judged per selected deployment instead of per group.
        """
        if kwargs.get("max_completion_tokens") is not None:
            return True
        drop = kwargs.get("additional_drop_params") or []
        return isinstance(drop, (list, tuple)) and "max_tokens" in drop

    async def async_pre_call_deployment_hook(
        self, kwargs: Dict[str, Any], call_type: Any
    ) -> Optional[dict]:
        try:
            requested = self._positive_int(kwargs.get("max_tokens"))
            if requested is None:
                return None
            if not self._deployment_uses_completion_tokens(kwargs):
                return None

            # A configured deployment ceiling (or a caller-supplied value) must
            # never be RAISED by the rename, so keep whichever is tighter.
            existing = self._positive_int(kwargs.get("max_completion_tokens"))
            resolved = min(requested, existing) if existing is not None else requested

            modified = dict(kwargs)
            modified.pop("max_tokens", None)
            modified["max_completion_tokens"] = resolved

            if verbose or verbose_full:
                print(f"MaxTokensRenameHook: {kwargs.get('model')}: "
                      f"max_tokens={requested} -> max_completion_tokens={resolved}",
                      flush=True)
            return modified
        except Exception as e:
            # Never break a request over a rename.
            if verbose_full:
                import traceback
                print(f"MaxTokensRenameHook: error in pre_call: {e}", flush=True)
                traceback.print_exc()
            return None


# Create the hook instance that LiteLLM will use
max_tokens_rename_hook = MaxTokensRenameHook()

if verbose or verbose_full:
    print(f"HOOK EXPORT: max_tokens_rename_hook created successfully: "
          f"{type(max_tokens_rename_hook)}", flush=True)
