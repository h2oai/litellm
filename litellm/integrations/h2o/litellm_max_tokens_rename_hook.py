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

  * the call is a chat completion, and
  * the request carries a positive integer `max_tokens`, and
  * the selected deployment is one that discards or rejects `max_tokens`,
    i.e. it lists `max_tokens` in `additional_drop_params`, or it is an Azure
    route.

A configured `max_completion_tokens` is deliberately NOT a trigger on its own.
`convert_model_to_litellm_config` sets it for every reasoning model, Azure or
not (`use_completion_tokens = is_reasoning_model or is_azure_provider`), and
only the Azure branch adds the drop. Treating it as a trigger would strip
`max_tokens` from non-Azure providers that need it, leaving those requests with
no ceiling at all, which is worse than the bug being fixed. It also cannot be
distinguished from a caller-supplied value once kwargs are merged.

Deployments that natively accept `max_tokens` (Anthropic, Bedrock, vLLM,
non-2025 Azure) are left untouched, including when they share a model group
with an Azure deployment.

Two scope limits worth knowing:

  * CHAT COMPLETIONS ONLY, enforced via `call_type`. The dispatch is NOT
    chat-specific: `wrapper_async` runs it for every @client-decorated async
    entrypoint. `litellm.anthropic_messages` (which backs /v1/messages, and
    which bridges Azure and other non-Anthropic providers) declares
    `max_tokens: int` as a REQUIRED parameter, so popping it there makes
    litellm's own wrapper raise
    `TypeError: anthropic_messages() missing 1 required positional argument`
    on the following `await original_function(*args, **kwargs)`. That is
    outside this hook's try/except and cannot be caught here, and the
    traceback never names this hook. `atext_completion` (/v1/completions has
    no `max_completion_tokens`) and `aembedding` reach the same dispatch.

  * ASYNC PATH ONLY. The dispatch lives in the @client decorator's ASYNC
    wrapper; the sync wrapper does not dispatch it, so a direct sync
    `litellm.completion()` or `Router._completion()` bypasses this hook. That
    is fine for how the hook is deployed: it is registered only in the proxy
    config, and the proxy maps /chat/completions to `acompletion`
    (proxy/route_llm_request.py), so all proxied traffic is async.

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
    def _is_chat_completion(call_type: Any) -> bool:
        """True only for the chat-completion call types.

        `call_type` is a `CallTypes` enum member (or None for an unrecognised
        entrypoint), so compare on its value and tolerate a bare string.
        """
        return getattr(call_type, "value", call_type) in ("completion", "acompletion")

    @staticmethod
    def _deployment_discards_max_tokens(kwargs: Dict[str, Any]) -> bool:
        """True when the SELECTED deployment would discard or reject
        `max_tokens`, so the caller's limit only survives as
        `max_completion_tokens`.

        Read straight off the merged kwargs rather than the router, so a mixed
        model group is judged per selected deployment instead of per group.
        """
        drop = kwargs.get("additional_drop_params") or []
        if isinstance(drop, (list, tuple)) and "max_tokens" in drop:
            return True
        # Azure 2025+ rejects max_tokens outright. Reasoning Azure entries are
        # exempt from the drop above but still need the rename, so recognise
        # the route itself. custom_llm_provider is not always populated at this
        # point, so the prefixed model string is the primary signal.
        if kwargs.get("custom_llm_provider") == "azure":
            return True
        model = kwargs.get("model")
        return isinstance(model, str) and model.startswith("azure/")

    async def async_pre_call_deployment_hook(
        self, kwargs: Dict[str, Any], call_type: Any
    ) -> Optional[dict]:
        try:
            if not self._is_chat_completion(call_type):
                return None
            requested = self._positive_int(kwargs.get("max_tokens"))
            if requested is None:
                return None
            if not self._deployment_discards_max_tokens(kwargs):
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
