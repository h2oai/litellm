#!/usr/bin/env python3
"""
LiteLLM Pre-Call Hook: rename client `max_tokens` to `max_completion_tokens`
on deployments that only accept the latter.

WHY THIS EXISTS
---------------
Azure 2025+ API versions reject `max_tokens` for chat models, so
`launch_litellm.py:convert_model_to_litellm_config` configures those
deployments with `max_completion_tokens` and adds
`additional_drop_params: ["max_tokens"]`. litellm's generic Azure chat
transform lists BOTH fields as supported and forwards both unchanged, so
without that drop the client's `max_tokens` and our configured
`max_completion_tokens` arrive together and Azure 400s:

    AzureException BadRequestError - Setting 'max_tokens' and
    'max_completion_tokens' at the same time is not supported.

The drop prevents that 400, but it DISCARDS the caller's requested output
limit instead of honoring it, so the deployment default applies instead. A
client asking for 50 tokens gets the deployment ceiling (e.g. 16384), and
`max_tokens` silently has no effect on cost or latency. Measured through the
h2oGPTe OpenAI-compatible API: a 50-token request returned 1666 tokens.

This hook closes that gap. It runs BEFORE litellm's param mapping, and
therefore before `additional_drop_params` takes effect, moving the caller's
value into the field the deployment actually accepts. The drop is left in
place as a fallback for images whose litellm predates this hook.

SCOPE
-----
Only fires when BOTH hold:

  * the request carries a positive integer `max_tokens`, and
  * the targeted model has at least one deployment that uses
    `max_completion_tokens` (either configured directly in `litellm_params`,
    or implied by `max_tokens` being in that deployment's
    `additional_drop_params`).

Deployments that legitimately take `max_tokens` (Anthropic, Bedrock, vLLM,
non-2025 Azure) are left completely untouched.

Only the top-level request body is rewritten. A `max_tokens` nested in
`extra_body` is a provider passthrough the caller set deliberately, and
rewriting it could change a field the upstream treats differently; both the
OpenAI SDK and h2oGPTe send `max_tokens` top-level.

INTERACTION WITH THE CAP HOOK
-----------------------------
`litellm_max_tokens_cap_hook` clips both `max_tokens` and
`max_completion_tokens` down to the deployment ceiling. The two hooks are
order-independent: clipping then renaming, and renaming then clipping,
produce the same value.

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
    """Move client `max_tokens` onto `max_completion_tokens` for deployments
    that only accept the latter, so the caller's limit is honored instead of
    silently dropped."""

    def __init__(self):
        super().__init__()
        self.enabled = True
        if verbose or verbose_full:
            print("MaxTokensRenameHook: Initialized", flush=True)

    def _uses_completion_tokens(self, model: str) -> bool:
        """True when `model` has a deployment that expects
        `max_completion_tokens` rather than `max_tokens`.

        Returns False when the router isn't reachable (e.g. called outside the
        proxy server context) so the request is left untouched.
        """
        if not model:
            return False
        try:
            from litellm.proxy.proxy_server import llm_router
        except Exception:
            return False
        if llm_router is None:
            return False

        try:
            for deployment in (llm_router.model_list or []):
                if deployment.get("model_name") != model:
                    continue
                params = deployment.get("litellm_params") or {}
                if params.get("max_completion_tokens") is not None:
                    return True
                drop = params.get("additional_drop_params") or []
                if isinstance(drop, (list, tuple)) and "max_tokens" in drop:
                    return True
        except Exception:
            return False
        return False

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

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: Dict[str, Any],
        call_type: str,
    ) -> Dict[str, Any]:
        try:
            requested = self._positive_int(data.get("max_tokens"))
            if requested is None:
                return data
            model = data.get("model", "")
            if not self._uses_completion_tokens(model):
                return data

            # A caller that sent both fields already expressed the tighter
            # intent explicitly; keep the smaller of the two rather than
            # letting the rename raise their ceiling.
            existing = self._positive_int(data.get("max_completion_tokens"))
            resolved = min(requested, existing) if existing is not None else requested

            data.pop("max_tokens", None)
            data["max_completion_tokens"] = resolved

            if verbose or verbose_full:
                print(f"MaxTokensRenameHook: {model}: max_tokens={requested} -> "
                      f"max_completion_tokens={resolved}", flush=True)
        except Exception as e:
            # Never break a request over a rename.
            if verbose_full:
                import traceback
                print(f"MaxTokensRenameHook: error in pre_call: {e}", flush=True)
                traceback.print_exc()
        return data


# Create the hook instance that LiteLLM will use
max_tokens_rename_hook = MaxTokensRenameHook()

if verbose or verbose_full:
    print(f"HOOK EXPORT: max_tokens_rename_hook created successfully: "
          f"{type(max_tokens_rename_hook)}", flush=True)
