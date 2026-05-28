#!/usr/bin/env python3
"""
LiteLLM Pre-Call Hook: Cap client-supplied max_tokens to model's configured limit.

WHY THIS EXISTS
---------------
LiteLLM's `litellm_params.max_tokens` (or `max_completion_tokens`) on a
model_list entry is a DEFAULT — it's filled in when the client request
doesn't include max_tokens, but it does NOT cap requests that DO include
a higher value. That gap was demonstrated by user reports against
Bedrock Llama-3.2-11B-Vision-Instruct (and similar):

    litellm.BadRequestError: BedrockException -
      "The maximum tokens you requested exceeds the model limit of 8192.
       Try again with a maximum tokens value that is lower than 8192."

When `max_max_new_tokens: 8192` is set on a model_lock entry, h2ogpt's
`gen.py:get_max_max_new_tokens` clips the value before sending — but
that only protects h2ogpt's own request path. Other callers (or any
agent that sends max_tokens explicitly) still bypass the YAML default
and hit the upstream rejection.

This hook closes the gap: at request time, look up the configured
output-token cap for the targeted model and clip the inbound request to
it, so upstream APIs never receive an over-limit value regardless of
what the client asked for.

ENFORCEMENT SOURCE
------------------
The cap comes from EITHER:

  * `model_info.max_output_tokens` on the registered model_list entry
    (set by `launch_litellm.py:convert_model_to_litellm_config` when the
    model_lock entry has `max_max_new_tokens` or `max_output_seq_len`)
  * `litellm_params.max_tokens` / `max_completion_tokens` (fallback)

Resolution order is "tightest wins" — if both are set, the lower value
is the effective cap.

WHAT IT MODIFIES
----------------
For Anthropic-style providers it caps `max_tokens` in `data`,
`extra_body`, `litellm_params`, and `litellm_params.extra_body`.
For OpenAI-reasoning-style providers it caps `max_completion_tokens`
in the same locations.

It does NOT add max_tokens when missing — that's already the proxy
YAML default's job. It only clips DOWN when the request value exceeds
the cap.

NEVER PROPAGATES
----------------
Any unexpected error during cap lookup is swallowed and the request
proceeds unmodified. Caps are an enhancement, not a hard requirement —
breaking the request because a cap couldn't be derived would be worse
than letting it through.
"""

import os
from typing import Any, Dict, Optional

from litellm.integrations.custom_logger import CustomLogger

verbose = os.getenv('H2OGPT_VERBOSE', '0') == '1'
verbose_full = os.getenv('H2OGPT_VERBOSE_FULL', '0') == '1'


# Token-limit-enforcement parameter names. Some providers use
# max_completion_tokens (newer Azure/OpenAI reasoning), others use
# max_tokens (everyone else). Both can co-exist in the same payload.
_MAX_TOKEN_FIELDS = ("max_tokens", "max_completion_tokens")


class MaxTokensCapHook(CustomLogger):
    """Server-side cap on client-supplied max_tokens / max_completion_tokens."""

    def __init__(self):
        super().__init__()
        self.enabled = True
        if verbose or verbose_full:
            print(f"🔒 MaxTokensCapHook: Initialized", flush=True)

    def _resolve_cap(self, model: str) -> Optional[int]:
        """Look up the effective max-output cap for `model` from the
        litellm router's registered config. Returns None if no cap is
        configured (or if the router isn't accessible — e.g. we're being
        called outside the proxy server context).
        """
        if not model:
            return None
        try:
            from litellm.proxy.proxy_server import llm_router
        except Exception:
            return None
        if llm_router is None:
            return None

        # Tightest of (model_info.max_output_tokens, litellm_params.max_tokens,
        # litellm_params.max_completion_tokens) wins.
        candidates = []
        try:
            for deployment in (llm_router.model_list or []):
                if deployment.get("model_name") != model:
                    continue
                model_info = deployment.get("model_info") or {}
                v = model_info.get("max_output_tokens")
                if isinstance(v, int) and v > 0:
                    candidates.append(v)
                lp = deployment.get("litellm_params") or {}
                for f in _MAX_TOKEN_FIELDS:
                    v = lp.get(f)
                    if isinstance(v, int) and v > 0:
                        candidates.append(v)
                # Walk every matching deployment — same model_name can have
                # multiple deployments with different caps, and litellm routes
                # between them at request time. Use the TIGHTEST seen as a
                # safe ceiling (so a request that ends up routed to the
                # smaller deployment can't exceed its limit).
        except Exception:
            return None
        return min(candidates) if candidates else None

    def _cap_in(self, container: Dict[str, Any], cap: int, label: str, modified: list) -> None:
        if not isinstance(container, dict):
            return
        for f in _MAX_TOKEN_FIELDS:
            if f in container:
                v = container[f]
                if isinstance(v, int) and v > cap:
                    container[f] = cap
                    modified.append(f"{label}.{f}: {v} -> {cap}")

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: Dict[str, Any],
        call_type: str,
    ) -> Dict[str, Any]:
        try:
            model = data.get("model", "")
            cap = self._resolve_cap(model)
            if cap is None:
                return data

            modified: list = []
            self._cap_in(data, cap, "data", modified)
            self._cap_in(data.get("extra_body") or {}, cap, "extra_body", modified)
            litellm_params = data.get("litellm_params") or {}
            self._cap_in(litellm_params, cap, "litellm_params", modified)
            self._cap_in(litellm_params.get("extra_body") or {}, cap,
                         "litellm_params.extra_body", modified)

            if modified and (verbose or verbose_full):
                print(f"🔒 MaxTokensCapHook: capped to {cap} for {model}: {modified}",
                      flush=True)
        except Exception as e:
            # Caps are advisory — never break the request because of them.
            if verbose_full:
                import traceback
                print(f"🔒 MaxTokensCapHook: error in pre_call: {e}", flush=True)
                traceback.print_exc()
        return data


# Create the hook instance that LiteLLM will use
max_tokens_cap_hook = MaxTokensCapHook()

if verbose or verbose_full:
    print(f"🔒 HOOK EXPORT: max_tokens_cap_hook created successfully: "
          f"{type(max_tokens_cap_hook)}", flush=True)
