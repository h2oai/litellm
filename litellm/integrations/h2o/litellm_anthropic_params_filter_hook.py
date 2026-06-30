#!/usr/bin/env python3
"""
LiteLLM Hook to Filter Anthropic-Specific Parameters and Limit max_tokens

This hook performs six functions:

1. Filters Anthropic-specific parameters for non-Anthropic models:
   - context_management: Anthropic beta feature for context management
   - enable_caching: Custom parameter for Anthropic caching hook
   - thinking: Anthropic extended thinking parameter (removed for non-Anthropic models)
   - Other Anthropic-specific beta parameters

2. Forces temperature=1 when thinking is enabled for Anthropic models:
   - Anthropic API requires temperature=1 when thinking is enabled
   - Prevents: "temperature may only be set to 1 when thinking is enabled"

3. Removes top_p when temperature is also set for Anthropic models:
   - Anthropic API does not allow both temperature and top_p to be specified
   - When both are present, keeps temperature and removes top_p
   - Prevents: "temperature and top_p cannot both be specified"

4. Ensures max_tokens > thinking.budget_tokens for Anthropic models:
   - Anthropic API requires max_tokens to be greater than thinking budget
   - Auto-adjusts max_tokens to thinking.budget_tokens + 8192 (response buffer)
   - Prevents: "max_tokens must be greater than thinking.budget_tokens"

5. Limits max_tokens for Azure models with lower token limits:
   - gpt-4o: max_tokens capped to 8192
   - gpt-4o-mini: max_tokens capped to 8192

6. Filters unsupported anthropic-beta flags for Bedrock models:
   - Bedrock only supports a whitelist of beta flags
   - Unsupported flags like output-128k, prompt-caching, mcp-servers cause errors
   - Strips unsupported beta flags from extra_headers and anthropic_beta params
   - Prevents: "invalid beta flag" errors from Bedrock API

This prevents errors like:
  AzureException - Unrecognized request argument supplied: context_management
  AzureException - max_tokens is too large
  AnthropicException - temperature may only be set to 1 when thinking is enabled
  AnthropicException - temperature and top_p cannot both be specified
  AnthropicException - max_tokens must be greater than thinking.budget_tokens
  BedrockException - invalid beta flag
"""

import os
from typing import Dict, Any, List, Optional, Set
from litellm.integrations.custom_logger import CustomLogger

verbose = os.getenv('H2OGPT_VERBOSE', '0') == '1'
verbose_full = os.getenv('H2OGPT_VERBOSE_FULL', '0') == '1'


class AnthropicParamsFilterHook(CustomLogger):
    """Custom LiteLLM hook to filter out Anthropic-specific parameters for non-Anthropic models."""

    def __init__(self):
        super().__init__()
        self.enabled = True
        if verbose or verbose_full:
            print(f"🧹 AnthropicParamsFilterHook: Initialized", flush=True)

    def _is_anthropic_provider(self, model: str) -> bool:
        """Check if the model uses Anthropic or Bedrock Anthropic provider."""
        if not model:
            return False

        model_str = str(model).lower()

        # Check for native Anthropic models
        if model_str.startswith('anthropic/'):
            return True

        # Check for Bedrock Anthropic models
        if model_str.startswith('bedrock/') and 'anthropic.claude' in model_str:
            return True

        # Check for models that contain claude (may be configured without prefix)
        if 'claude' in model_str:
            return True

        return False

    def _is_bedrock_provider(self, model: str) -> bool:
        """Check if the model uses AWS Bedrock provider."""
        if not model:
            return False

        model_str = str(model).lower()
        return model_str.startswith('bedrock/')

    # Whitelist of beta flags supported by AWS Bedrock.
    # Bedrock rejects any beta flag not in its whitelist with "invalid beta flag" error.
    # See: https://docs.anthropic.com/en/api/claude-on-amazon-bedrock
    # See: https://github.com/BerriAI/litellm/issues/16726
    BEDROCK_SUPPORTED_BETA_FLAGS: Set[str] = {
        # Computer use
        'computer-use-2024-10-22',
        'computer-use-2025-01-24',
        # Token-efficient tools
        'token-efficient-tools-2025-02-19',
        # Interleaved thinking
        'interleaved-thinking-2025-05-14',
        # Tool search
        'tool-search-tool-2025-10-19',
        # Extended context window
        'context-1m-2025-08-07',
        # Effort control
        'effort-2025-11-24',
        # Claude code
        'claude-code-20250219',
        # Fine-grained tool streaming
        'fine-grained-tool-streaming-2025-05-14',
    }

    def _filter_bedrock_beta_flags(self, data: Dict[str, Any], model: str) -> None:
        """
        Filter unsupported anthropic-beta flags for Bedrock models.

        AWS Bedrock only supports a strict whitelist of beta flags. Unsupported flags
        (like output-128k-2025-02-19, prompt-caching-2024-07-31, mcp-servers-2025-12-04)
        cause "invalid beta flag" errors.

        This method:
        1. Filters anthropic-beta from extra_headers (HTTP header format, comma-separated)
        2. Filters anthropic_beta from request body (list format)
        3. Checks all data locations (top-level, extra_body, litellm_params, etc.)
        """
        if not self._is_bedrock_provider(model):
            return

        removed_flags = []

        # Helper to filter a comma-separated beta header string
        def filter_beta_header(header_value: str) -> Optional[str]:
            if not header_value:
                return header_value
            flags = [f.strip() for f in header_value.split(',') if f.strip()]
            supported = [f for f in flags if f in self.BEDROCK_SUPPORTED_BETA_FLAGS]
            unsupported = [f for f in flags if f not in self.BEDROCK_SUPPORTED_BETA_FLAGS]
            if unsupported:
                removed_flags.extend(unsupported)
            return ','.join(supported) if supported else None

        # Helper to filter a list of beta flags
        def filter_beta_list(beta_list: List[str]) -> List[str]:
            supported = [f for f in beta_list if f in self.BEDROCK_SUPPORTED_BETA_FLAGS]
            unsupported = [f for f in beta_list if f not in self.BEDROCK_SUPPORTED_BETA_FLAGS]
            if unsupported:
                removed_flags.extend(unsupported)
            return supported

        # --- Filter anthropic-beta from extra_headers (HTTP header format) ---

        # Location 1: data.extra_headers
        extra_headers = data.get('extra_headers', {})
        if extra_headers and isinstance(extra_headers, dict):
            if 'anthropic-beta' in extra_headers:
                filtered = filter_beta_header(extra_headers['anthropic-beta'])
                if filtered:
                    extra_headers['anthropic-beta'] = filtered
                else:
                    del extra_headers['anthropic-beta']

        # Location 2: data.headers
        headers = data.get('headers', {})
        if headers and isinstance(headers, dict):
            if 'anthropic-beta' in headers:
                filtered = filter_beta_header(headers['anthropic-beta'])
                if filtered:
                    headers['anthropic-beta'] = filtered
                else:
                    del headers['anthropic-beta']

        # Location 3: litellm_params.extra_headers
        litellm_params = data.get('litellm_params', {})
        if litellm_params and isinstance(litellm_params, dict):
            lp_extra_headers = litellm_params.get('extra_headers', {})
            if lp_extra_headers and isinstance(lp_extra_headers, dict):
                if 'anthropic-beta' in lp_extra_headers:
                    filtered = filter_beta_header(lp_extra_headers['anthropic-beta'])
                    if filtered:
                        lp_extra_headers['anthropic-beta'] = filtered
                    else:
                        del lp_extra_headers['anthropic-beta']

            # Location 4: litellm_params.headers
            lp_headers = litellm_params.get('headers', {})
            if lp_headers and isinstance(lp_headers, dict):
                if 'anthropic-beta' in lp_headers:
                    filtered = filter_beta_header(lp_headers['anthropic-beta'])
                    if filtered:
                        lp_headers['anthropic-beta'] = filtered
                    else:
                        del lp_headers['anthropic-beta']

        # --- Filter anthropic_beta from request body (list format) ---

        # Location 5: data.anthropic_beta (list of beta flags in request body)
        if 'anthropic_beta' in data:
            beta_val = data['anthropic_beta']
            if isinstance(beta_val, list):
                filtered = filter_beta_list(beta_val)
                if filtered:
                    data['anthropic_beta'] = filtered
                else:
                    del data['anthropic_beta']
            elif isinstance(beta_val, str):
                filtered = filter_beta_header(beta_val)
                if filtered:
                    data['anthropic_beta'] = filtered
                else:
                    del data['anthropic_beta']

        # Location 6: extra_body.anthropic_beta
        extra_body = data.get('extra_body', {})
        if extra_body and isinstance(extra_body, dict):
            if 'anthropic_beta' in extra_body:
                beta_val = extra_body['anthropic_beta']
                if isinstance(beta_val, list):
                    filtered = filter_beta_list(beta_val)
                    if filtered:
                        extra_body['anthropic_beta'] = filtered
                    else:
                        del extra_body['anthropic_beta']
                elif isinstance(beta_val, str):
                    filtered = filter_beta_header(beta_val)
                    if filtered:
                        extra_body['anthropic_beta'] = filtered
                    else:
                        del extra_body['anthropic_beta']

        # Location 7: litellm_params.extra_body.anthropic_beta
        if litellm_params and isinstance(litellm_params, dict):
            lb_extra_body = litellm_params.get('extra_body', {})
            if lb_extra_body and isinstance(lb_extra_body, dict):
                if 'anthropic_beta' in lb_extra_body:
                    beta_val = lb_extra_body['anthropic_beta']
                    if isinstance(beta_val, list):
                        filtered = filter_beta_list(beta_val)
                        if filtered:
                            lb_extra_body['anthropic_beta'] = filtered
                        else:
                            del lb_extra_body['anthropic_beta']
                    elif isinstance(beta_val, str):
                        filtered = filter_beta_header(beta_val)
                        if filtered:
                            lb_extra_body['anthropic_beta'] = filtered
                        else:
                            del lb_extra_body['anthropic_beta']

        if removed_flags:
            if verbose or verbose_full:
                print(f"🛡️ AnthropicParamsFilterHook: Filtered unsupported Bedrock beta flags for {model}: {removed_flags}", flush=True)
            else:
                # Always print this at info level since it prevents errors
                print(f"🛡️ Filtered unsupported Bedrock beta flags: {removed_flags}", flush=True)

    def _needs_max_tokens_limit(self, model: str) -> bool:
        """Check if model needs max_tokens limiting when used via Anthropic passthrough."""
        if not model:
            return False

        model_str = str(model).lower()

        # Models that need max_tokens capped to 8192 when accessed via Anthropic passthrough
        limited_models = {
            'gpt-4o',
            'gpt-4o-mini',
        }

        return model_str in limited_models

    def _limit_max_tokens(self, data: Dict[str, Any], model: str, max_limit: int = 8192) -> None:
        """
        Limit max_tokens for specific models to prevent 'max_tokens is too large' errors.

        Some Azure models have lower max_tokens limits than the defaults used by
        Anthropic passthrough. This method caps max_tokens to a safe value.
        """
        if not self._needs_max_tokens_limit(model):
            return

        modified = []

        # Check and limit max_tokens in data
        if 'max_tokens' in data:
            original = data['max_tokens']
            if original and original > max_limit:
                data['max_tokens'] = max_limit
                modified.append(f"data.max_tokens: {original} -> {max_limit}")

        # Check and limit in extra_body
        extra_body = data.get('extra_body', {})
        if extra_body and isinstance(extra_body, dict):
            if 'max_tokens' in extra_body:
                original = extra_body['max_tokens']
                if original and original > max_limit:
                    extra_body['max_tokens'] = max_limit
                    modified.append(f"extra_body.max_tokens: {original} -> {max_limit}")

        # Check and limit in litellm_params
        litellm_params = data.get('litellm_params', {})
        if litellm_params:
            if 'max_tokens' in litellm_params:
                original = litellm_params['max_tokens']
                if original and original > max_limit:
                    litellm_params['max_tokens'] = max_limit
                    modified.append(f"litellm_params.max_tokens: {original} -> {max_limit}")

            # Check in litellm_params.extra_body
            lb_extra_body = litellm_params.get('extra_body', {})
            if lb_extra_body and isinstance(lb_extra_body, dict):
                if 'max_tokens' in lb_extra_body:
                    original = lb_extra_body['max_tokens']
                    if original and original > max_limit:
                        lb_extra_body['max_tokens'] = max_limit
                        modified.append(f"litellm_params.extra_body.max_tokens: {original} -> {max_limit}")

        if modified:
            if verbose or verbose_full:
                print(f"🔧 AnthropicParamsFilterHook: Limited max_tokens for {model}: {modified}", flush=True)

    def _filter_anthropic_params(self, data: Dict[str, Any], model: str) -> None:
        """
        Remove Anthropic-specific parameters from the request data for non-Anthropic models.

        This prevents errors when Anthropic-specific parameters are sent to other providers.

        Anthropic-specific parameters that may be present:
        - context_management: Anthropic beta feature for context management
        - enable_caching: Custom parameter for controlling Anthropic caching hook
        - thinking: Anthropic extended thinking parameter
        - output_config: Anthropic adaptive-thinking effort control. LiteLLM's
          Anthropic/Bedrock-Claude transformations map ``reasoning_effort`` to
          ``output_config={"effort": ...}`` for Claude 4.5/4.6/4.7 / Opus 4.5.
          The OpenAI/Azure code paths neither produce nor strip it, so if it
          reaches an Azure deployment (e.g. via a routing/fallback group that
          mixes Claude and Azure models, or a model-group default) Azure rejects
          the whole request with:
            AzureException - Unrecognized request argument supplied: output_config
          Filtering it here keeps it Anthropic-only.

        New Anthropic-only params should be added to ``anthropic_params`` below so
        they are only ever passed through to actual Anthropic/Bedrock-Claude
        models — never leaked to OpenAI/Azure/other providers.
        """
        if self._is_anthropic_provider(model):
            # Model is Anthropic, don't filter anything
            return

        anthropic_params = [
            'context_management',
            'enable_caching',  # This is handled by caching hook, but filter as safety net
            'thinking',  # Anthropic extended thinking parameter
            'output_config',  # Anthropic adaptive-thinking effort (reasoning_effort -> output_config)
        ]

        removed_params = []

        # Filter top-level data dict
        for param in anthropic_params:
            if param in data:
                data.pop(param)
                removed_params.append(f"data.{param}")

        # Filter extra_body
        extra_body = data.get('extra_body', {})
        if extra_body and isinstance(extra_body, dict):
            for param in anthropic_params:
                if param in extra_body:
                    extra_body.pop(param)
                    removed_params.append(f"extra_body.{param}")

        # Filter litellm_params.extra_body
        litellm_params = data.get('litellm_params', {})
        if litellm_params:
            lb_extra_body = litellm_params.get('extra_body', {})
            if lb_extra_body and isinstance(lb_extra_body, dict):
                for param in anthropic_params:
                    if param in lb_extra_body:
                        lb_extra_body.pop(param)
                        removed_params.append(f"litellm_params.extra_body.{param}")

        if removed_params:
            if verbose or verbose_full:
                print(f"🧹 AnthropicParamsFilterHook: Filtered Anthropic params for {model}: {removed_params}", flush=True)

    def _has_thinking_enabled(self, data: Dict[str, Any]) -> bool:
        """
        Check if thinking parameter is present and enabled in the request.

        Thinking parameter format:
        {
            "type": "enabled",
            "budget_tokens": 16384
        }
        """
        # Check top-level data dict
        if 'thinking' in data:
            thinking = data.get('thinking', {})
            if isinstance(thinking, dict) and thinking.get('type') == 'enabled':
                return True

        # Check extra_body
        extra_body = data.get('extra_body', {})
        if extra_body and isinstance(extra_body, dict):
            if 'thinking' in extra_body:
                thinking = extra_body.get('thinking', {})
                if isinstance(thinking, dict) and thinking.get('type') == 'enabled':
                    return True

        # Check litellm_params.extra_body
        litellm_params = data.get('litellm_params', {})
        if litellm_params:
            lb_extra_body = litellm_params.get('extra_body', {})
            if lb_extra_body and isinstance(lb_extra_body, dict):
                if 'thinking' in lb_extra_body:
                    thinking = lb_extra_body.get('thinking', {})
                    if isinstance(thinking, dict) and thinking.get('type') == 'enabled':
                        return True

        return False

    def _force_temperature_for_thinking(self, data: Dict[str, Any], model: str) -> None:
        """
        Force temperature=1 when thinking is enabled for Anthropic models.

        Anthropic API requires temperature to be exactly 1 when thinking is enabled.
        This prevents the error: "temperature may only be set to 1 when thinking is enabled"
        """
        if not self._is_anthropic_provider(model):
            # Not an Anthropic model, nothing to do
            return

        if self._is_no_sampling_params_model(model):
            # Newer Claude models reject temperature entirely — never force it.
            return

        if not self._has_thinking_enabled(data):
            # Thinking not enabled, nothing to do
            return

        modified = []

        # Set in top-level data dict
        if 'temperature' not in data or data['temperature'] != 1:
            original = data.get('temperature', 'not set')
            data['temperature'] = 1
            modified.append(f"data.temperature: {original} -> 1")

        # Set in extra_body
        extra_body = data.get('extra_body', {})
        if extra_body and isinstance(extra_body, dict):
            if 'temperature' in extra_body and extra_body['temperature'] != 1:
                original = extra_body['temperature']
                extra_body['temperature'] = 1
                modified.append(f"extra_body.temperature: {original} -> 1")

        # Set in litellm_params
        litellm_params = data.get('litellm_params', {})
        if litellm_params:
            if 'temperature' in litellm_params and litellm_params['temperature'] != 1:
                original = litellm_params['temperature']
                litellm_params['temperature'] = 1
                modified.append(f"litellm_params.temperature: {original} -> 1")

            # Set in litellm_params.extra_body
            lb_extra_body = litellm_params.get('extra_body', {})
            if lb_extra_body and isinstance(lb_extra_body, dict):
                if 'temperature' in lb_extra_body and lb_extra_body['temperature'] != 1:
                    original = lb_extra_body['temperature']
                    lb_extra_body['temperature'] = 1
                    modified.append(f"litellm_params.extra_body.temperature: {original} -> 1")

        if modified:
            if verbose or verbose_full:
                print(f"🧠 AnthropicParamsFilterHook: Forced temperature=1 for thinking mode ({model}): {modified}", flush=True)

    # Models that reject "temperature AND top_p both specified" with HTTP 400 even
    # when temperature is implicit. For these we always strip top_p, regardless of
    # whether temperature is present in the visible request data — litellm or the
    # router may inject temperature in a sub-dict the hook can't reach. List
    # mirrors src.enums.anthropic_stronger_exclusions; duplicated here because the
    # hook runs in the litellm conda env which can't import from h2ogpt's src tree.
    _STRONGER_EXCLUSIONS_FRAGMENTS = (
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
        "claude-opus-4-1",
        "claude-opus-4-5",
        "claude-opus-4-6",
        "claude-sonnet-4-6",
    )

    def _is_stronger_exclusion_model(self, model: str) -> bool:
        if not model:
            return False
        s = str(model).lower()
        return any(frag in s for frag in self._STRONGER_EXCLUSIONS_FRAGMENTS)

    # Newer Claude models that reject the sampling params temperature/top_p/top_k
    # ENTIRELY with HTTP 400 ("`temperature` is deprecated for this model."), and
    # reject the legacy extended-thinking enabled/budget_tokens API ("use
    # thinking.type.adaptive and output_config.effort"). They use adaptive
    # thinking. Handled like OpenAI o1 / gpt-5 reasoning models: all sampling
    # params are stripped, and thinking is forced to adaptive. Mirrors
    # src.enums.anthropic_no_sampling_params; duplicated here because the hook
    # runs in the litellm env and can't import h2ogpt's src tree.
    _NO_SAMPLING_PARAMS_FRAGMENTS = (
        "claude-fable-5",
        "claude-mythos",       # mythos-5 and mythos-preview
        "claude-opus-4-7",
        "claude-opus-4-8",
        "claude-sonnet-5",
    )

    def _is_no_sampling_params_model(self, model: str) -> bool:
        if not model:
            return False
        s = str(model).lower()
        return any(frag in s for frag in self._NO_SAMPLING_PARAMS_FRAGMENTS)

    def _remove_sampling_params_for_anthropic(self, data: Dict[str, Any], model: str) -> None:
        """Strip temperature/top_p/top_k for newer Claude models that reject them.

        Fable 5 / Mythos, Opus 4.7+, and Sonnet 5 return HTTP 400 if any of
        temperature, top_p, or top_k is present (each is "deprecated for this
        model"). Remove all three from every location litellm/the router might
        place them — the same dicts handled by _remove_top_p_for_anthropic.
        """
        if not self._is_anthropic_provider(model):
            return

        removed = []

        def _strip(d: Dict[str, Any], path_label: str) -> None:
            if not isinstance(d, dict):
                return
            for param in ("temperature", "top_p", "top_k"):
                if param in d:
                    removed.append(f"{path_label}.{param}={d[param]}")
                    del d[param]

        extra_body = data.get('extra_body', {})
        litellm_params = data.get('litellm_params', {})
        _strip(data, "data")
        _strip(extra_body, "extra_body")
        _strip(litellm_params, "litellm_params")
        if isinstance(litellm_params, dict):
            _strip(litellm_params.get('extra_body', {}), "litellm_params.extra_body")

        if removed and (verbose or verbose_full):
            print(f"🧹 AnthropicParamsFilterHook: Removed deprecated sampling params for Anthropic model ({model}): {removed}", flush=True)

    @staticmethod
    def _budget_tokens_to_effort(budget_tokens: Any) -> Optional[str]:
        """Map a legacy thinking budget_tokens value onto an adaptive effort tier.

        The Anthropic guidance for these models is "use thinking.type.adaptive
        AND output_config.effort", so when we drop budget_tokens we preserve the
        user's intended reasoning depth by translating it to an effort level
        rather than silently falling back to the default.
        """
        try:
            budget = int(budget_tokens)
        except (TypeError, ValueError):
            return None
        if budget <= 0:
            return None
        if budget >= 16384:
            return "high"
        if budget >= 4096:
            return "medium"
        return "low"

    def _force_adaptive_thinking_for_no_sampling(self, data: Dict[str, Any], model: str) -> None:
        """Convert legacy enabled/budget_tokens thinking to adaptive for newer Claude.

        Fable 5 / Mythos, Opus 4.7+, Sonnet 5 reject
        ``thinking={"type": "enabled", "budget_tokens": N}`` (HTTP 400, "use
        thinking.type.adaptive and output_config.effort"). Rewrite any such
        thinking block to ``{"type": "adaptive"}`` so the request is accepted, and
        translate the dropped ``budget_tokens`` into ``output_config.effort`` (when
        no output_config is already present in that dict) so reasoning depth is
        preserved rather than silently reset to the default.
        """
        if not self._is_anthropic_provider(model):
            return

        modified = []

        def _convert(d: Dict[str, Any], path_label: str) -> None:
            if not isinstance(d, dict):
                return
            thinking = d.get('thinking')
            if isinstance(thinking, dict) and thinking.get('type') == 'enabled':
                effort = self._budget_tokens_to_effort(thinking.get('budget_tokens'))
                d['thinking'] = {'type': 'adaptive'}
                if effort is not None and 'output_config' not in d:
                    d['output_config'] = {'effort': effort}
                    path_label += f" (effort={effort})"
                modified.append(path_label)

        extra_body = data.get('extra_body', {})
        litellm_params = data.get('litellm_params', {})
        _convert(data, "data")
        _convert(extra_body, "extra_body")
        _convert(litellm_params, "litellm_params")
        if isinstance(litellm_params, dict):
            _convert(litellm_params.get('extra_body', {}), "litellm_params.extra_body")

        if modified and (verbose or verbose_full):
            print(f"🧠 AnthropicParamsFilterHook: Converted thinking to adaptive for Anthropic model ({model}): {modified}", flush=True)

    def _remove_top_p_for_anthropic(self, data: Dict[str, Any], model: str) -> None:
        """
        Remove top_p when temperature is also set for Anthropic models.

        Anthropic API does not allow both temperature and top_p to be specified
        simultaneously, so when both are present we keep temperature and remove
        top_p. This prevents the error:
        "temperature and top_p cannot both be specified".

        For stricter reasoning-era models (opus 4.1+, sonnet/haiku 4.5+, opus
        4.5+/4.6+, sonnet 4.6+) the API rejects the request even when only
        top_p is in our visible payload — litellm/the router may inject
        temperature in a sub-dict the hook can't reach. Strip top_p
        unconditionally for those.
        """
        if not self._is_anthropic_provider(model):
            # Not an Anthropic model, nothing to do
            return

        is_strict = self._is_stronger_exclusion_model(model)
        removed = []

        def _strip_top_p(d: Dict[str, Any], path_label: str, require_temp: bool) -> None:
            if not isinstance(d, dict):
                return
            if 'top_p' not in d:
                return
            if require_temp and 'temperature' not in d:
                return
            removed.append(f"{path_label}.top_p={d['top_p']}")
            del d['top_p']

        extra_body = data.get('extra_body', {})
        litellm_params = data.get('litellm_params', {})

        # Strict reasoning-era models (opus 4.1+, sonnet/haiku 4.5+,
        # opus 4.5+/4.6+, sonnet 4.6+) reject top_p outright — litellm
        # or the router may inject temperature in a sub-dict we can't
        # see, so strip top_p UNCONDITIONALLY for those (matches the
        # docstring's "Strip top_p unconditionally for those" intent
        # and tests/test_langchain_1x_runtime.py
        # ::test_anthropic_hook_strips_top_p_for_strict_models_with_only_top_p).
        # For non-strict Anthropic models the API only errors when
        # BOTH temperature and top_p are present in the same dict, so
        # we keep the historical "drop top_p when temp is in the same
        # dict" behavior.
        require_temp = not is_strict

        _strip_top_p(data, "data", require_temp)
        _strip_top_p(extra_body, "extra_body", require_temp)
        _strip_top_p(litellm_params, "litellm_params", require_temp)
        if isinstance(litellm_params, dict):
            _strip_top_p(litellm_params.get('extra_body', {}), "litellm_params.extra_body", require_temp)

        # Cross-location for non-strict models: temperature in data, top_p in extra_body
        if not is_strict and 'temperature' in data:
            _strip_top_p(extra_body, "extra_body (temperature in data)", require_temp=False)

        if removed:
            if verbose or verbose_full:
                tag = "strict" if is_strict else "lenient"
                print(f"🧹 AnthropicParamsFilterHook: Removed top_p ({tag}) for Anthropic model ({model}): {removed}", flush=True)

    def _get_thinking_budget_tokens(self, data: Dict[str, Any]) -> int:
        """
        Get the thinking budget_tokens value from the request.

        Returns:
            The budget_tokens value if thinking is enabled, 0 otherwise
        """
        # Check top-level data dict
        if 'thinking' in data:
            thinking = data.get('thinking', {})
            if isinstance(thinking, dict) and thinking.get('type') == 'enabled':
                return thinking.get('budget_tokens', 0)

        # Check extra_body
        extra_body = data.get('extra_body', {})
        if extra_body and isinstance(extra_body, dict):
            if 'thinking' in extra_body:
                thinking = extra_body.get('thinking', {})
                if isinstance(thinking, dict) and thinking.get('type') == 'enabled':
                    return thinking.get('budget_tokens', 0)

        # Check litellm_params.extra_body
        litellm_params = data.get('litellm_params', {})
        if litellm_params:
            lb_extra_body = litellm_params.get('extra_body', {})
            if lb_extra_body and isinstance(lb_extra_body, dict):
                if 'thinking' in lb_extra_body:
                    thinking = lb_extra_body.get('thinking', {})
                    if isinstance(thinking, dict) and thinking.get('type') == 'enabled':
                        return thinking.get('budget_tokens', 0)

        return 0

    def _ensure_max_tokens_for_thinking(self, data: Dict[str, Any], model: str) -> None:
        """
        Ensure max_tokens is greater than thinking.budget_tokens for Anthropic models.

        Anthropic API requires: max_tokens > thinking.budget_tokens
        This prevents the error: "max_tokens must be greater than thinking.budget_tokens"

        Strategy:
        - If max_tokens is not set or too small, set it to thinking.budget_tokens + 8192
        - The buffer (8192) allows room for the actual response after thinking
        """
        if not self._is_anthropic_provider(model):
            # Not an Anthropic model, nothing to do
            return

        thinking_budget = self._get_thinking_budget_tokens(data)
        if thinking_budget == 0:
            # No thinking budget, nothing to do
            return

        # Minimum buffer for response after thinking (in tokens)
        # 8192 tokens is ~6000 words, enough for comprehensive responses
        RESPONSE_BUFFER = 8192
        required_max_tokens = thinking_budget + RESPONSE_BUFFER

        modified = []

        # Check and fix max_tokens in top-level data dict
        current_max_tokens = data.get('max_tokens', 0)
        if current_max_tokens == 0 or current_max_tokens <= thinking_budget:
            original = current_max_tokens if current_max_tokens > 0 else 'not set'
            data['max_tokens'] = required_max_tokens
            modified.append(f"data.max_tokens: {original} -> {required_max_tokens} (thinking_budget={thinking_budget} + buffer={RESPONSE_BUFFER})")

        # Check extra_body
        extra_body = data.get('extra_body', {})
        if extra_body and isinstance(extra_body, dict):
            if 'max_tokens' in extra_body:
                current = extra_body['max_tokens']
                if current <= thinking_budget:
                    extra_body['max_tokens'] = required_max_tokens
                    modified.append(f"extra_body.max_tokens: {current} -> {required_max_tokens}")

        # Check litellm_params
        litellm_params = data.get('litellm_params', {})
        if litellm_params:
            if 'max_tokens' in litellm_params:
                current = litellm_params['max_tokens']
                if current <= thinking_budget:
                    litellm_params['max_tokens'] = required_max_tokens
                    modified.append(f"litellm_params.max_tokens: {current} -> {required_max_tokens}")

            # Check in litellm_params.extra_body
            lb_extra_body = litellm_params.get('extra_body', {})
            if lb_extra_body and isinstance(lb_extra_body, dict):
                if 'max_tokens' in lb_extra_body:
                    current = lb_extra_body['max_tokens']
                    if current <= thinking_budget:
                        lb_extra_body['max_tokens'] = required_max_tokens
                        modified.append(f"litellm_params.extra_body.max_tokens: {current} -> {required_max_tokens}")

        if modified:
            if verbose or verbose_full:
                print(f"🧠 AnthropicParamsFilterHook: Ensured max_tokens > thinking.budget_tokens ({model}): {modified}", flush=True)

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Dict[str, Any],
        cache: Any,
        data: Dict[str, Any],
        call_type: str
    ) -> Dict[str, Any]:
        """
        Pre-call hook to filter out Anthropic-specific parameters for non-Anthropic models,
        force temperature=1 when thinking is enabled, remove top_p when temperature is set,
        ensure max_tokens requirements, limit max_tokens for models with restrictions,
        and filter unsupported beta flags for Bedrock.

        This is called before the request is sent to the LLM provider.
        We modify the data dict in-place to:
        1. Force temperature=1 when thinking is enabled for Anthropic models
        2. Remove top_p when temperature is also set for Anthropic models
        3. Ensure max_tokens > thinking.budget_tokens for Anthropic models with thinking
        4. Remove Anthropic-specific parameters if the model is not an Anthropic model
        5. Filter unsupported anthropic-beta flags for Bedrock models
        6. Limit max_tokens for Azure models that have lower limits

        Args:
            user_api_key_dict: User authentication info
            cache: LiteLLM cache instance
            data: Request data dictionary (mutable)
            call_type: Type of call ("completion", etc.)

        Returns:
            Modified data dict with temperature fixed, top_p removed for Anthropic,
            max_tokens ensured, Anthropic-specific parameters filtered,
            Bedrock beta flags whitelisted, and max_tokens limited
        """
        try:
            model = data.get('model', '')

            if verbose_full:
                print(f"🧹 AnthropicParamsFilterHook: async_pre_call_hook called with call_type={call_type}, model={model}", flush=True)

            if self._is_no_sampling_params_model(model):
                # Newer Claude (Fable 5 / Mythos, Opus 4.7+, Sonnet 5): temperature,
                # top_p and top_k are all deprecated (HTTP 400), and the legacy
                # enabled/budget_tokens thinking API is rejected. Convert thinking
                # to adaptive and strip all sampling params — never force
                # temperature=1 (which would itself 400). Handled like OpenAI o1 /
                # gpt-5 reasoning models.
                # ORDERING IS LOAD-BEARING: the adaptive conversion must run before
                # _ensure_max_tokens_for_thinking (below) so the latter sees no
                # enabled/budget_tokens block and is a no-op (adaptive has no
                # budget); otherwise a stale enabled block would inflate max_tokens.
                self._force_adaptive_thinking_for_no_sampling(data, model)
                self._remove_sampling_params_for_anthropic(data, model)
            else:
                # Force temperature=1 when thinking is enabled for Anthropic models
                # This must be done BEFORE filtering Anthropic params to preserve the thinking parameter for Anthropic models
                self._force_temperature_for_thinking(data, model)

                # Remove top_p when temperature is also set for Anthropic models
                # Anthropic API does not allow both to be specified simultaneously
                # This must be done AFTER _force_temperature_for_thinking which may set temperature
                self._remove_top_p_for_anthropic(data, model)

            # Ensure max_tokens > thinking.budget_tokens for Anthropic models
            # This must be done BEFORE filtering Anthropic params to access the thinking budget
            self._ensure_max_tokens_for_thinking(data, model)

            # Filter Anthropic-specific parameters for non-Anthropic models
            self._filter_anthropic_params(data, model)

            # Filter unsupported beta flags for Bedrock models
            # Bedrock only supports a whitelist of beta flags - unsupported ones cause
            # "invalid beta flag" errors (see https://github.com/BerriAI/litellm/issues/16726)
            self._filter_bedrock_beta_flags(data, model)

            # Limit max_tokens for models that need it (e.g., gpt-4o, gpt-4o-mini)
            self._limit_max_tokens(data, model)

            return data

        except Exception as e:
            print(f"🧹 AnthropicParamsFilterHook: Error in async_pre_call_hook: {e}", flush=True)
            import traceback
            traceback.print_exc()
            # On error, return data unchanged to not break the request
            return data


# Create the hook instance that LiteLLM will use
if verbose or verbose_full:
    print(f"🧹 HOOK EXPORT: Creating anthropic_params_filter_hook instance", flush=True)

anthropic_params_filter_hook = AnthropicParamsFilterHook()

if verbose or verbose_full:
    print(f"🧹 HOOK EXPORT: anthropic_params_filter_hook created successfully: {type(anthropic_params_filter_hook)}", flush=True)
    print(f"🧹 HOOK EXPORT: Hook filters: context_management, enable_caching, thinking for non-Anthropic models", flush=True)
    print(f"🧠 HOOK EXPORT: Hook forces: temperature=1 when thinking is enabled for Anthropic models", flush=True)
    print(f"🧹 HOOK EXPORT: Hook removes: top_p when temperature is set for Anthropic models", flush=True)
    print(f"🔧 HOOK EXPORT: Hook limits: max_tokens to 8192 for gpt-4o, gpt-4o-mini", flush=True)
    print(f"🛡️ HOOK EXPORT: Hook filters: unsupported anthropic-beta flags for Bedrock models", flush=True)
