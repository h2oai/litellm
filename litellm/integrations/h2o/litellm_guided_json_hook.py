#!/usr/bin/env python3
"""
LiteLLM Hook to translate vLLM-style ``guided_json`` into per-provider structured output.

h2oGPT expresses "return JSON matching this schema" with a single vLLM-native
``guided_json`` parameter (a JSON Schema dict), passed through ``extra_body``.
That parameter only means something to a real vLLM server. When the same request
is routed through the LiteLLM proxy to a non-vLLM provider (OpenAI/Azure, Gemini,
Anthropic/Bedrock, ...), ``guided_json`` is meaningless and is silently stripped
by ``AnthropicCachingHook._remove_vllm_params`` — so the model is never told the
schema and can legitimately return ``{}``.

This hook runs FIRST (register it before the caching/params hooks) and converts
``guided_json`` into the best structured-output mechanism the target model/provider
actually supports, in three tiers:

1. **Strict schema** — if ``litellm.supports_response_schema(model)``:
   set ``response_format={"type":"json_schema","json_schema":{...}}``. LiteLLM
   then translates natively per provider (OpenAI/Azure structured outputs, Gemini
   ``response_schema``, Anthropic/Bedrock ``json_tool_call`` tool).

2. **JSON mode** — elif the provider lists ``response_format`` in
   ``get_supported_openai_params``: set ``response_format={"type":"json_object"}``
   and inject the schema (with required keys) into the prompt so the model knows
   the shape.

3. **Prompt only** — else: inject the schema + required-keys instruction into the
   prompt and leave ``response_format`` off (the provider can't enforce it).

In every case the raw ``guided_json`` (and sibling vLLM-only ``guided_*`` /
``stop_token_ids``) is removed from the request so it cannot error downstream.

This is the proxy-side root-cause fix that complements the prompt-only stopgap in
h2oai/h2ogpt_internal#889.
"""

import os
import json
from typing import Any, Dict, List, Optional, Tuple

from litellm.integrations.custom_logger import CustomLogger

verbose = os.getenv('H2OGPT_VERBOSE', '0') == '1'
verbose_full = os.getenv('H2OGPT_VERBOSE_FULL', '0') == '1'

# vLLM-only guided-decoding params that are meaningless to non-vLLM providers.
VLLM_GUIDED_PARAMS = (
    'guided_json',
    'guided_regex',
    'guided_choice',
    'guided_grammar',
    'guided_whitespace_pattern',
    'guided_decoding_backend',
)

# Prompt wording kept identical to h2ogpt's src/enums.py (json_schema_instruction0
# / format_required_keys_instruction) so prompt-only behavior matches the backend.
JSON_SCHEMA_INSTRUCTION = (
    'Ensure you follow this JSON schema, and ensure to use the same key names as '
    'the schema:\n```json\n%s\n```'
)


class GuidedJsonHook(CustomLogger):
    """Translate ``guided_json`` into per-provider structured output before the call."""

    def __init__(self):
        super().__init__()
        self.enabled = True
        if verbose or verbose_full:
            print(f"🧩 GuidedJsonHook: Initialized", flush=True)

    # ------------------------------------------------------------------ helpers

    def _extra_body_locations(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return the dict containers that may hold extra_body params, in priority order.

        h2ogpt sends params via the OpenAI client ``extra_body``, which the SDK
        flattens into the top-level request body; LiteLLM may also surface them
        under ``extra_body`` or ``litellm_params.extra_body``. Mirror the three
        locations AnthropicCachingHook checks.
        """
        locations: List[Dict[str, Any]] = [data]
        extra_body = data.get('extra_body')
        if isinstance(extra_body, dict):
            locations.append(extra_body)
        litellm_params = data.get('litellm_params')
        if isinstance(litellm_params, dict):
            lb = litellm_params.get('extra_body')
            if isinstance(lb, dict):
                locations.append(lb)
        return locations

    def _pop_guided_json(self, data: Dict[str, Any]) -> Optional[Any]:
        """Pop guided_json (and sibling vLLM-only guided params) from all locations.

        Returns the guided_json schema (first one found), or None.
        """
        guided_json = None
        for container in self._extra_body_locations(data):
            for param in VLLM_GUIDED_PARAMS:
                if param in container:
                    value = container.pop(param)
                    if param == 'guided_json' and guided_json is None:
                        guided_json = value
            # vLLM-only sampling param that also errors on most providers
            container.pop('stop_token_ids', None)
        return guided_json

    @staticmethod
    def _coerce_schema(guided_json: Any) -> Optional[Dict[str, Any]]:
        """Coerce guided_json (dict or JSON string) into a schema dict, or None."""
        if isinstance(guided_json, str):
            try:
                guided_json = json.loads(guided_json)
            except (json.JSONDecodeError, TypeError):
                return None
        return guided_json if isinstance(guided_json, dict) else None

    @staticmethod
    def _properties_only(schema: Dict[str, Any]) -> Any:
        """Reduce a wrapped object schema to just its ``properties`` for prompting.

        Mirrors src/gen.py: the validation scaffolding (type/required/$defs/...)
        confuses weaker prompt-only models, so show only the shape.
        """
        if isinstance(schema, dict) and 'properties' in schema:
            return schema['properties']
        return schema

    @staticmethod
    def _required_keys(schema: Dict[str, Any]) -> List[str]:
        required = schema.get('required') if isinstance(schema, dict) else None
        if isinstance(required, (list, tuple)):
            return [k for k in required if isinstance(k, str)]
        return []

    @classmethod
    def _is_strict_safe(cls, schema: Dict[str, Any]) -> bool:
        """True only if the schema satisfies OpenAI strict mode requirements.

        OpenAI strict structured outputs require ``additionalProperties: false`` and
        every property listed in ``required``. Setting strict on a schema with
        optional keys (e.g. the classifier's optional ``rationale``) would error,
        so only enable strict when it's provably safe.
        """
        if not isinstance(schema, dict):
            return False
        props = schema.get('properties')
        if not isinstance(props, dict) or not props:
            return False
        if schema.get('additionalProperties', True) is not False:
            return False
        return set(cls._required_keys(schema)) >= set(props.keys())

    def _inject_schema_into_prompt(self, data: Dict[str, Any], schema: Dict[str, Any]) -> None:
        """Append a schema instruction (+ required keys) to the last user message.

        Appending to the user turn (rather than adding a system message) is the most
        portable choice — some reasoning models reject system turns.
        """
        properties_json = json.dumps(self._properties_only(schema))
        instruction = '\n\n' + (JSON_SCHEMA_INSTRUCTION % properties_json)
        required_keys = self._required_keys(schema)
        if required_keys:
            instruction += (
                '\nAll of these keys are required and must be present with a valid '
                'value: ' + ', '.join('"%s"' % k for k in required_keys) + '.'
            )

        messages = data.get('messages')
        if not isinstance(messages, list) or not messages:
            data['messages'] = [{'role': 'user', 'content': instruction.lstrip()}]
            return

        # Append to the last user-role message; fall back to the last message.
        target_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], dict) and messages[i].get('role') == 'user':
                target_idx = i
                break
        if target_idx is None:
            target_idx = len(messages) - 1

        msg = messages[target_idx]
        if not isinstance(msg, dict):
            messages.append({'role': 'user', 'content': instruction.lstrip()})
            return
        content = msg.get('content')
        if isinstance(content, str):
            msg['content'] = content + instruction
        elif isinstance(content, list):
            # Multimodal content: append a text part.
            content.append({'type': 'text', 'text': instruction})
        else:
            msg['content'] = instruction.lstrip()

    @staticmethod
    def _provider_for(model: str) -> Optional[str]:
        try:
            import litellm
            _, provider, _, _ = litellm.get_llm_provider(model=model)
            return provider
        except Exception:
            return None

    @staticmethod
    def _supports_response_schema(model: str, provider: Optional[str]) -> bool:
        try:
            import litellm
            return bool(litellm.supports_response_schema(model=model, custom_llm_provider=provider))
        except Exception:
            return False

    @staticmethod
    def _supports_json_mode(model: str, provider: Optional[str]) -> bool:
        try:
            import litellm
            params = litellm.get_supported_openai_params(model=model, custom_llm_provider=provider) or []
            return 'response_format' in params
        except Exception:
            return False

    # --------------------------------------------------------------- main hook

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Dict[str, Any],
        cache: Any,
        data: Dict[str, Any],
        call_type: str,
    ) -> Dict[str, Any]:
        """Translate guided_json -> per-provider structured output. Mutates and returns data."""
        try:
            if call_type not in ('completion', 'text_completion', None):
                return data

            guided_json_raw = self._pop_guided_json(data)
            if guided_json_raw is None:
                return data  # nothing to do; sibling params already cleaned

            model = data.get('model', '') or ''
            schema = self._coerce_schema(guided_json_raw)

            # Respect an explicit json_schema response_format from the caller — we
            # only removed the (now redundant) guided_json above.
            existing_rf = data.get('response_format')
            if isinstance(existing_rf, dict) and existing_rf.get('type') == 'json_schema':
                if verbose or verbose_full:
                    print(f"🧩 GuidedJsonHook: caller already set json_schema for {model}; only removed guided_json", flush=True)
                return data

            if schema is None:
                if verbose or verbose_full:
                    print(f"🧩 GuidedJsonHook: guided_json not a usable schema for {model}; removed", flush=True)
                return data

            provider = self._provider_for(model)

            if self._supports_response_schema(model, provider):
                # Tier 1: native strict/structured schema.
                strict = self._is_strict_safe(schema)
                data['response_format'] = {
                    'type': 'json_schema',
                    'json_schema': {
                        'name': 'response',
                        'schema': schema,
                        'strict': strict,
                    },
                }
                if verbose or verbose_full:
                    print(f"🧩 GuidedJsonHook: {model} ({provider}) -> json_schema (strict={strict})", flush=True)
            elif self._supports_json_mode(model, provider):
                # Tier 2: JSON mode + schema in prompt.
                data['response_format'] = {'type': 'json_object'}
                self._inject_schema_into_prompt(data, schema)
                if verbose or verbose_full:
                    print(f"🧩 GuidedJsonHook: {model} ({provider}) -> json_object + prompt schema", flush=True)
            else:
                # Tier 3: prompt only; provider can't enforce response_format.
                data.pop('response_format', None)
                self._inject_schema_into_prompt(data, schema)
                if verbose or verbose_full:
                    print(f"🧩 GuidedJsonHook: {model} ({provider}) -> prompt-only schema", flush=True)

            return data

        except Exception as e:
            print(f"🧩 GuidedJsonHook: Error in async_pre_call_hook: {e}", flush=True)
            import traceback
            traceback.print_exc()
            # On error, return data unchanged to not break the request.
            return data


# Create the hook instance that LiteLLM will use
if verbose or verbose_full:
    print(f"🧩 HOOK EXPORT: Creating guided_json_hook instance", flush=True)

guided_json_hook = GuidedJsonHook()
