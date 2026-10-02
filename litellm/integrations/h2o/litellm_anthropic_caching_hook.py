#!/usr/bin/env python3
"""
LiteLLM Hook for Dynamic Anthropic Prompt Caching

This hook automatically adds cache_control breakpoints to messages when:
1. The model is using Anthropic or Bedrock Anthropic provider
2. The request includes `enable_caching=True` in extra_body
3. The messages format supports caching

Supports:
- Native Anthropic API (anthropic/claude-*)
- AWS Bedrock Anthropic (bedrock/anthropic.claude-*)
- Automatic cache_control injection based on message structure
- Compatible with both streaming and non-streaming requests

Cache Strategy:
- System prompts: Always cached (highest priority)
- Recent user messages: Cached in reverse order (up to 3 messages)
- Maximum cache control points: 4 for Bedrock, 3 for native Anthropic

Based on process_messages() from gpt_langchain.py and add_cache_points_to_bedrock_messages()
"""

import os
import copy
from typing import Dict, Any, List, Optional
from litellm.integrations.custom_logger import CustomLogger

verbose = os.getenv('H2OGPT_VERBOSE', '0') == '1'
verbose_full = os.getenv('H2OGPT_VERBOSE_FULL', '0') == '1'


class AnthropicCachingHook(CustomLogger):
    """Custom LiteLLM hook to automatically add cache_control to Anthropic messages."""

    def __init__(self):
        super().__init__()
        self.enabled = True
        if verbose or verbose_full:
            print(f"💾 AnthropicCachingHook: Initialized", flush=True)

    def _is_anthropic_provider(self, model: str) -> bool:
        """
        Check if the model uses Anthropic or Bedrock Anthropic provider.

        Uses an extensive whitelist of known Anthropic model names to avoid
        false positives when non-Anthropic models use the Anthropic passthrough endpoint.
        """
        if not model:
            return False

        model_str = str(model).lower()

        # Extensive list of known Anthropic model names (native and bedrock)
        # This prevents non-Anthropic models from triggering caching when using
        # the Anthropic passthrough endpoint
        anthropic_models = {
            # === Claude 4 Family (Latest) ===
            # Opus 4.5
            'claude-opus-4-5-20251101',
            'anthropic/claude-opus-4-5-20251101',
            'claude-opus-4-5',
            'anthropic/claude-opus-4-5',
            'claude-opus-4.5',
            'anthropic/claude-opus-4.5',
            # Opus 4.1
            'claude-opus-4-1-20250805',
            'anthropic/claude-opus-4-1-20250805',
            'claude-opus-4.1',
            'anthropic/claude-opus-4.1',
            'claude-opus-4',
            'anthropic/claude-opus-4',
            # Sonnet 4.5
            'claude-sonnet-4-5-20250929',
            'anthropic/claude-sonnet-4-5-20250929',
            'claude-sonnet-4-5',
            'anthropic/claude-sonnet-4-5',
            'claude-sonnet-4.5',
            'anthropic/claude-sonnet-4.5',
            # Sonnet 4
            'claude-sonnet-4-20250514',
            'anthropic/claude-sonnet-4-20250514',
            'claude-sonnet-4',
            'anthropic/claude-sonnet-4',
            # Haiku 4.5
            'claude-haiku-4-5-20251001',
            'anthropic/claude-haiku-4-5-20251001',
            'claude-haiku-4-5',
            'anthropic/claude-haiku-4-5',
            'claude-haiku-4.5',
            'anthropic/claude-haiku-4.5',
            'claude-haiku-4',
            'anthropic/claude-haiku-4',

            # === Claude 3.7 Family ===
            'claude-3-7-sonnet-20250219',
            'anthropic/claude-3-7-sonnet-20250219',
            'claude-3-7-sonnet-20250219-litellm',
            'anthropic/claude-3-7-sonnet-20250219-litellm',
            'claude-3.7-sonnet',
            'anthropic/claude-3.7-sonnet',

            # === Claude 3.5 Family ===
            # Sonnet 3.5
            'claude-3-5-sonnet-20241022',
            'anthropic/claude-3-5-sonnet-20241022',
            'claude-3-5-sonnet-20240620',
            'anthropic/claude-3-5-sonnet-20240620',
            'claude-3.5-sonnet',
            'anthropic/claude-3.5-sonnet',
            # Haiku 3.5
            'claude-3-5-haiku-20241022',
            'anthropic/claude-3-5-haiku-20241022',
            'claude-3.5-haiku',
            'anthropic/claude-3.5-haiku',

            # === Claude 3 Family (Original) ===
            # Opus 3
            'claude-3-opus-20240229',
            'anthropic/claude-3-opus-20240229',
            'claude-3-opus',
            'anthropic/claude-3-opus',
            # Sonnet 3
            'claude-3-sonnet-20240229',
            'anthropic/claude-3-sonnet-20240229',
            'claude-3-sonnet',
            'anthropic/claude-3-sonnet',
            # Haiku 3
            'claude-3-haiku-20240307',
            'anthropic/claude-3-haiku-20240307',
            'claude-3-haiku',
            'anthropic/claude-3-haiku',

            # === Claude 2 Family ===
            'claude-2.1',
            'anthropic/claude-2.1',
            'claude-2.0',
            'anthropic/claude-2.0',
            'claude-2',
            'anthropic/claude-2',

            # === Claude Instant Family ===
            'claude-instant-1.2',
            'anthropic/claude-instant-1.2',
            'claude-instant-1.1',
            'anthropic/claude-instant-1.1',
            'claude-instant-1.0',
            'anthropic/claude-instant-1.0',
            'claude-instant-1',
            'anthropic/claude-instant-1',
            'claude-instant',
            'anthropic/claude-instant',

            # === Bedrock Anthropic Models ===
            # Claude 4 on Bedrock (with regional prefixes)
            'bedrock/anthropic.claude-opus-4-5-20251101-v1:0',
            'bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0',
            'bedrock/global.anthropic.claude-opus-4-5-20251101-v1:0',
            'bedrock/anthropic.claude-opus-4-1-20250805-v1:0',
            'bedrock/us.anthropic.claude-opus-4-1-20250805-v1:0',
            'bedrock/global.anthropic.claude-opus-4-1-20250805-v1:0',
            'bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0',
            'bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0',
            'bedrock/global.anthropic.claude-sonnet-4-5-20250929-v1:0',
            'bedrock/anthropic.claude-sonnet-4-20250514-v1:0',
            'bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0',
            'bedrock/global.anthropic.claude-sonnet-4-20250514-v1:0',
            'bedrock/anthropic.claude-haiku-4-5-20251001-v1:0',
            'bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0',
            'bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0',
            # Claude 3.7 on Bedrock
            'bedrock/anthropic.claude-3-7-sonnet-20250219-v1:0',
            'bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0',
            'bedrock/global.anthropic.claude-3-7-sonnet-20250219-v1:0',
            # Claude 3.5 on Bedrock (with regional prefixes)
            'bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0',
            'bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0',
            'bedrock/global.anthropic.claude-3-5-sonnet-20241022-v2:0',
            'bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0',
            'bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0',
            'bedrock/global.anthropic.claude-3-5-sonnet-20240620-v1:0',
            'bedrock/anthropic.claude-3-5-haiku-20241022-v1:0',
            'bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0',
            'bedrock/global.anthropic.claude-3-5-haiku-20241022-v1:0',
            # Claude 3 on Bedrock (with regional prefixes)
            'bedrock/anthropic.claude-3-opus-20240229-v1:0',
            'bedrock/us.anthropic.claude-3-opus-20240229-v1:0',
            'bedrock/global.anthropic.claude-3-opus-20240229-v1:0',
            'bedrock/anthropic.claude-3-sonnet-20240229-v1:0',
            'bedrock/us.anthropic.claude-3-sonnet-20240229-v1:0',
            'bedrock/global.anthropic.claude-3-sonnet-20240229-v1:0',
            'bedrock/anthropic.claude-3-haiku-20240307-v1:0',
            'bedrock/us.anthropic.claude-3-haiku-20240307-v1:0',
            'bedrock/global.anthropic.claude-3-haiku-20240307-v1:0',
            # Claude 2 on Bedrock (with regional prefixes)
            'bedrock/anthropic.claude-v2:1',
            'bedrock/us.anthropic.claude-v2:1',
            'bedrock/global.anthropic.claude-v2:1',
            'bedrock/anthropic.claude-v2:0',
            'bedrock/us.anthropic.claude-v2:0',
            'bedrock/global.anthropic.claude-v2:0',
            'bedrock/anthropic.claude-v2',
            'bedrock/us.anthropic.claude-v2',
            'bedrock/global.anthropic.claude-v2',
            # Claude Instant on Bedrock (with regional prefixes and versions)
            'bedrock/anthropic.claude-instant-v1:2',
            'bedrock/us.anthropic.claude-instant-v1:2',
            'bedrock/global.anthropic.claude-instant-v1:2',
            'bedrock/anthropic.claude-instant-v1:1',
            'bedrock/us.anthropic.claude-instant-v1:1',
            'bedrock/global.anthropic.claude-instant-v1:1',
            'bedrock/anthropic.claude-instant-v1:0',
            'bedrock/us.anthropic.claude-instant-v1:0',
            'bedrock/global.anthropic.claude-instant-v1:0',
            'bedrock/anthropic.claude-instant-v1',
            'bedrock/us.anthropic.claude-instant-v1',
            'bedrock/global.anthropic.claude-instant-v1',

            # === Vertex AI Anthropic Models ===
            'vertex_ai/claude-3-5-sonnet@20240620',
            'vertex_ai/claude-3-5-sonnet-v2@20241022',
            'vertex_ai/claude-3-5-haiku@20241022',
            'vertex_ai/claude-3-opus@20240229',
            'vertex_ai/claude-3-sonnet@20240229',
            'vertex_ai/claude-3-haiku@20240307',

            # === Generic/Alias Names ===
            'claude-opus',
            'anthropic/claude-opus',
            'claude-sonnet',
            'anthropic/claude-sonnet',
            'claude-haiku',
            'anthropic/claude-haiku',
        }

        # Check if model exactly matches any known Anthropic model
        if model_str in anthropic_models:
            return True

        # Check for bedrock/ prefix with anthropic.claude in the model ID
        # This handles any new bedrock anthropic models not in the list above
        if model_str.startswith('bedrock/') and 'anthropic.claude' in model_str:
            return True

        # Check for anthropic/ prefix (handles any new models with this prefix)
        if model_str.startswith('anthropic/'):
            return True

        return False

    def _is_bedrock_provider(self, model: str) -> bool:
        """Check if the model uses Bedrock provider."""
        if not model:
            return False

        model_str = str(model).lower()
        return model_str.startswith('bedrock/')

    def _remove_enable_caching_param(self, data: Dict[str, Any]) -> bool:
        """
        Remove enable_caching parameter from all locations in the data dict.

        This prevents the parameter from being sent to any provider (Anthropic or otherwise),
        since it's a custom parameter for controlling this hook's behavior.

        Returns:
            True if enable_caching was found and was set to True, False otherwise
        """
        enable_caching = False

        # Location 1: Top-level data dict (when unpacked by LiteLLM)
        if 'enable_caching' in data:
            enable_caching = data.pop('enable_caching', False)
            if verbose or verbose_full:
                print(f"💾 AnthropicCachingHook: Found and removed enable_caching={enable_caching} from top-level data", flush=True)

        # Location 2: extra_body
        extra_body = data.get('extra_body', {})
        if extra_body and 'enable_caching' in extra_body:
            enable_caching = extra_body.pop('enable_caching', False)
            if verbose or verbose_full:
                print(f"💾 AnthropicCachingHook: Found and removed enable_caching={enable_caching} from extra_body", flush=True)

        # Location 3: litellm_params.extra_body
        litellm_params = data.get('litellm_params', {})
        litellm_extra_body = litellm_params.get('extra_body', {})
        if litellm_extra_body and 'enable_caching' in litellm_extra_body:
            enable_caching = litellm_extra_body.pop('enable_caching', False)
            if verbose or verbose_full:
                print(f"💾 AnthropicCachingHook: Found and removed enable_caching={enable_caching} from litellm_params.extra_body", flush=True)

        return enable_caching

    def _remove_vllm_params(self, data: Dict[str, Any]) -> None:
        """
        Remove VLLM-specific parameters from extra_body if present.

        This is a safety net in case the is_litellm detection logic fails in gen.py/gpt_langchain.py
        and VLLM parameters make it through to Anthropic endpoints.

        VLLM parameters that are incompatible with Anthropic API:
        - stop_token_ids
        - repetition_penalty
        - guided_json
        - guided_regex
        - guided_choice
        - guided_grammar
        - guided_whitespace_pattern
        """
        vllm_params = [
            'stop_token_ids',
            'repetition_penalty',
            'guided_json',
            'guided_regex',
            'guided_choice',
            'guided_grammar',
            'guided_whitespace_pattern',
        ]

        removed_params = []

        # Check top-level extra_body
        extra_body = data.get('extra_body', {})
        if extra_body and isinstance(extra_body, dict):
            for param in vllm_params:
                if param in extra_body:
                    extra_body.pop(param)
                    removed_params.append(f"extra_body.{param}")

        # Check litellm_params.extra_body
        litellm_params = data.get('litellm_params', {})
        if litellm_params:
            lb_extra_body = litellm_params.get('extra_body', {})
            if lb_extra_body and isinstance(lb_extra_body, dict):
                for param in vllm_params:
                    if param in lb_extra_body:
                        lb_extra_body.pop(param)
                        removed_params.append(f"litellm_params.extra_body.{param}")

        if removed_params and (verbose or verbose_full):
            model = data.get('model', 'unknown')
            print(f"💾 AnthropicCachingHook: Removed VLLM params for {model}: {removed_params}", flush=True)

    def _check_enable_caching(self, data: Dict[str, Any]) -> bool:
        """
        Check if caching is enabled for this request WITHOUT modifying the data.

        This is a read-only version for use in logging/monitoring contexts.
        Use _should_enable_caching() in the request processing path.

        Returns:
            True if enable_caching is True and model is Anthropic, False otherwise
        """
        # Check if model is Anthropic
        model = data.get('model', '')
        if not self._is_anthropic_provider(model):
            return False

        # Check for enable_caching flag (read-only, no modification)
        if 'enable_caching' in data:
            return bool(data.get('enable_caching', False))

        extra_body = data.get('extra_body', {})
        if extra_body and 'enable_caching' in extra_body:
            return bool(extra_body.get('enable_caching', False))

        litellm_params = data.get('litellm_params', {})
        if litellm_params:
            lb_extra_body = litellm_params.get('extra_body', {})
            if lb_extra_body and 'enable_caching' in lb_extra_body:
                return bool(lb_extra_body.get('enable_caching', False))

        return False

    def _remove_cache_control_from_messages(self, data: Dict[str, Any]) -> None:
        """
        Remove all cache_control and cachePoint markers from messages and system prompts.

        This prevents cache-related parameters from being sent to non-Anthropic providers
        that might try to use their own native caching APIs (like Vertex AI).

        Removes:
        - cache_control from message content items (Anthropic format)
        - cachePoint from message content (Bedrock format)
        - cache_control from system prompt (Anthropic format)
        - cachePoint from system prompt (Bedrock format)
        """
        removed_count = 0

        # Remove from messages array
        messages = data.get('messages', [])
        if messages and isinstance(messages, list):
            for message in messages:
                if not isinstance(message, dict):
                    continue

                content = message.get('content')

                # Handle list content (structured format)
                if isinstance(content, list):
                    filtered_content = []
                    for item in content:
                        if isinstance(item, dict):
                            # Remove cache_control if present
                            if 'cache_control' in item:
                                item.pop('cache_control')
                                removed_count += 1
                            # Skip cachePoint items entirely (Bedrock format)
                            if 'cachePoint' not in item:
                                filtered_content.append(item)
                            else:
                                removed_count += 1
                        else:
                            filtered_content.append(item)
                    message['content'] = filtered_content

        # Remove from system prompt
        system = data.get('system')
        if system:
            if isinstance(system, list):
                filtered_system = []
                for item in system:
                    if isinstance(item, dict):
                        # Remove cache_control if present
                        if 'cache_control' in item:
                            item.pop('cache_control')
                            removed_count += 1
                        # Skip cachePoint items entirely (Bedrock format)
                        if 'cachePoint' not in item:
                            filtered_system.append(item)
                        else:
                            removed_count += 1
                    else:
                        filtered_system.append(item)
                data['system'] = filtered_system

        if removed_count > 0:
            model = data.get('model', 'unknown')
            if verbose or verbose_full:
                print(f"💾 AnthropicCachingHook: Removed {removed_count} cache markers from non-Anthropic model {model}", flush=True)

    def _should_enable_caching(self, data: Dict[str, Any]) -> bool:
        """
        Determine if caching should be enabled for this request.

        Checks:
        1. Model is using Anthropic/Bedrock Anthropic provider
        2. enable_caching flag in extra_body is True

        IMPORTANT: This function REMOVES the enable_caching flag from all locations
        regardless of model type to prevent it from being sent to any provider.
        """
        # Check if model is Anthropic
        model = data.get('model', '')
        if verbose or verbose_full:
            print(f"💾 AnthropicCachingHook: _should_enable_caching called with model={model}", flush=True)

        # ALWAYS remove enable_caching parameter regardless of model type
        # This prevents it from being passed to non-Anthropic providers
        enable_caching = self._remove_enable_caching_param(data)

        if not self._is_anthropic_provider(model):
            if verbose or verbose_full:
                print(f"💾 AnthropicCachingHook: Model {model} is not Anthropic provider, caching disabled (but parameter removed)", flush=True)
            return False

        if verbose or verbose_full:
            print(f"💾 AnthropicCachingHook: Final enable_caching value: {enable_caching}", flush=True)
            print(f"💾 AnthropicCachingHook: All data keys: {list(data.keys())}", flush=True)

        return enable_caching

    def _process_messages_native_anthropic(self, messages: List[Dict], max_cache_controls: int) -> List[Dict]:
        """
        Process messages for native Anthropic API format.

        Based on process_messages() from gpt_langchain.py H2OChatAnthropic3.
        Adds cache_control to user messages in reverse order.

        IMPORTANT: This function receives the REMAINING cache breakpoints available
        for messages after accounting for the system prompt:
        - Total: 4 cache breakpoints (Anthropic limit)
        - System: 1 breakpoint (handled separately by caller)
        - Messages: 2-3 breakpoints (passed as max_cache_controls by caller)
        - Current: Automatically cached by Anthropic

        Args:
            messages: List of message dictionaries
            max_cache_controls: Remaining cache controls available for messages
                               (caller should subtract 1 if system prompt exists)

        Returns:
            Processed messages with cache_control added
        """
        processed_messages = []
        cache_control_count = 0

        for message in reversed(messages):
            if message["role"] == "user":
                # Process user message content
                if isinstance(message["content"], str):
                    # Simple string content - convert to structured format with cache_control
                    content = [{
                        "type": "text",
                        "text": message["content"]
                    }]
                    if cache_control_count < max_cache_controls:
                        content[0]["cache_control"] = {"type": "ephemeral"}
                        cache_control_count += 1
                elif isinstance(message["content"], list):
                    # Structured content - add cache_control to first item in reverse order
                    content = []
                    for item in reversed(message["content"]):
                        if isinstance(item, dict):
                            item_copy = item.copy()
                            if cache_control_count < max_cache_controls:
                                item_copy["cache_control"] = {"type": "ephemeral"}
                                cache_control_count += 1
                            content.append(item_copy)
                        else:
                            content.append(item)
                    content.reverse()  # Restore original order within the message
                else:
                    content = message["content"]

                processed_messages.append({
                    "role": "user",
                    "content": content
                })
            else:
                # Keep non-user messages unchanged
                processed_messages.append(message)

        # Restore original message order
        processed_messages.reverse()
        return processed_messages

    def _process_system_native_anthropic(self, system: Any) -> Any:
        """
        Process system message for native Anthropic API format.

        Adds cache_control to system message if it's a string or list.

        Args:
            system: System message (string or list)

        Returns:
            Processed system message with cache_control
        """
        if not system:
            return system

        if isinstance(system, str):
            # Convert string system to structured format with cache_control
            return [{
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"}
            }]
        elif isinstance(system, list):
            # System is already structured - add cache_control to last item
            # Use shallow copy for performance, only deep copy the last item we modify
            system_copy = system.copy()
            if system_copy and isinstance(system_copy[-1], dict):
                # Make a copy of the last dict item before modifying it
                system_copy[-1] = system_copy[-1].copy()
                system_copy[-1]["cache_control"] = {"type": "ephemeral"}
            return system_copy

        return system

    def _process_messages_bedrock(self, messages: List[Dict], max_cache_controls: int) -> List[Dict]:
        """
        Process messages for Bedrock Converse API format.

        Based on add_cache_points_to_bedrock_messages() from gpt_langchain.py.
        Adds cachePoint markers to user messages in reverse order.

        IMPORTANT: This function receives the REMAINING cache breakpoints available
        for messages after accounting for the system prompt:
        - Total: 4 cache breakpoints (Bedrock limit)
        - System: 1 breakpoint (handled separately by caller)
        - Messages: 3 breakpoints (passed as max_cache_controls by caller)

        Args:
            messages: List of message dictionaries in Bedrock format
            max_cache_controls: Remaining cache controls available for messages
                               (caller should subtract 1 if system prompt exists)

        Returns:
            Processed messages with cachePoint added
        """
        # Use shallow copy for performance - only copy the messages we modify
        processed_messages = messages.copy()
        cache_control_count = 0

        # Find all user messages
        user_messages = []
        for i, message in enumerate(processed_messages):
            if message.get("role") == "user":
                user_messages.append((i, message))

        # Cache up to max_cache_controls recent user messages
        max_user_messages_to_cache = min(max_cache_controls, len(user_messages))

        # Only select the most recent messages up to the limit
        if max_user_messages_to_cache > 0 and user_messages:
            recent_user_messages = user_messages[-max_user_messages_to_cache:]
        else:
            recent_user_messages = []

        for msg_index, message in recent_user_messages:
            if cache_control_count >= max_cache_controls:
                break

            content = message.get("content", [])
            if isinstance(content, list):
                # Make a shallow copy of the message before modifying it
                processed_messages[msg_index] = message.copy()
                # Also copy the content list before appending
                processed_messages[msg_index]["content"] = content.copy()
                # Add cachePoint to the content list
                processed_messages[msg_index]["content"].append({"cachePoint": {"type": "default"}})
                cache_control_count += 1

        return processed_messages

    def _process_system_bedrock(self, system: Any) -> Any:
        """
        Process system message for Bedrock Converse API format.

        Adds cachePoint marker to system message.

        Args:
            system: System message (list format for Bedrock)

        Returns:
            Processed system message with cachePoint
        """
        if not system:
            return system

        if isinstance(system, str):
            # Convert string to Bedrock format with cachePoint
            return [
                {
                    "type": "text",
                    "text": system
                },
                {
                    "cachePoint": {"type": "default"}
                }
            ]
        elif isinstance(system, list):
            # System is already in list format - add cachePoint
            # Use shallow copy for performance since we're only appending
            system_copy = system.copy()
            system_copy.append({"cachePoint": {"type": "default"}})
            return system_copy

        return system

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Dict[str, Any],
        cache: Any,
        data: Dict[str, Any],
        call_type: str
    ) -> Dict[str, Any]:
        """
        Pre-call hook to inject cache_control into Anthropic requests.

        This is called before the request is sent to the LLM provider.
        We modify the data dict in-place to add cache_control markers.

        Args:
            user_api_key_dict: User authentication info
            cache: LiteLLM cache instance
            data: Request data dictionary (mutable)
            call_type: Type of call ("completion", etc.)

        Returns:
            Modified data dict with cache_control added
        """
        try:
            # Check if caching should be enabled FIRST to avoid unnecessary processing
            if not self._should_enable_caching(data):
                # For non-Anthropic models, remove any existing cache_control markers
                # that may have been added by client code to prevent provider errors
                model = data.get('model', '')
                if not self._is_anthropic_provider(model):
                    self._remove_cache_control_from_messages(data)
                if verbose_full:
                    print(f"💾 AnthropicCachingHook: Caching not enabled for this request, skipping processing", flush=True)
                return data

            # Safety net: Remove VLLM-specific parameters from extra_body if present
            # These parameters are incompatible with Anthropic API and should not be sent
            # This is a defensive measure in case the is_litellm detection logic fails
            self._remove_vllm_params(data)

            # Only process completion-like calls
            supported_call_types = [
                "completion", "acompletion", "text_completion", "chat_completion",
                "message", "messages", "anthropic_message", "anthropic_messages"
            ]
            if call_type not in supported_call_types:
                if verbose_full:
                    print(f"💾 AnthropicCachingHook: Call type {call_type} not supported, skipping", flush=True)
                return data

            if verbose_full:
                print(f"💾 AnthropicCachingHook: async_pre_call_hook called with call_type={call_type}, model={data.get('model', 'unknown')}", flush=True)
                print(f"💾 AnthropicCachingHook: data keys: {list(data.keys())}", flush=True)
                # Print extra_body if it exists
                if 'extra_body' in data:
                    print(f"💾 AnthropicCachingHook: extra_body = {data['extra_body']}", flush=True)
                if 'litellm_params' in data and 'extra_body' in data['litellm_params']:
                    print(f"💾 AnthropicCachingHook: litellm_params.extra_body = {data['litellm_params']['extra_body']}", flush=True)

            model = data.get('model', '')
            is_bedrock = self._is_bedrock_provider(model)

            if verbose or verbose_full:
                provider_type = "Bedrock" if is_bedrock else "Native Anthropic"
                print(f"💾 AnthropicCachingHook: Enabling prompt caching for {provider_type} model: {model}", flush=True)

            # Get messages and system prompt
            messages = data.get('messages', [])
            system = data.get('system', '')

            # Check if system message is in messages array (OpenAI format)
            # LiteLLM will convert it later, but we need to handle it now for caching
            system_in_messages = False
            if messages and isinstance(messages, list) and len(messages) > 0:
                if messages[0].get('role') == 'system':
                    system_in_messages = True
                    if not system:  # If system wasn't extracted yet
                        system = messages[0].get('content', '')
                        if verbose or verbose_full:
                            print(f"💾 AnthropicCachingHook: Found system message in messages array (OpenAI format)", flush=True)

            if not messages:
                if verbose_full:
                    print(f"💾 AnthropicCachingHook: No messages to process", flush=True)
                return data

            # Process based on provider type
            if is_bedrock:
                # Bedrock Converse API format
                max_cache_controls = 4  # Bedrock supports up to 4 cache points

                # Process system message first (highest priority)
                if system:
                    processed_system = self._process_system_bedrock(system)
                    data['system'] = processed_system
                    if verbose or verbose_full:
                        print(f"💾 AnthropicCachingHook: Added cachePoint to Bedrock system message", flush=True)

                # Process user messages (remaining cache points)
                processed_messages = self._process_messages_bedrock(messages, max_cache_controls - (1 if system else 0))
                data['messages'] = processed_messages

                if verbose or verbose_full:
                    cache_points = sum(1 for msg in processed_messages if isinstance(msg.get('content'), list) and any(isinstance(item, dict) and 'cachePoint' in item for item in msg.get('content', [])))
                    if system:
                        cache_points += 1
                    print(f"💾 AnthropicCachingHook: Added {cache_points} cachePoints to Bedrock request", flush=True)
            else:
                # Native Anthropic API format
                # Start with max_cache_controls=3 for messages (not 4) because:
                # - Total: 4 cache breakpoints (Anthropic limit)
                # - System: 1 breakpoint (if present, subtracted below)
                # - Prior messages: Up to 2-3 breakpoints (depending on system presence)
                # - Current message: Automatically cached by Anthropic
                max_cache_controls = 3

                # Process system message first (highest priority, doesn't count against message limit)
                if system:
                    if system_in_messages:
                        # System is in messages array - add cache_control to first message
                        if messages and messages[0].get('role') == 'system':
                            content = messages[0].get('content', '')
                            if isinstance(content, str):
                                messages[0]['content'] = [{
                                    "type": "text",
                                    "text": content,
                                    "cache_control": {"type": "ephemeral"}
                                }]
                            elif isinstance(content, list) and len(content) > 0:
                                if isinstance(content[-1], dict):
                                    content[-1]["cache_control"] = {"type": "ephemeral"}
                            if verbose or verbose_full:
                                print(f"💾 AnthropicCachingHook: Added cache_control to system message in messages array", flush=True)
                    else:
                        # System is separate parameter
                        processed_system = self._process_system_native_anthropic(system)
                        data['system'] = processed_system
                        if verbose or verbose_full:
                            print(f"💾 AnthropicCachingHook: Added cache_control to native Anthropic system message", flush=True)

                # Process user messages (remaining cache controls after accounting for system)
                # Filter out system message if it's in the array
                messages_to_process = messages if not system_in_messages else messages[1:]
                # Subtract 1 from max_cache_controls if system prompt exists (reserve 1 for system)
                messages_cache_limit = max_cache_controls - (1 if system else 0)
                processed_messages = self._process_messages_native_anthropic(messages_to_process, messages_cache_limit)

                # Reconstruct messages array
                if system_in_messages:
                    data['messages'] = [messages[0]] + processed_messages
                else:
                    data['messages'] = processed_messages

                if verbose or verbose_full:
                    cache_controls = 0
                    for msg in processed_messages:
                        if msg.get('role') == 'user' and isinstance(msg.get('content'), list):
                            cache_controls += sum(1 for item in msg['content'] if isinstance(item, dict) and 'cache_control' in item)
                    if system:
                        cache_controls += 1
                    print(f"💾 AnthropicCachingHook: Added {cache_controls} cache_controls to native Anthropic request", flush=True)

            # Final verification: check if enable_caching is still in data
            if verbose or verbose_full:
                print(f"💾 AnthropicCachingHook: Final data.extra_body keys: {list(data.get('extra_body', {}).keys())}", flush=True)
                print(f"💾 AnthropicCachingHook: Final data.extra_body: {data.get('extra_body', {})}", flush=True)
                if 'enable_caching' in data.get('extra_body', {}):
                    print(f"💾 AnthropicCachingHook: ⚠️  WARNING: enable_caching still present in extra_body after hook!", flush=True)

            return data

        except Exception as e:
            print(f"💾 AnthropicCachingHook: Error in async_pre_call_hook: {e}", flush=True)
            import traceback
            traceback.print_exc()
            # On error, return data unchanged to not break the request
            return data

    def log_success_event(self, kwargs: dict, response_obj: Any, start_time: float, end_time: float):
        """Log successful responses to show cache usage."""
        try:
            # Only log if caching was enabled for this request
            # Use _check_enable_caching (read-only) to avoid race condition with other hooks
            if not self._check_enable_caching(kwargs):
                return

            # Check for cache usage in response
            if hasattr(response_obj, 'usage') and response_obj.usage:
                usage = response_obj.usage

                # Native Anthropic format
                cache_creation = getattr(usage, 'cache_creation_input_tokens', 0) or 0
                cache_read = getattr(usage, 'cache_read_input_tokens', 0) or 0

                # Also check dict format
                if hasattr(usage, '__dict__'):
                    usage_dict = usage.__dict__
                    cache_creation = usage_dict.get('cache_creation_input_tokens', cache_creation) or 0
                    cache_read = usage_dict.get('cache_read_input_tokens', cache_read) or 0

                if cache_creation > 0 or cache_read > 0:
                    model = kwargs.get('model', 'unknown')
                    input_tokens = getattr(usage, 'input_tokens', 0) or 0
                    print(f"💾 AnthropicCachingHook: Cache usage for {model}:", flush=True)
                    print(f"   - Input tokens: {input_tokens}", flush=True)
                    print(f"   - Cache creation: {cache_creation}", flush=True)
                    print(f"   - Cache read: {cache_read}", flush=True)
                    if cache_read > 0:
                        savings_pct = (cache_read / (input_tokens + cache_creation)) * 100 if (input_tokens + cache_creation) > 0 else 0
                        print(f"   - Cache efficiency: {savings_pct:.1f}% of tokens from cache", flush=True)

        except Exception as e:
            if verbose_full:
                print(f"💾 AnthropicCachingHook: Error in log_success_event: {e}", flush=True)


# Create the hook instance that LiteLLM will use
if verbose or verbose_full:
    print(f"💾 HOOK EXPORT: Creating anthropic_caching_hook instance", flush=True)

anthropic_caching_hook = AnthropicCachingHook()

if verbose or verbose_full:
    print(f"💾 HOOK EXPORT: anthropic_caching_hook created successfully: {type(anthropic_caching_hook)}", flush=True)
    print(f"💾 HOOK EXPORT: Hook supports: Anthropic (native), Bedrock Anthropic", flush=True)
    print(f"💾 HOOK EXPORT: Activation: Set enable_caching=True in extra_body", flush=True)
