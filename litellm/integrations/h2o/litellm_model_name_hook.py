"""
LiteLLM Hook to Override Response Model Name

This hook modifies the response to return the configured model_name
instead of the underlying provider's model identifier.

For example:
- User requests: "gpt-5"
- Provider returns: "gpt-5-2025-08-07"
- This hook changes response to: "gpt-5"

The original provider model name is preserved in metadata for debugging.
"""

import os
from typing import Optional, Dict, Any, Union
from litellm.integrations.custom_logger import CustomLogger


class ModelNameOverrideHook(CustomLogger):
    """
    Custom hook to override response model name with the requested model_name.
    """

    def __init__(self):
        super().__init__()
        self.verbose = os.getenv('H2OGPT_VERBOSE_FULL', '0') == '1'
        self.debug = self.verbose  # Only debug when verbose mode is enabled
        if self.verbose:
            print(f"🔧 ModelNameOverrideHook initialized", flush=True)

    def _strip_model_suffix(self, model_name: str) -> str:
        """Strip __nofallback and __fallback suffixes from model name."""
        if model_name and ("__nofallback" in model_name or "__fallback" in model_name):
            return model_name.replace("__nofallback", "").replace("__fallback", "")
        return model_name

    def _process_response(self, response: Any, requested_model: str) -> Any:
        """Process response to strip model suffix - works for both sync and async."""
        if self.debug:
            print(f"🔍 _process_response: requested_model={requested_model}, has response.model={hasattr(response, 'model')}", flush=True)

        if not hasattr(response, "model"):
            return response

        provider_model = response.model
        if self.debug:
            print(f"🔍 _process_response: provider_model={provider_model}", flush=True)

        # Special handling for agent_auto: keep the actual model used
        if requested_model in ["agent_auto"]:
            if self.verbose:
                print(f"🤖 {requested_model} routed to: {provider_model}", flush=True)
            return response

        # If requested model has routing suffixes, strip them and use the clean name
        if "__nofallback" in requested_model or "__fallback" in requested_model:
            clean_model = self._strip_model_suffix(requested_model)
            if self.verbose:
                print(f"🧹 Stripping suffix from requested: {requested_model} -> {clean_model}", flush=True)
            response.model = clean_model
            if self.debug:
                print(f"🧹 After setting: response.model={response.model}", flush=True)
            return response

        # For regular models: override to show the configured model_name
        if provider_model and provider_model != requested_model:
            if self.verbose:
                print(f"🔄 Model name override: {provider_model} -> {requested_model}", flush=True)
            response.model = requested_model

        return response

    def logging_hook(self, kwargs: dict, result: Any, call_type: str):
        """
        Called BEFORE other callbacks - this is the right place to modify the response.
        This method is called by the logging framework and the modified result is used
        for subsequent logging callbacks.
        """
        try:
            if self.debug:
                print(f"🔍 logging_hook called: call_type={call_type}", flush=True)

            requested_model = kwargs.get("model")
            if requested_model and result:
                result = self._process_response(result, requested_model)

            return kwargs, result
        except Exception as e:
            print(f"⚠️ Model name hook error in logging_hook: {e}", flush=True)
            return kwargs, result

    async def async_logging_hook(self, kwargs: dict, result: Any, call_type: str):
        """
        Async version of logging_hook - called BEFORE other async callbacks.
        """
        try:
            if self.debug:
                print(f"🔍 async_logging_hook called: call_type={call_type}", flush=True)

            requested_model = kwargs.get("model")
            if requested_model and result:
                result = self._process_response(result, requested_model)

            return kwargs, result
        except Exception as e:
            print(f"⚠️ Model name hook error in async_logging_hook: {e}", flush=True)
            return kwargs, result

    async def async_post_call_success_hook(
        self,
        data: Dict[str, Any],
        user_api_key_dict: Dict[str, Any],
        response: Any,
    ):
        """
        Modify the response to use the appropriate model_name.

        For regular models: Use the requested model_name instead of provider's model.
        For agent_auto: Use the actual selected model from the cascade (provider model).
        """
        try:
            # Get the requested model name from the request data
            requested_model = data.get("model")

            if self.debug:
                print(f"🔍 Hook called: requested_model={requested_model}", flush=True)

            if not requested_model:
                return response

            # Get the current response model name (provider's model)
            if hasattr(response, "model"):
                provider_model = response.model

                if self.debug:
                    print(f"🔍 Hook: provider_model={provider_model}, requested_model={requested_model}", flush=True)

                # Special handling for agent_auto: keep the actual model used
                if requested_model == "agent_auto":
                    if self.verbose:
                        print(f"🤖 {requested_model} routed to: {provider_model}", flush=True)
                    # For agent_auto, we want to show which actual model was used
                    # Store agent_auto in the header for reference
                    if hasattr(response, "_hidden_params"):
                        if "additional_headers" not in response._hidden_params:
                            response._hidden_params["additional_headers"] = {}
                        response._hidden_params["additional_headers"]["x-litellm-requested-model"] = requested_model
                    else:
                        response._hidden_params = {
                            "additional_headers": {
                                "x-litellm-requested-model": requested_model
                            }
                        }
                    return response

                # If requested model has routing suffixes, strip them and use the clean name
                # e.g., "o4-mini__nofallback" -> "o4-mini"
                if "__nofallback" in requested_model or "__fallback" in requested_model:
                    clean_model = requested_model.replace("__nofallback", "").replace("__fallback", "")
                    if self.debug:
                        print(f"🧹 Stripping suffix: {requested_model} -> {clean_model}", flush=True)
                    # Store the original requested model in header for reference
                    if hasattr(response, "_hidden_params"):
                        if "additional_headers" not in response._hidden_params:
                            response._hidden_params["additional_headers"] = {}
                        response._hidden_params["additional_headers"]["x-litellm-requested-model"] = requested_model
                    else:
                        response._hidden_params = {
                            "additional_headers": {
                                "x-litellm-requested-model": requested_model
                            }
                        }
                    # Override with the clean LiteLLM model name
                    response.model = clean_model
                    return response

                # For regular models: override to show the configured model_name
                # Only override if they're different
                if provider_model and provider_model != requested_model:
                    if self.verbose:
                        print(f"🔄 Model name override: {provider_model} -> {requested_model}", flush=True)

                    # Store original provider model in metadata
                    if hasattr(response, "_hidden_params"):
                        if "additional_headers" not in response._hidden_params:
                            response._hidden_params["additional_headers"] = {}
                        response._hidden_params["additional_headers"]["x-litellm-provider-model"] = provider_model
                    else:
                        response._hidden_params = {
                            "additional_headers": {
                                "x-litellm-provider-model": provider_model
                            }
                        }

                    # Override the model name in the response
                    response.model = requested_model

            return response

        except Exception as e:
            # Don't break the request if hook fails
            print(f"⚠️ Model name override hook error: {e}", flush=True)
            return response


# Instantiate the hook
model_name_override_hook = ModelNameOverrideHook()

# Register the hook globally for client-side routers
try:
    import litellm
    import os
    if model_name_override_hook not in litellm.success_callback:
        litellm.success_callback.append(model_name_override_hook)
        if os.getenv('H2OGPT_VERBOSE_FULL', '0') == '1':
            print(f"🔧 Registered model_name_override_hook globally for client-side routers", flush=True)
except Exception as e:
    print(f"⚠️ Failed to register model_name_override_hook globally: {e}", flush=True)
