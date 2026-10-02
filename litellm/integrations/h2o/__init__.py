"""h2oai custom LiteLLM integrations.

Hooks in this package extend LiteLLM's behaviour for h2oai-specific
needs and live alongside upstream integrations under
``litellm.integrations.*``. Each hook is a callable suitable for
registration via the LiteLLM proxy config ``callbacks`` list, e.g.::

    callbacks:
      - litellm.integrations.h2o.litellm_model_name_hook.model_name_override_hook
      - litellm.integrations.h2o.litellm_anthropic_caching_hook.anthropic_caching_hook
      - litellm.integrations.h2o.litellm_anthropic_params_filter_hook.anthropic_params_filter_hook
      - litellm.integrations.h2o.litellm_max_tokens_cap_hook.max_tokens_cap_hook
      - litellm.integrations.h2o.litellm_max_tokens_resolution_hook.max_tokens_resolution_hook
      - litellm.integrations.h2o.litellm_dedup_tool_call_ids_hook.dedup_tool_call_ids_hook
      - litellm.integrations.h2o.litellm_web_search_hook.web_search_hook
      - litellm.integrations.h2o.litellm_router_hook.router_hook
      - litellm.integrations.h2o.litellm_oauth_auth_hook.oauth_auth_hook

See individual modules for hook documentation.
"""
