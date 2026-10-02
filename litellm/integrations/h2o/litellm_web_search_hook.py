#!/usr/bin/env python3
"""
Custom LiteLLM hook to intercept web search tool calls and use SERP API server-side.

This hook intercepts web_search_20250305 tool calls from Claude Code SDK and:
1. Captures the web search request before it reaches the LLM provider
2. Uses SERPAPI_API_KEY to perform the actual web search
3. Returns search results without sending the tool call to the provider
4. Works with ANY model (Azure, VLLM, etc.) that supports tools
"""

import os
import json
import asyncio
from typing import Optional, Dict, Any, List
from litellm.integrations.custom_logger import CustomLogger

verbose = os.getenv('H2OGPT_VERBOSE', '0') == '1'
verbose_full = os.getenv('H2OGPT_VERBOSE_FULL', '0') == '1'

class WebSearchInterceptorHook(CustomLogger):
    """Custom LiteLLM hook to intercept and handle web search tool calls server-side."""
    
    # Anthropic models that have native web search support - skip hook for these
    ANTHROPIC_MODELS_WITH_NATIVE_SEARCH = [
        # Current models from documentation with native web search
        'claude-opus-4-20250514',
        'claude-opus-4-20250514-reasoning',
        'claude-opus-4-5-20251101',
        'claude-opus-4-1-20250805',
        'claude-sonnet-4-20250514',
        'claude-sonnet-4-20250514-reasoning',
        'claude-3-7-sonnet-20250219',
        'claude-3-7-sonnet-20250219-reasoning',
        'claude-3-5-sonnet-latest',
        'claude-3-5-haiku-latest',
        'claude-sonnet-4-5-20250929',
        'claude-haiku-4-5-20251001',
    ]
    
    def _should_skip_anthropic_model(self, model_name: str) -> bool:
        """Check if this Anthropic model should be skipped (has native web search)."""
        model_str = str(model_name)

        if verbose_full:
            print(f"🔍 WebSearchInterceptorHook: Checking if model should be skipped: {model_str}", flush=True)

        # Check specific models from documentation - exact matches only
        for native_model in self.ANTHROPIC_MODELS_WITH_NATIVE_SEARCH:
            if native_model in model_str:
                if verbose_full:
                    print(f"🔍 WebSearchInterceptorHook: Model {model_str} matches native search model {native_model} - SKIPPING", flush=True)
                return True

        # Future-proofing: Only skip newer claude-opus-4-* and claude-sonnet-4-* models
        # This is more restrictive to avoid catching older models that need SERP hook
        import re
        if re.match(r'.*claude-opus-4-.*', model_str):
            if verbose_full:
                print(f"🔍 WebSearchInterceptorHook: Model {model_str} matches claude-opus-4-* pattern - SKIPPING", flush=True)
            return True
        if re.match(r'.*claude-sonnet-4-.*', model_str):
            if verbose_full:
                print(f"🔍 WebSearchInterceptorHook: Model {model_str} matches claude-sonnet-4-* pattern - SKIPPING", flush=True)
            return True

        # Also check for claude-3-7-sonnet-* pattern (newer 3.7 models)
        if re.match(r'.*claude-3-7-sonnet-.*', model_str):
            if verbose_full:
                print(f"🔍 WebSearchInterceptorHook: Model {model_str} matches claude-3-7-sonnet-* pattern - SKIPPING", flush=True)
            return True

        if verbose_full:
            print(f"🔍 WebSearchInterceptorHook: Model {model_str} does NOT match skip patterns - USING HOOK", flush=True)
        return False
    
    def __init__(self):
        super().__init__()
        self.serp_api_key = os.environ.get('SERPAPI_API_KEY')
        print(f"🔍 WebSearchInterceptorHook: FRESH INIT v2 - with SERP API: {'✅ Available' if self.serp_api_key else '❌ Missing'}", flush=True)
        
        # Check if methods exist and are callable
        methods = ['log_pre_api_call', 'log_success_event', 'async_pre_call_hook', 'log_failure_event', 'async_log_success_event', 'async_log_failure_event']
        for method in methods:
            if hasattr(self, method) and callable(getattr(self, method)):
                print(f"🔍 WebSearchInterceptorHook: Method {method} exists and callable: ✅", flush=True)
            else:
                print(f"🔍 WebSearchInterceptorHook: Method {method} missing or not callable: ❌", flush=True)
    
    def log_pre_api_call(self, model, messages, kwargs):
        """Called before API call - log for debugging."""
        if self._should_skip_anthropic_model(model):
            return
        # Skip logging for health checks and non-LLM calls
        if self._should_skip_hook_processing(kwargs, kwargs.get('call_type', '')):
            return
        if verbose_full:
            print(f"🔍 WebSearchInterceptorHook: log_pre_api_call called with model={model}", flush=True)
        tools = kwargs.get("tools", [])
        if tools:
            if verbose_full:
                print(f"🔍 WebSearchInterceptorHook: Tools in kwargs: {len(tools)} tools", flush=True)
            for i, tool in enumerate(tools):
                if verbose_full:
                    print(f"🔍 WebSearchInterceptorHook: Tool {i}: {tool.get('type', 'unknown')} - {tool.get('name', 'unnamed')}", flush=True)
                if tool.get('type') == 'web_search_20250305':
                    if verbose_full:
                        print(f"🔍 WebSearchInterceptorHook: FOUND web_search_20250305 in log_pre_api_call!", flush=True)
    
    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Called on successful completion."""
        if self._should_skip_anthropic_model(kwargs.get('model', '')):
            return
        # Skip logging for health checks and non-LLM calls
        if self._should_skip_hook_processing(kwargs, kwargs.get('call_type', '')):
            return
        if verbose_full:
            print(f"🔍 WebSearchInterceptorHook: log_success_event called", flush=True)
    
    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """Called on failed completion."""
        if self._should_skip_anthropic_model(kwargs.get('model', '')):
            return
        # Skip logging for health checks and non-LLM calls
        if self._should_skip_hook_processing(kwargs, kwargs.get('call_type', '')):
            return
        if verbose_full:
            print(f"🔍 WebSearchInterceptorHook: log_failure_event called", flush=True)
    
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Called on successful completion (async)."""
        if self._should_skip_anthropic_model(kwargs.get('model', '')):
            return
        # Skip logging for health checks and non-LLM calls
        if self._should_skip_hook_processing(kwargs, kwargs.get('call_type', '')):
            return
        if verbose_full:
            print(f"🔍 WebSearchInterceptorHook: async_log_success_event called", flush=True)
    
    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """Called on failed completion (async)."""
        if self._should_skip_anthropic_model(kwargs.get('model', '')):
            return
        # Skip logging for health checks and non-LLM calls
        if self._should_skip_hook_processing(kwargs, kwargs.get('call_type', '')):
            return
        if verbose_full:
            print(f"🔍 WebSearchInterceptorHook: async_log_failure_event called", flush=True)
    
    async def async_pre_call_hook(self, user_api_key_dict, cache, data: Dict[str, Any], call_type: str):
        """
        Intercept requests and handle web search tools server-side.
        
        Args:
            user_api_key_dict: User authentication info
            cache: LiteLLM cache instance
            data: Request data dictionary (mutable)
            call_type: Type of call ("completion", etc.)
        
        Returns:
            Modified data dict or None to proceed normally
        """
        try:
            if verbose_full:
                print(f"🔍 WebSearchInterceptorHook: async_pre_call_hook called with call_type={call_type}, model={data.get('model', 'unknown')}", flush=True)
            
            # Skip hook processing for health checks and other non-LLM endpoints
            # These don't need web search processing and shouldn't trigger cost tracking
            if self._should_skip_hook_processing(data, call_type):
                if verbose_full:
                    print(f"🔍 WebSearchInterceptorHook: Skipping hook processing for endpoint/call_type: {call_type}", flush=True)
                return data
            
            # Preserve authentication metadata to fix cost tracking callback error
            self._preserve_authentication_metadata(data, user_api_key_dict)
            
            # Only process calls that might have tools - expanded for all possible types
            # Note: /chat/completions uses "completion", /v1/messages uses "text_completion"  
            # /anthropic/v1/messages uses "pass_through_endpoint" and bypasses hooks entirely
            # "acompletion" is async completion used by LiteLLM proxy
            supported_call_types = [
                "completion", "acompletion", "text_completion", "chat_completion", 
                "message", "messages", "anthropic_message", "anthropic_messages",
                "pass_through_endpoint"  # Add pass-through support
            ]
            if call_type not in supported_call_types:
                if verbose_full:
                    print(f"🔍 WebSearchInterceptorHook: Skipping unsupported call_type: {call_type}", flush=True)
                return data
            
            # Skip interception only for Anthropic models that have native web search support
            model_name = data.get('model', '')
            if self._should_skip_anthropic_model(model_name):
                if verbose_full:
                    print(f"🔍 WebSearchInterceptorHook: Skipping Anthropic model {model_name} - using native web search", flush=True)
                return data
            
            # Check if request has tools
            tools = data.get("tools", [])
            if verbose_full:
                print(f"🔍 WebSearchInterceptorHook: Found {len(tools)} tools in request", flush=True)
            if not tools:
                return data
            
            # Look for web_search_20250305 tool calls
            web_search_tools = []
            other_tools = []
            
            for tool in tools:
                if isinstance(tool, dict) and tool.get("type") == "web_search_20250305":
                    web_search_tools.append(tool)
                    if verbose_full:
                        print(f"🔍 WebSearchInterceptorHook: Found web_search_20250305 tool: {tool}", flush=True)
                else:
                    other_tools.append(tool)
            
            # If no web search tools, proceed normally
            if not web_search_tools:
                if verbose_full:
                    print(f"🔍 WebSearchInterceptorHook: No web search tools found, proceeding normally", flush=True)
                return data

            if verbose_full:
                print(f"🔍 WebSearchInterceptorHook: Intercepting {len(web_search_tools)} web search tools for model: {data.get('model', 'unknown')}", flush=True)
            
            # Always remove web search tools from the request to prevent provider errors
            data["tools"] = other_tools if other_tools else None
            if not data["tools"]:
                data.pop("tools", None)  # Remove empty tools array

            if verbose_full:
                print(f"🔍 WebSearchInterceptorHook: Removed web search tools, {len(other_tools)} other tools remaining", flush=True)
            
            # If we have SERP API key, perform web search and inject results
            if self.serp_api_key:
                return await self._handle_web_search_tools(data, web_search_tools, other_tools, user_api_key_dict, cache)
            else:
                if verbose_full:
                   print("🔍 WebSearchInterceptorHook: No SERPAPI_API_KEY found - web search disabled", flush=True)
                return data
            
        except Exception as e:
            print(f"🔍 WebSearchInterceptorHook: Error in async_pre_call_hook: {e}", flush=True)
            import traceback
            traceback.print_exc()
            # On error, at least remove web search tools to prevent provider errors
            try:
                tools = data.get("tools", [])
                filtered_tools = [t for t in tools if not (isinstance(t, dict) and t.get("type") == "web_search_20250305")]
                data["tools"] = filtered_tools if filtered_tools else None
                if not data["tools"]:
                    data.pop("tools", None)
            except:
                pass
            return data
    
    def _should_skip_hook_processing(self, data: Dict[str, Any], call_type: str) -> bool:
        """
        Determine if hook processing should be skipped for this request.
        
        Skip processing for:
        - Health checks and status endpoints
        - Non-LLM API calls 
        - Requests without models (likely infrastructure calls)
        """
        # Skip if no model specified (likely health check or admin endpoint)
        model = data.get('model', '').strip()
        if not model:
            return True
            
        # Skip for specific non-LLM call types
        non_llm_call_types = ['health_check', 'status', 'admin', 'key_generate', 'models_list']
        if call_type in non_llm_call_types:
            return True
            
        # Skip if request looks like a health check (no messages, no tools, etc.)
        messages = data.get('messages', [])
        tools = data.get('tools', [])
        if not messages and not tools:
            return True
            
        return False
    
    def _preserve_authentication_metadata(self, data: Dict[str, Any], user_api_key_dict):
        """
        Preserve authentication metadata to fix cost tracking callback error.
        
        The cost tracking callback requires user_api_key, user_id, team_id, or end_user_id
        to be present in the metadata. This method preserves existing metadata without 
        adding litellm_params that would cause provider API errors.
        
        LiteLLM should already have this metadata from the proxy authentication process,
        but we ensure it's preserved if the hook modifies the request.
        """
        try:
            # Only preserve metadata if it already exists - don't create litellm_params
            # as this can cause "Extra inputs are not permitted" errors with providers
            existing_litellm_params = data.get("litellm_params")
            if existing_litellm_params is None:
                # No existing litellm_params - the proxy should handle authentication metadata
                # through its internal mechanisms. Just log what we have from user_api_key_dict.
                if verbose_full and user_api_key_dict:
                    auth_info = []
                    if hasattr(user_api_key_dict, 'api_key') and user_api_key_dict.api_key:
                        auth_info.append("api_key")
                    if hasattr(user_api_key_dict, 'user_id') and user_api_key_dict.user_id:
                        auth_info.append("user_id")
                    if hasattr(user_api_key_dict, 'team_id') and user_api_key_dict.team_id:
                        auth_info.append("team_id")
                    if hasattr(user_api_key_dict, 'end_user_id') and user_api_key_dict.end_user_id:
                        auth_info.append("end_user_id")
                    print(f"🔍 WebSearchInterceptorHook: Authentication available from user_api_key_dict: {auth_info}", flush=True)
                return
            
            # If litellm_params already exists, preserve and enhance metadata 
            existing_metadata = existing_litellm_params.get("metadata", {})
            
            # Only add missing authentication info - don't override existing values
            if user_api_key_dict:
                if not existing_metadata.get("user_api_key") and hasattr(user_api_key_dict, 'api_key') and user_api_key_dict.api_key:
                    existing_metadata["user_api_key"] = user_api_key_dict.api_key
                
                if not existing_metadata.get("user_api_key_user_id") and hasattr(user_api_key_dict, 'user_id') and user_api_key_dict.user_id:
                    existing_metadata["user_api_key_user_id"] = user_api_key_dict.user_id
                
                if not existing_metadata.get("user_api_key_team_id") and hasattr(user_api_key_dict, 'team_id') and user_api_key_dict.team_id:
                    existing_metadata["user_api_key_team_id"] = user_api_key_dict.team_id
                
                if not existing_metadata.get("user_api_key_end_user_id") and hasattr(user_api_key_dict, 'end_user_id') and user_api_key_dict.end_user_id:
                    existing_metadata["user_api_key_end_user_id"] = user_api_key_dict.end_user_id
                
                if not existing_metadata.get("user_api_key_org_id") and hasattr(user_api_key_dict, 'org_id') and user_api_key_dict.org_id:
                    existing_metadata["user_api_key_org_id"] = user_api_key_dict.org_id
                
                if not existing_metadata.get("user_api_key_alias") and hasattr(user_api_key_dict, 'key_alias') and user_api_key_dict.key_alias:
                    existing_metadata["user_api_key_alias"] = user_api_key_dict.key_alias
                
                if not existing_metadata.get("user_api_key_user_email") and hasattr(user_api_key_dict, 'user_email') and user_api_key_dict.user_email:
                    existing_metadata["user_api_key_user_email"] = user_api_key_dict.user_email
                
                if not existing_metadata.get("user_api_key_team_alias") and hasattr(user_api_key_dict, 'team_alias') and user_api_key_dict.team_alias:
                    existing_metadata["user_api_key_team_alias"] = user_api_key_dict.team_alias
            
            # Update the metadata back into litellm_params
            existing_litellm_params["metadata"] = existing_metadata
            
            if verbose_full:
                preserved_keys = [k for k in existing_metadata.keys() if k.startswith("user_api_key")]
                print(f"🔍 WebSearchInterceptorHook: Enhanced existing metadata with keys: {preserved_keys}", flush=True)
                
        except Exception as e:
            print(f"🔍 WebSearchInterceptorHook: Error preserving authentication metadata: {e}", flush=True)
            # Non-critical error, continue processing
            pass
    
    async def _handle_web_search_tools(self, data: Dict[str, Any], web_search_tools: List[Dict], 
                                     other_tools: List[Dict], user_api_key_dict, cache) -> Dict[str, Any]:
        """
        Handle web search tools by performing searches server-side and modifying the request.
        
        Strategy: Extract search queries from messages, perform searches, inject results into context.
        """
        try:
            # Extract potential search queries from the last user message
            messages = data.get("messages", [])
            if not messages:
                if verbose_full:
                    print("🔍 WebSearchInterceptorHook: No messages found", flush=True)
                data["tools"] = other_tools
                return data
            
            # Get the last user message
            last_message = None
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    last_message = msg
                    break
            
            if not last_message:
                if verbose_full:
                    print("🔍 WebSearchInterceptorHook: No user message found", flush=True)
                data["tools"] = other_tools
                return data
            
            query = last_message.get("content", "")
            if not query or len(query.strip()) < 3:
                if verbose_full:
                    print("🔍 WebSearchInterceptorHook: Query too short for web search", flush=True)
                data["tools"] = other_tools
                return data

            if verbose_full:
                print(f"🔍 WebSearchInterceptorHook: Performing search for: {query[:100]}...", flush=True)
            
            # Perform the web search
            search_results = await self._perform_serp_search(query)
            
            if search_results:
                # Inject search results into the conversation context
                search_context = f"\n\n[Web Search Results for: {query}]\n{search_results}\n[End of Web Search Results]\n\n"
                
                # Add search results to the last user message
                modified_content = last_message["content"] + search_context
                
                # Create new messages with injected search results
                new_messages = messages.copy()
                for i, msg in enumerate(new_messages):
                    if msg.get("role") == "user" and msg.get("content") == last_message["content"]:
                        new_messages[i] = {**msg, "content": modified_content}
                        break
                
                data["messages"] = new_messages
                if verbose_full:
                    print(f"🔍 WebSearchInterceptorHook: Injected search results into context", flush=True)
            else:
                if verbose_full:
                    print("🔍 WebSearchInterceptorHook: No search results obtained", flush=True)
            
            # Remove web search tools from the request (they're now handled)
            data["tools"] = other_tools
            
            return data
            
        except Exception as e:
            print(f"🔍 WebSearchInterceptorHook: Error handling web search: {e}", flush=True)
            # On error, remove web search tools and proceed
            data["tools"] = other_tools
            return data
    
    async def _perform_serp_search(self, query: str, max_results: int = 5) -> Optional[str]:
        """
        Perform web search using SERP API.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            Formatted search results string or None if search fails
        """
        try:
            import aiohttp
            
            params = {
                "q": query,
                "api_key": self.serp_api_key,
                "num": max_results,
                "hl": "en",
                "gl": "us"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get("https://serpapi.com/search", params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._format_search_results(data)
                    else:
                        print(f"🔍 WebSearchInterceptorHook: SERP API error: {response.status}", flush=True)
                        return None
                        
        except ImportError:
            print("🔍 WebSearchInterceptorHook: aiohttp not available, trying requests", flush=True)
            return await self._perform_serp_search_sync(query, max_results)
        except Exception as e:
            print(f"🔍 WebSearchInterceptorHook: SERP search error: {e}", flush=True)
            return None
    
    async def _perform_serp_search_sync(self, query: str, max_results: int = 5) -> Optional[str]:
        """Fallback synchronous SERP search using requests."""
        try:
            import requests
            
            params = {
                "q": query,
                "api_key": self.serp_api_key,
                "num": max_results,
                "hl": "en",
                "gl": "us"
            }
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: requests.get("https://serpapi.com/search", params=params, timeout=10)
            )
            
            if response.status_code == 200:
                data = response.json()
                return self._format_search_results(data)
            else:
                print(f"🔍 WebSearchInterceptorHook: SERP API error: {response.status_code}", flush=True)
                return None
                
        except Exception as e:
            print(f"🔍 WebSearchInterceptorHook: Sync SERP search error: {e}", flush=True)
            return None
    
    def _format_search_results(self, serp_data: Dict[str, Any]) -> str:
        """Format SERP API results into readable text."""
        try:
            results = []
            
            # Extract organic results
            organic_results = serp_data.get("organic_results", [])
            for i, result in enumerate(organic_results[:5], 1):
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                link = result.get("link", "")
                
                if title and snippet:
                    results.append(f"{i}. {title}\n   {snippet}\n   Source: {link}\n")
            
            # Extract answer box if available
            answer_box = serp_data.get("answer_box")
            if answer_box:
                answer = answer_box.get("answer") or answer_box.get("snippet")
                if answer:
                    results.insert(0, f"Quick Answer: {answer}\n\n")
            
            # Extract knowledge graph if available
            knowledge_graph = serp_data.get("knowledge_graph")
            if knowledge_graph:
                title = knowledge_graph.get("title")
                description = knowledge_graph.get("description")
                if title and description:
                    results.insert(0, f"Knowledge: {title} - {description}\n\n")
            
            if results:
                return "".join(results)
            else:
                return "No search results found."
                
        except Exception as e:
            print(f"🔍 WebSearchInterceptorHook: Error formatting results: {e}", flush=True)
            return "Error formatting search results."


# Create the hook instance that LiteLLM will use
print(f"🔍 HOOK EXPORT: Creating web_search_hook instance", flush=True)
web_search_hook = WebSearchInterceptorHook()
print(f"🔍 HOOK EXPORT: web_search_hook created successfully: {type(web_search_hook)}", flush=True)

# Debug: Print what will be available for import
print(f"🔍 HOOK EXPORT: Available in module namespace: {[name for name in globals().keys() if not name.startswith('_')]}", flush=True)