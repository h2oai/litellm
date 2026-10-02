#!/usr/bin/env python3
"""
LiteLLM Pre-Call Hook: Deduplicate tool messages with the same tool_call_id.

WHY THIS EXISTS
---------------
When the Claude Code Router pattern is used (h2ogpt's
`api_server/agent_tools/claude_tool_runner.py` sets
`ANTHROPIC_BASE_URL=<litellm-proxy>` so the Claude SDK posts Anthropic-format
messages to litellm), the Claude SDK's internal session resume / context
compaction can produce *duplicate `tool_use_id`s* across the conversation
history it replays. Litellm converts those to `tool_call_id`s when targeting
OpenAI-style providers (Azure / OpenAI / vLLM-OpenAI). Azure's
`/chat/completions` then rejects the request with:

    litellm.ContentPolicyViolationError: AzureException -
      Invalid parameter: Duplicate value for 'tool_call_id' of 'call_xxxx'

(Azure's exception mapper buckets the 400 as `ContentPolicyViolationError`
even though the underlying error is a duplicate-id validation, not policy.)

Upstream BerriAI/litellm PR #23104 (merged 2026-03-11) added "Case D"
deduplication to `sanitize_messages_for_tool_calling()` — but it is only
called inside `anthropic_messages_pt()`, which is the OpenAI→Anthropic
*outbound* conversion path. The Anthropic→OpenAI *inbound* conversion
path (used by Claude Code Router → Azure) has no equivalent dedup, so
bumping litellm alone does not fix this.

This hook closes the gap at the proxy boundary: regardless of which
client path produced the duplicate, dedup `tool_call_id`s in the messages
list before the request leaves litellm for the upstream provider.

DEDUP STRATEGY
--------------
Mirrors PR #23104 exactly:

  - Walk `data['messages']` in order.
  - Track seen `tool_call_id` -> message index within each contiguous
    tool-result block.
  - When the same `tool_call_id` appears twice in a block, mark the
    earlier occurrence for removal (last-wins). Last-wins matches
    upstream because the duplicate arises from history replay where the
    *latest* entry represents the final state.
  - Reset the per-block tracker on any non-tool message (user / assistant
    / system) — that's a turn boundary. Tool/function messages with no
    `tool_call_id` are malformed and do NOT reset the block (otherwise
    they'd mask real within-block duplicates).

NEVER PROPAGATES
----------------
Any unexpected error during dedup is swallowed and the request proceeds
unmodified. Dedup is a guardrail, not a hard requirement — failing the
request because dedup couldn't run would be worse than letting it through
and getting the upstream-API error the operator can already see.
"""

import os
from typing import Any, Dict, List, Set

from litellm.integrations.custom_logger import CustomLogger

verbose = os.getenv('H2OGPT_VERBOSE', '0') == '1'
verbose_full = os.getenv('H2OGPT_VERBOSE_FULL', '0') == '1'


def _dedup_tool_call_ids(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a new messages list with duplicate `tool_call_id` tool-result
    messages removed (last-wins) within each contiguous tool-result block.
    If no duplicates, returns the same list unchanged (cheap fast path).
    """
    if not isinstance(messages, list) or len(messages) < 2:
        return messages

    seen_in_block: Dict[str, int] = {}
    duplicates_to_remove: Set[int] = set()

    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            # Non-dict messages don't participate in dedup but DO act as a
            # turn boundary, since their semantics are unknown.
            seen_in_block = {}
            continue
        role = msg.get("role")
        if role in ("tool", "function"):
            tcid = msg.get("tool_call_id")
            if tcid:
                if tcid in seen_in_block:
                    # Mark the earlier occurrence for removal (keep latest).
                    duplicates_to_remove.add(seen_in_block[tcid])
                seen_in_block[tcid] = idx
            # Tool/function with no tool_call_id is malformed — don't reset
            # the block, otherwise we'd mask real within-block duplicates.
        else:
            # user / assistant / system → conversational turn boundary.
            seen_in_block = {}

    if not duplicates_to_remove:
        return messages
    return [m for i, m in enumerate(messages) if i not in duplicates_to_remove]


class DedupToolCallIdsHook(CustomLogger):
    """Server-side dedup of duplicate `tool_call_id` tool-result messages.

    See module docstring for the full why. Applied at the litellm proxy
    pre-call boundary so it covers every provider the proxy routes to —
    Azure, OpenAI, Anthropic, Bedrock, vLLM-OpenAI, etc."""

    def __init__(self):
        super().__init__()
        self.enabled = True
        if verbose or verbose_full:
            print("🧹 DedupToolCallIdsHook: Initialized", flush=True)

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: Dict[str, Any],
        call_type: str,
    ) -> Dict[str, Any]:
        try:
            messages = data.get("messages")
            if not messages:
                return data
            new_messages = _dedup_tool_call_ids(messages)
            if new_messages is not messages:
                # We dropped duplicates — log so operators can see it.
                dropped = len(messages) - len(new_messages)
                model = data.get("model", "")
                # Always log when we actually drop — this is rare and
                # important for debugging the user-visible 400 it would
                # otherwise produce.
                print(
                    f"🧹 DedupToolCallIdsHook: dropped {dropped} duplicate "
                    f"tool_call_id message(s) for model={model!r} "
                    f"(would otherwise 400 upstream).",
                    flush=True,
                )
                data["messages"] = new_messages
        except Exception as e:
            # Dedup is advisory — never break the request because of it.
            if verbose_full:
                import traceback
                print(f"🧹 DedupToolCallIdsHook: error in pre_call: {e}",
                      flush=True)
                traceback.print_exc()
        return data


# Create the hook instance that LiteLLM will use
dedup_tool_call_ids_hook = DedupToolCallIdsHook()

if verbose or verbose_full:
    print(f"🧹 HOOK EXPORT: dedup_tool_call_ids_hook created successfully: "
          f"{type(dedup_tool_call_ids_hook)}", flush=True)
