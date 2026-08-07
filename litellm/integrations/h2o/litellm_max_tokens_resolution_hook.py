#!/usr/bin/env python3
"""LiteLLM Deployment Hook: resolve the two output-token fields onto one.

WHY THIS EXISTS
---------------
The OpenAI chat schema carries two output-token fields — the original
``max_tokens`` and its replacement ``max_completion_tokens``. A request routinely
arrives with BOTH, because a proxy deployment configures one as a ceiling in
``litellm_params`` while the client sends the other. What then reaches the
provider depends on the order litellm happens to iterate, and it is wrong two
different ways:

  * SILENT CEILING OVERRIDE. The provider maps that collapse the pair assign the
    same output key from two branches, so the one iterated LAST wins, and
    ``max_completion_tokens`` is second in the ``get_optional_params``
    signature. Measured on the unmodified tree with
    ``max_tokens=50`` + ``max_completion_tokens=64000``:

        anthropic     -> max_tokens: 64000
        bedrock       -> maxTokens: 64000
        gemini        -> max_output_tokens: 64000
        openai o3     -> max_completion_tokens: 64000
        openai gpt-5  -> max_completion_tokens: 64000

    The caller's 50 is discarded on every one. ``openai_like`` is worse:
    ``replace_max_completion_tokens_with_max_tokens`` overwrites
    unconditionally, so the ceiling always wins. This is the defect reported in
    h2oai/h2ogpte#11992 — 50 output tokens requested, 1666 returned,
    ``finish_reason: "stop"`` — and it is NOT Azure-specific.

  * BOTH FIELDS ON THE WIRE. Providers that forward both send both, and Azure
    2025+ api_versions reject that outright:

        AzureException BadRequestError - Setting 'max_tokens' and
        'max_completion_tokens' at the same time is not supported.

    h2ogpt worked around that with ``additional_drop_params: ["max_tokens"]``,
    which traded the 400 for the first defect: the caller's value was dropped
    and the deployment ceiling applied instead.

WHY A HOOK, AND WHY THIS HOOK POINT
-----------------------------------
Everything here could live in core litellm (``get_optional_params`` plus a
method on each provider config). It deliberately does not: ``h2o-main`` is
rebuilt as ``<upstream-tag>`` + the h2o file delta on every version bump, so
every edited upstream file is a standing rebase cost, while a file added under
``integrations/h2o/`` is additive and costs nothing. Verified against the
unmodified tree that a hook can reach everything it needs and that the collapse
survives into provider mapping, so no core edit buys any capability here.

``async_pre_call_deployment_hook`` is the right hook point rather than
``async_pre_call_hook``:

  * it runs AFTER the router selects a concrete deployment, so a MIXED model
    group is judged per member. Our generated config really has one —
    ``agent_auto`` spans azure, bedrock, anthropic, gemini, mistral and
    openrouter under a single ``model_name`` — and a group-level decision would
    have to be wrong for at least one member. Verified live: the azure member
    gets ``max_completion_tokens``, the bedrock member ``maxTokens``, and a
    sibling carrying ``use_max_completion_tokens: false`` gets ``max_tokens``.
  * it runs BEFORE param mapping, which is what lets it fix the last-wins
    ordering above rather than only renaming a field.
  * its ``kwargs`` are the selected deployment's merged ``litellm_params``, so
    ``api_version``, ``additional_drop_params`` and the directive are all
    readable without a router lookup.

TWO SCOPE LIMITS, BOTH DELIBERATE
---------------------------------
CHAT COMPLETIONS ONLY, enforced on ``call_type``. That dispatch is NOT
chat-specific — ``wrapper_async`` runs it for every ``@client``-decorated async
entrypoint. Observed call types reaching it: ``acompletion``,
``anthropic_messages``, ``atext_completion``. ``litellm.anthropic_messages``
(which backs ``/v1/messages``) declares ``max_tokens`` as a REQUIRED parameter,
so popping it there makes litellm's own wrapper raise

    TypeError: anthropic_messages() missing 1 required positional argument: 'max_tokens'

on the following ``await original_function(...)`` — outside this hook's
``try/except``, with a traceback that never names this hook. Reproduced, hence
the gate. ``atext_completion`` is excluded for a different reason: ``/v1/
completions`` has no ``max_completion_tokens`` at all.

ASYNC PATH ONLY. The dispatch lives in the ``@client`` decorator's ASYNC
wrapper, so a direct sync ``litellm.completion()`` bypasses this. That is the
one capability a core implementation would add, and it is not one we use: this
hook is registered only in the proxy config, and the proxy maps
``/chat/completions`` to ``acompletion``
(``proxy/route_llm_request.py``), so all proxied traffic is async.

INTERACTION WITH THE CAP HOOK
-----------------------------
``litellm_max_tokens_cap_hook`` clips both fields down to the deployment ceiling
from ``async_pre_call_hook`` (pre-routing), so it runs before this. The two are
order-independent: clip-then-resolve and resolve-then-clip produce the same
value, because both reduce and this one takes a minimum. Verified live — a
client asking for 99999 against a 16384 ceiling goes out as
``max_completion_tokens: 16384``.

NEVER PROPAGATES
----------------
Any unexpected error leaves the request exactly as it arrived. Failing a request
because a field could not be resolved would be worse than the defect.
"""

import os
from typing import Any, Dict, List, Optional, Sequence

from litellm.integrations.custom_logger import CustomLogger

verbose = os.getenv("H2OGPT_VERBOSE", "0") == "1"
verbose_full = os.getenv("H2OGPT_VERBOSE_FULL", "0") == "1"

MAX_TOKENS_PARAM = "max_tokens"
MAX_COMPLETION_TOKENS_PARAM = "max_completion_tokens"
MAX_TOKENS_PARAMS = (MAX_TOKENS_PARAM, MAX_COMPLETION_TOKENS_PARAM)

# The directive an operator sets per model_lock entry; h2ogpt's launch_litellm
# forwards it into litellm_params. Popped here so it never reaches a provider —
# it is not a recognized litellm param, so left in place it rides through to the
# request body and Azure/OpenAI reject the unrecognized argument.
DIRECTIVE_PARAM = "use_max_completion_tokens"

# Only these call types are chat completions. See the module docstring for what
# popping max_tokens does to the others.
CHAT_CALL_TYPES = ("completion", "acompletion")

# First api_version year in which Azure chat completions reject `max_tokens` in
# favour of `max_completion_tokens`:
#     Unsupported parameter: 'max_tokens' is not supported with this model.
#     Use 'max_completion_tokens' instead.
# Observed on 2025-04-01-preview against a gpt-4o deployment — a plain chat
# model, not only the o-series. Set at the year rather than a specific preview
# date because Azure rolled this across the 2025 preview versions and the v1
# API, and because `max_completion_tokens` is the field Azure documents as
# current for every 2025 version.
AZURE_YEAR_REQUIRING_MAX_COMPLETION_TOKENS = 2025

# api_version values that mean "the v1 API" rather than a dated preview.
AZURE_V1_API_VERSIONS = frozenset({"preview", "latest", "v1"})


def _usable_int(value: Any) -> Optional[int]:
    """``value`` as a positive int, or None if it is not a usable limit.

    ``bool`` is rejected explicitly because it is an ``int`` subclass. Floats are
    accepted and rounded, because a float ``max_tokens`` really does reach
    litellm — ``AnthropicConfig.map_openai_params`` coerces one with
    ``max(1, int(round(value)))`` — and treating one as unusable would silently
    unbound the request. Strings are NOT coerced: a value litellm cannot
    interpret is left for the provider to reject.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):  # NaN / inf
            return None
        if value <= 0:
            return None
        return max(1, int(round(value)))
    return value if value > 0 else None


def _directive_target(kwargs: Dict[str, Any]) -> Optional[str]:
    """The field an explicit ``use_max_completion_tokens`` asks for.

    Only the exact booleans count. A stray value — a quoted ``"false"`` out of a
    YAML config — must read as "not given" rather than be coerced by truthiness
    into the opposite of what it looks like, since ``bool("false")`` is True.
    """
    directive = kwargs.get(DIRECTIVE_PARAM)
    if directive is True:
        return MAX_COMPLETION_TOKENS_PARAM
    if directive is False:
        return MAX_TOKENS_PARAM
    return None


def _resolved_azure_api_version(api_version: Any) -> Any:
    """The api_version the request will actually be sent with.

    A deployment may omit it, in which case litellm falls back to
    ``litellm.api_version``, then ``AZURE_API_VERSION``, then
    ``litellm.AZURE_DEFAULT_API_VERSION`` (a 2025 version today). Mirror that
    chain — including its ``or``-based falsiness — or an Azure deployment with no
    configured api_version resolves the field against nothing and sends
    ``max_tokens`` to a version that rejects it.
    """
    if api_version:
        return api_version
    import litellm

    try:
        from litellm.secret_managers.main import get_secret
    except Exception:
        get_secret = None  # type: ignore[assignment]
    env_version = None
    if get_secret is not None:
        try:
            env_version = get_secret("AZURE_API_VERSION")
        except Exception:
            env_version = None
    return (getattr(litellm, "api_version", None)
            or env_version
            or getattr(litellm, "AZURE_DEFAULT_API_VERSION", None))


def _azure_target(api_version: Any) -> Optional[str]:
    """Azure's output-token field, from the api_version.

    Returns None for anything unrecognizable — including a non-str api_version
    (an int year, bytes, a list). This runs on every Azure chat request, so it
    must not turn a misconfigured api_version into a traceback.
    """
    api_version = _resolved_azure_api_version(api_version)
    if not isinstance(api_version, str) or not api_version:
        return None
    if api_version in AZURE_V1_API_VERSIONS:
        return MAX_COMPLETION_TOKENS_PARAM
    year = api_version.split("-")[0]
    if len(year) != 4 or not year.isdigit():
        return None
    if int(year) >= AZURE_YEAR_REQUIRING_MAX_COMPLETION_TOKENS:
        return MAX_COMPLETION_TOKENS_PARAM
    return MAX_TOKENS_PARAM


def _is_reasoning_model(model: str) -> bool:
    """o-series / gpt-5, asked of litellm's own detectors rather than a name list
    maintained here.

    Those configs rename ``max_tokens`` to ``max_completion_tokens`` themselves,
    but they write the key the generic mapping then overwrites when BOTH fields
    are present — which is why they appear in the last-wins table above.
    Declaring the target here is what makes a request carrying both resolve to
    one field holding the tighter value.
    """
    import litellm

    try:
        if litellm.AzureOpenAIO1Config().is_o_series_model(model=model):
            return True
    except Exception:
        pass
    try:
        if litellm.AzureOpenAIGPT5Config.is_model_gpt_5_model(model=model):
            return True
    except Exception:
        pass
    return False


def _provider_and_model(kwargs: Dict[str, Any]) -> tuple:
    """(provider, bare model) for the selected deployment.

    ``custom_llm_provider`` is not always populated at this point, so the
    prefixed model string is the fallback signal.
    """
    model = kwargs.get("model")
    if not isinstance(model, str):
        return None, None
    provider = kwargs.get("custom_llm_provider")
    if not provider and "/" in model:
        provider = model.split("/", 1)[0]
    bare = model.split("/", 1)[1] if provider and model.startswith(f"{provider}/") else model
    return provider, bare


def _supported_params(model: str, provider: Optional[str]) -> Optional[List[str]]:
    """What this model/provider accepts, or None when litellm cannot say.

    Used so a target the provider does not accept is never forced — forcing one
    would leave the request with no output-token ceiling at all, which is worse
    than the defect being fixed.
    """
    import litellm

    try:
        params = litellm.get_supported_openai_params(
            model=model, custom_llm_provider=provider)
    except Exception:
        return None
    return list(params) if params else None


def _eligible(
    param: str,
    supported: Optional[Sequence[str]],
    drop_list: Optional[Sequence[str]],
) -> bool:
    """A field can be a target only if the provider accepts it and the operator
    has not dropped it. An operator who put a param in
    ``additional_drop_params`` meant it, and moving a value onto that field
    would defeat the drop."""
    if isinstance(drop_list, (list, tuple)) and param in drop_list:
        return False
    if supported is None:
        return True  # litellm could not tell us; do not block on that
    return param in supported


class MaxTokensResolutionHook(CustomLogger):
    """Collapse ``max_tokens`` / ``max_completion_tokens`` onto the one field the
    selected deployment accepts, keeping the tighter value."""

    def __init__(self):
        super().__init__()
        self.enabled = True
        if verbose or verbose_full:
            print("MaxTokensResolutionHook: Initialized", flush=True)

    # -- the decision ------------------------------------------------------

    def _target_field(
        self,
        kwargs: Dict[str, Any],
        supported: Optional[List[str]],
        drop_list: Optional[Sequence[str]],
        present: List[str],
    ) -> Optional[str]:
        """Which single field should survive, or None to change nothing."""
        provider, bare_model = _provider_and_model(kwargs)

        # 1. An explicit directive outranks every detection below. That is what
        #    makes the model_lock flag a control rather than a suggestion.
        target = _directive_target(kwargs)

        # 2. Otherwise, what the provider itself requires.
        if target is None and bare_model and _is_reasoning_model(bare_model):
            target = MAX_COMPLETION_TOKENS_PARAM
        if target is None and provider == "azure":
            target = _azure_target(kwargs.get("api_version"))

        if target is not None and _eligible(target, supported, drop_list):
            return target

        # 3. No usable preference. One field is already canonical; two are not —
        #    leaving both is what makes the last-wins provider maps pick the
        #    looser value, so collapse onto whichever field is eligible,
        #    preferring `max_tokens` since every provider understands it.
        if len(present) < 2:
            return None
        for candidate in MAX_TOKENS_PARAMS:
            if _eligible(candidate, supported, drop_list):
                return candidate
        return None

    # -- the hook ----------------------------------------------------------

    async def async_pre_call_deployment_hook(
        self, kwargs: Dict[str, Any], call_type: Any
    ) -> Optional[dict]:
        try:
            # The directive is not a recognized litellm param, so it must come
            # off the request on EVERY call type — including the ones below that
            # we otherwise leave alone — or it is forwarded to the provider in
            # the body and rejected as an unrecognized argument.
            modified: Optional[Dict[str, Any]] = None
            if DIRECTIVE_PARAM in kwargs:
                modified = dict(kwargs)
                modified.pop(DIRECTIVE_PARAM, None)

            if getattr(call_type, "value", call_type) not in CHAT_CALL_TYPES:
                return modified

            present = [p for p in MAX_TOKENS_PARAMS if p in kwargs]
            if not present:
                return modified

            values = [
                v for v in (_usable_int(kwargs[p]) for p in present)
                if v is not None
            ]
            if not values:
                # Nothing usable to move or tighten. Leave the request exactly as
                # it arrived so the provider rejects it as loudly as it would
                # have without us — rewriting it here would turn a
                # garbage-in/error-out request into an unbounded one.
                return modified
            resolved = min(values)

            _, bare_model = _provider_and_model(kwargs)
            provider, _ = _provider_and_model(kwargs)
            supported = _supported_params(bare_model or "", provider)
            drop_list = kwargs.get("additional_drop_params")

            target = self._target_field(kwargs, supported, drop_list, present)
            if target is None:
                return modified

            if modified is None:
                modified = dict(kwargs)
            for param in MAX_TOKENS_PARAMS:
                if param != target:
                    modified.pop(param, None)
            modified[target] = resolved

            if (verbose or verbose_full) and (
                len(present) > 1 or present[0] != target
                or kwargs.get(target) != resolved
            ):
                print(f"MaxTokensResolutionHook: {kwargs.get('model')}: "
                      f"{ {p: kwargs.get(p) for p in present} } -> "
                      f"{target}={resolved}", flush=True)
            return modified
        except Exception as e:
            # Never break a request over a field-name resolution.
            if verbose_full:
                import traceback
                print(f"MaxTokensResolutionHook: error in pre_call: {e}",
                      flush=True)
                traceback.print_exc()
            return None


# Create the hook instance that LiteLLM will use
max_tokens_resolution_hook = MaxTokensResolutionHook()

if verbose or verbose_full:
    print(f"HOOK EXPORT: max_tokens_resolution_hook created successfully: "
          f"{type(max_tokens_resolution_hook)}", flush=True)
