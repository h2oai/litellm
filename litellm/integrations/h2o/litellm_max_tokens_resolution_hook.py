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

  * BOTH FIELDS ON THE WIRE. Providers that forward both send both, and that is
    rejected outright:

        AzureException BadRequestError - Setting 'max_tokens' and
        'max_completion_tokens' at the same time is not supported.

    Confirmed against the live ``h2ogpt2`` Azure deployment, and NOT
    Azure-specific — raw OpenAI returns the same 400 for ``gpt-4o-mini``.
    h2ogpt worked around it with ``additional_drop_params: ["max_tokens"]``,
    which traded the 400 for the first defect: the caller's value was dropped
    and the deployment ceiling applied instead.

    Worth stating precisely what is NOT true, because it shaped an earlier
    version of this file: "Azure 2025+ rejects ``max_tokens``" does not hold in
    general. That deployment accepts EITHER field on its own, on api-version
    2024-02-01, 2024-08-01-preview and 2025-04-01-preview alike; only the pair
    fails. Azure's wording is "not supported with **this model**", so the
    single-field rejection is model-specific — which is why the reasoning-model
    preference below is keyed on the model rather than only on the api_version,
    and why the load-bearing behaviour here is the collapse plus tighter-wins
    rather than the field-name routing.

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
the gate.

``atext_completion`` (``/v1/completions``) is handled separately rather than
skipped: that route has no ``max_completion_tokens`` at all, so it never gets the
rename — but it does still get the pair COLLAPSED onto ``max_tokens``. Skipping it
outright meant that removing h2ogpt's ``additional_drop_params`` workaround left an
Azure text-completion deployment mapping BOTH fields, moving toward the same 400
the drop existed to prevent. Collapsing is safe there in a way it is not for
``anthropic_messages``, because ``atext_completion`` does not declare
``max_tokens`` as a required parameter.

ASYNC PATH ONLY. The dispatch lives in the ``@client`` decorator's ASYNC
wrapper, so a direct sync ``litellm.completion()`` bypasses this entirely — and
the consequence is worse than "no resolution": with the core plumbing reverted,
nothing strips the directive on that path either, so a sync call carrying
``use_max_completion_tokens`` yields BOTH token fields plus
``extra_body: {"use_max_completion_tokens": ...}``. That is the one capability a
core implementation would add.

It is not a path we use: this hook is registered only in the proxy config, the
proxy maps ``/chat/completions`` to ``acompletion``
(``proxy/route_llm_request.py``), and the ``/health`` chat probe also goes
through ``litellm.acompletion`` (``health_check_helpers.py``). Anything that
later embeds litellm in-process with a sync ``completion()`` call — a guardrail,
a new route — would need the directive stripped some other way.

INTERACTION WITH THE CAP HOOK — A COUPLED CONTRACT
--------------------------------------------------
``litellm_max_tokens_cap_hook`` clips both fields down to the deployment ceiling
from ``async_pre_call_hook`` (pre-routing), so it runs before this. The two are
order-independent: clip-then-resolve and resolve-then-clip produce the same
value, because both reduce and this one takes a minimum. Verified live — a client
asking for 99999 against a 16384 ceiling goes out as
``max_completion_tokens: 16384``.

They are COUPLED, not merely adjacent, and this file is why. The cap hook used to
clip ``isinstance(v, int)`` only, so a float sailed past the ceiling — survivable
while nothing normalised floats, because the provider then rejected the float and
the request failed loudly. Once this hook began coercing floats to ints for every
provider (see ``_usable_int``), that gap turned a rejected request into an
ACCEPTED over-cap one:

    model_info.max_output_tokens = 8192, no litellm_params ceiling
      client max_tokens=99999    -> cap clips -> 8192
      client max_tokens=99999.0  -> cap SKIPS -> 99999   (over the cap)

So ``_cap_in`` now clips floats too. If the coercion here is ever widened again —
strings, Decimal — the cap hook has to widen with it, or the ceiling it exists to
enforce is bypassable. The order-independence tests import the REAL cap hook and
assert absolute values, not just symmetry, because a symmetry-only assertion
cannot catch a symmetric bug and is exactly how this one got through.

NEVER PROPAGATES, BUT NOT FAIL-OPEN FOR THE DIRECTIVE
-----------------------------------------------------
Any unexpected error leaves the TOKEN FIELDS exactly as they arrived — failing a
request because a field could not be resolved would be worse than the defect.
The directive is different: it is stripped before the ``try`` and the ``except``
returns that stripped copy, because ``use_max_completion_tokens`` is not a
recognized litellm param and exists only for this hook to consume. Returning the
original kwargs on error would leak it into the request body and turn an internal
bug into a 400 on every request to that deployment.
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

# The legacy /v1/completions route. It has no `max_completion_tokens` at all, so
# it is NOT a chat call type — but it still needs the pair COLLAPSED, always onto
# `max_tokens`. Without this, removing h2ogpt's `additional_drop_params` workaround
# left an Azure text-completion deployment mapping BOTH fields
# ({'max_tokens': 50, 'max_completion_tokens': 16384}), which moves toward the very
# 400 the drop existed to prevent. Collapsing here is safe in a way that
# `anthropic_messages` is not: `atext_completion` does not declare `max_tokens` as
# a required parameter, so removing the other field cannot make litellm's own
# wrapper raise.
TEXT_COMPLETION_CALL_TYPES = ("text_completion", "atext_completion")

# Providers whose model string really is an OpenAI model id, so litellm's
# substring-based o-series / gpt-5 detection is meaningful. Everywhere else the
# name is operator-chosen and a match would be a coincidence — see rule 2.
REASONING_NAME_PROVIDERS = ("azure", "openai", "azure_ai", "azure_text")

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

    Truthiness is NEVER used: ``bool("false")`` is True, so a quoted ``"false"``
    out of a hand-written YAML config would mean the opposite of what it reads
    like. But treating every non-boolean as "not given" was not good enough
    either — on an Azure 2025 deployment the provider rule then supplies
    ``max_completion_tokens`` anyway, so ``"false"`` / ``0`` / ``""`` still ended
    up meaning the inverse of the operator's intent, silently.

    So the shapes an operator plausibly writes are recognised explicitly, and
    anything else is a logged no-op rather than a silent guess. h2ogpt's own
    config generation only ever emits real booleans, so the string forms can only
    arrive from a hand-written litellm config.
    """
    if DIRECTIVE_PARAM not in kwargs:
        return None
    directive = kwargs[DIRECTIVE_PARAM]

    if directive is True or directive is False:
        return (MAX_COMPLETION_TOKENS_PARAM if directive
                else MAX_TOKENS_PARAM)
    if isinstance(directive, int) and directive in (0, 1):
        return (MAX_COMPLETION_TOKENS_PARAM if directive == 1
                else MAX_TOKENS_PARAM)
    if isinstance(directive, str):
        normalised = directive.strip().lower()
        if normalised in ("true", "yes", "1"):
            return MAX_COMPLETION_TOKENS_PARAM
        if normalised in ("false", "no", "0"):
            return MAX_TOKENS_PARAM

    if directive is not None and (verbose or verbose_full):
        print(f"MaxTokensResolutionHook: ignoring unrecognised "
              f"{DIRECTIVE_PARAM}={directive!r} — expected a boolean", flush=True)
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

    # get_secret_STR, matching litellm's own api_version chain (main.py). Plain
    # `get_secret` performs a secret-manager fetch when one is configured, and
    # applies bool coercion — neither is wanted for a version string, in a hook
    # that runs on every Azure request.
    try:
        from litellm.secret_managers.main import get_secret_str
    except Exception:
        get_secret_str = None  # type: ignore[assignment]
    env_version = None
    if get_secret_str is not None:
        try:
            env_version = get_secret_str("AZURE_API_VERSION")
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
    # Pre-2025: NO preference, deliberately.
    #
    # An earlier revision returned `max_tokens` here, on the reasoning that older
    # api_versions predate `max_completion_tokens`. Measured against the live
    # `h2ogpt2` deployment, that is not true — api-version 2024-02-01 and
    # 2024-08-01-preview both accept `max_completion_tokens` on its own and
    # return finish_reason=length with the requested 50 tokens. So renaming a
    # lone `max_completion_tokens` there would mutate a request that works, for
    # no measured benefit.
    #
    # Returning None does not weaken the fix: when BOTH fields are present a
    # collapse is still required, and the no-preference branch in the hook
    # already collapses onto `max_tokens`, which every provider understands.
    return None


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
        #
        # Reasoning detection is delegated to litellm's own `is_o_series_model` /
        # `is_model_gpt_5_model`, which are SUBSTRING matches ("o1" in model). On a
        # first-party route the deployment name is the model name, so that is fine.
        # On a self-hosted route it is operator-chosen: `hosted_vllm/o1-local`
        # matched and got `max_completion_tokens`, which TGI and older vLLM simply
        # ignore — the ceiling silently disappears, and
        # `get_supported_openai_params` is no guard because it reports that field
        # supported for EVERY openai-compatible provider regardless of what the
        # server accepts. So this rule is scoped to the providers whose names
        # really are OpenAI model ids.
        if (
            target is None
            and bare_model
            and provider in REASONING_NAME_PROVIDERS
            and _is_reasoning_model(bare_model)
        ):
            target = MAX_COMPLETION_TOKENS_PARAM
        if target is None and provider == "azure":
            target = _azure_target(kwargs.get("api_version"))

        if target is not None:
            if _eligible(target, supported, drop_list):
                return target
            # The preferred field is dropped or unsupported. Fall through with no
            # preference rather than returning it: an operator who wrote both
            # `use_max_completion_tokens: false` AND
            # `additional_drop_params: ["max_tokens"]` has given contradictory
            # config, and honouring the directive literally would leave the value
            # on the field litellm is about to strip — the caller's ceiling would
            # vanish with no error. Falling through lets rule 3 below move it
            # somewhere that survives, which respects the drop (the dropped field
            # is still never sent) and keeps the limit.
            target = None

        # 3. The caller's own field is being DROPPED. This was the original
        #    signal in the first attempt at this (h2oai/litellm#25): a deployment
        #    carrying `additional_drop_params: ["max_tokens"]` accepts only the
        #    other field, so moving the value across is what keeps the caller's
        #    limit instead of letting the drop destroy it — which is the whole
        #    defect. Applies on ANY provider, not just Azure: an operator can put
        #    that drop on a vllm or openai-compatible deployment too, and
        #    h2ogpt's own `_drop()` writes into the same list.
        #
        #    Not in tension with "never resurrect a dropped param" — the target
        #    still has to be eligible, so a field the operator dropped is never
        #    the destination.
        if target is None and not any(
            _eligible(p, supported, drop_list) for p in present
        ):
            for candidate in MAX_TOKENS_PARAMS:
                if candidate not in present and _eligible(
                    candidate, supported, drop_list
                ):
                    return candidate

        # 4. No usable preference. One field is already canonical; two are not —
        #    leaving both is what makes the last-wins provider maps pick the
        #    looser value, so collapse onto whichever field is eligible,
        #    preferring `max_tokens` since every provider understands it.
        if len(present) < 2:
            return None
        for candidate in MAX_TOKENS_PARAMS:
            if _eligible(candidate, supported, drop_list):
                return candidate

        # Both fields present and NEITHER is eligible — e.g. an mt-only provider
        # (xai, watsonx, replicate, azure_text, petals) whose deployment also
        # drops max_tokens. "Collapse the pair" is the load-bearing guarantee, so
        # still collapse: emitting both leaves `UnsupportedParamsError` on the
        # table at `drop_params: false`, and with the proxy's `drop_params: true`
        # it makes no difference to the ceiling (litellm strips both either way).
        # Keep the caller's tighter value on the field they are most likely to
        # have meant.
        return present[0]

    # -- the hook ----------------------------------------------------------

    async def async_pre_call_deployment_hook(
        self, kwargs: Dict[str, Any], call_type: Any
    ) -> Optional[dict]:
        # Stripping the directive happens FIRST and outside the try, and the
        # except below returns this same value rather than None.
        #
        # "Leave the request exactly as it arrived" is the right failure policy
        # for the token fields — but it is the WRONG one for the directive, which
        # exists only because this hook removes it. Returning None on an
        # unexpected error makes litellm keep the original kwargs, so an
        # unrecognized `use_max_completion_tokens` reaches the provider body and
        # is rejected: measured against an unpatched proxy as
        #     {"max_tokens": 50, "use_max_completion_tokens": false, ...}
        # So a bug in the resolution below must not be able to turn into a 400 on
        # every request to that deployment.
        modified: Optional[Dict[str, Any]] = None
        if DIRECTIVE_PARAM in kwargs:
            modified = dict(kwargs)
            modified.pop(DIRECTIVE_PARAM, None)

        try:
            resolved_call_type = getattr(call_type, "value", call_type)
            forced_target: Optional[str] = None
            if resolved_call_type in TEXT_COMPLETION_CALL_TYPES:
                # /v1/completions: collapse, but only ever onto max_tokens.
                forced_target = MAX_TOKENS_PARAM
            elif resolved_call_type not in CHAT_CALL_TYPES:
                return modified

            # An explicitly-None field counts as ABSENT, not as garbage: `None` is
            # the OpenAI SDK's default and litellm strips None-valued params, so it
            # never reaches the wire and must not block the resolution. Verified
            # that ordinary requests never carry a None-valued token key at all —
            # only a client explicitly passing None does.
            present = [p for p in MAX_TOKENS_PARAMS
                       if p in kwargs and kwargs[p] is not None]
            if not present:
                return modified

            values = [
                v for v in (_usable_int(kwargs[p]) for p in present)
                if v is not None
            ]
            if len(values) != len(present):
                # At least one field carries a value that is present, non-None and
                # NOT a usable limit — 0, -1, "50", True. Change NOTHING, even if
                # another field is usable.
                #
                # Resolving here would do two harmful things at once: discard the
                # garbage field (so the deployment ceiling silently applies, which
                # is h2ogpte#11992's exact symptom) AND suppress the loud error the
                # request would otherwise get. Measured on azure/gpt-4o-mini with a
                # 64000 ceiling: `max_tokens: "50"` used to produce both fields and
                # an Azure 400; resolving it produced `max_completion_tokens:
                # 64000` and a 200. Garbage in must stay error out — that is rule 3,
                # and it has to hold in the MIXED case too, not just when every
                # field is unusable.
                return modified
            resolved = min(values)

            provider, bare_model = _provider_and_model(kwargs)
            supported = _supported_params(bare_model or "", provider)
            drop_list = kwargs.get("additional_drop_params")

            if forced_target is not None:
                target = (forced_target
                          if _eligible(forced_target, supported, drop_list)
                          else None)
            else:
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
            # Never break a request over a field-name resolution — but DO keep the
            # directive stripped (see the note above the try). Returning None here
            # would hand litellm the original kwargs and leak the directive to the
            # provider, turning an internal error into a 400 on every request.
            if verbose_full:
                import traceback
                print(f"MaxTokensResolutionHook: error in pre_call: {e}",
                      flush=True)
                traceback.print_exc()
            return modified


# Create the hook instance that LiteLLM will use
max_tokens_resolution_hook = MaxTokensResolutionHook()

if verbose or verbose_full:
    print(f"HOOK EXPORT: max_tokens_resolution_hook created successfully: "
          f"{type(max_tokens_resolution_hook)}", flush=True)
