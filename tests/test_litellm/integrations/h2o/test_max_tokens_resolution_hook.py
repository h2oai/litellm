"""Tests for the h2o MaxTokensResolutionHook.

The hook collapses `max_tokens` / `max_completion_tokens` onto the one field the
selected deployment accepts, keeping the tighter value. It replaces an earlier
implementation that lived in core litellm (`get_optional_params` plus a method on
each provider config); everything moved into the hook because `h2o-main` is
rebuilt as `<upstream-tag>` + the h2o delta on every version bump, so an edited
upstream file is a standing rebase cost while a file under `integrations/h2o/` is
additive. Verified against the unmodified tree that the hook can reach
everything it needs, so no core edit buys any capability.

Two defects motivate it, both reproduced on the unmodified tree:

1. SILENT CEILING OVERRIDE. The provider maps that collapse the pair assign the
   same output key from two branches, so the one iterated LAST wins, and
   `max_completion_tokens` is second. With max_tokens=50 + a 64000 ceiling:
       anthropic -> max_tokens: 64000     bedrock -> maxTokens: 64000
       gemini    -> max_output_tokens: 64000
       o3/gpt-5  -> max_completion_tokens: 64000
   That is h2oai/h2ogpte#11992 (50 requested, 1666 returned), and it is not
   Azure-specific.

2. BOTH FIELDS ON THE WIRE. Azure 2025+ rejects the pair outright.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
from litellm.integrations.h2o.litellm_max_tokens_resolution_hook import (
    DIRECTIVE_PARAM,
    MAX_COMPLETION_TOKENS_PARAM,
    MAX_TOKENS_PARAM,
    MaxTokensResolutionHook,
    _azure_target,
    _directive_target,
    _usable_int,
)

TOKEN_FIELDS = ("max_tokens", "max_completion_tokens", "maxTokens",
                "max_output_tokens")


@pytest.fixture
def hook():
    return MaxTokensResolutionHook()


async def run_hook(hook, call_type="acompletion", **kwargs):
    """Drive the hook the way `wrapper_async` does, returning the kwargs the
    request would go on with."""
    result = await hook.async_pre_call_deployment_hook(kwargs, call_type)
    return kwargs if result is None else result


def tokens(params):
    return {k: v for k, v in params.items() if k in TOKEN_FIELDS}


AZURE_2025 = {"model": "azure/gpt-4o-mini", "api_version": "2025-04-01-preview"}


# --------------------------------------------------------------------------
# The two defects
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_callers_limit_beats_the_deployment_ceiling(hook):
    """Defect 1. Before this, the ceiling silently replaced the request."""
    out = await run_hook(hook, **AZURE_2025,
                         max_tokens=50, max_completion_tokens=64000)
    assert tokens(out) == {"max_completion_tokens": 50}


@pytest.mark.asyncio
async def test_a_ceiling_is_never_raised_by_a_client(hook):
    out = await run_hook(hook, **AZURE_2025,
                         max_tokens=99999, max_completion_tokens=8192)
    assert tokens(out) == {"max_completion_tokens": 8192}


@pytest.mark.asyncio
async def test_only_one_field_survives(hook):
    """Defect 2. Both on the wire is what Azure 2025+ rejects."""
    out = await run_hook(hook, **AZURE_2025,
                         max_tokens=50, max_completion_tokens=64000)
    assert len(tokens(out)) == 1


@pytest.mark.asyncio
async def test_a_deployment_that_accepts_the_callers_field_is_untouched(hook):
    """anthropic takes max_tokens natively; nothing to do."""
    out = await run_hook(hook, model="anthropic/claude-sonnet-4-5-20250929",
                         max_tokens=50)
    assert tokens(out) == {"max_tokens": 50}


# --------------------------------------------------------------------------
# Azure, by api_version, in both directions
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "api_version",
    ["2025-04-01-preview", "2025-01-01-preview", "2026-02-01",
     "preview", "latest", "v1"],
)
async def test_azure_2025_and_v1_use_completion_tokens(hook, api_version):
    out = await run_hook(hook, model="azure/gpt-4o-mini",
                         api_version=api_version, max_tokens=50)
    assert tokens(out) == {"max_completion_tokens": 50}


@pytest.mark.asyncio
@pytest.mark.parametrize("api_version",
                         ["2024-08-01-preview", "2024-02-01", "2023-12-01-preview"])
async def test_pre_2025_azure_leaves_a_lone_field_alone(hook, api_version):
    """No preference pre-2025, on evidence.

    An earlier revision renamed a lone `max_completion_tokens` to `max_tokens`
    here, reasoning that older api_versions predate the newer field. Measured
    against the live `h2ogpt2` deployment that is false: api-version 2024-02-01
    and 2024-08-01-preview both accept `max_completion_tokens` alone and return
    finish_reason=length with the requested 50 tokens. Renaming would mutate a
    request that already works.
    """
    out = await run_hook(hook, model="azure/gpt-4o-mini",
                         api_version=api_version, max_completion_tokens=50)
    assert tokens(out) == {"max_completion_tokens": 50}


@pytest.mark.asyncio
@pytest.mark.parametrize("api_version", ["2024-08-01-preview", "2024-02-01"])
async def test_pre_2025_azure_still_collapses_the_pair(hook, api_version):
    """Having no preference must not weaken the actual fix: the pair is what
    live Azure rejects, on every api-version tested, so it still has to become
    one field with the tighter value."""
    out = await run_hook(hook, model="azure/gpt-4o-mini",
                         api_version=api_version,
                         max_tokens=50, max_completion_tokens=64000)
    assert tokens(out) == {"max_tokens": 50}


@pytest.mark.parametrize(
    "api_version, expected",
    [
        ("2025-04-01-preview", MAX_COMPLETION_TOKENS_PARAM),
        ("preview", MAX_COMPLETION_TOKENS_PARAM),
        ("2024-08-01-preview", None),
        ("garbage", None),
        (2025, None),
        (b"2025-04-01", None),
        (["2025"], None),
    ],
)
def test_azure_target_never_raises_on_a_malformed_api_version(
    api_version, expected
):
    """This runs on every Azure chat request; a misconfigured api_version must
    not turn the hook into a traceback."""
    assert _azure_target(api_version) == expected


@pytest.mark.parametrize("api_version", ["", None])
def test_a_missing_api_version_falls_back_the_way_litellm_does(api_version):
    """A deployment may omit api_version, in which case litellm falls back to
    litellm.api_version / AZURE_API_VERSION / AZURE_DEFAULT_API_VERSION — a 2025
    version today. Without mirroring that chain, such a deployment resolved the
    field against nothing and sent max_tokens to a version that rejects it.
    Caught by the h2ogpt cross-model matrix, not by hand."""
    assert litellm.AZURE_DEFAULT_API_VERSION.startswith("2025")
    assert _azure_target(api_version) == MAX_COMPLETION_TOKENS_PARAM


@pytest.mark.asyncio
async def test_azure_with_no_api_version_still_uses_completion_tokens(hook):
    out = await run_hook(hook, model="azure/gpt-4o-mini", max_tokens=50)
    assert tokens(out) == {"max_completion_tokens": 50}


# --------------------------------------------------------------------------
# Reasoning models — asked of litellm's own detectors, not a name list here
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    ["openai/o3", "openai/gpt-5", "azure/o4-mini", "azure/gpt-5.2",
     "azure/o_series/my-deployment"],
)
async def test_reasoning_models_get_completion_tokens(hook, model):
    """Their own transforms rename max_tokens, but write the key the generic
    mapping then overwrites when both are present — which is why they showed up
    in the last-wins table."""
    out = await run_hook(hook, model=model,
                         max_tokens=50, max_completion_tokens=64000)
    assert tokens(out) == {"max_completion_tokens": 50}


# --------------------------------------------------------------------------
# The directive is the authoritative control
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "directive, expected",
    [(True, "max_completion_tokens"), (False, "max_tokens")],
)
async def test_directive_overrides_the_api_version_detection(
    hook, directive, expected
):
    out = await run_hook(hook, **AZURE_2025, max_tokens=50,
                         **{DIRECTIVE_PARAM: directive})
    assert tokens(out) == {expected: 50}


@pytest.mark.asyncio
async def test_directive_works_on_a_provider_with_no_detection(hook):
    """A non-Azure deployment that needs the newer field can just say so."""
    out = await run_hook(hook, model="hosted_vllm/my-model", max_tokens=50,
                         **{DIRECTIVE_PARAM: True})
    assert tokens(out) == {"max_completion_tokens": 50}


@pytest.mark.asyncio
async def test_directive_overrides_even_reasoning_detection(hook):
    """A documented foot-gun, and the point of it being 100% a control."""
    out = await run_hook(hook, model="azure/gpt-5.2",
                         api_version="2025-04-01-preview", max_tokens=50,
                         **{DIRECTIVE_PARAM: False})
    assert tokens(out) == {"max_tokens": 50}


@pytest.mark.asyncio
async def test_the_directive_never_reaches_the_provider(hook):
    """It is not a recognized litellm param, so left in place it rides through
    to the request body — measured against an unpatched proxy, which sent
    `{"max_tokens": 50, "use_max_completion_tokens": false, ...}` to the
    deployment, where Azure rejects the unrecognized argument."""
    out = await run_hook(hook, **AZURE_2025, max_tokens=50,
                         **{DIRECTIVE_PARAM: False})
    assert DIRECTIVE_PARAM not in out


@pytest.mark.asyncio
@pytest.mark.parametrize("call_type",
                         ["anthropic_messages", "atext_completion",
                          "aembedding", None])
async def test_the_directive_is_stripped_even_on_gated_call_types(
    hook, call_type
):
    """Otherwise a deployment carrying the directive would leak it on any
    non-chat entrypoint that shares this dispatch."""
    out = await run_hook(hook, call_type=call_type, **AZURE_2025,
                         max_tokens=50, **{DIRECTIVE_PARAM: True})
    assert DIRECTIVE_PARAM not in out


@pytest.mark.parametrize(
    "value, expected",
    [
        (True, MAX_COMPLETION_TOKENS_PARAM),
        (False, MAX_TOKENS_PARAM),
        (1, MAX_COMPLETION_TOKENS_PARAM),
        (0, MAX_TOKENS_PARAM),
        ("true", MAX_COMPLETION_TOKENS_PARAM),
        ("false", MAX_TOKENS_PARAM),
        ("True", MAX_COMPLETION_TOKENS_PARAM),
        ("FALSE", MAX_TOKENS_PARAM),
        (" false ", MAX_TOKENS_PARAM),
        ("yes", MAX_COMPLETION_TOKENS_PARAM),
        ("no", MAX_TOKENS_PARAM),
        ("1", MAX_COMPLETION_TOKENS_PARAM),
        ("0", MAX_TOKENS_PARAM),
    ],
)
def test_the_directive_recognises_what_an_operator_plausibly_writes(value, expected):
    """Truthiness is never used — `bool("false")` is True — but treating every
    non-boolean as "not given" was not good enough either: on an Azure 2025
    deployment the provider rule then supplied `max_completion_tokens` anyway, so
    `"false"` still meant the INVERSE of the operator's intent, silently."""
    assert _directive_target({DIRECTIVE_PARAM: value}) == expected


@pytest.mark.parametrize("junk", ["", [], {}, 2, -1, "maybe", 1.5, None])
def test_an_unrecognisable_directive_is_a_no_op(junk):
    """Not a guess. Anything outside the recognised shapes falls through to
    provider detection and is logged under H2OGPT_VERBOSE."""
    assert _directive_target({DIRECTIVE_PARAM: junk}) is None


def test_an_absent_directive_is_a_no_op():
    assert _directive_target({}) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("written, expected",
                         [("false", "max_tokens"), (0, "max_tokens"),
                          ("no", "max_tokens")])
async def test_a_string_false_no_longer_means_its_opposite(hook, written, expected):
    """Regression for the silent inversion: on Azure 2025 these used to come out
    as `max_completion_tokens` because the api_version rule supplied it once the
    directive was discarded."""
    out = await run_hook(hook, **AZURE_2025, max_tokens=50,
                         **{DIRECTIVE_PARAM: written})
    assert tokens(out) == {expected: 50}


# --------------------------------------------------------------------------
# Rule 3, the mixed case — garbage next to a usable value
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("garbage", ["50", 0, -5, True, False])
async def test_garbage_beside_a_usable_value_changes_nothing(hook, garbage):
    """The mixed case, which rule 3 used to miss.

    `values` filtered the unusable entry out but the pop loop still removed the
    field carrying it, so on azure/gpt-4o-mini with a 64000 ceiling a client
    sending `max_tokens: "50"` got `max_completion_tokens: 64000` and a 200 —
    h2ogpte#11992's exact symptom — where before it got both fields and a loud
    Azure 400. Garbage in must stay error out even when another field is usable.
    """
    out = await run_hook(hook, **AZURE_2025,
                         max_tokens=garbage, max_completion_tokens=64000)
    assert tokens(out) == {"max_tokens": garbage,
                           "max_completion_tokens": 64000}


@pytest.mark.asyncio
async def test_an_explicit_none_counts_as_absent_not_as_garbage(hook):
    """`None` is the OpenAI SDK default and litellm strips None-valued params, so
    it never reaches the wire. Blocking on it would disable the resolution for any
    client that passes it explicitly."""
    out = await run_hook(hook, **AZURE_2025,
                         max_tokens=None, max_completion_tokens=64000)
    assert tokens(out) == {"max_completion_tokens": 64000}


# --------------------------------------------------------------------------
# Rule 2 is scoped to providers whose model string is an OpenAI model id
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    ["hosted_vllm/o1-local", "hosted_vllm/my-o3-finetune",
     "openrouter/o1-preview-clone", "hosted_vllm/gpt-5-lookalike"],
)
async def test_a_self_hosted_name_that_looks_reasoning_is_not_renamed(hook, model):
    """litellm's detection is a substring match, and on a self-hosted route the
    name is operator-chosen. TGI and older vLLM ignore `max_completion_tokens`, so
    renaming there makes the ceiling silently disappear — and
    `get_supported_openai_params` is no guard, since it reports that field
    supported for every openai-compatible provider."""
    out = await run_hook(hook, model=model, max_tokens=50)
    assert tokens(out) == {"max_tokens": 50}


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["openai/o3", "openai/gpt-5", "azure/o4-mini"])
async def test_a_first_party_reasoning_name_still_is_renamed(hook, model):
    out = await run_hook(hook, model=model,
                         max_tokens=50, max_completion_tokens=64000)
    assert tokens(out) == {"max_completion_tokens": 50}


@pytest.mark.asyncio
async def test_the_pair_is_collapsed_even_when_no_field_is_eligible(hook):
    """An mt-only provider whose deployment also drops max_tokens. Emitting both
    leaves UnsupportedParamsError on the table at drop_params=false, and "collapse
    the pair" is the load-bearing guarantee."""
    out = await run_hook(hook, model="xai/grok-4",
                         max_tokens=50, max_completion_tokens=64000,
                         additional_drop_params=["max_tokens"])
    assert len(tokens(out)) == 1


# --------------------------------------------------------------------------
# The call_type gate — this is what broke the earlier hook attempt
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call_type",
    ["anthropic_messages", "atext_completion", "aembedding", "aimage_generation",
     None, "unknown_future_entrypoint"],
)
async def test_non_chat_call_types_keep_their_max_tokens(hook, call_type):
    """The dispatch is NOT chat-specific: `wrapper_async` runs it for every
    @client-decorated async entrypoint. `litellm.anthropic_messages` declares
    max_tokens as a REQUIRED parameter, so popping it makes litellm's own
    wrapper raise `TypeError: anthropic_messages() missing 1 required positional
    argument` on the following `await original_function(...)` — outside this
    hook's try/except, with a traceback that never names it. And
    /v1/completions has no max_completion_tokens at all.
    """
    out = await run_hook(hook, call_type=call_type, **AZURE_2025, max_tokens=50)
    assert tokens(out) == {"max_tokens": 50}


@pytest.mark.asyncio
@pytest.mark.parametrize("call_type", ["completion", "acompletion"])
async def test_chat_call_types_are_resolved(hook, call_type):
    out = await run_hook(hook, call_type=call_type, **AZURE_2025, max_tokens=50)
    assert tokens(out) == {"max_completion_tokens": 50}


@pytest.mark.asyncio
async def test_a_call_types_enum_member_is_accepted(hook):
    """`wrapper_async` passes a CallTypes member, not a bare string."""
    from litellm.types.utils import CallTypes

    out = await run_hook(hook, call_type=CallTypes.acompletion, **AZURE_2025,
                         max_tokens=50)
    assert tokens(out) == {"max_completion_tokens": 50}


# --------------------------------------------------------------------------
# Value handling
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value, expected",
                         [(50, 50), (50.0, 50), (49.6, 50), (0.4, 1),
                          (0, None), (-1, None), ("50", None), (True, None),
                          (False, None), (None, None),
                          (float("nan"), None), (float("inf"), None)])
def test_usable_int(value, expected):
    assert _usable_int(value) == expected


@pytest.mark.asyncio
async def test_a_float_ceiling_is_coerced_not_discarded(hook):
    """A float max_tokens really does reach litellm —
    `AnthropicConfig.map_openai_params` coerces one with
    `max(1, int(round(value)))`. Treating it as unusable would silently unbound
    the request."""
    out = await run_hook(hook, **AZURE_2025, max_tokens=50.0)
    assert tokens(out) == {"max_completion_tokens": 50}


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [0, -1, "50", True, False])
async def test_an_unusable_value_is_left_exactly_as_it_arrived(hook, bad):
    """Garbage in must stay error out. Rewriting or stripping it would turn a
    request the provider would reject into an unbounded one."""
    out = await run_hook(hook, **AZURE_2025, max_tokens=bad)
    assert out["max_tokens"] == bad or out["max_tokens"] is bad
    assert "max_completion_tokens" not in out


@pytest.mark.asyncio
async def test_no_token_fields_is_a_no_op(hook):
    out = await run_hook(hook, **AZURE_2025, temperature=0.5)
    assert tokens(out) == {}
    assert out["temperature"] == 0.5


# --------------------------------------------------------------------------
# An operator's drop list outranks the resolution
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_dropped_field_is_never_used_as_a_target(hook):
    """An operator who dropped a param meant it; moving a value onto that field
    would defeat the drop. h2ogpt's own `_drop()` writes that list."""
    out = await run_hook(hook, **AZURE_2025, max_tokens=50,
                         additional_drop_params=["max_completion_tokens"])
    assert tokens(out) == {"max_tokens": 50}


@pytest.mark.asyncio
async def test_the_directive_cannot_resurrect_a_dropped_field(hook):
    out = await run_hook(hook, **AZURE_2025, max_tokens=50,
                         additional_drop_params=["max_completion_tokens"],
                         **{DIRECTIVE_PARAM: True})
    assert tokens(out) == {"max_tokens": 50}


@pytest.mark.asyncio
async def test_both_fields_collapse_onto_the_one_not_dropped(hook):
    out = await run_hook(hook, **AZURE_2025,
                         max_tokens=50, max_completion_tokens=64000,
                         additional_drop_params=["max_tokens"])
    assert tokens(out) == {"max_completion_tokens": 50}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    ["hosted_vllm/my-model", "openai/gpt-4o-mini",
     "anthropic/claude-sonnet-4-5-20250929"],
)
async def test_a_dropped_caller_field_moves_rather_than_vanishes(hook, model):
    """h2oai/litellm#25's original signal, which this must not lose.

    A deployment carrying `additional_drop_params: ["max_tokens"]` accepts only
    the other field. Leaving the request alone means the drop destroys the
    caller's limit — the exact defect. Applies on ANY provider, not just Azure:
    an operator can put that drop on a vllm or openai-compatible deployment, and
    h2ogpt's own `_drop()` writes into the same list.
    """
    out = await run_hook(hook, model=model, max_tokens=50,
                         additional_drop_params=["max_tokens"])
    assert tokens(out) == {"max_completion_tokens": 50}


@pytest.mark.asyncio
async def test_a_dropped_completion_tokens_field_moves_the_other_way(hook):
    out = await run_hook(hook, model="hosted_vllm/my-model",
                         max_completion_tokens=50,
                         additional_drop_params=["max_completion_tokens"])
    assert tokens(out) == {"max_tokens": 50}


@pytest.mark.asyncio
async def test_nothing_moves_when_every_field_is_dropped(hook):
    """There is nowhere to put it, and inventing a destination would defeat the
    drop."""
    out = await run_hook(hook, model="hosted_vllm/my-model", max_tokens=50,
                         additional_drop_params=["max_tokens",
                                                 "max_completion_tokens"])
    assert tokens(out) == {"max_tokens": 50}


@pytest.mark.asyncio
async def test_a_directive_pointing_at_a_dropped_field_still_saves_the_limit(hook):
    """Contradictory config must not silently unbound the request.

    `use_max_completion_tokens: false` says "send max_tokens";
    `additional_drop_params: ["max_tokens"]` says "strip max_tokens". Honouring
    the directive literally leaves the value on the field litellm is about to
    strip, so the caller's ceiling vanishes with no error. The value moves to the
    field that survives instead — which still respects the drop, since the
    dropped field is never sent.
    """
    out = await run_hook(hook, model="hosted_vllm/m", max_tokens=50,
                         additional_drop_params=["max_tokens"],
                         **{DIRECTIVE_PARAM: False})
    assert tokens(out) == {"max_completion_tokens": 50}


@pytest.mark.asyncio
async def test_a_reasoning_preference_that_is_dropped_keeps_the_callers_field(hook):
    """The mirror case: the o-series preference is max_completion_tokens, but the
    operator dropped it. The caller's own field is eligible, so it stays — no
    resurrection of the dropped field."""
    out = await run_hook(hook, model="openai/o3", max_tokens=50,
                         additional_drop_params=["max_completion_tokens"])
    assert tokens(out) == {"max_tokens": 50}


@pytest.mark.asyncio
async def test_nothing_moves_when_there_is_nowhere_valid_to_move_it(hook):
    """`azure_text` has no max_completion_tokens at all, so a dropped
    `max_tokens` has no valid destination. Leave the request as it arrived rather
    than inventing a field the route does not have."""
    out = await run_hook(hook, model="azure_text/gpt-35-turbo-instruct",
                         api_version="2025-04-01-preview", max_tokens=50,
                         additional_drop_params=["max_tokens"])
    assert tokens(out) == {"max_tokens": 50}


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_drop_list",
                         ["max_tokens", {"max_tokens": True}, 5, None])
async def test_a_malformed_drop_list_does_not_raise(hook, bad_drop_list):
    """litellm's own `_should_drop_param` also requires a list, so a non-list
    drop list is a no-op there too — this must agree rather than crash."""
    out = await run_hook(hook, **AZURE_2025, max_tokens=50,
                         additional_drop_params=bad_drop_list)
    assert tokens(out) == {"max_completion_tokens": 50}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    ["hosted_vllm/Qwen/Qwen3-Next-80B-A3B", "hosted_vllm/moonshotai/Kimi-K2.5",
     "hosted_vllm/deepseek-ai/DeepSeek-R1", "mistral/pixtral-large-2502",
     "bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0",
     "hosted_vllm/openai/gpt-oss-120b",
     "hosted_vllm/meta-llama/Llama-3.3-70B-Instruct"],
)
async def test_no_false_positive_reasoning_detection(hook, model):
    """Reasoning detection is delegated to litellm's substring-based
    `is_o_series_model` / `is_model_gpt_5_model`, which could in principle fire on
    an unrelated name. Pin every model route we actually serve."""
    out = await run_hook(hook, model=model, max_tokens=50)
    assert tokens(out) == {"max_tokens": 50}, f"{model} misread as reasoning"


@pytest.mark.asyncio
async def test_the_callers_kwargs_dict_is_never_mutated(hook):
    """`wrapper_async` keeps using its own kwargs when the hook returns None, so
    mutating in place would be an invisible side effect."""
    original = dict(AZURE_2025, max_tokens=50)
    snapshot = dict(original)
    await hook.async_pre_call_deployment_hook(original, "acompletion")
    assert original == snapshot


@pytest.mark.asyncio
async def test_a_request_with_no_model_does_not_raise(hook):
    out = await run_hook(hook, max_tokens=50, max_completion_tokens=64000)
    assert tokens(out) == {"max_tokens": 50}


# --------------------------------------------------------------------------
# End to end through acompletion, per provider
# --------------------------------------------------------------------------


PROVIDER_ROUTES = [
    ("azure/gpt-4o-mini", "2025-04-01-preview", "max_completion_tokens"),
    ("openai/gpt-4o-mini", None, "max_tokens"),
    ("openai/o3", None, "max_completion_tokens"),
    ("anthropic/claude-sonnet-4-5-20250929", None, "max_tokens"),
    ("bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0", None, "maxTokens"),
    ("gemini/gemini-2.5-pro", None, "max_output_tokens"),
    ("hosted_vllm/Qwen/Qwen3-Next-80B-A3B", None, "max_tokens"),
    ("openrouter/openai/gpt-oss-120b", None, "max_tokens"),
    ("mistral/pixtral-large-2502", None, "max_tokens"),
]


@pytest.fixture
def mapped_params(monkeypatch):
    """What get_optional_params returned for each request — i.e. what the
    provider is actually sent."""
    import litellm.main

    captured = []
    original = litellm.main.get_optional_params

    def spy(**kwargs):
        result = original(**kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(litellm.main, "get_optional_params", spy)
    return captured


@pytest.fixture
def registered(monkeypatch, hook):
    monkeypatch.setattr(litellm, "callbacks", [hook])
    return hook


@pytest.mark.asyncio
@pytest.mark.parametrize("model, api_version, expected", PROVIDER_ROUTES)
async def test_end_to_end_the_callers_limit_reaches_each_provider(
    registered, mapped_params, model, api_version, expected
):
    """The whole path, per provider: the hook's collapse must survive into param
    mapping, which is what makes it fix the last-wins ordering rather than only
    renaming a field."""
    # The proxy runs with `drop_params: true`; mirror it so a provider that
    # does not accept a sampling param drops it instead of raising.
    call = {"api_key": "fake-key", "mock_response": "ok",
            "drop_params": True}
    if api_version:
        call["api_version"] = api_version
        call["api_base"] = "https://example.openai.azure.com"
    await litellm.acompletion(
        model=model, messages=[{"role": "user", "content": "hi"}],
        max_tokens=50, max_completion_tokens=64000, **call)
    assert tokens(mapped_params[-1]) == {expected: 50}


@pytest.mark.asyncio
async def test_end_to_end_v1_messages_still_works(registered):
    """`/v1/messages` is what the earlier hook attempt broke. It must return
    normally with the hook registered."""
    result = await litellm.anthropic_messages(
        model="anthropic/claude-sonnet-4-5-20250929",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=50, api_key="fake-key", mock_response="ok")
    assert result is not None


@pytest.mark.asyncio
async def test_end_to_end_legacy_text_completions_still_works(
    registered, mapped_params
):
    """/v1/completions has no max_completion_tokens; it must be left alone."""
    await litellm.atext_completion(
        model="azure/gpt-35-turbo-instruct", prompt="hi", max_tokens=50,
        api_key="fake-key", api_base="https://example.openai.azure.com",
        api_version="2025-04-01-preview", mock_response="ok")
    assert tokens(mapped_params[-1]) == {"max_tokens": 50}


@pytest.mark.asyncio
async def test_end_to_end_mixed_model_group_is_judged_per_member(
    registered, mapped_params
):
    """Our generated config really has one: `agent_auto` spans azure, bedrock and
    anthropic under a single model_name, and a group-level decision would have to
    be wrong for at least one member."""
    from litellm import Router

    router = Router(model_list=[
        {"model_name": "agent_auto", "model_info": {"id": "az"},
         "litellm_params": {"model": "azure/gpt-4o-mini", "api_key": "k",
                            "api_base": "https://example.openai.azure.com",
                            "api_version": "2025-04-01-preview",
                            "max_completion_tokens": 16384}},
        {"model_name": "agent_auto", "model_info": {"id": "az-forced"},
         "litellm_params": {"model": "azure/gpt-4o-mini", "api_key": "k",
                            "api_base": "https://example.openai.azure.com",
                            "api_version": "2025-04-01-preview",
                            DIRECTIVE_PARAM: False}},
        {"model_name": "agent_auto", "model_info": {"id": "bed"},
         "litellm_params": {
             "model": "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
             "max_tokens": 64000}},
    ])
    for deployment_id, expected in [("az", {"max_completion_tokens": 50}),
                                    ("az-forced", {"max_tokens": 50}),
                                    ("bed", {"maxTokens": 50})]:
        await router.acompletion(
            model=deployment_id, messages=[{"role": "user", "content": "hi"}],
            max_tokens=50, mock_response="ok")
        assert tokens(mapped_params[-1]) == expected, deployment_id


@pytest.mark.asyncio
@pytest.mark.parametrize("model, api_version, expected", PROVIDER_ROUTES)
@pytest.mark.parametrize(
    "set_name, extra",
    [
        ("tools", {"tools": [{"type": "function", "function": {
            "name": "w", "description": "w",
            "parameters": {"type": "object", "properties": {}}}}]}),
        ("tools_parallel", {"tools": [{"type": "function", "function": {
            "name": "w", "description": "w",
            "parameters": {"type": "object", "properties": {}}}}],
            "parallel_tool_calls": True}),
        ("sampling", {"temperature": 0.2, "top_p": 0.9,
                      "frequency_penalty": 0.5, "presence_penalty": 0.5}),
        ("seed_stop_user", {"seed": 7, "stop": ["END"], "user": "t"}),
        ("response_format", {"response_format": {"type": "json_object"}}),
    ],
)
async def test_end_to_end_other_params_are_not_collateral_damage(
    registered, mapped_params, model, api_version, expected, set_name, extra
):
    """Adding an output-token limit must leave every OTHER mapped param
    byte-identical — the tripwire for the resolution perturbing function
    calling, structured output or sampling."""
    # The proxy runs with `drop_params: true`; mirror it so a provider that
    # does not accept a sampling param drops it instead of raising.
    call = {"api_key": "fake-key", "mock_response": "ok",
            "drop_params": True}
    if api_version:
        call["api_version"] = api_version
        call["api_base"] = "https://example.openai.azure.com"
    messages = [{"role": "user", "content": "hi"}]

    await litellm.acompletion(model=model, messages=messages, **call, **extra)
    without = {k: repr(v) for k, v in mapped_params[-1].items()
               if k not in TOKEN_FIELDS}

    await litellm.acompletion(model=model, messages=messages, max_tokens=50,
                              **call, **extra)
    with_limit = {k: repr(v) for k, v in mapped_params[-1].items()
                  if k not in TOKEN_FIELDS}

    assert without == with_limit, f"{model} [{set_name}]"


# --------------------------------------------------------------------------
# Interaction with the cap hook
# --------------------------------------------------------------------------


def _run_the_real_cap_hook(params, cap):
    """Drive the ACTUAL MaxTokensCapHook, not a local reimplementation of it.

    The previous version of this section reimplemented the clipping inline and
    asserted only that clip-then-resolve == resolve-then-clip. That symmetry
    assertion was vacuous: mutation-tested, it passed unchanged with `min`
    replaced by `max` AND with `_target_field` stubbed to None. A symmetry-only
    check cannot catch a symmetric bug — which is exactly how the float-bypass
    below got through. So: real hook, absolute values.
    """
    from litellm.integrations.h2o.litellm_max_tokens_cap_hook import (
        MaxTokensCapHook,
    )

    cap_hook = MaxTokensCapHook()
    modified = []
    cap_hook._cap_in(params, cap, "data", modified)
    return params


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_params, cap, expected",
    [
        # client tighter than the cap -> the client wins
        ({"max_tokens": 50, "max_completion_tokens": 64000}, 16384,
         {"max_completion_tokens": 50}),
        # client looser than the cap -> the cap wins
        ({"max_tokens": 99999, "max_completion_tokens": 64000}, 16384,
         {"max_completion_tokens": 16384}),
        ({"max_tokens": 99999}, 8192, {"max_completion_tokens": 8192}),
        ({"max_tokens": 5, "max_completion_tokens": 7}, 16384,
         {"max_completion_tokens": 5}),
        # A FLOAT over the cap must still be capped. This is the regression the
        # symmetry-only test could not see: the cap hook clipped `isinstance(v,
        # int)` only, so a float bypassed it, and once this hook began coercing
        # floats for every provider that turned a provider-rejected request into
        # an accepted OVER-CAP one.
        ({"max_tokens": 99999.0}, 8192, {"max_completion_tokens": 8192}),
        ({"max_tokens": 99999.0, "max_completion_tokens": 64000}, 8192,
         {"max_completion_tokens": 8192}),
    ],
)
async def test_cap_hook_then_resolution_lands_on_the_absolute_value(
    hook, request_params, cap, expected
):
    """The real cap hook runs first (it is an async_pre_call_hook, pre-routing),
    then this resolution. Assert the VALUE, not just symmetry."""
    params = _run_the_real_cap_hook(dict(request_params, **AZURE_2025), cap)
    out = await run_hook(hook, **params)
    assert tokens(out) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_params, cap",
    [
        ({"max_tokens": 50, "max_completion_tokens": 64000}, 16384),
        ({"max_tokens": 99999, "max_completion_tokens": 64000}, 16384),
        ({"max_tokens": 99999}, 8192),
        ({"max_tokens": 99999.0}, 8192),
    ],
)
async def test_cap_hook_and_resolution_are_order_independent(
    hook, request_params, cap
):
    """Which one a deployment sees first depends on callback registration, so both
    orders must land on the same value. Kept as a SUPPLEMENT to the absolute
    assertions above, never as the only check."""
    clip_first = _run_the_real_cap_hook(dict(request_params, **AZURE_2025), cap)
    clip_first = await run_hook(hook, **clip_first)

    resolve_first = await run_hook(hook, **dict(request_params, **AZURE_2025))
    resolve_first = _run_the_real_cap_hook(dict(resolve_first), cap)

    assert tokens(clip_first) == tokens(resolve_first)


# --------------------------------------------------------------------------
# Never propagates
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_internal_error_leaves_the_token_fields_untouched(hook, monkeypatch):
    """Failing a request because a field could not be resolved would be worse
    than the defect."""
    monkeypatch.setattr(
        "litellm.integrations.h2o.litellm_max_tokens_resolution_hook."
        "_usable_int",
        lambda value: (_ for _ in ()).throw(RuntimeError("boom")))
    out = await run_hook(hook, **AZURE_2025, max_tokens=50)
    assert tokens(out) == {"max_tokens": 50}


@pytest.mark.asyncio
async def test_an_internal_error_still_strips_the_directive(hook, monkeypatch):
    """The fail-safe must not be fail-OPEN for the directive.

    "Leave the request as it arrived" is right for the token fields and wrong for
    `use_max_completion_tokens`, which exists only because this hook removes it.
    Returning None on error hands litellm the original kwargs, so the
    unrecognized key reaches the provider body and is rejected — measured against
    an unpatched proxy as `{"max_tokens": 50, "use_max_completion_tokens": false}`.
    An internal bug must not become a 400 on every request to that deployment.
    """
    monkeypatch.setattr(
        "litellm.integrations.h2o.litellm_max_tokens_resolution_hook."
        "_usable_int",
        lambda value: (_ for _ in ()).throw(RuntimeError("boom")))
    out = await run_hook(hook, **AZURE_2025, max_tokens=50,
                         **{DIRECTIVE_PARAM: False})
    assert DIRECTIVE_PARAM not in out
    assert tokens(out) == {"max_tokens": 50}


@pytest.mark.asyncio
async def test_an_error_before_the_call_type_gate_still_strips_the_directive(
    hook, monkeypatch
):
    """Same guarantee when the throw happens on a non-chat call type."""
    monkeypatch.setattr(
        "litellm.integrations.h2o.litellm_max_tokens_resolution_hook."
        "CHAT_CALL_TYPES",
        property(lambda self: (_ for _ in ()).throw(RuntimeError("boom"))))
    out = await run_hook(hook, call_type="anthropic_messages", **AZURE_2025,
                         max_tokens=50, **{DIRECTIVE_PARAM: True})
    assert DIRECTIVE_PARAM not in out
