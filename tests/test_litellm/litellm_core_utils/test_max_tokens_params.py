"""Tests for max_tokens / max_completion_tokens canonicalization.

Two defects motivate this, both reproduced against the pre-change tree:

1. SILENT CEILING OVERRIDE (non-Azure). A deployment configured with a
   `max_completion_tokens` ceiling plus a client sending `max_tokens` produced
   BOTH fields in `non_default_params`, and the provider maps that collapse
   them assign the same output key from two branches, so the one iterated last
   won. `max_completion_tokens` is iterated second, so the ceiling overwrote
   the caller's request:

       anthropic    in={'max_tokens': 50, 'max_completion_tokens': 64000}
                    -> {'max_tokens': 64000}
       bedrock      -> {'maxTokens': 64000}
       openai o3    -> {'max_completion_tokens': 64000}

2. BOTH FIELDS ON THE WIRE (Azure). Providers that forward both unchanged send
   both, and Azure 2025+ rejects that:

       AzureException BadRequestError - Setting 'max_tokens' and
       'max_completion_tokens' at the same time is not supported.

   Which h2ogpt worked around with `additional_drop_params: ["max_tokens"]`,
   trading the 400 for defect 1.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

import litellm
from litellm.litellm_core_utils.max_tokens_params import (
    MAX_COMPLETION_TOKENS_PARAM,
    MAX_TOKENS_PARAM,
    preferred_param_for_directive,
    resolve_max_tokens_params,
)

BOTH_SUPPORTED = [MAX_TOKENS_PARAM, MAX_COMPLETION_TOKENS_PARAM, "temperature"]


def _token_params(optional_params: dict) -> dict:
    """The output-token fields only, including bedrock's renamed form."""
    return {
        k: v
        for k, v in optional_params.items()
        if k in (MAX_TOKENS_PARAM, MAX_COMPLETION_TOKENS_PARAM, "maxTokens")
    }


# --------------------------------------------------------------------------
# resolve_max_tokens_params — the rules in isolation
# --------------------------------------------------------------------------


def test_tighter_value_wins_when_both_present():
    """Rule 1. A client's tighter request is not widened to the ceiling."""
    params = {"max_tokens": 50, "max_completion_tokens": 64000}
    target = resolve_max_tokens_params(params, BOTH_SUPPORTED)
    assert params == {target: 50}


def test_tighter_value_wins_regardless_of_which_field_holds_it():
    """Rule 1 is not an artifact of field order — a tight ceiling still wins."""
    params = {"max_tokens": 64000, "max_completion_tokens": 50}
    target = resolve_max_tokens_params(params, BOTH_SUPPORTED)
    assert params == {target: 50}


def test_both_present_collapses_to_one_field():
    """Rule 2. Leaving both is what 400s Azure and what feeds defect 1."""
    params = {"max_tokens": 50, "max_completion_tokens": 64000}
    resolve_max_tokens_params(params, BOTH_SUPPORTED)
    assert len(set(params) & {MAX_TOKENS_PARAM, MAX_COMPLETION_TOKENS_PARAM}) == 1


def test_preference_renames_a_single_field():
    params = {"max_tokens": 50}
    target = resolve_max_tokens_params(
        params, BOTH_SUPPORTED, preferred_param=MAX_COMPLETION_TOKENS_PARAM
    )
    assert target == MAX_COMPLETION_TOKENS_PARAM
    assert params == {"max_completion_tokens": 50}


def test_preference_renames_in_reverse_too():
    """A pre-2025 Azure deployment needs the opposite direction."""
    params = {"max_completion_tokens": 50}
    target = resolve_max_tokens_params(
        params, BOTH_SUPPORTED, preferred_param=MAX_TOKENS_PARAM
    )
    assert target == MAX_TOKENS_PARAM
    assert params == {"max_tokens": 50}


def test_single_field_with_no_preference_is_untouched():
    """Deployments that natively accept what the caller sent stay untouched."""
    params = {"max_tokens": 50, "temperature": 0.5}
    assert resolve_max_tokens_params(params, BOTH_SUPPORTED) is None
    assert params == {"max_tokens": 50, "temperature": 0.5}


def test_unsupported_preference_is_ignored_not_forced():
    """Forcing a field the provider rejects would strip the ceiling entirely,
    which is worse than the defect being fixed."""
    params = {"max_tokens": 50}
    target = resolve_max_tokens_params(
        params,
        supported_params=[MAX_TOKENS_PARAM],
        preferred_param=MAX_COMPLETION_TOKENS_PARAM,
    )
    assert target is None
    assert params == {"max_tokens": 50}


def test_no_token_fields_is_a_no_op():
    params = {"temperature": 0.5}
    assert resolve_max_tokens_params(params, BOTH_SUPPORTED) is None
    assert params == {"temperature": 0.5}


@pytest.mark.parametrize("bad", [0, -1, None, "50", True, False])
def test_never_invents_a_ceiling_from_an_unusable_value(bad):
    """Rule 3. A max_tokens of 0/None/"50"/True must not become a
    max_completion_tokens that truncates every response."""
    params = {"max_tokens": bad}
    resolve_max_tokens_params(
        params, BOTH_SUPPORTED, preferred_param=MAX_COMPLETION_TOKENS_PARAM
    )
    assert params.get(MAX_COMPLETION_TOKENS_PARAM) is None


@pytest.mark.parametrize("bad", [0, -1, "50", True, False, float("nan"), float("inf")])
def test_an_unusable_value_is_left_exactly_as_it_arrived(bad):
    """Rule 3's other half, and a regression guard.

    An earlier revision stripped an unusable value from a field the provider
    rejects, on the theory that a 0 is not a ceiling worth keeping. That was
    wrong for every non-int shape: a client sending ``max_tokens: "50"`` had its
    ceiling silently deleted and the request went out UNBOUNDED, where before it
    was forwarded and rejected loudly. Garbage in must stay error out.
    """
    params = {"max_tokens": bad}
    assert (
        resolve_max_tokens_params(
            params, BOTH_SUPPORTED, preferred_param=MAX_COMPLETION_TOKENS_PARAM
        )
        is None
    )
    assert params == {"max_tokens": bad} or (
        # NaN != NaN, so compare identity for that one
        isinstance(bad, float) and params["max_tokens"] is bad
    )


@pytest.mark.parametrize(
    "value, expected",
    [(50.0, 50), (49.6, 50), (0.4, 1), (1e3, 1000)],
)
def test_a_float_ceiling_is_coerced_not_discarded(value, expected):
    """A float ``max_tokens`` really does reach litellm —
    ``AnthropicConfig.map_openai_params`` coerces one with
    ``max(1, int(round(value)))``. An earlier revision treated floats as
    unusable and deleted them, silently unbounding the request."""
    params = {"max_tokens": value}
    target = resolve_max_tokens_params(
        params, BOTH_SUPPORTED, preferred_param=MAX_COMPLETION_TOKENS_PARAM
    )
    assert target == MAX_COMPLETION_TOKENS_PARAM
    assert params == {"max_completion_tokens": expected}


# --------------------------------------------------------------------------
# Rule 4 — an operator's drop list outranks the resolution
# --------------------------------------------------------------------------


def test_a_dropped_field_is_never_used_as_a_target():
    """Regression: an earlier revision moved the value onto the preferred field
    even when the operator had listed it in additional_drop_params, defeating
    the drop and putting back a param they had explicitly removed."""
    params = {"max_tokens": 50}
    target = resolve_max_tokens_params(
        params,
        BOTH_SUPPORTED,
        preferred_param=MAX_COMPLETION_TOKENS_PARAM,
        additional_drop_params=["max_completion_tokens"],
    )
    assert target is None
    assert params == {"max_tokens": 50}


def test_both_fields_collapse_onto_the_one_that_is_not_dropped():
    params = {"max_tokens": 50, "max_completion_tokens": 64000}
    target = resolve_max_tokens_params(
        params,
        BOTH_SUPPORTED,
        additional_drop_params=["max_tokens"],
    )
    assert target == MAX_COMPLETION_TOKENS_PARAM
    assert params == {"max_completion_tokens": 50}


def test_nothing_happens_when_both_fields_are_dropped():
    params = {"max_tokens": 50, "max_completion_tokens": 64000}
    assert (
        resolve_max_tokens_params(
            params,
            BOTH_SUPPORTED,
            preferred_param=MAX_COMPLETION_TOKENS_PARAM,
            additional_drop_params=["max_tokens", "max_completion_tokens"],
        )
        is None
    )
    assert params == {"max_tokens": 50, "max_completion_tokens": 64000}


def test_drop_list_is_respected_end_to_end():
    optional_params = litellm.get_optional_params(
        model="gpt-4o-mini",
        custom_llm_provider="azure",
        api_version="2025-04-01-preview",
        drop_params=True,
        max_tokens=50,
        additional_drop_params=["max_completion_tokens"],
    )
    assert _token_params(optional_params) == {"max_tokens": 50}


def test_azure_text_completions_route_is_not_renamed():
    """`azure_text` is the legacy /completions endpoint, which has no
    `max_completion_tokens` at all. It resolves to a different provider config,
    so the Azure api_version rule must not reach it — the same class of hazard
    that made the earlier deployment-hook approach break /v1/messages."""
    optional_params = litellm.get_optional_params(
        model="gpt-35-turbo-instruct",
        custom_llm_provider="azure_text",
        api_version="2025-04-01-preview",
        drop_params=True,
        max_tokens=50,
    )
    assert _token_params(optional_params) == {"max_tokens": 50}


@pytest.mark.parametrize("api_version", [2025, b"2025-04-01", ["2025"], {}, 3.5])
def test_a_non_string_api_version_yields_no_preference_rather_than_raising(api_version):
    """This lookup runs on every Azure chat request, so a misconfigured
    api_version must not turn param mapping into a traceback. An earlier
    revision raised `unhashable type` on a list/dict api_version."""
    assert (
        litellm.AzureOpenAIConfig().get_preferred_max_tokens_param(
            model="gpt-4o-mini", api_version=api_version
        )
        is None
    )


def test_unusable_value_alongside_a_usable_one_is_discarded():
    params = {"max_tokens": 0, "max_completion_tokens": 64000}
    resolve_max_tokens_params(
        params, BOTH_SUPPORTED, preferred_param=MAX_COMPLETION_TOKENS_PARAM
    )
    assert params == {"max_completion_tokens": 64000}


@pytest.mark.parametrize(
    "directive, expected",
    [(True, MAX_COMPLETION_TOKENS_PARAM), (False, MAX_TOKENS_PARAM), (None, None)],
)
def test_preferred_param_for_directive(directive, expected):
    assert preferred_param_for_directive(directive) == expected


@pytest.mark.parametrize("stray", ["false", "true", "", 0, 1, [], "no"])
def test_only_the_exact_booleans_count_as_a_directive(stray):
    """A stray value — a quoted `"false"` out of a YAML config, say — must read
    as "not given" rather than being coerced by truthiness into the opposite of
    what it looks like. `bool("false")` is True, which would send
    `max_completion_tokens` to a deployment whose operator wrote `false`."""
    assert preferred_param_for_directive(stray) is None


# --------------------------------------------------------------------------
# get_optional_params — defect 1, per provider
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider, model",
    [
        ("anthropic", "claude-sonnet-4-5-20250929"),
        ("bedrock", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"),
        ("hosted_vllm", "my-vllm-model"),
        ("openai", "gpt-4o"),
        ("openai", "o3"),
        ("openai", "gpt-5"),
    ],
)
def test_client_limit_survives_a_configured_ceiling(provider, model):
    """Regression for defect 1: pre-change this returned 64000 for anthropic,
    bedrock, o3 and gpt-5, i.e. the caller's 50 was silently discarded."""
    optional_params = litellm.get_optional_params(
        model=model,
        custom_llm_provider=provider,
        drop_params=True,
        max_tokens=50,
        max_completion_tokens=64000,
    )
    assert list(_token_params(optional_params).values()) == [50]


@pytest.mark.parametrize(
    "provider, model",
    [
        ("anthropic", "claude-sonnet-4-5-20250929"),
        ("hosted_vllm", "my-vllm-model"),
        ("openai", "gpt-4o"),
    ],
)
def test_only_one_token_field_reaches_the_provider(provider, model):
    optional_params = litellm.get_optional_params(
        model=model,
        custom_llm_provider=provider,
        drop_params=True,
        max_tokens=50,
        max_completion_tokens=64000,
    )
    assert len(_token_params(optional_params)) == 1


# --------------------------------------------------------------------------
# get_optional_params — defect 2, Azure by api_version
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "api_version",
    ["2025-04-01-preview", "2025-01-01-preview", "2026-02-01", "preview", "latest", "v1"],
)
def test_azure_2025_renames_client_max_tokens(api_version):
    """The measured defect: Azure 2025+ rejects max_tokens, and dropping it
    (the previous workaround) applied the deployment ceiling instead of the
    caller's 50."""
    optional_params = litellm.get_optional_params(
        model="gpt-4o-mini",
        custom_llm_provider="azure",
        api_version=api_version,
        drop_params=True,
        max_tokens=50,
    )
    assert _token_params(optional_params) == {"max_completion_tokens": 50}


def test_azure_2025_collapses_both_to_the_tighter_value():
    """Exactly the shipped h2ogpt config shape: a max_completion_tokens
    ceiling on the deployment plus a client max_tokens."""
    optional_params = litellm.get_optional_params(
        model="gpt-4o-mini",
        custom_llm_provider="azure",
        api_version="2025-04-01-preview",
        drop_params=True,
        max_tokens=50,
        max_completion_tokens=16384,
    )
    assert _token_params(optional_params) == {"max_completion_tokens": 50}


@pytest.mark.parametrize("api_version", ["2024-08-01-preview", "2023-12-01-preview"])
def test_pre_2025_azure_gets_max_tokens(api_version):
    """Older api_versions predate max_completion_tokens, so the rename has to
    run in the other direction."""
    optional_params = litellm.get_optional_params(
        model="gpt-4o-mini",
        custom_llm_provider="azure",
        api_version=api_version,
        drop_params=True,
        max_completion_tokens=50,
    )
    assert _token_params(optional_params) == {"max_tokens": 50}


def test_azure_with_no_api_version_uses_the_litellm_default():
    """litellm.AZURE_DEFAULT_API_VERSION is a 2025 version, so the default
    path must land on max_completion_tokens rather than 400."""
    assert litellm.AZURE_DEFAULT_API_VERSION.startswith("2025")
    optional_params = litellm.get_optional_params(
        model="gpt-4o-mini",
        custom_llm_provider="azure",
        drop_params=True,
        max_tokens=50,
    )
    assert _token_params(optional_params) == {"max_completion_tokens": 50}


def test_unparseable_api_version_leaves_the_callers_field_alone():
    optional_params = litellm.get_optional_params(
        model="gpt-4o-mini",
        custom_llm_provider="azure",
        api_version="not-a-version",
        drop_params=True,
        max_tokens=50,
    )
    assert _token_params(optional_params) == {"max_tokens": 50}


# --------------------------------------------------------------------------
# use_max_completion_tokens — the authoritative override
# --------------------------------------------------------------------------


def test_directive_true_forces_max_completion_tokens_off_azure():
    """A non-Azure deployment that requires the newer field can say so,
    without the provider needing to be recognized."""
    optional_params = litellm.get_optional_params(
        model="my-openai-compatible-model",
        custom_llm_provider="hosted_vllm",
        drop_params=True,
        max_tokens=50,
        use_max_completion_tokens=True,
    )
    assert _token_params(optional_params) == {"max_completion_tokens": 50}


def test_directive_false_overrides_the_azure_api_version_detection():
    """An operator whose 2025-versioned deployment still wants max_tokens
    gets max_tokens. The flag is the control, detection is the fallback."""
    optional_params = litellm.get_optional_params(
        model="gpt-4o-mini",
        custom_llm_provider="azure",
        api_version="2025-04-01-preview",
        drop_params=True,
        max_tokens=50,
        use_max_completion_tokens=False,
    )
    assert _token_params(optional_params) == {"max_tokens": 50}


def test_directive_true_overrides_pre_2025_azure_detection():
    optional_params = litellm.get_optional_params(
        model="gpt-4o-mini",
        custom_llm_provider="azure",
        api_version="2024-08-01-preview",
        drop_params=True,
        max_tokens=50,
        use_max_completion_tokens=True,
    )
    assert _token_params(optional_params) == {"max_completion_tokens": 50}


def test_directive_false_collapses_both_onto_max_tokens():
    optional_params = litellm.get_optional_params(
        model="gpt-4o-mini",
        custom_llm_provider="azure",
        api_version="2025-04-01-preview",
        drop_params=True,
        max_tokens=50,
        max_completion_tokens=16384,
        use_max_completion_tokens=False,
    )
    assert _token_params(optional_params) == {"max_tokens": 50}


def test_directive_is_never_sent_to_the_provider():
    """It selects a field; it is not itself a param any provider accepts."""
    optional_params = litellm.get_optional_params(
        model="gpt-4o-mini",
        custom_llm_provider="azure",
        api_version="2025-04-01-preview",
        drop_params=True,
        max_tokens=50,
        use_max_completion_tokens=True,
    )
    assert "use_max_completion_tokens" not in optional_params
    assert "use_max_completion_tokens" not in (
        optional_params.get("extra_body") or {}
    )


def test_directive_is_a_recognized_litellm_param():
    """So it is stripped from the provider-bound params rather than forwarded
    as a model-specific extra."""
    from litellm.types.utils import all_litellm_params

    assert "use_max_completion_tokens" in all_litellm_params


def test_directive_survives_deployment_config_validation():
    """It has to reach get_optional_params from a model_list entry."""
    from litellm.types.router import LiteLLM_Params

    params = LiteLLM_Params(
        model="azure/gpt-4o-mini", use_max_completion_tokens=False
    )
    assert params.use_max_completion_tokens is False


# --------------------------------------------------------------------------
# provider preference declarations
# --------------------------------------------------------------------------


def test_base_config_declares_no_preference_by_default():
    """Providers that accept either field must not be disturbed."""
    assert (
        litellm.AnthropicConfig().get_preferred_max_tokens_param(
            model="claude-sonnet-4-5-20250929"
        )
        is None
    )


@pytest.mark.parametrize(
    "api_version, expected",
    [
        ("2025-04-01-preview", "max_completion_tokens"),
        ("2025-01-01-preview", "max_completion_tokens"),
        ("preview", "max_completion_tokens"),
        ("2024-08-01-preview", "max_tokens"),
        ("2023-05-15", "max_tokens"),
        ("garbage", None),
        (None, None),
    ],
)
def test_azure_preference_by_api_version(api_version, expected):
    assert (
        litellm.AzureOpenAIConfig().get_preferred_max_tokens_param(
            model="gpt-4o-mini", api_version=api_version
        )
        == expected
    )


def test_azure_gpt5_prefers_completion_tokens_regardless_of_api_version():
    """MRO would otherwise reach AzureOpenAIConfig's api_version rule first."""
    assert (
        litellm.AzureOpenAIGPT5Config().get_preferred_max_tokens_param(
            model="gpt-5", api_version="2024-08-01-preview"
        )
        == "max_completion_tokens"
    )


def test_o_series_prefers_completion_tokens():
    assert (
        litellm.OpenAIOSeriesConfig().get_preferred_max_tokens_param(model="o3")
        == "max_completion_tokens"
    )


# --------------------------------------------------------------------------
# Router plumbing — a deployment's litellm_params must reach the resolution
# --------------------------------------------------------------------------


@pytest.fixture
def mapped_params(monkeypatch):
    """Capture what get_optional_params returns for each request.

    The logging callbacks' `optional_params` is captured before provider
    mapping runs, so it cannot be used to assert on the mapped result.
    """
    import litellm.main

    captured = []
    original = litellm.main.get_optional_params

    def spy(**kwargs):
        result = original(**kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(litellm.main, "get_optional_params", spy)
    return captured


def _deployment(**litellm_params):
    base = {
        "model": "azure/gpt-4o-mini",
        "api_key": "fake-key",
        "api_base": "https://example.openai.azure.com",
        "api_version": "2025-04-01-preview",
    }
    base.update(litellm_params)
    return {"model_name": "target", "litellm_params": base}


@pytest.mark.asyncio
async def test_router_azure_deployment_honors_client_max_tokens(mapped_params):
    """The shipped config shape. Pre-change this reached Azure as
    max_tokens=50 AND max_completion_tokens=16384 (a 400), which h2ogpt then
    worked around by dropping max_tokens — applying 16384 instead of 50."""
    from litellm import Router

    router = Router(model_list=[_deployment(max_completion_tokens=16384)])
    await router.acompletion(
        model="target",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=50,
        mock_response="ok",
    )
    assert _token_params(mapped_params[-1]) == {"max_completion_tokens": 50}


@pytest.mark.asyncio
async def test_router_deployment_directive_overrides_api_version(mapped_params):
    """`use_max_completion_tokens: false` on the deployment is authoritative
    even though the api_version says otherwise."""
    from litellm import Router

    router = Router(
        model_list=[_deployment(use_max_completion_tokens=False, max_tokens=16384)]
    )
    await router.acompletion(
        model="target",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=50,
        mock_response="ok",
    )
    assert _token_params(mapped_params[-1]) == {"max_tokens": 50}


@pytest.mark.asyncio
async def test_router_mixed_model_group_is_judged_per_deployment(mapped_params):
    """Our generated config has one: `agent_auto` spans azure, bedrock,
    anthropic and others under a single model_name. A group-level decision
    would send max_completion_tokens to the members that need max_tokens."""
    from litellm import Router

    router = Router(
        model_list=[
            {
                "model_name": "mixed",
                "litellm_params": {
                    "model": "azure/gpt-4o-mini",
                    "api_key": "fake-key",
                    "api_base": "https://example.openai.azure.com",
                    "api_version": "2025-04-01-preview",
                    "max_completion_tokens": 16384,
                },
                "model_info": {"id": "azure-member"},
            },
            {
                "model_name": "mixed",
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-4-5-20250929",
                    "api_key": "fake-key",
                    "max_tokens": 64000,
                },
                "model_info": {"id": "anthropic-member"},
            },
        ]
    )
    for deployment_id, expected in [
        ("azure-member", {"max_completion_tokens": 50}),
        ("anthropic-member", {"max_tokens": 50}),
    ]:
        # Address the member by its model_info id so the assertion is about
        # that deployment rather than whichever one the strategy shuffles to.
        await router.acompletion(
            model=deployment_id,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=50,
            mock_response="ok",
        )
        assert _token_params(mapped_params[-1]) == expected, deployment_id
