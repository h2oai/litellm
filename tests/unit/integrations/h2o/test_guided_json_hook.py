"""Tests for the h2o GuidedJsonHook structured-output translation.

The hook converts vLLM-style ``guided_json`` (passed through ``extra_body``) into
the best structured-output mechanism the target provider supports:

- strict/native ``json_schema`` when ``supports_response_schema`` is True,
- ``json_object`` + schema-in-prompt when only json mode is supported,
- prompt-only schema injection otherwise,

and always removes the raw ``guided_json`` so it cannot error downstream.
"""

import json

import pytest

import litellm
from litellm.integrations.h2o.litellm_guided_json_hook import GuidedJsonHook


# A classifier-shaped schema with an OPTIONAL key (rationale) — strict mode must
# therefore be disabled for it (OpenAI strict requires every prop in `required`).
CLASSIFIER_SCHEMA = {
    "type": "object",
    "properties": {
        "classification": {"type": "string", "enum": ["a", "b", "c"]},
        "rationale": {"type": "string"},
    },
    "required": ["classification"],
    "additionalProperties": False,
}

# A fully-required schema — strict mode IS safe here.
STRICT_SAFE_SCHEMA = {
    "type": "object",
    "properties": {"x": {"type": "string"}},
    "required": ["x"],
    "additionalProperties": False,
}


@pytest.fixture
def hook():
    return GuidedJsonHook()


@pytest.fixture
def caps(monkeypatch):
    """Control litellm capability introspection. Returns a setter."""
    state = {"schema": True, "json": True, "provider": "openai"}

    def set_caps(schema=None, json_mode=None, provider=None):
        if schema is not None:
            state["schema"] = schema
        if json_mode is not None:
            state["json"] = json_mode
        if provider is not None:
            state["provider"] = provider

    monkeypatch.setattr(
        litellm, "get_llm_provider",
        lambda model, *a, **k: (model, state["provider"], None, None),
    )
    monkeypatch.setattr(
        litellm, "supports_response_schema",
        lambda model, custom_llm_provider=None: state["schema"],
    )
    monkeypatch.setattr(
        litellm, "get_supported_openai_params",
        lambda model, custom_llm_provider=None: (["response_format"] if state["json"] else []),
    )
    return set_caps


async def _run(hook, data):
    return await hook.async_pre_call_hook(
        user_api_key_dict={}, cache=None, data=data, call_type="completion"
    )


def _last_user_text(data):
    for m in reversed(data["messages"]):
        if m.get("role") == "user":
            c = m["content"]
            return c if isinstance(c, str) else json.dumps(c)
    return ""


# --- Tier 1: strict-capable provider -> json_schema -----------------------

async def test_tier1_json_schema_strict_safe(hook, caps):
    caps(schema=True)
    data = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}],
            "extra_body": {"guided_json": STRICT_SAFE_SCHEMA}}
    out = await _run(hook, data)
    rf = out["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["schema"] == STRICT_SAFE_SCHEMA
    assert rf["json_schema"]["strict"] is True
    assert "guided_json" not in out.get("extra_body", {})


async def test_tier1_strict_disabled_for_optional_keys(hook, caps):
    caps(schema=True)
    data = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}],
            "extra_body": {"guided_json": CLASSIFIER_SCHEMA}}
    out = await _run(hook, data)
    assert out["response_format"]["type"] == "json_schema"
    # rationale is optional -> strict must be False or OpenAI rejects the request
    assert out["response_format"]["json_schema"]["strict"] is False


# --- Tier 2: json mode only -> json_object + prompt schema -----------------

async def test_tier2_json_object_and_prompt(hook, caps):
    caps(schema=False, json_mode=True)
    data = {"model": "some-json-model", "messages": [{"role": "user", "content": "classify this"}],
            "extra_body": {"guided_json": CLASSIFIER_SCHEMA}}
    out = await _run(hook, data)
    assert out["response_format"] == {"type": "json_object"}
    text = _last_user_text(out)
    assert "JSON schema" in text
    assert "classification" in text  # required key named
    assert "guided_json" not in out.get("extra_body", {})


# --- Tier 3: no support -> prompt only -------------------------------------

async def test_tier3_prompt_only(hook, caps):
    caps(schema=False, json_mode=False)
    data = {"model": "no-json-model", "messages": [{"role": "user", "content": "classify this"}],
            "extra_body": {"guided_json": CLASSIFIER_SCHEMA}}
    out = await _run(hook, data)
    assert "response_format" not in out  # provider can't enforce it
    text = _last_user_text(out)
    assert "classification" in text
    assert "required" in text.lower()


# --- guided_json found/removed in each location ----------------------------

@pytest.mark.parametrize("place", ["top", "extra_body", "litellm_params"])
async def test_guided_json_removed_from_all_locations(hook, caps, place):
    caps(schema=True)
    data = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    if place == "top":
        data["guided_json"] = STRICT_SAFE_SCHEMA
    elif place == "extra_body":
        data["extra_body"] = {"guided_json": STRICT_SAFE_SCHEMA}
    else:
        data["litellm_params"] = {"extra_body": {"guided_json": STRICT_SAFE_SCHEMA}}
    out = await _run(hook, data)
    assert out["response_format"]["type"] == "json_schema"
    assert "guided_json" not in out
    assert "guided_json" not in out.get("extra_body", {})
    assert "guided_json" not in out.get("litellm_params", {}).get("extra_body", {})


# --- sibling vLLM-only params are stripped ---------------------------------

async def test_sibling_vllm_params_stripped(hook, caps):
    caps(schema=True)
    data = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}],
            "extra_body": {"guided_json": STRICT_SAFE_SCHEMA, "guided_regex": "x",
                           "stop_token_ids": [1, 2], "guided_decoding_backend": "outlines"}}
    out = await _run(hook, data)
    eb = out.get("extra_body", {})
    for p in ("guided_json", "guided_regex", "stop_token_ids", "guided_decoding_backend"):
        assert p not in eb


# --- no guided_json -> untouched ------------------------------------------

async def test_no_guided_json_is_noop(hook, caps):
    caps(schema=True)
    data = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.5}
    out = await _run(hook, data)
    assert "response_format" not in out
    assert out["messages"] == [{"role": "user", "content": "hi"}]


# --- explicit json_schema response_format respected -----------------------

async def test_explicit_response_format_respected(hook, caps):
    caps(schema=True)
    explicit = {"type": "json_schema", "json_schema": {"name": "mine", "schema": STRICT_SAFE_SCHEMA}}
    data = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}],
            "response_format": explicit, "extra_body": {"guided_json": CLASSIFIER_SCHEMA}}
    out = await _run(hook, data)
    assert out["response_format"] == explicit  # unchanged
    assert "guided_json" not in out.get("extra_body", {})  # but guided_json removed


# --- guided_json as a JSON string is coerced ------------------------------

async def test_guided_json_string_is_coerced(hook, caps):
    caps(schema=True)
    data = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}],
            "extra_body": {"guided_json": json.dumps(STRICT_SAFE_SCHEMA)}}
    out = await _run(hook, data)
    assert out["response_format"]["json_schema"]["schema"] == STRICT_SAFE_SCHEMA
