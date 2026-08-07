"""Canonicalize the two OpenAI output-token fields onto a single field.

WHY THIS EXISTS
---------------
The OpenAI chat schema carries two output-token fields: the original
``max_tokens`` and its replacement ``max_completion_tokens``. A request can
arrive with both — most commonly because a proxy deployment configures one of
them as a ceiling in ``litellm_params`` while the client sends the other. When
that happens, what reaches the provider depends on the order the provider's
``map_openai_params`` happens to iterate, and the result is wrong in two
different ways:

  * Providers that collapse both onto one field do so with LAST-WINS
    semantics, because both branches assign the same output key. ``anthropic``
    (``max_tokens``) and ``bedrock`` converse (``maxTokens``) both do this, and
    ``max_completion_tokens`` is iterated second, so a client asking for 50
    output tokens against a deployment configured with a 64000 ceiling gets
    64000 — the caller's limit is silently discarded. ``openai_like`` is worse:
    it replaces unconditionally, so the ceiling always wins.

  * Providers that forward both fields unchanged send both to the API.
    Azure 2025+ API versions reject that outright:

        AzureException BadRequestError - Setting 'max_tokens' and
        'max_completion_tokens' at the same time is not supported.

Neither failure mode is provider-specific in nature, so neither belongs in a
provider transform. This module resolves both fields to one, once, before the
provider mapping runs.

THE RULES
---------
1. TIGHTER WINS. When more than one usable value is present, the smallest is
   kept. A deployment ceiling can therefore never be RAISED by a client, and a
   client's tighter request can never be widened to the ceiling.

2. ONE FIELD OUT. When a target field is known, only that field survives. The
   target comes from the provider config
   (``BaseConfig.get_preferred_max_tokens_param``) or from an explicit
   per-request/per-deployment ``use_max_completion_tokens`` directive, which
   overrides the provider's own detection.

3. NEVER SILENTLY DROP A CEILING. If no usable value can be derived — a
   ``max_tokens`` of ``0``, ``"50"``, ``True``, ``None`` — NOTHING is changed.
   The request goes on exactly as it would have without this module, and the
   provider rejects it as loudly as it always did. Rewriting it here would
   turn a garbage-in/error-out request into an unbounded one, which is a worse
   failure than the one being fixed. Numeric values ARE usable and are coerced
   to a positive int, matching what ``AnthropicConfig.map_openai_params``
   already does with a float ``max_tokens``.

4. NEVER RESURRECT A DROPPED PARAM. A field listed in
   ``additional_drop_params`` is off limits as a target. An operator who
   dropped a param meant it, and moving a value onto that field would defeat
   the drop.

OUT OF SCOPE: ``extra_body``
----------------------------
A copy of either field inside ``extra_body`` is NOT considered here, and
deliberately so — ``extra_body`` is not a mappable chat param, so it never
reaches ``non_default_params`` and cannot be read at this layer. It is merged
into the request body further downstream, which means a caller who puts an
output-token field there can still override what this resolves. That is the
pre-existing behaviour of that channel (a caller reaching for ``extra_body`` is
explicitly bypassing param mapping), and the h2o ``max_tokens`` cap hook is
where ``extra_body`` is policed.
"""

from typing import Any, Dict, List, Optional, Sequence

MAX_TOKENS_PARAM = "max_tokens"
MAX_COMPLETION_TOKENS_PARAM = "max_completion_tokens"

MAX_TOKENS_PARAMS = (MAX_TOKENS_PARAM, MAX_COMPLETION_TOKENS_PARAM)


def preferred_param_for_directive(
    use_max_completion_tokens: Optional[bool],
) -> Optional[str]:
    """Translate the ``use_max_completion_tokens`` directive to a field name.

    ``None`` means "no directive given" — fall back to provider detection.
    Only the exact booleans are honoured; anything else (a stray ``"false"``
    string from a YAML config, say) is treated as "not given" rather than
    coerced by truthiness into the opposite of what it reads like.
    """
    if use_max_completion_tokens is True:
        return MAX_COMPLETION_TOKENS_PARAM
    if use_max_completion_tokens is False:
        return MAX_TOKENS_PARAM
    return None


def _usable_int(value: Any) -> Optional[int]:
    """Return ``value`` as a positive int, or None if it isn't a usable limit.

    ``bool`` is rejected explicitly because it is an ``int`` subclass. Floats
    are accepted and rounded, because a float ``max_tokens`` really does reach
    litellm — ``AnthropicConfig.map_openai_params`` coerces one with
    ``max(1, int(round(value)))``, and this mirrors that so the two agree.
    Strings are NOT coerced: a value litellm cannot interpret must be left for
    the provider to reject (rule 3).
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


def _is_dropped(param: str, additional_drop_params: Optional[Sequence[str]]) -> bool:
    """Mirror of ``litellm.utils._should_drop_param`` for these two fields."""
    return (
        additional_drop_params is not None
        and isinstance(additional_drop_params, (list, tuple))
        and param in additional_drop_params
    )


def _eligible(
    param: str,
    supported_params: Sequence[str],
    additional_drop_params: Optional[Sequence[str]],
) -> bool:
    """A field can only be a target if the provider accepts it (rule 2) and the
    operator has not dropped it (rule 4)."""
    return param in supported_params and not _is_dropped(
        param, additional_drop_params
    )


def resolve_max_tokens_params(
    non_default_params: Dict[str, Any],
    supported_params: Sequence[str],
    preferred_param: Optional[str] = None,
    additional_drop_params: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """Collapse ``max_tokens`` / ``max_completion_tokens`` in place.

    Args:
        non_default_params: mutated in place; the params about to be mapped.
        supported_params: the params this model/provider accepts. A preference
            for a field the provider does not accept is ignored rather than
            forced, so the resolution can never leave a request with no
            output-token ceiling at all.
        preferred_param: the field to keep, or None to let the caller's own
            field stand when only one was sent.
        additional_drop_params: this deployment's drop list; a field in it is
            never used as a target.

    Returns:
        The field that survived, or None when nothing was changed.
    """
    present = [p for p in MAX_TOKENS_PARAMS if p in non_default_params]
    if not present:
        return None

    values = [
        v
        for v in (_usable_int(non_default_params[p]) for p in present)
        if v is not None
    ]
    if not values:
        # Rule 3 — nothing usable. Leave the request exactly as it was so the
        # provider rejects it as loudly as it would have without us.
        return None
    resolved = min(values)

    target: Optional[str] = None
    if preferred_param is not None and _eligible(
        preferred_param, supported_params, additional_drop_params
    ):
        target = preferred_param
    elif len(present) == 1:
        # One field, no provider preference: the caller's choice already is
        # canonical.
        return None
    else:
        # Both fields, no provider preference. Collapsing is still required:
        # leaving both is what makes the last-wins provider maps pick the looser
        # value. Prefer `max_tokens` — every provider understands it, and the
        # reasoning-model configs that want `max_completion_tokens` rename it
        # themselves downstream.
        for candidate in MAX_TOKENS_PARAMS:
            if _eligible(candidate, supported_params, additional_drop_params):
                target = candidate
                break
        if target is None:
            # Both fields are unsupported or dropped: there is nothing to move
            # the value onto, and inventing one would defeat the drop.
            return None

    for param in MAX_TOKENS_PARAMS:
        if param != target:
            non_default_params.pop(param, None)
    non_default_params[target] = resolved
    return target


__all__ = [
    "MAX_COMPLETION_TOKENS_PARAM",
    "MAX_TOKENS_PARAM",
    "MAX_TOKENS_PARAMS",
    "preferred_param_for_directive",
    "resolve_max_tokens_params",
]
