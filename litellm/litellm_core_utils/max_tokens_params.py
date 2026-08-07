"""Canonicalize the two OpenAI output-token fields onto a single field.

WHY THIS EXISTS
---------------
The OpenAI chat schema carries two output-token fields: the original
``max_tokens`` and its replacement ``max_completion_tokens``. A request can
arrive with both — most commonly because a proxy deployment configures one of
them as a ceiling in ``litellm_params`` while the client sends the other. When
that happens today, what reaches the provider depends on the order the
provider's ``map_openai_params`` happens to iterate, and the result is wrong in
two different ways:

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
1. TIGHTER WINS. When both fields carry a usable value, the smaller one is
   kept. A deployment ceiling can therefore never be RAISED by a client, and a
   client's tighter request can never be widened to the ceiling.

2. ONE FIELD OUT. When a preferred field is known, only that field survives.
   The preference comes from the provider config
   (``BaseConfig.get_preferred_max_tokens_param``), or from an explicit
   per-request/per-deployment ``use_max_completion_tokens`` directive, which
   overrides the provider's own detection.

3. NEVER INVENT A VALUE. If neither field holds a positive integer, nothing is
   changed. A ``max_tokens`` of ``0``, ``None``, a string, or a bool must not
   become a ``max_completion_tokens`` that truncates every response.
"""

from typing import Any, Dict, Optional, Sequence

MAX_TOKENS_PARAM = "max_tokens"
MAX_COMPLETION_TOKENS_PARAM = "max_completion_tokens"

MAX_TOKENS_PARAMS = (MAX_TOKENS_PARAM, MAX_COMPLETION_TOKENS_PARAM)


def preferred_param_for_directive(
    use_max_completion_tokens: Optional[bool],
) -> Optional[str]:
    """Translate the ``use_max_completion_tokens`` directive to a field name.

    ``None`` means "no directive given" — fall back to provider detection.
    """
    if use_max_completion_tokens is None:
        return None
    return (
        MAX_COMPLETION_TOKENS_PARAM
        if use_max_completion_tokens
        else MAX_TOKENS_PARAM
    )


def _positive_int(value: Any) -> Optional[int]:
    """Return ``value`` as a positive int, or None if it isn't one.

    ``bool`` is rejected explicitly because it is an ``int`` subclass.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value > 0 else None


def resolve_max_tokens_params(
    non_default_params: Dict[str, Any],
    supported_params: Sequence[str],
    preferred_param: Optional[str] = None,
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

    Returns:
        The field that survived, or None when nothing was changed.
    """
    present = [p for p in MAX_TOKENS_PARAMS if p in non_default_params]
    if not present:
        return None

    values = [
        v
        for v in (_positive_int(non_default_params[p]) for p in present)
        if v is not None
    ]
    if not values:
        # Rule 3 — nothing usable to move or tighten. Still strip a field the
        # provider is known to reject: an unusable value (0, a string) is not a
        # ceiling worth preserving, and leaving it on a field the provider
        # rejects turns a garbage request into a guaranteed 400.
        if preferred_param is not None and preferred_param in supported_params:
            for param in MAX_TOKENS_PARAMS:
                if param != preferred_param:
                    non_default_params.pop(param, None)
        return None
    resolved = min(values)

    target: Optional[str] = None
    if preferred_param is not None and preferred_param in supported_params:
        target = preferred_param
    elif len(present) == 1:
        # One field, no preference: the caller's choice already is canonical.
        return None
    else:
        # Both fields with no provider preference. Collapsing is still
        # required — leaving both is what makes the last-wins provider maps
        # pick the looser value. Prefer `max_tokens`: every provider
        # understands it, and the reasoning-model configs that want
        # `max_completion_tokens` rename it themselves downstream.
        target = (
            MAX_TOKENS_PARAM
            if MAX_TOKENS_PARAM in supported_params
            else MAX_COMPLETION_TOKENS_PARAM
        )

    for param in MAX_TOKENS_PARAMS:
        if param != target:
            non_default_params.pop(param, None)
    non_default_params[target] = resolved
    return target
