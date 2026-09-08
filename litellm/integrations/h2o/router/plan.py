"""Plan-JSON contract: the wire format exchanged between policy and hook.

The router's policy emits a ``Plan`` describing how a single inbound
request should be answered. The hook reads the Plan and either (for
``mode=route_single``) rewrites the outgoing call to use the chosen
worker, or hands the Plan to a multi-call dispatcher in :mod:`modes`.

These dataclasses are the only contract; the policy module and the
mode dispatchers can be swapped freely as long as they produce or
consume these shapes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional

PlanMode = Literal[
    "route_single",
    "cascade",
    "parallel_aggregate",
    "best_of_n",
    "debate",
    "hierarchical_delegate",
]

Aggregator = Literal[
    "identity",
    "judge_fuse",
    "vote",
    "score_select",
    "debate_referee",
]


@dataclass
class WorkerCall:
    """A single worker invocation inside a Plan."""

    # LiteLLM-routable model name (e.g. ``claude-opus-4-7`` or
    # ``anthropic/claude-opus-4-7``). The hook rewrites
    # ``data["model"]`` to this value for ``mode=route_single``.
    model: str

    # Optional per-call parameter overrides applied to the request data
    # (``temperature``, ``max_tokens``, etc.). Anything not set here
    # inherits from the original request.
    params_override: Optional[Dict[str, Any]] = None

    # Free-form capability hint propagated to the worker. Unused by
    # the skeleton; deploy-time policies may use it to signal intent.
    role_hint: Optional[str] = None


@dataclass
class Plan:
    """A routing decision."""

    mode: PlanMode

    # Ordered list of worker calls. Length semantics by mode:
    #
    # * ``route_single``                  exactly 1
    # * ``cascade``                       >=2, called in order
    # * ``parallel_aggregate``,           >=2, all called concurrently
    #   ``best_of_n``, ``debate``
    # * ``hierarchical_delegate``         typically 1 sub-router target
    workers: List[WorkerCall] = field(default_factory=list)

    # How to combine multi-worker outputs. Required for non-``route_single``
    # modes; ignored otherwise.
    aggregator: Optional[Aggregator] = None

    # Maximum additional spend (USD) this Plan is allowed beyond the
    # cheapest worker's cost. ``None`` = unbounded. Advisory for
    # the skeleton; deploy-time dispatchers may enforce.
    escalation_budget_usd: Optional[float] = None

    # Free-form metadata threaded through for diagnostics.
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def passthrough(cls, model: str) -> "Plan":
        """Trivial Plan that routes to the given model unchanged.

        This is the safe default the skeleton policy returns when no
        deploy-time replacement is configured.
        """
        return cls(
            mode="route_single",
            workers=[WorkerCall(model=model)],
            aggregator="identity",
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str) -> "Plan":
        data = json.loads(raw)
        workers = [WorkerCall(**w) for w in data.get("workers", [])]
        return cls(
            mode=data["mode"],
            workers=workers,
            aggregator=data.get("aggregator"),
            escalation_budget_usd=data.get("escalation_budget_usd"),
            metadata=data.get("metadata", {}),
        )
