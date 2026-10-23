"""Multi-call mode dispatchers (skeleton).

Each dispatcher takes a ``Plan`` and the inbound request data and
returns a dict that the hook hands back to LiteLLM. The skeleton
implementations all raise ``NotImplementedError`` -- the contract is
present so deploy-time replacements are drop-in.

Replace this file at deploy time to enable multi-call routing. The
expected return value is either:

* the request data dict (possibly mutated to point at a single chosen
  worker, letting LiteLLM finish the call), or
* a fully synthesized response shaped like a LiteLLM completion (the
  dispatcher having done its own out-of-band calls).

The hook tolerates ``NotImplementedError`` by falling through to
pass-through unless ``H2O_ROUTER_FAIL_OPEN=false`` is set.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict

from .plan import Plan, PlanMode


async def execute_cascade(plan: Plan, request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Try workers in ``plan.workers`` order; accept the first that
    satisfies the verifier."""
    raise NotImplementedError(
        "cascade dispatch is not implemented in the skeleton. "
        "Replace litellm.integrations.h2o.router.modes at deploy time."
    )


async def execute_parallel_aggregate(
    plan: Plan, request_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Call all workers concurrently; combine via ``plan.aggregator``."""
    raise NotImplementedError(
        "parallel_aggregate dispatch is not implemented in the skeleton. "
        "Replace litellm.integrations.h2o.router.modes at deploy time."
    )


async def execute_best_of_n(plan: Plan, request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Call all workers; return the highest-scoring response."""
    raise NotImplementedError(
        "best_of_n dispatch is not implemented in the skeleton. "
        "Replace litellm.integrations.h2o.router.modes at deploy time."
    )


async def execute_debate(plan: Plan, request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Multi-round critique-and-refine with a referee aggregator."""
    raise NotImplementedError(
        "debate dispatch is not implemented in the skeleton. "
        "Replace litellm.integrations.h2o.router.modes at deploy time."
    )


async def execute_hierarchical_delegate(
    plan: Plan, request_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Dispatch to a recursive sub-router (typically an agent endpoint
    with its own worker pool)."""
    raise NotImplementedError(
        "hierarchical_delegate dispatch is not implemented in the skeleton. "
        "Replace litellm.integrations.h2o.router.modes at deploy time."
    )


# Dispatch table consumed by ``litellm_router_hook.H2ORouterHook``.
# Keep keys in sync with the non-``route_single`` values of ``PlanMode``.
DISPATCHERS: Dict[PlanMode, Callable[[Plan, Dict[str, Any]], Awaitable[Dict[str, Any]]]] = {
    "cascade": execute_cascade,
    "parallel_aggregate": execute_parallel_aggregate,
    "best_of_n": execute_best_of_n,
    "debate": execute_debate,
    "hierarchical_delegate": execute_hierarchical_delegate,
}
