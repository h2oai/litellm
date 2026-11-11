"""Routing policy: decide what ``Plan`` to use for a given request.

The default implementation is a pure pass-through that returns
``Plan.passthrough(expressed_model)``. The basic policy uses only
information LiteLLM provides on the inbound request (the messages, the
expressed model, etc.) -- no external calls, no model inference.

To plug in a real policy at deploy time, choose one of:

1. **Module override** -- set ``H2O_ROUTER_POLICY_MODULE`` to the import
   path of a module that exposes ``decide_plan(data, expressed_model) -> Plan``.
   The module is imported on first use and its ``decide_plan`` is used
   in place of the default.

   .. code-block:: bash

      export H2O_ROUTER_POLICY_MODULE=my_company.router.policy

2. **File replacement** -- overwrite this file in the installed
   ``litellm/integrations/h2o/router/`` directory with one that contains
   a real ``decide_plan`` function. The hook imports ``decide_plan`` from
   this module name, so a drop-in replacement Just Works.

The policy must not raise; it returns a ``Plan`` for every request,
falling back to ``Plan.passthrough(...)`` whenever a real decision is
unavailable. The hook's fail-open wrapper catches exceptions as a
defensive measure but a well-behaved policy never relies on it.
"""

from __future__ import annotations

import importlib
import logging
import os
from typing import Any, Callable, Dict, Optional

from .plan import Plan

logger = logging.getLogger(__name__)

DecidePlanFn = Callable[[Dict[str, Any], str], Plan]


def basic_decide_plan(data: Dict[str, Any], expressed_model: str) -> Plan:
    """The default in-process policy.

    Returns a pass-through Plan that routes to the expressed model with
    no rewrites and no multi-call orchestration. Uses only information
    already present on the inbound request -- no external calls.
    """
    return Plan.passthrough(expressed_model)


def _load_external_policy() -> Optional[DecidePlanFn]:
    """Try to load a deploy-time policy from ``H2O_ROUTER_POLICY_MODULE``.

    Returns ``None`` on any failure -- import error, missing attribute,
    non-callable attribute -- and logs a warning. The hook then falls
    back to :func:`basic_decide_plan`.
    """
    module_name = os.environ.get("H2O_ROUTER_POLICY_MODULE", "").strip()
    if not module_name:
        return None
    try:
        mod = importlib.import_module(module_name)
    except ImportError as exc:
        logger.warning(
            "h2o-router: could not import H2O_ROUTER_POLICY_MODULE=%r: %s",
            module_name,
            exc,
        )
        return None
    fn = getattr(mod, "decide_plan", None)
    if not callable(fn):
        logger.warning(
            "h2o-router: %r has no callable decide_plan; using basic policy.",
            module_name,
        )
        return None
    logger.info("h2o-router: external policy loaded from %r", module_name)
    return fn  # type: ignore[return-value]


# Resolved at import time. Re-resolve programmatically by calling
# ``_load_external_policy()`` again (e.g. in tests).
_external_decide_plan: Optional[DecidePlanFn] = _load_external_policy()


def decide_plan(data: Dict[str, Any], expressed_model: str) -> Plan:
    """Top-level routing decision used by the hook.

    Dispatches to the deploy-time policy if one was successfully loaded
    via ``H2O_ROUTER_POLICY_MODULE``; otherwise delegates to
    :func:`basic_decide_plan`.
    """
    if _external_decide_plan is not None:
        return _external_decide_plan(data, expressed_model)
    return basic_decide_plan(data, expressed_model)
