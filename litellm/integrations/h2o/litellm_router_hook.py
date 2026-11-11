"""Router hook for the LiteLLM proxy (skeleton implementation).

``H2ORouterHook`` is a ``CustomLogger`` that runs before each LLM call,
consults a routing policy, and either:

* (``mode=route_single``) rewrites ``data["model"]`` to point at the
  chosen worker and lets LiteLLM continue the call, or
* (multi-call modes) delegates to a dispatcher in
  :mod:`router.modes`.

In this skeleton, only ``route_single`` runs end-to-end; the multi-call
dispatchers raise ``NotImplementedError`` and the hook falls through to
pass-through unless ``H2O_ROUTER_FAIL_OPEN=false``.

Both the policy (:mod:`router.policy`) and the dispatchers
(:mod:`router.modes`) are intentionally swappable so deploy-time
replacements can be drop-in.

Registration (LiteLLM proxy ``config.yaml``):

.. code-block:: yaml

    callbacks:
      - litellm.integrations.h2o.litellm_router_hook.router_hook

Environment variables:

* ``H2O_ROUTER_POLICY_MODULE`` -- import path of a module providing a
  ``decide_plan(data, expressed_model) -> Plan`` function. Unset = use
  the basic pass-through policy.
* ``H2O_ROUTER_FAIL_OPEN`` -- if ``"true"`` (default), any exception in
  policy or dispatcher degrades to pass-through. Set ``"false"`` to
  surface errors.
* ``H2O_ROUTER_LOG_DECISIONS`` -- if ``"true"`` (default), log each
  routing decision through the LiteLLM verbose logger.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger

from .router import modes
from .router.policy import decide_plan


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


class H2ORouterHook(CustomLogger):
    """LiteLLM ``CustomLogger`` that consults a routing policy and acts
    on the resulting ``Plan``."""

    def __init__(self) -> None:
        super().__init__()
        self.log_decisions = _bool_env("H2O_ROUTER_LOG_DECISIONS", True)
        self.fail_open = _bool_env("H2O_ROUTER_FAIL_OPEN", True)

    async def async_pre_call_hook(  # type: ignore[override]
        self,
        user_api_key_dict,
        cache,
        data: Dict[str, Any],
        call_type: str,
    ) -> Dict[str, Any]:
        expressed_model = data.get("model") or ""
        if not expressed_model:
            return data

        try:
            plan = decide_plan(data, expressed_model)
        except Exception as exc:
            if self.fail_open:
                verbose_proxy_logger.warning(
                    "h2o-router: policy raised (%r); passing request through.", exc
                )
                return data
            raise

        if self.log_decisions:
            verbose_proxy_logger.info(
                "h2o-router: %s -> mode=%s, workers=%s",
                expressed_model,
                plan.mode,
                [w.model for w in plan.workers],
            )

        if plan.mode == "route_single":
            target = plan.workers[0] if plan.workers else None
            if target and target.model and target.model != expressed_model:
                data["model"] = target.model
            if target and target.params_override:
                for k, v in target.params_override.items():
                    data[k] = v
            return data

        dispatcher = modes.DISPATCHERS.get(plan.mode)
        if dispatcher is None:
            if self.fail_open:
                verbose_proxy_logger.warning(
                    "h2o-router: unknown plan mode %r; passing through.", plan.mode
                )
                return data
            raise ValueError(f"unknown plan mode {plan.mode!r}")

        try:
            result = await dispatcher(plan, data)
        except NotImplementedError:
            if self.fail_open:
                verbose_proxy_logger.info(
                    "h2o-router: mode %r not implemented in skeleton; passing "
                    "through. Replace litellm.integrations.h2o.router.modes at "
                    "deploy time to enable.",
                    plan.mode,
                )
                return data
            raise
        except Exception as exc:
            if self.fail_open:
                verbose_proxy_logger.warning(
                    "h2o-router: mode %r dispatcher raised (%r); passing through.",
                    plan.mode,
                    exc,
                )
                return data
            raise

        return result if isinstance(result, dict) else data


# Module-level singleton matching the registration style of the other h2o hooks.
router_hook = H2ORouterHook()
