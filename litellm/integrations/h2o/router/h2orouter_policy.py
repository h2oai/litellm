"""A deploy-time policy that routes with a trained H2ORouter router.

WHY THIS IS THE WHOLE INTEGRATION. The proxy keeps every capability LiteLLM has -- streaming, fallbacks,
retries, budgets, guardrails, caching, and its OWN routing strategies (``simple-shuffle``,
``least-busy``, ``usage-based-routing``, ``latency-based-routing``, ``cost-based-routing``) -- and our
contribution is one ``async_pre_call_hook`` that rewrites ``data["model"]`` before the call goes out.
That is deliberately the smallest possible deviation: everything downstream of the rewrite is stock
LiteLLM, so an upgrade of the fork does not have to re-examine our routing, and our routing does not
have to re-implement anything LiteLLM already does.

**A REQUEST FOR A MODEL WE DO NOT KNOW IS NOT TOUCHED.** If the expressed model is not the name of a
loaded H2ORouter router, this returns a pass-through Plan and LiteLLM behaves exactly as it would without us
-- including using its own routing strategy for that model group. The proxy therefore still works as a
plain generic model router with the learned router simply not in the path. That property is the reason
the hook is a rewrite rather than a replacement.

Wire it up in the proxy `config.yaml`:

.. code-block:: yaml

    callbacks:
      - litellm.integrations.h2o.litellm_router_hook.router_hook

and point the hook at this module:

.. code-block:: bash

    export H2O_ROUTER_POLICY_MODULE=litellm.integrations.h2o.router.h2orouter_policy
    export H2OROUTER_ARTIFACTS=/artifacts            # where trained routers were saved
    # or, to consult a running H2ORouter service instead of loading in-process:
    export H2OROUTER_ROUTER_URL=http://h2orouter:8080

TWO WAYS TO REACH A ROUTER, and the trade is latency against coupling:

``in-process`` (default when ``H2OROUTER_ARTIFACTS`` is set)
    Loads the saved routers once and decides locally. **No network hop per request**, which matters
    because this runs inside the request path. Needs `h2orouter` installed in the proxy's environment.

``http`` (when ``H2OROUTER_ROUTER_URL`` is set)
    Asks the H2ORouter service. Costs a local round trip per request, but the proxy needs neither the package
    nor the artefacts, and routers can be retrained without restarting the proxy. Prefer this when the
    two are operated by different people.

FAILING OPEN IS THE RIGHT DEFAULT HERE and it is worth saying why: a routing decision is an
optimisation, not a correctness requirement. If the router cannot be reached, answering with the model
the caller named is a good answer; refusing to answer is not. Every degradation is logged once with the
reason, so a silently mis-routing proxy is not the failure mode -- a loudly pass-through one is.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, Optional

from .plan import Plan, WorkerCall

logger = logging.getLogger(__name__)

#: Routers, by name. Populated on first use and reused; loading is the expensive part, deciding is not.
_ROUTERS: Dict[str, Any] = {}
_LOADED = False
_LOCK = threading.Lock()
#: Names we have already complained about, so a per-request failure logs once rather than per request.
_WARNED: set = set()


def _warn_once(key: str, msg: str, *args) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        logger.warning(msg, *args)


def _load_local() -> None:
    """Load every router saved under ``H2OROUTER_ARTIFACTS``. Never raises."""
    global _LOADED
    with _LOCK:
        if _LOADED:
            return
        _LOADED = True
        root = os.environ.get("H2OROUTER_ARTIFACTS")
        if not root:
            return
        try:
            from h2orouter.api import persist

            persist.ARTIFACT_DIR = root
            res = persist.load_all()
            for item in res.get("loaded", []):
                meta = item.get("meta") or {}
                # ONLY PUBLISHED ROUTERS ROUTE, matching the standalone service. An unpublished
                # router has been fitted but has no pinned operating point, so acting on it would
                # route at whatever dial setting the last call happened to leave behind.
                if not meta.get("published"):
                    logger.info(
                        "h2o-router: %s is trained but not published; not routing with it",
                        item.get("name"),
                    )
                    continue
                _ROUTERS[item["name"]] = {
                    "router": item["router"],
                    "models": meta.get("models", []),
                }
            for s in res.get("skipped", []):
                logger.warning("h2o-router: skipped %s (%s)", s.get("name"), s.get("reason"))
            logger.info("h2o-router: loaded %d H2ORouter router(s) from %s", len(_ROUTERS), root)
        except ImportError:
            _warn_once(
                "import",
                "h2o-router: H2OROUTER_ARTIFACTS is set but `h2orouter` is not installed in this "
                "environment; passing every request through. Install it, or set H2OROUTER_ROUTER_URL to "
                "consult the H2ORouter service over HTTP instead.",
            )
        except Exception as exc:  # noqa: BLE001 - a bad artefact must not take the proxy down
            _warn_once("load", "h2o-router: could not load routers from %s (%r)", root, exc)


def _prompt_of(data: Dict[str, Any]) -> str:
    """The user text the routing decision is made from.

    Only user turns: a long system prompt is identical across every request in a deployment, so
    including it would push every question towards the same features and flatten the decision.
    """
    parts = []
    for m in data.get("messages") or []:
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):  # the multimodal content-block form
            parts.extend(b.get("text", "") for b in c if isinstance(b, dict))
    return "\n".join(p for p in parts if p)


def _decide_http(name: str, prompt: str) -> Optional[str]:
    """Ask a running H2ORouter service. Returns the chosen model, or None to pass through."""
    url = os.environ.get("H2OROUTER_ROUTER_URL", "").rstrip("/")
    if not url:
        return None
    try:
        import httpx

        r = httpx.post(
            f"{url}/v1/routers/{name}/route",
            json={"prompts": [prompt]},
            timeout=float(os.environ.get("H2OROUTER_ROUTER_TIMEOUT", "5")),
        )
        if r.status_code != 200:
            _warn_once(f"http-{name}", "h2o-router: %s returned HTTP %s", name, r.status_code)
            return None
        return (r.json().get("routed_to") or [None])[0]
    except Exception as exc:  # noqa: BLE001 - the router being down must not break the proxy
        _warn_once(f"http-{name}-exc", "h2o-router: could not reach %s (%r)", url, exc)
        return None


def _ensure_proxy_targets() -> None:
    """Delegate to the package, which owns the registry. See `h2orouter.api.common`.

    Kept as a thin wrapper so this module still works when `h2orouter` is absent -- the proxy then
    has no targets of ours to sync, and every request passes through untouched.
    """
    try:
        from h2orouter.api.common import ensure_proxy_targets
    except ImportError:
        return
    try:
        ensure_proxy_targets()
    except Exception as exc:  # noqa: BLE001 - never fail a request over a registry sync
        _warn_once("sync", "h2o-router: could not sync targets into the proxy router (%r)", exc)


def _passthrough_unknown(name: str) -> bool:
    """Register an unknown model name with the proxy so a plain provider call just works.

    THE DOCUMENTED PROMISE WAS FALSE. Both the guide and `GET /v1/targets` say a public model whose
    provider key is in the environment needs no registration -- "unregistered pool names pass straight
    through to LiteLLM". That is true of the litellm LIBRARY and NOT of the proxy: the proxy resolves
    `model` against its own `model_list`, and this image ships an empty one to keep `docker run`
    credential-free. So a customer who set OPENAI_API_KEY exactly as documented and asked for `gpt-5`
    got `Invalid model name passed in model=gpt-5`, and the API told them the opposite of the truth.

    Adding the deployment here makes the promise true: the hook runs BEFORE model resolution, so a name
    registered now resolves a moment later. litellm infers the provider from the name and reads the key
    from the environment exactly as it does in the library.

    This does not invent credentials and it does not hide failures -- an unroutable name still fails,
    but with the provider's own error ("no API key for anthropic") rather than a proxy-level rejection
    that reads as though the model does not exist. Set H2OROUTER_PASSTHROUGH_UNKNOWN_MODELS=false for a
    proxy that should only ever serve models an operator listed.
    """
    if os.environ.get("H2OROUTER_PASSTHROUGH_UNKNOWN_MODELS", "true").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return False
    try:
        import litellm
        from litellm.proxy import proxy_server as ps
        from litellm.types.router import Deployment, LiteLLM_Params
    except ImportError:
        return False
    try:
        if ps.llm_router is None:
            ps.llm_router = litellm.Router(model_list=[])
        if name in set(ps.llm_router.get_model_names() or []):
            return False
        ps.llm_router.upsert_deployment(
            Deployment(model_name=name, litellm_params=LiteLLM_Params(model=name))
        )
        logger.info("h2o-router: passed %r through to LiteLLM as an unlisted model", name)
        return True
    except Exception as exc:  # noqa: BLE001 - an unroutable name is the provider's error to give
        _warn_once(f"pt-{name}", "h2o-router: could not pass %r through (%r)", name, exc)
        return False


def _in_process(name: str):
    """The router by that name in THIS process's store, if it is published. Never raises.

    Returns None when the control plane is not mounted here, when no such router exists, or when it
    exists but is not published -- an unpublished router has no pinned operating point, so acting on
    it would route at whatever dial the last call happened to leave behind.
    """
    try:
        from h2orouter.api.store import STORE
    except ImportError:
        return None
    rec = STORE.get(name)
    if rec is None or rec.get("router") is None or not rec.get("published"):
        return None
    return rec


def _record(router, chosen):
    """Count the decision. NOTHING ELSE IN THE STACK CAN: by the time the proxy logs the call the
    model has already been rewritten, so it records a call to the chosen model and has no idea a
    router picked it over three others. Never costs a request -- a counter that can fail a call is
    not worth having."""
    try:
        from h2orouter.api import usage

        usage.record(router, chosen)
    except Exception:  # noqa: BLE001
        pass


def _decide_from(rec, data: Dict[str, Any], expressed_model: str) -> Plan:
    """Route with an in-process record. Any failure degrades to pass-through."""
    prompt = _prompt_of(data)
    if not prompt:
        _record(expressed_model, None)
        return Plan.passthrough(expressed_model)
    try:
        k = int(rec["router"].predict([prompt])[0])
        target = list(rec["spec"].models)[k]
    except Exception as exc:  # noqa: BLE001
        _warn_once(f"inproc-{expressed_model}", "h2o-router: %s failed to decide (%r)", expressed_model, exc)
        _record(expressed_model, None)
        return Plan.passthrough(expressed_model)
    # AND THE MODEL WE JUST CHOSE has to be routable too. Passing through only the EXPRESSED name is
    # not enough: the whole point of the hook is that the name going out is different from the name
    # coming in, so it is the REWRITTEN one the proxy must be able to resolve. Missing this is why
    # the preset walkthrough still failed with `Invalid model name passed in model=gpt-5` on a fresh
    # container -- the router chose correctly and handed the proxy something it did not know.
    _passthrough_unknown(target)
    _record(expressed_model, target)
    return Plan(mode="route_single", workers=[WorkerCall(model=target, role_hint="h2orouter")])


def decide_plan(data: Dict[str, Any], expressed_model: str) -> Plan:
    """Route with a H2ORouter router when the caller named one; otherwise change nothing.

    The pass-through branch is the important one. It is what lets this hook be installed permanently
    on a proxy that mostly serves ordinary models: those requests take the same path they always did,
    and LiteLLM's own routing strategy for that model group is untouched.
    """
    # Targets registered with US must exist in the PROXY'S router or it rejects them by name; this
    # is a set difference after the first request. See the note on `_ensure_proxy_targets`.
    _ensure_proxy_targets()

    # SAME PROCESS FIRST. In the consolidated deployment the control plane is mounted onto this very
    # proxy, so the router the customer trained a moment ago is already in memory. Reading it here
    # means no network hop, no artefact reload, and -- the part that actually matters -- no window in
    # which `POST /v1/routers/x/publish` has returned but `model: "x"` is not yet routable.
    rec = _in_process(expressed_model)
    if rec is not None:
        return _decide_from(rec, data, expressed_model)

    # not one of ours -- make sure the proxy can actually serve it before passing through
    _passthrough_unknown(expressed_model)

    if os.environ.get("H2OROUTER_ROUTER_URL"):
        target = _decide_http(expressed_model, _prompt_of(data))
        if target and target != expressed_model:
            _passthrough_unknown(target)
            _record(expressed_model, target)
            return Plan(mode="route_single", workers=[WorkerCall(model=target)])
        return Plan.passthrough(expressed_model)

    _load_local()
    rec = _ROUTERS.get(expressed_model)
    if rec is None:
        return Plan.passthrough(
            expressed_model
        )  # not ours: leave the request exactly as it came in

    prompt = _prompt_of(data)
    if not prompt:
        # nothing to read means nothing to route on -- a tool-result-only turn, say. Guessing from an
        # empty string would send every such request to whichever model wins the empty prompt.
        return Plan.passthrough(expressed_model)
    try:
        k = int(rec["router"].predict([prompt])[0])
        target = rec["models"][k]
    except Exception as exc:  # noqa: BLE001
        _warn_once(
            f"predict-{expressed_model}",
            "h2o-router: %s failed to decide (%r)",
            expressed_model,
            exc,
        )
        return Plan.passthrough(expressed_model)

    return Plan(mode="route_single", workers=[WorkerCall(model=target, role_hint="h2orouter")])


def reload_routers() -> int:
    """Forget the loaded routers so the next request re-reads the artefacts directory.

    Retraining writes a new artefact; without this the proxy would serve the old router until it was
    restarted, which is the thing `reload_models_on_demand` exists to avoid for model registration.
    Returns the number of routers loaded.
    """
    global _LOADED
    with _LOCK:
        _ROUTERS.clear()
        _LOADED = False
        _WARNED.clear()
    _load_local()
    return len(_ROUTERS)
