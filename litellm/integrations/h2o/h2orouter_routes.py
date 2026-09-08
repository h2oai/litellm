"""Mount the H2ORouter control plane onto the proxy's own FastAPI app.

WHY THIS LIVES IN THE FORK. The router deploys as ONE service: the LiteLLM proxy, extended. Something
has to add our routes to the proxy's `app`, and it should be the fork rather than a customer's
bootstrap script -- so that installing the fork with `h2orouter` present is all it takes, and so that
this project and h2ogpt_internal both inherit it on upgrade.

WHY IT IS OPTIONAL AND SILENT. A proxy without `h2orouter` installed is the normal case for most
deployments, and it must start exactly as it does today. So the import failing is not an error, it is
the absence of a feature, and nothing is logged above debug for it.

WHAT IS ADDED AND WHAT IS NOT. `h2orouter.mount` decides, from the real route table rather than from
a list anyone wrote down: 26 routes keep their natural paths, the Files API and two meta routes move
under `/v1/h2orouter`, and `POST /v1/chat/completions`, `GET /v1/models`, `GET /health` and `GET /v1/files`
are left alone because they are the proxy's. Routing happens in `litellm_router_hook`, not in a
completion endpoint of ours.
"""

from __future__ import annotations

from litellm._logging import verbose_proxy_logger


def mount_h2orouter(app) -> list:
    """Add the H2ORouter control plane to `app` if `h2orouter` is installed. Never raises.

    Returns the list of routes added, empty when the package is absent.
    """
    try:
        from h2orouter.mount import mount
    except ImportError:
        verbose_proxy_logger.debug(
            "h2o-h2orouter: `h2orouter` is not installed; the proxy's own routes are unchanged."
        )
        return []
    try:
        added = mount(app)
    except Exception as exc:  # noqa: BLE001 - a broken extension must not stop the proxy booting
        verbose_proxy_logger.warning(
            "h2o-h2orouter: could not mount the control plane (%r); the proxy starts without it.", exc
        )
        return []
    verbose_proxy_logger.info(
        "h2o-h2orouter: mounted %d route(s); router control plane at /v1/routers and /v1/h2orouter.",
        len(added),
    )
    return added
