"""Skeleton routing subpackage for the h2oai LiteLLM proxy hook.

Components:

* :mod:`plan` -- the ``Plan`` / ``WorkerCall`` dataclasses and the
  ``PlanMode`` / ``Aggregator`` literal types. This is the wire format
  the hook and the policy agree on.
* :mod:`policy` -- ``decide_plan(data, expressed_model) -> Plan``.
  The default implementation is pass-through; it can be replaced at
  deploy time by setting ``H2O_ROUTER_POLICY_MODULE`` to a module that
  exposes a ``decide_plan`` callable, or by overwriting this file.
* :mod:`modes` -- dispatcher stubs for the multi-call modes (cascade,
  parallel_aggregate, best_of_n, debate, hierarchical_delegate). The
  skeleton implementations raise ``NotImplementedError``; replace this
  file at deploy time to enable a given mode.

The skeleton is intentionally minimal: only ``mode=route_single`` runs
end-to-end. Everything else has signature-stable stubs so a deploy-time
replacement is drop-in.
"""
