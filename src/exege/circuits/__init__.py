"""circuits -- which paths through a network a behaviour runs along.

    from exege.circuits import field_mean, open_patchable, prune

    system = open_patchable(None, "toy-transformer")
    base, source = system.states([0]), system.states([1])
    metric = field_mean(system.grid(), "tracer", where=system.grid().within(0, 90, 3000))
    circuit = prune(system, base, source, metric, tau=0.01)
    circuit.edges(), circuit.recovery

Running a model is an adapter's job (see ``Patchable``); this package designs the patches
and reads the answers.

- ``graph``     senders, receivers, edges; Vi-CD's reduced graph of a residual network
- ``contract``  ``Patchable`` (edge and component patching), the optional ``EdgeGradients``
- ``metrics``   what a circuit is for: a scalar of a step's fields; area-weighted means
- ``discover``  attribution (exact or EAP), greedy pruning, faithfulness curves, and how
                alike circuits are (Jaccard, edge frequency, the stable core)

Needs the ``circuits`` extra (numpy). Names load on first use, as in ``exege.latents``.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from exege.circuits.contract import (
        ComponentPatch,
        EdgeGradients,
        EdgePatch,
        Patchable,
        open_patchable,
    )
    from exege.circuits.discover import (
        Attribution,
        Circuit,
        FaithfulnessCurve,
        attribute,
        core,
        edge_frequency,
        endpoints,
        faithfulness_curve,
        jaccard,
        patched,
        prune,
        top_k,
    )
    from exege.circuits.graph import Graph, residual_graph
    from exege.circuits.metrics import LinearMetric, field_mean, recovery

_LAZY = {
    "ComponentPatch": "contract",
    "EdgeGradients": "contract",
    "EdgePatch": "contract",
    "Patchable": "contract",
    "open_patchable": "contract",
    "Attribution": "discover",
    "Circuit": "discover",
    "FaithfulnessCurve": "discover",
    "attribute": "discover",
    "core": "discover",
    "edge_frequency": "discover",
    "endpoints": "discover",
    "faithfulness_curve": "discover",
    "jaccard": "discover",
    "patched": "discover",
    "prune": "discover",
    "top_k": "discover",
    "Graph": "graph",
    "residual_graph": "graph",
    "LinearMetric": "metrics",
    "field_mean": "metrics",
    "recovery": "metrics",
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{_LAZY[name]}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_LAZY])


__all__ = [
    "Attribution",
    "Circuit",
    "ComponentPatch",
    "EdgeGradients",
    "EdgePatch",
    "FaithfulnessCurve",
    "Graph",
    "LinearMetric",
    "Patchable",
    "attribute",
    "core",
    "edge_frequency",
    "endpoints",
    "faithfulness_curve",
    "field_mean",
    "jaccard",
    "open_patchable",
    "patched",
    "prune",
    "recovery",
    "residual_graph",
    "top_k",
]
