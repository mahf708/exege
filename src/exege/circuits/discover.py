"""Finding circuits: which edges a behaviour runs through.

The behaviour is a metric of one step on a *base* state; its counterfactual is the same
metric on a *source* state with the evidence removed (Vi-CD's inpainted image; for an
emulator, the same day with a feature taken out, another day, a climatology). Patching an
edge to the source's value and watching the metric is the test of whether it matters.

- ``attribute``            a score per edge: the metric's change when that edge alone is
                           patched, exactly (one run per edge) or to first order (EAP, one
                           gradient, when the system offers ``EdgeGradients``)
- ``prune``                Vi-CD's greedy pass (ACDC's order): from the output backwards,
                           patch each edge and keep it patched when the metric moves by less
                           than ``tau`` of the clean-vs-corrupt gap
- ``faithfulness_curve``   keep the top-k edges by a score, patch the rest: recovery vs size
- ``edge_frequency``, ``core``, ``jaccard``  how alike circuits from different runs,
                           classes or seeds are (Vi-CD's RQ2)

Every result says what it was computed from (``settings``) and keeps the base and source
metric values, so a recovery is never reported without the gap it is a fraction of.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from exege.circuits.contract import EdgeGradients, EdgePatch, Patchable
from exege.circuits.graph import Graph
from exege.circuits.metrics import LinearMetric, recovery
from exege.core.errors import RequestError
from exege.core.extras import missing_extra

try:
    import numpy as np
except ImportError as exc:
    raise missing_extra("numpy", "circuits") from exc

Metric = Callable[[Mapping[str, np.ndarray]], float]


def patched(system: Patchable, base: Any, source: Any, mask: np.ndarray, metric: Metric) -> float:
    """The metric of one step on ``base`` with the edges ``mask`` leaves at 0 patched."""
    return float(metric(system.step(base, EdgePatch(source, mask))))


def endpoints(system: Patchable, base: Any, source: Any, metric: Metric) -> tuple[float, float]:
    """The metric on the base and on the source: what a circuit's recovery is measured
    between."""
    return float(metric(system.step(base))), float(metric(system.step(source)))


@dataclass(frozen=True)
class Attribution:
    """``scores[s, r]``: the metric's change when edge ``(s, r)`` alone carries the source's
    value (negative: the edge holds the metric up). Zero off the graph."""

    graph: Graph
    scores: np.ndarray
    method: str
    clean: float
    corrupt: float
    settings: Mapping[str, Any] = field(default_factory=dict)

    def ranked(self, n: int | None = None) -> list[tuple[str, float]]:
        """Edges by the size of their score, largest first."""
        edges = self.graph.edge_index()
        order = sorted(edges, key=lambda e: -abs(self.scores[e]))
        return [(self.graph.edge_name(e), float(self.scores[e])) for e in order[:n]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "clean": self.clean,
            "corrupt": self.corrupt,
            "scores": dict(self.ranked()),
            **dict(self.settings),
        }


def attribute(
    system: Patchable, base: Any, source: Any, metric: Metric, method: str = "auto"
) -> Attribution:
    """Score every edge. ``method``: ``"patch"`` (exact, one step per edge), ``"eap"``
    (first order, needs ``EdgeGradients`` and a ``LinearMetric``), or ``"auto"`` (EAP when
    it can be had)."""
    g = system.graph()
    can_eap = isinstance(system, EdgeGradients) and isinstance(metric, LinearMetric)
    if method == "auto":
        method = "eap" if can_eap else "patch"
    clean, corrupt = endpoints(system, base, source, metric)
    if method == "eap":
        if not can_eap:
            raise RequestError("EAP needs a system with EdgeGradients and a LinearMetric")
        grad = np.asarray(system.edge_gradients(base, source, metric.weights), dtype=np.float64)
        if grad.shape != g.shape:
            raise RequestError(f"edge gradients must be {g.shape}, got {grad.shape}")
        scores = -np.where(g.edges, grad, 0.0)  # mask 1 -> 0 is a step of -1
    elif method == "patch":
        scores = np.zeros(g.shape)
        full = g.mask()
        for e in g.edge_index():
            m = full.copy()
            m[e] = 0.0
            scores[e] = patched(system, base, source, m, metric) - clean
    else:
        raise RequestError(f"method must be 'auto', 'patch' or 'eap', not {method!r}")
    return Attribution(g, scores, method, clean, corrupt, {"metric": _describe(metric)})


@dataclass(frozen=True)
class Circuit:
    """A set of edges, and how well it alone reproduces the base run.

    ``value`` is the metric with every edge outside the circuit patched to the source;
    ``recovery`` puts it on the scale where the base run is 1 and the source 0.
    """

    graph: Graph
    mask: np.ndarray
    value: float
    clean: float
    corrupt: float
    method: str
    settings: Mapping[str, Any] = field(default_factory=dict)
    log: tuple[tuple[str, float], ...] = ()

    @property
    def n_edges(self) -> int:
        return int((self.mask > 0.5).sum())

    @property
    def fraction(self) -> float:
        return self.n_edges / self.graph.n_edges

    @property
    def recovery(self) -> float:
        return recovery(self.value, self.clean, self.corrupt)

    def edges(self) -> list[str]:
        return self.graph.names(self.mask)

    def components(self) -> list[str]:
        """Senders with at least one edge in the circuit."""
        keep = (self.mask > 0.5).any(axis=1)
        return [s for s, k in zip(self.graph.senders, keep, strict=True) if k]

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "n_edges": self.n_edges,
            "of": self.graph.n_edges,
            "value": self.value,
            "clean": self.clean,
            "corrupt": self.corrupt,
            "recovery": self.recovery,
            "edges": self.edges(),
            **dict(self.settings),
        }


def prune(
    system: Patchable,
    base: Any,
    source: Any,
    metric: Metric,
    tau: float = 0.01,
    *,
    absolute: bool = False,
) -> Circuit:
    """Greedy pruning by activation patching (ACDC; Vi-CD with a regression metric).

    Receivers are visited from the output back and, for each, its senders from the latest
    back. An edge is patched to the source and stays patched when the metric moves by less
    than ``tau`` -- a fraction of ``|clean - corrupt|``, or an absolute amount with
    ``absolute=True`` -- from the current circuit's value. One step per edge.
    """
    g = system.graph()
    clean, corrupt = endpoints(system, base, source, metric)
    gap = abs(clean - corrupt)
    if not absolute and gap == 0:
        raise RequestError("the base and the source give the same metric: nothing to explain")
    limit = tau if absolute else tau * gap
    mask = g.mask()
    current = clean
    log = []
    for j in reversed(range(len(g.receivers))):
        for i in reversed(g.upstream(j)):
            mask[i, j] = 0.0
            value = patched(system, base, source, mask, metric)
            if abs(value - current) < limit:
                current = value
                log.append((g.edge_name((i, j)), value))
            else:
                mask[i, j] = 1.0
    settings = {"tau": tau, "absolute": absolute, "metric": _describe(metric)}
    return Circuit(g, mask, current, clean, corrupt, "prune", settings, tuple(log))


def top_k(scores: np.ndarray, graph: Graph, k: int) -> np.ndarray:
    """The mask of the k edges with the largest scores in size (ties broken by edge order)."""
    edges = graph.edge_index()
    order = sorted(range(len(edges)), key=lambda n: (-abs(scores[edges[n]]), n))
    mask = np.zeros(graph.shape)
    for n in order[: max(0, int(k))]:
        mask[edges[n]] = 1.0
    return mask


@dataclass(frozen=True)
class FaithfulnessCurve:
    sizes: tuple[int, ...]
    values: tuple[float, ...]
    clean: float
    corrupt: float
    method: str

    @property
    def recoveries(self) -> tuple[float, ...]:
        return tuple(recovery(v, self.clean, self.corrupt) for v in self.values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "clean": self.clean,
            "corrupt": self.corrupt,
            "sizes": list(self.sizes),
            "values": list(self.values),
            "recoveries": list(self.recoveries),
        }


def faithfulness_curve(
    system: Patchable,
    base: Any,
    source: Any,
    metric: Metric,
    attribution: Attribution,
    sizes: Sequence[int],
) -> FaithfulnessCurve:
    """Keep the top-k edges by ``attribution`` and patch the rest, for each k: Vi-CD's
    sparsity-against-faithfulness comparison of methods."""
    g = attribution.graph
    values = tuple(
        patched(system, base, source, top_k(attribution.scores, g, k), metric) for k in sizes
    )
    return FaithfulnessCurve(
        tuple(int(k) for k in sizes),
        values,
        attribution.clean,
        attribution.corrupt,
        attribution.method,
    )


def _masks(circuits: Sequence[Circuit | np.ndarray]) -> list[np.ndarray]:
    return [np.asarray(c.mask if isinstance(c, Circuit) else c) > 0.5 for c in circuits]


def jaccard(a: Circuit | np.ndarray, b: Circuit | np.ndarray) -> float:
    """Edges in both over edges in either; 1 for two empty circuits."""
    ma, mb = _masks([a, b])
    union = (ma | mb).sum()
    return 1.0 if union == 0 else float((ma & mb).sum() / union)


def edge_frequency(circuits: Sequence[Circuit | np.ndarray]) -> np.ndarray:
    """The fraction of circuits each edge is in."""
    if not circuits:
        raise RequestError("no circuits")
    return np.mean(_masks(circuits), axis=0)


def core(circuits: Sequence[Circuit | np.ndarray], at: float = 0.9) -> np.ndarray:
    """The edges in at least ``at`` of the circuits (Vi-CD's stable core is 90%)."""
    return (edge_frequency(circuits) >= at).astype(np.float64)


def _describe(metric: Metric) -> Any:
    to_dict = getattr(metric, "to_dict", None)
    return to_dict() if callable(to_dict) else getattr(metric, "__name__", type(metric).__name__)
