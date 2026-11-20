"""The computational graph a circuit is a part of.

A residual network is read as senders that *write* into the residual stream (the input
embedding, every attention head, every MLP) and receivers that *read* it (each block's
attention input, its MLP input, the output). An edge ``(sender, receiver)`` exists when the
sender is upstream of the receiver, and carries that sender's additive contribution to
what the receiver reads (Elhage et al. 2021). A circuit is a subset of the edges.

``residual_graph`` builds the reduced graph of Vi-CD (Zukowska et al. 2026): heads stay
separate senders, but the heads of one block share a single receiver, its attention input,
because their queries, keys and values all read the same normalized stream. That divides
the number of edges by about the number of heads and loses no dependency.

Masks are ``(n_senders, n_receivers)`` arrays. In a mask, 1 is "this edge carries the
value of the run being patched" and 0 is "it carries the value of the source run";
entries off the graph mean nothing and are kept at 0.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from exege.core.errors import RequestError
from exege.core.extras import missing_extra

try:
    import numpy as np
except ImportError as exc:
    raise missing_extra("numpy", "circuits") from exc

INPUT, OUTPUT = "embed", "out"


@dataclass(frozen=True, eq=False)
class Graph:
    """Senders, receivers, and which pairs are edges, in topological order.

    ``edges[i, j]`` is True when sender ``i`` feeds receiver ``j``. Senders are listed in
    the order they are computed and receivers likewise, so walking receivers backwards
    visits the output first (the order greedy pruning uses).
    """

    senders: tuple[str, ...]
    receivers: tuple[str, ...]
    edges: np.ndarray

    def __post_init__(self) -> None:
        senders, receivers = tuple(self.senders), tuple(self.receivers)
        edges = np.array(self.edges, dtype=bool)
        if edges.shape != (len(senders), len(receivers)):
            raise ValueError(f"edges must be {(len(senders), len(receivers))}, got {edges.shape}")
        for kind, names in (("sender", senders), ("receiver", receivers)):
            if len(set(names)) != len(names):
                raise ValueError(f"{kind} names must be unique")
        edges.setflags(write=False)
        object.__setattr__(self, "senders", senders)
        object.__setattr__(self, "receivers", receivers)
        object.__setattr__(self, "edges", edges)

    @property
    def n_edges(self) -> int:
        return int(self.edges.sum())

    @property
    def shape(self) -> tuple[int, int]:
        return self.edges.shape

    def sender(self, name: str) -> int:
        try:
            return self.senders.index(name)
        except ValueError:
            raise RequestError(f"no sender {name!r}; senders are {_some(self.senders)}") from None

    def receiver(self, name: str) -> int:
        try:
            return self.receivers.index(name)
        except ValueError:
            raise RequestError(
                f"no receiver {name!r}; receivers are {_some(self.receivers)}"
            ) from None

    def upstream(self, receiver: str | int) -> list[int]:
        """Indices of the senders feeding a receiver, in computation order."""
        j = receiver if isinstance(receiver, int) else self.receiver(receiver)
        return [int(i) for i in np.flatnonzero(self.edges[:, j])]

    def edge_index(self) -> list[tuple[int, int]]:
        """Every edge as ``(sender, receiver)`` indices, receivers in order."""
        return [(int(i), int(j)) for j, i in zip(*np.nonzero(self.edges.T), strict=True)]

    def edge_name(self, edge: tuple[int, int]) -> str:
        return f"{self.senders[edge[0]]}->{self.receivers[edge[1]]}"

    def names(self, mask: np.ndarray | None = None) -> list[str]:
        """Names of the edges a mask keeps (all edges without one)."""
        keep = self.edges if mask is None else self.check_mask(mask) > 0.5
        return [self.edge_name(e) for e in self.edge_index() if keep[e]]

    def mask(self, fill: float = 1.0) -> np.ndarray:
        """A float mask holding ``fill`` on every edge and 0 off the graph."""
        return self.edges.astype(np.float64) * float(fill)

    def mask_of(self, edges: Iterable[str]) -> np.ndarray:
        """The 0/1 mask keeping the named ``"sender->receiver"`` edges."""
        m = np.zeros(self.shape)
        for name in edges:
            s, sep, r = name.partition("->")
            if not sep:
                raise RequestError(f"an edge is named 'sender->receiver', not {name!r}")
            i, j = self.sender(s), self.receiver(r)
            if not self.edges[i, j]:
                raise RequestError(f"{name} is not an edge of this graph")
            m[i, j] = 1.0
        return m

    def check_mask(self, mask: Any) -> np.ndarray:
        m = np.asarray(mask, dtype=np.float64)
        if m.shape != self.shape:
            raise RequestError(f"a mask must be {self.shape} (senders, receivers), got {m.shape}")
        if not np.isfinite(m).all():
            raise RequestError("a mask must be finite")
        return np.where(self.edges, m, 0.0)

    def check_components(self, components: Sequence[str]) -> tuple[str, ...]:
        for c in components:
            self.sender(c)
        return tuple(components)

    def to_dict(self) -> dict[str, Any]:
        return {
            "senders": list(self.senders),
            "receivers": list(self.receivers),
            "edges": [self.edge_name(e) for e in self.edge_index()],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Graph:
        senders, receivers = tuple(d["senders"]), tuple(d["receivers"])
        edges = np.zeros((len(senders), len(receivers)), dtype=bool)
        for name in d["edges"]:
            s, _, r = name.partition("->")
            if s not in senders or r not in receivers:
                raise RequestError(f"edge {name!r} names a node the graph does not have")
            edges[senders.index(s), receivers.index(r)] = True
        return cls(senders, receivers, edges)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Graph)
            and self.senders == other.senders
            and self.receivers == other.receivers
            and bool(np.array_equal(self.edges, other.edges))
        )

    def __hash__(self) -> int:
        return hash((self.senders, self.receivers, self.edges.tobytes()))


def head(layer: int, h: int) -> str:
    return f"a{layer}.h{h}"


def mlp(layer: int) -> str:
    return f"m{layer}"


def attn_in(layer: int) -> str:
    return f"a{layer}"


def residual_graph(depth: int, heads: int, *, mlps: bool = True) -> Graph:
    """Vi-CD's reduced graph of a pre-norm residual network.

    Senders: ``embed``, then per block ``a{i}.h{h}`` for each head and ``m{i}``. Receivers:
    per block ``a{i}`` (the attention input, shared by its heads) and ``m{i}``, then
    ``out``. A block's MLP reads its own heads; its attention reads only earlier blocks.
    """
    if depth < 1 or heads < 1:
        raise RequestError("a residual graph needs at least one block and one head")
    senders = [INPUT]
    receivers: list[str] = []
    feeds: list[int] = []  # how many senders precede each receiver
    for i in range(depth):
        receivers.append(attn_in(i))
        feeds.append(len(senders))
        senders += [head(i, h) for h in range(heads)]
        if mlps:
            receivers.append(mlp(i))
            feeds.append(len(senders))
            senders.append(mlp(i))
    receivers.append(OUTPUT)
    feeds.append(len(senders))
    edges = np.zeros((len(senders), len(receivers)), dtype=bool)
    for j, n in enumerate(feeds):
        edges[:n, j] = True
    return Graph(tuple(senders), tuple(receivers), edges)


def _some(names: Sequence[str], n: int = 6) -> str:
    shown = ", ".join(names[:n])
    return shown + (f", ... ({len(names)} in all)" if len(names) > n else "")
