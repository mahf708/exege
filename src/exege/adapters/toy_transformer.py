"""A toy residual network whose circuits are planted. For tests and docs.

Not a model of anything: two blocks of two attention heads and an MLP over a small lat-lon
grid, with fixed attention patterns and no normalization, so what every edge carries can
be worked out by hand. Fields ``tracer`` and ``ripple``; residual channels::

    c0  tracer          written by embed
    c1  ripple          written by embed, and by m1
    c2  ripple, copied  a0.h0 (pattern: itself)
    c3  a distractor    a0.h1 (pattern: the global mean of c0), m0 (relu of c2), a1.h1
                        (pattern: the eastern neighbour of c1)
    c4  tracer, west    a1.h0 (pattern: the western neighbour, reading c0)

    tracer' = (1 - courant) * c0 + courant * c4      upwind advection, eastward
    ripple' = c1 + 0.1 * c3

So a metric of ``tracer'`` runs through exactly three edges -- ``embed->a1`` (a1.h0 reads
c0), ``a1.h0->out`` and ``embed->out`` -- and nothing else; a metric of ``ripple'`` runs
through most of the graph. A planted wave ``cos(s * lon)`` in the tracer moves east by
``arg(1 - courant + courant * exp(-i s dlon))`` per step: the discrete advection law.

A state is a batch of fields drawn from a seed and the start (positions only), and reads
no source, so the options are keyword-only. Edge gradients are central differences: the
network is piecewise linear, so they are exact away from a ReLU's kink.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from exege.circuits.contract import ComponentPatch, EdgePatch, Patch
from exege.circuits.graph import Graph, residual_graph
from exege.core.errors import RequestError
from exege.core.extras import missing_extra
from exege.latents.grid import Grid

try:
    import numpy as np
except ImportError as exc:
    raise missing_extra("numpy", "circuits") from exc

FIELDS = ("tracer", "ripple")
WIDTH = 5
PLANTED = ("embed->a1", "a1.h0->out", "embed->out")  # the circuit of any tracer metric


@dataclass(frozen=True)
class ToyState:
    """``x[b, node, field]``, read-only. Opaque to exege."""

    x: np.ndarray


class ToyTransformer:
    """``Patchable`` and ``EdgeGradients``."""

    def __init__(
        self,
        *,
        n_lat: int = 3,
        n_lon: int = 12,
        n_times: int = 16,
        courant: float = 0.5,
        seed: int = 0,
    ) -> None:
        if n_lat < 1 or n_lon < 3:
            raise RequestError("the toy needs at least 1 latitude and 3 longitudes")
        self.n_lat, self.n_lon, self.n_times = n_lat, n_lon, n_times
        self.courant, self.seed = float(courant), seed
        lat = np.linspace(-60, 60, n_lat) if n_lat > 1 else np.zeros(1)
        lon = np.arange(n_lon) * 360.0 / n_lon
        LA, LO = np.meshgrid(lat, lon, indexing="ij")
        self._grid = Grid(lat=LA.ravel(), lon=LO.ravel(), shape=(n_lat, n_lon))
        self._graph = residual_graph(2, 2)
        n = n_lat * n_lon
        idx = np.arange(n).reshape(n_lat, n_lon)
        west = np.zeros((n, n))
        east = np.zeros((n, n))
        west[idx.ravel(), np.roll(idx, 1, axis=1).ravel()] = 1.0
        east[idx.ravel(), np.roll(idx, -1, axis=1).ravel()] = 1.0
        e = np.eye(WIDTH)
        # (pattern, reads, writes) per head, in graph order
        self._heads = {
            "a0.h0": (np.eye(n), e[1], e[2]),
            "a0.h1": (np.full((n, n), 1.0 / n), e[0], e[3]),
            "a1.h0": (west, e[0], e[4]),
            "a1.h1": (east, e[1], e[3]),
        }
        self._mlps = {"m0": (e[2], e[3] * 0.5), "m1": (e[3], e[1] * 0.2)}  # relu(x.r) * w

    # -- the contract --

    def graph(self) -> Graph:
        return self._graph

    def grid(self) -> Grid:
        return self._grid

    def field_names(self) -> tuple[str, ...]:
        return FIELDS

    def states(self, starts: Sequence[str | int]) -> ToyState:
        xs = []
        for s in starts:
            if isinstance(s, str) or not 0 <= int(s) < self.n_times:
                raise RequestError(
                    f"the toy's starts are positions 0..{self.n_times - 1}, not {s!r}"
                )
            rng = np.random.default_rng([self.seed, int(s)])
            xs.append(rng.normal(size=(self._grid.n_nodes, len(FIELDS))))
        if not xs:
            raise RequestError("no starts")
        return _frozen(np.stack(xs))

    def fields(self, state: ToyState) -> dict[str, np.ndarray]:
        return {f: state.x[:, :, k].copy() for k, f in enumerate(FIELDS)}

    def perturb(self, state: ToyState, delta: Mapping[str, np.ndarray]) -> ToyState:
        x = state.x.copy()
        for name, d in delta.items():
            if name not in FIELDS:
                raise RequestError(f"the toy has no field {name!r}; it has {', '.join(FIELDS)}")
            d = np.asarray(d, dtype=np.float64)
            if d.shape not in ((self._grid.n_nodes,), x.shape[:2]):
                raise RequestError(f"a delta is (n_nodes,) or (batch, n_nodes), got {d.shape}")
            x[:, :, FIELDS.index(name)] += d
        return _frozen(x)

    def step(self, state: ToyState, patch: Patch | None = None) -> dict[str, np.ndarray]:
        out = self._forward(state, patch)
        return {
            "tracer": (1 - self.courant) * out[..., 0] + self.courant * out[..., 4],
            "ripple": out[..., 1] + 0.1 * out[..., 3],
        }

    def edge_gradients(self, state: ToyState, source: ToyState, weights) -> np.ndarray:
        g = self._graph
        grad = np.zeros(g.shape)
        eps = 1e-6

        def value(mask):
            out = self.step(state, EdgePatch(source, mask))
            return sum(
                float((np.asarray(out[f]) @ np.asarray(w)).mean()) for f, w in weights.items()
            )

        for e in g.edge_index():
            up, down = g.mask(), g.mask()
            up[e] += eps
            down[e] -= eps
            grad[e] = (value(up) - value(down)) / (2 * eps)
        return grad

    # -- the network --

    def _senders(self, x: np.ndarray, receive, swap=None) -> np.ndarray:
        """Run the blocks; ``receive(j, outs)`` builds receiver j's input from the sender
        outputs so far; ``swap`` maps a sender index to the output it is to write."""
        g = self._graph
        outs: list[np.ndarray] = []

        def emit(value):
            i = len(outs)
            outs.append(swap[i] if swap is not None and i in swap else value)

        emit(np.concatenate([x, np.zeros(x.shape[:2] + (WIDTH - x.shape[2],))], axis=2))
        for layer in range(2):
            a_in = receive(g.receiver(f"a{layer}"), outs)
            for h in range(2):
                pattern, read, write = self._heads[f"a{layer}.h{h}"]
                emit(np.einsum("mn,bn->bm", pattern, a_in @ read)[..., None] * write)
            m_in = receive(g.receiver(f"m{layer}"), outs)
            read, write = self._mlps[f"m{layer}"]
            emit(np.maximum(m_in @ read, 0.0)[..., None] * write)
        return np.stack(outs), receive(g.receiver("out"), outs)

    def _forward(self, state: ToyState, patch: Patch | None) -> np.ndarray:
        g = self._graph

        def plain(j, outs):
            return sum(outs[i] for i in g.upstream(j))

        if patch is None:
            return self._senders(state.x, plain)[1]
        if not isinstance(patch.source, ToyState) or patch.source.x.shape != state.x.shape:
            raise RequestError("a patch's source must be a state of the same batch size")
        src = self._senders(patch.source.x, plain)[0]
        if isinstance(patch, EdgePatch):
            m = g.check_mask(patch.mask)

            def mixed(j, outs):
                return sum(src[i] + m[i, j] * (outs[i] - src[i]) for i in g.upstream(j))

            return self._senders(state.x, mixed)[1]
        if isinstance(patch, ComponentPatch):
            names = g.check_components(patch.components)
            swap = {g.sender(c): src[g.sender(c)] for c in names}
            return self._senders(state.x, plain, swap)[1]
        raise RequestError(f"unknown patch {type(patch).__name__}")


def _frozen(x: np.ndarray) -> ToyState:
    x = np.array(x, dtype=np.float64)
    x.setflags(write=False)
    return ToyState(x)
