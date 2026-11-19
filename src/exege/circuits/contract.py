"""What a system must provide for its circuits to be found: ``Patchable``.

Like ``latents.Intervenable``, a patch is a *run*: the adapter owns the model and exege
designs what is swapped. Nothing of the model's framework crosses the protocol -- states are
opaque, fields and masks are numpy -- so the code that searches for circuits runs on numpy
alone and is tested on a toy whose answers are planted (``adapters/toy_transformer.py``).

One step is one forward call: ``step(state)`` returns what it wrote, per field,
``(batch, n_nodes)``. A patched step runs ``state`` (the *base*) with some of the network's
values taken from a second forward call on ``patch.source``, a state of the same batch size:

- ``EdgePatch(source, mask)`` -- edge patching (Vi-CD; ACDC). Every receiver reads
  ``sum_s source_out[s] + mask[s, r] * (base_out[s] - source_out[s])``, where ``base_out``
  is what sender ``s`` wrote *in this patched run* and ``source_out`` what it wrote in a
  plain run on the source. A mask of ones is exactly ``step(state)``, of zeros exactly
  ``step(source)``; a float mask in between is allowed (it is what gradients are taken in).
- ``ComponentPatch(source, components)`` -- activation patching of whole components
  (LAWFUL): the named senders write what they wrote on the source, and everything else is
  computed as usual from there. No components is ``step(state)``; every sender is
  ``step(source)``.

``EdgeGradients`` is optional: the derivative of a linear functional of the output with
respect to the edge mask, at the base (all ones). It is what edge attribution patching
(EAP) needs, and a system that cannot take gradients simply does not offer it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from exege.circuits.graph import Graph
from exege.core import registry
from exege.core.errors import AdapterError
from exege.core.extras import missing_extra
from exege.latents.grid import Grid

try:
    import numpy as np
except ImportError as exc:
    raise missing_extra("numpy", "circuits") from exc


@dataclass(frozen=True)
class EdgePatch:
    """Edges carry the base run's value where ``mask`` is 1 and the source's where 0."""

    source: Any
    mask: np.ndarray


@dataclass(frozen=True)
class ComponentPatch:
    """The named senders write what they wrote on ``source``."""

    source: Any
    components: tuple[str, ...]


Patch = EdgePatch | ComponentPatch


@runtime_checkable
class Patchable(Protocol):
    """A system that can be run one step with parts of its forward pass swapped.

    ``states`` hands out an opaque batch of initial states (one per start, a position or a
    label); ``fields`` reads a state's physical fields, ``(batch, n_nodes)`` per name, and
    ``perturb`` adds ``delta`` to them (``(n_nodes,)`` or ``(batch, n_nodes)`` per name)
    and returns a new state, leaving the old one alone. ``step`` writes ``field_names()``.
    A state is never modified by any method. What cannot be had -- a start, a field, a
    component, a mask of the wrong shape, a source of another batch size -- is a
    ``RequestError``, raised before anything is run.
    """

    def graph(self) -> Graph: ...

    def grid(self) -> Grid: ...

    def field_names(self) -> tuple[str, ...]: ...

    def states(self, starts: Sequence[str | int]) -> Any: ...

    def fields(self, state: Any) -> Mapping[str, np.ndarray]: ...

    def perturb(self, state: Any, delta: Mapping[str, np.ndarray]) -> Any: ...

    def step(self, state: Any, patch: Patch | None = None) -> Mapping[str, np.ndarray]: ...


@runtime_checkable
class EdgeGradients(Protocol):
    """``d/d mask [ mean_b sum_f sum_n weights[f][n] * step(state, EdgePatch(source,
    mask))[f][b, n] ]`` at ``mask = 1``, as ``(n_senders, n_receivers)``, zero off the graph.
    """

    def edge_gradients(
        self, state: Any, source: Any, weights: Mapping[str, np.ndarray]
    ) -> np.ndarray: ...


def open_patchable(source: str | None, adapter: str, **options: Any) -> Patchable:
    """Open a system whose circuits can be searched, through the adapter registry."""
    where = None if source is None else str(source)
    built = registry.create(adapter, source=where, options=options)
    if not isinstance(built, Patchable):
        raise AdapterError(f"adapter {adapter!r} does not implement Patchable")
    return built
