"""LAWFUL on a patchable system: how consistent a model is with a law, and which of its
components carry that consistency.

- ``measure``      M over the family, for the plain model or with components patched
- ``deviations``   LAWFUL's eq. 3 for a set of components C: ``d_suf = M0 - M_suf`` (the
                   perturbation routed through C alone) and ``d_nec = M0 - M_nec`` (through
                   every component but C), with optional source-randomized controls
- ``search``       the incremental greedy search of LAWFUL section 4.1: for each tau in
                   increasing order, add the component that most lowers ``d_suf`` until C is
                   tau-consistent (``d_suf <= (1 - tau) M0`` and ``d_nec >= tau M0``), then drop
                   what this tau added that minimality does not need

Components are the graph's senders other than the input embedding: patching the embedding
would route the whole perturbation, and LAWFUL's vertices are heads and MLPs.

The unpatched consistency ``M0`` is the model run on the perturbed state; a patched run
keeps the base state as input and takes C's outputs from a run on the perturbed one (the
*source*). Either way the law reads the tendency response (see ``exege.laws.law``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from exege.circuits.contract import ComponentPatch, Patchable
from exege.circuits.graph import INPUT
from exege.core.errors import RequestError
from exege.core.extras import missing_extra
from exege.laws.law import Law, consistency

try:
    import numpy as np
except ImportError as exc:
    raise missing_extra("numpy", "laws") from exc


@dataclass(frozen=True)
class LawResult:
    """The family, both sides at each alpha, and M. ``patched`` names what was patched
    (``None``: the plain model), ``mode`` how."""

    law: Mapping[str, Any]
    alphas: tuple[float, ...]
    model: tuple[float, ...]
    expected: tuple[float, ...]
    M: float
    mode: str = "model"
    patched: tuple[str, ...] | None = None
    score: str = "eq10"

    def rows(self) -> list[dict[str, float]]:
        return [
            {"alpha": a, "model": f, "law": g}
            for a, f, g in zip(self.alphas, self.model, self.expected, strict=True)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "law": dict(self.law),
            "mode": self.mode,
            "score": self.score,
            "M": self.M,
            "patched": None if self.patched is None else list(self.patched),
            "rows": self.rows(),
        }


def components(system: Patchable) -> tuple[str, ...]:
    """What LAWFUL patches: every sender but the input embedding."""
    return tuple(s for s in system.graph().senders if s != INPUT)


def _minus(a: Mapping[str, np.ndarray], b: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: np.asarray(a[k], dtype=np.float64) - np.asarray(b[k], dtype=np.float64) for k in a}


def measure(
    system: Patchable,
    base: Any,
    law: Law,
    patched: Sequence[str] | None = None,
    *,
    mode: str = "model",
    source_base: Any = None,
    score: str = "eq10",
) -> LawResult:
    """M of ``system`` against ``law`` on the batch ``base``.

    With ``patched`` None, the model runs on each perturbed state. Otherwise the base state
    runs with the components in ``patched`` taken from a run on the perturbed state -- or,
    for a control, on ``source_base`` perturbed the same way.
    """
    grid = system.grid()
    fields = system.fields(base)
    base_out = system.step(base)
    if patched is not None:
        patched = system.graph().check_components(tuple(patched))
        if INPUT in patched:
            raise RequestError("the input embedding is not a component LAWFUL patches")
    src_root = base if source_base is None else source_base
    model, expected = [], []
    for alpha in law.alphas:
        delta = law.perturbation(grid, fields, alpha)
        if patched is None:
            moved = system.perturb(base, delta)
            out = system.step(moved)
            moved_fields = system.fields(moved)
            names = tuple(out)
            response = _minus(
                _minus({k: out[k] for k in names}, {k: moved_fields[k] for k in names}),
                _minus({k: base_out[k] for k in names}, {k: fields[k] for k in names}),
            )
        else:
            source = system.perturb(src_root, delta)
            response = _minus(system.step(base, ComponentPatch(source, patched)), base_out)
        model.append(float(law.observe(grid, fields, delta, response, alpha)))
        expected.append(float(law.expect(grid, fields, alpha)))
    M = consistency(model, expected, score)
    return LawResult(
        law.to_dict(),
        tuple(law.alphas),
        tuple(model),
        tuple(expected),
        M,
        mode,
        None if patched is None else tuple(patched),
        score,
    )


@dataclass(frozen=True)
class Deviations:
    """LAWFUL's eq. 3 for one set of components, with controls when asked for."""

    components: tuple[str, ...]
    M0: float
    M_suf: float
    M_nec: float
    random_suf: tuple[float, ...] = ()
    random_nec: tuple[float, ...] = ()

    @property
    def d_suf(self) -> float:
        return self.M0 - self.M_suf

    @property
    def d_nec(self) -> float:
        return self.M0 - self.M_nec

    def consistent(self, tau: float) -> bool:
        return self.d_suf <= (1 - tau) * self.M0 and self.d_nec >= tau * self.M0

    def to_dict(self) -> dict[str, Any]:
        d = {
            "components": list(self.components),
            "M0": self.M0,
            "M_suf": self.M_suf,
            "M_nec": self.M_nec,
            "d_suf": self.d_suf,
            "d_nec": self.d_nec,
        }
        if self.random_suf:
            d["d_suf_random"] = [self.M0 - m for m in self.random_suf]
            d["d_nec_random"] = [self.M0 - m for m in self.random_nec]
        return d


def deviations(
    system: Patchable,
    base: Any,
    law: Law,
    comps: Sequence[str],
    *,
    M0: float | None = None,
    controls: Sequence[Any] = (),
    score: str = "eq10",
) -> Deviations:
    """Sufficiency and necessity of ``comps``. Each entry of ``controls`` is a batch of
    other states, of the base's size, whose perturbed runs supply the patched components
    instead: LAWFUL's source-randomized control (pass states that differ from the base's)."""
    every = components(system)
    comps = tuple(comps)
    rest = tuple(c for c in every if c not in comps)
    if M0 is None:
        M0 = measure(system, base, law, score=score).M
    M_suf = measure(system, base, law, comps, mode="sufficiency", score=score).M
    M_nec = measure(system, base, law, rest, mode="necessity", score=score).M
    r_suf = tuple(
        measure(system, base, law, comps, mode="control", source_base=c, score=score).M
        for c in controls
    )
    r_nec = tuple(
        measure(system, base, law, rest, mode="control", source_base=c, score=score).M
        for c in controls
    )
    return Deviations(comps, M0, M_suf, M_nec, r_suf, r_nec)


@dataclass(frozen=True)
class LawCircuit:
    tau: float
    deviations: Deviations
    units: tuple[str, ...] = ()

    @property
    def components(self) -> tuple[str, ...]:
        return self.deviations.components

    def to_dict(self) -> dict[str, Any]:
        d = {"tau": self.tau, "size": len(self.components), **self.deviations.to_dict()}
        if self.units:
            d["units"] = list(self.units)
        return d


@dataclass(frozen=True)
class Search:
    M0: float
    circuits: tuple[LawCircuit, ...]
    of: int
    settings: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "M0": self.M0,
            "of": self.of,
            "circuits": [c.to_dict() for c in self.circuits],
            **dict(self.settings),
        }


def blocks(system: Patchable) -> dict[str, tuple[str, ...]]:
    """The components of each block -- its heads and its MLP -- as search groups: a coarse
    first pass (8 candidates instead of 56) before searching inside the blocks it finds."""
    out: dict[str, list[str]] = {}
    for c in components(system):
        layer = c.split(".")[0][1:]  # a3.h2 -> 3, m3 -> 3
        out.setdefault(f"block{layer}", []).append(c)
    return {k: tuple(v) for k, v in out.items()}


def search(
    system: Patchable,
    base: Any,
    law: Law,
    taus: Sequence[float] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    *,
    candidates: Sequence[str] | None = None,
    groups: Mapping[str, Sequence[str]] | None = None,
    controls: Sequence[Any] = (),
    max_size: int | None = None,
    score: str = "eq10",
) -> Search:
    """LAWFUL's incremental greedy search for tau-physically-consistent circuits.

    Circuits are nested (each tau starts from the last circuit). A tau no circuit within
    ``max_size`` units meets ends the search. The units added are single components
    (``candidates``, default all) or named ``groups`` of them patched together (``blocks``
    gives one per block). Choosing the next unit costs one patched sweep of the family per
    candidate (sufficiency); necessity is measured only for the circuits tested.
    """
    every = components(system)
    if groups is not None:
        units = {name: tuple(cs) for name, cs in groups.items()}
    else:
        units = {c: (c,) for c in (tuple(candidates) if candidates is not None else every)}
    for cs in units.values():
        system.graph().check_components(cs)
        if INPUT in cs:
            raise RequestError("the input embedding is not a component LAWFUL patches")
    M0 = measure(system, base, law, score=score).M
    if M0 <= 0:
        raise RequestError(f"the model is not consistent with the law (M0 = {M0:.3g}): no circuit")
    limit = len(units) if max_size is None else max_size
    m_suf: dict[tuple[str, ...], float] = {}
    m_nec: dict[tuple[str, ...], float] = {}

    def comps_of(us: Sequence[str]) -> tuple[str, ...]:
        chosen = {c for u in us for c in units[u]}
        return tuple(c for c in every if c in chosen)

    def suf(us: Sequence[str]) -> float:
        key = comps_of(us)
        if key not in m_suf:
            m_suf[key] = measure(system, base, law, key, mode="sufficiency", score=score).M
        return m_suf[key]

    def nec(us: Sequence[str]) -> float:
        key = comps_of(us)
        if key not in m_nec:
            rest = tuple(c for c in every if c not in key)
            m_nec[key] = measure(system, base, law, rest, mode="necessity", score=score).M
        return m_nec[key]

    def dev(us: Sequence[str]) -> Deviations:
        return Deviations(comps_of(us), M0, suf(us), nec(us))

    def consistent(us: Sequence[str], tau: float) -> bool:
        # sufficiency first: necessity is only measured when sufficiency already holds
        return M0 - suf(us) <= (1 - tau) * M0 and dev(us).consistent(tau)

    circuit: list[str] = []
    found = []
    for tau in sorted(taus):
        added: list[str] = []
        while not consistent(circuit, tau) and len(circuit) < limit:
            options = [u for u in units if u not in circuit]
            if not options:
                break
            best = max(options, key=lambda u: suf([*circuit, u]))
            circuit.append(best)
            added.append(best)
        if not consistent(circuit, tau):
            break
        for u in reversed(added):  # minimal among supersets of the previous circuit
            trial = [x for x in circuit if x != u]
            if consistent(trial, tau):
                circuit = trial
        final = dev(circuit)
        if controls:
            final = deviations(
                system, base, law, final.components, M0=M0, controls=controls, score=score
            )
        found.append(LawCircuit(float(tau), final, tuple(circuit) if groups is not None else ()))
    settings = {"law": law.to_dict(), "taus": list(taus), "score": score}
    if groups is not None:
        settings["groups"] = {k: list(v) for k, v in units.items()}
    return Search(M0, tuple(found), len(units), settings)
