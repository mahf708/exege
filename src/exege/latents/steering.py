"""Steering: change a feature inside a running model and see what it does.

Everything else in this package reads. Steering is the first time an adapter
*writes*: the model is run, and at one layer and time the latents it produced are
replaced by an edited copy, after which the forward pass continues.
``docs/steering.md`` is the design note; in short, arms are run from one initial state
with one noise seed, and compared pairwise with the first:

- ``control``         no hooks;
- ``reconstruction``  the latents are replaced by ``decode(encode(h))``: what passing
                      through the dictionary costs, with no feature touched;
- ``feature``         ``h + delta * direction`` for one feature of the basis;
- ``random``          the same ``delta`` along a random unit direction scaled to the
                      feature direction's length, drawn ``n_random`` times;
- ``comparison``      the same ``delta`` along a named direction from another basis (a
                      difference of means, a principal component), scaled the same way;
- ``matched``         the same steer applied to other features of the basis, chosen to
                      carry about as much as the feature does where the edit acts: real,
                      in-distribution directions, where a random one may not be.

The feature's response on each physical field is then set against the random draws as an
effect size and an empirical rank, and against the matched features by rank. Seeds repeat
the whole thing under other noise for an uncertainty; ``run_cases`` repeats it from other
initial states, and ``dose_curve`` over amounts, with any amount chosen leave-one-out.
What a system must provide is ``Intervenable``; the experiment itself knows no framework,
takes a ``Decomposition`` and returns an object. Needs the ``latents`` extra.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

from exege.core import registry
from exege.core.errors import AdapterError, RequestError
from exege.core.extras import missing_extra
from exege.latents.analysis import AnyRegion
from exege.latents.basis import Decomposition, Dictionary, basis_provenance, result_provenance
from exege.latents.grid import Grid, node_digest
from exege.latents.source import LatentInfo, check_basis_fits

try:
    import numpy as np
except ImportError as exc:
    raise missing_extra("numpy", "latents") from exc

MODES = ("add", "scale", "clamp", "toward", "excess")
REFERENCE_MODES = ("toward", "excess")
LOOPS = ("open", "closed")
STATISTICS = ("mean", "min", "max")
CONTROL, RECONSTRUCTION, FEATURE, RANDOM = "control", "reconstruction", "feature", "random"
COMPARISON, MATCHED = "comparison", "matched"
_BLOCK = 8192


# -- the contract -------------------------------------------------------------


@dataclass(frozen=True)
class Hook:
    """A place inside the forward pass where the latents may be replaced.

    ``edit`` receives the ``(n_nodes, n_channels)`` tensor the model produced at
    ``layer`` during forward step ``time`` (counted from 0 in a run) and returns what
    the pass continues with, of the same shape.
    """

    layer: int
    time: int
    edit: Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Rollout:
    """What a run produced.

    ``fields[name]`` is ``(steps, n_nodes)``: what forward step ``t`` wrote, NaN where
    the field means nothing (the ``lead=1`` convention of ``ReferenceFields``).
    ``latents[(layer, time)]`` is the tensor at each place in ``record``, taken *after*
    any hook there: what the pass actually continued with.
    """

    fields: Mapping[str, np.ndarray]
    latents: Mapping[tuple[int, int], np.ndarray] = field(default_factory=dict)


@runtime_checkable
class Intervenable(Protocol):
    """A system that can be run forward with its latents edited on the way.

    An optional capability of an adapter, asked for with ``isinstance`` and separate from
    ``LatentSource``: reading a recording is cheap and repeatable, running a model is
    neither, and most sources cannot. Nothing of the model's framework crosses it:
    hooks take and return numpy arrays, and the initial state is whatever
    ``initial_state`` handed out, opaque to exege.

    ``run`` must be deterministic given ``noise_seed``, and the noise it draws must not
    depend on what the hooks did: that is what makes arms comparable pairwise. Without a
    seed the run has no noise. A hook at a layer or time that does not exist is a
    ``RequestError``.
    """

    def info(self) -> LatentInfo: ...

    def grid(self) -> Grid: ...

    def initial_state(self, start: str | int = 0) -> Any: ...

    def run(
        self,
        initial_state: Any,
        steps: int,
        hooks: Sequence[Hook] = (),
        *,
        noise_seed: int | None = None,
        record: Sequence[tuple[int, int]] = (),
    ) -> Rollout: ...


def open_intervenable(source: str | None, adapter: str, **options: Any) -> Intervenable:
    """Open a system that can be run, through the adapter registry. ``source`` is None
    for one that reads nothing."""
    where = None if source is None else str(source)
    built = registry.create(adapter, source=where, options=options)
    if not isinstance(built, Intervenable):
        raise AdapterError(f"adapter {adapter!r} does not implement Intervenable")
    return built


# -- what is done -------------------------------------------------------------


def _scale_of(basis: Decomposition) -> float:
    """What a basis's decode multiplies its directions by: 1, except for a dictionary
    fitted on standardized latents."""
    if isinstance(basis, Dictionary):
        return float(basis.input_scale if basis.output_scale is None else basis.output_scale)
    return 1.0


def feature_direction(basis: Decomposition, feature: int) -> np.ndarray:
    """The direction of ``feature`` in the units of the layer the basis reads, float64.

    A dictionary's decoder row is in standardized units, so it is multiplied by the scale
    its decode applies. A transcoder writes another layer: its directions are not
    directions of the layer it reads, and cannot steer it.
    """
    if not 0 <= feature < basis.n_features:
        raise RequestError(f"feature {feature} is not in 0..{basis.n_features - 1}")
    directions = np.asarray(basis.directions(), dtype=np.float64)
    if directions.shape[1] != basis.n_channels:
        raise RequestError(
            f"this basis reads {basis.n_channels} channel(s) and writes {directions.shape[1]}: "
            "a transcoder's directions belong to another layer, and cannot steer this one"
        )
    direction = directions[feature]
    if not np.isfinite(direction).all():
        raise RequestError(f"the direction of feature {feature} holds values that are not finite")
    return direction * _scale_of(basis)


def array_digest(values: np.ndarray) -> str:
    """SHA-256 of an array's dtype, shape and bytes: names a reference without holding it."""
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256(f"{array.dtype.str}|{array.shape}|".encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class Steer:
    """One feature, changed at one layer, at some forward steps.

    ``mode`` says what ``amount`` does to the feature's activation ``f`` at each node:

    - ``add``     ``f + amount``;
    - ``scale``   ``amount * f``;
    - ``clamp``   ``amount``, whatever ``f`` was;
    - ``toward``  ``f + amount * (reference - f)``: 1 sets the activation to the reference
                  (the mean-ablation and resample-ablation of the literature, depending
                  on what the reference is), 0 leaves it;
    - ``excess``  ``f + (amount - 1) * max(f - reference, 0)``: only what the activation
                  holds above the reference is scaled, so 0 caps it at the reference
                  ("as it is on a quiet day"), 1 leaves it and 2 doubles the excess.

    ``reference`` is one value per node (``feature_reference`` reads one from an
    archive) and ``reference_meta`` says where it came from; only ``toward`` and
    ``excess`` take one. Each mode is, node by node, a change ``delta`` in the
    activation, and the layer moves by ``delta * direction``. ``nodes`` restricts the
    edit to those nodes (None: every node).

    ``loop`` says what ``delta`` is worked out from. ``open`` (the default): the
    control's latents, so every arm of a seed changes the same nodes by the same amounts
    however the runs drift apart -- a forcing applied from outside. ``closed``: the
    latents the arm itself produced, at the moment of the edit, so an edit acting at
    several steps reads what its earlier edits did -- the way steering inside a model's
    own forward pass works, and the way an amplification compounds. For ``add`` the two
    agree.
    """

    layer: int
    feature: int
    mode: str = "add"
    amount: float = 1.0
    times: tuple[int, ...] = (0,)
    nodes: tuple[int, ...] | None = None
    loop: str = "open"
    reference: np.ndarray | None = field(default=None, compare=False, repr=False)
    reference_meta: Mapping[str, Any] | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise RequestError(f"mode must be one of {', '.join(MODES)}, not {self.mode!r}")
        if not np.isfinite(self.amount):
            raise RequestError(f"the amount must be a finite number, not {self.amount!r}")
        if not self.times:
            raise RequestError("a steer needs at least one time to act at")
        if self.nodes is not None and not self.nodes:
            raise RequestError("a steer restricted to no nodes changes nothing")
        if self.loop not in LOOPS:
            raise RequestError(f"loop must be one of {', '.join(LOOPS)}, not {self.loop!r}")
        if self.mode in REFERENCE_MODES and self.reference is None:
            raise RequestError(f"mode {self.mode!r} needs a reference: one value per node")
        if self.mode not in REFERENCE_MODES and self.reference is not None:
            raise RequestError(f"mode {self.mode!r} takes no reference; toward and excess do")
        if self.reference is not None:
            reference = np.asarray(self.reference, dtype=np.float64)
            if reference.ndim != 1:
                raise RequestError(f"a reference is one value per node, not {reference.shape}")
            object.__setattr__(self, "reference", reference)

    @property
    def identity_amount(self) -> float | None:
        """The amount at which this mode changes nothing (None for ``clamp``)."""
        return {"add": 0.0, "scale": 1.0, "toward": 0.0, "excess": 1.0}.get(self.mode)

    def deltas(self, basis: Decomposition, latents: np.ndarray) -> np.ndarray:
        """The change in the feature's activation at each node, ``(n_nodes,)`` float64."""
        n_nodes = latents.shape[0]
        if self.reference is not None and self.reference.shape != (n_nodes,):
            raise RequestError(
                f"the reference holds {self.reference.size} value(s) for {n_nodes} node(s)"
            )
        if self.mode == "add":
            delta = np.full(n_nodes, self.amount, dtype=np.float64)
        else:
            active = basis.transform(latents, [self.feature])[:, 0]
            if self.mode == "scale":
                delta = (self.amount - 1.0) * active
            elif self.mode == "clamp":
                delta = self.amount - active
            elif self.mode == "toward":
                delta = self.amount * (self.reference - active)
            else:
                delta = (self.amount - 1.0) * np.maximum(active - self.reference, 0.0)
        if self.nodes is not None:
            kept = np.zeros(delta.shape, dtype=bool)
            kept[list(self.nodes)] = True
            delta = np.where(kept, delta, 0.0)
        return delta

    def to_dict(self) -> dict[str, Any]:
        reference = None
        if self.reference is not None:
            finite = self.reference[np.isfinite(self.reference)]
            reference = {
                **dict(self.reference_meta or {}), "sha256": array_digest(self.reference),
                "mean": float(finite.mean()) if finite.size else None,
            }  # fmt: skip
        return {
            "layer": self.layer, "feature": self.feature, "mode": self.mode,
            "amount": self.amount, "times": list(self.times),
            "nodes": None if self.nodes is None else list(self.nodes), "loop": self.loop,
            "reference": reference,
        }  # fmt: skip


def _region_dict(region: AnyRegion | None) -> dict[str, Any] | None:
    return None if region is None else {"kind": type(region).__name__.lower(), **asdict(region)}


@dataclass(frozen=True)
class Outcome:
    """How a field a run wrote is reduced to one number per step: its area-weighted
    ``mean``, or its ``min`` or ``max``, over the valid nodes of ``region`` (None: the
    globe). A global mean dilutes an edit that acts in one place to nothing; the minimum
    pressure in a box around a storm does not."""

    statistic: str = "mean"
    region: AnyRegion | None = None

    def __post_init__(self) -> None:
        if self.statistic not in STATISTICS:
            raise RequestError(
                f"an outcome is one of {', '.join(STATISTICS)}, not {self.statistic!r}"
            )

    def nodes(self, grid: Grid) -> np.ndarray:
        nodes = np.flatnonzero(grid.valid) if self.region is None else self.region.nodes(grid)
        if nodes.size == 0:
            where = "on the grid" if self.region is None else self.region.describe()
            raise RequestError(f"no valid nodes {where} to measure an outcome")
        return nodes

    def to_dict(self) -> dict[str, Any]:
        return {"statistic": self.statistic, "region": _region_dict(self.region)}


# -- the result ---------------------------------------------------------------


@dataclass(frozen=True)
class Run:
    """One arm under one noise seed. ``outcomes[name]`` is a field reduced to one number
    per step (by default its area-weighted global mean; see ``Outcome``);
    ``latent_rms[(layer, time)]`` the area-weighted RMS difference of the latents from
    the control's, there (zero for the control). ``label`` names a comparison arm's
    direction or a matched arm's feature."""

    arm: str
    seed: int
    draw: int | None
    outcomes: Mapping[str, np.ndarray]
    latent_rms: Mapping[tuple[int, int], float]
    label: str | None = None


@dataclass(frozen=True)
class Pairing:
    """One run against the control of the same seed: ``differences[name]`` is the series
    of the outcome minus the control's, and ``response[name]`` its mean over the steps
    from the first edit on."""

    arm: str
    seed: int
    draw: int | None
    differences: Mapping[str, np.ndarray]
    response: Mapping[str, float]
    label: str | None = None


@dataclass(frozen=True)
class Effect:
    """The feature's response on one field against what other edits of its size do.

    ``feature_response`` and ``reconstruction_response`` are signed means over seeds of
    the paired response (``feature_se``: the standard error across seeds, None for one).
    ``random_responses`` holds one seed-mean per draw, and everything compared uses
    magnitudes. ``rank`` is the feature's place among itself and the draws (1 is a larger
    response than every draw), ``p_value`` the share of the ``n + 1`` that match or beat it, and
    ``effect_size`` how many standard deviations of the draws' magnitudes it sits above
    their mean (None for fewer than two draws, or draws that do not vary).

    ``comparison_responses`` is the seed-mean response of each named comparison
    direction. ``matched_responses`` is one seed-mean per matched feature (in the order of
    ``matched_features``), and ``matched_rank`` the feature's place by magnitude among
    itself and them (None without matched features).
    """

    field: str
    feature_response: float
    feature_se: float | None
    reconstruction_response: float
    random_responses: np.ndarray
    effect_size: float | None
    rank: int
    p_value: float
    comparison_responses: Mapping[str, float] = field(default_factory=dict)
    matched_features: tuple[int, ...] = ()
    matched_responses: np.ndarray | None = None
    matched_rank: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field, "feature_response": self.feature_response,
            "feature_se": self.feature_se,
            "reconstruction_response": self.reconstruction_response,
            "random_responses": self.random_responses, "effect_size": self.effect_size,
            "random_mean_magnitude": float(np.abs(self.random_responses).mean()),
            "rank": self.rank, "rank_ties": "last", "n_draws": int(self.random_responses.size),
            "p_value": self.p_value,
            "comparison_responses": dict(self.comparison_responses),
            "matched_features": list(self.matched_features),
            "matched_responses": self.matched_responses, "matched_rank": self.matched_rank,
        }  # fmt: skip


@dataclass(frozen=True)
class SteeringResult:
    """Every run, every pairing against its control, and the summary, with the settings
    and the provenance of the system and the basis that produced them."""

    spec: Mapping[str, Any]
    provenance: Mapping[str, Any]
    runs: tuple[Run, ...]
    pairings: tuple[Pairing, ...]
    effects: Mapping[str, Effect]

    def arm(self, arm: str, seed: int, draw: int | None = None, label: str | None = None) -> Run:
        for run in self.runs:
            if (run.arm, run.seed, run.draw) == (arm, seed, draw) and label in (None, run.label):
                return run
        raise RequestError(f"no run of arm {arm!r} with seed {seed} and draw {draw}")

    def pairing(
        self, arm: str, seed: int, draw: int | None = None, label: str | None = None
    ) -> Pairing:
        for pairing in self.pairings:
            if (pairing.arm, pairing.seed, pairing.draw) == (arm, seed, draw) and label in (
                None,
                pairing.label,
            ):
                return pairing
        raise RequestError(f"no pairing of arm {arm!r} with seed {seed} and draw {draw}")

    def to_dict(self) -> dict[str, Any]:
        def place(key: tuple[int, int]) -> str:
            return f"{key[0]}:{key[1]}"

        return {
            "spec": dict(self.spec),
            "provenance": dict(self.provenance),
            "effects": {name: e.to_dict() for name, e in self.effects.items()},
            "runs": [
                {
                    "arm": r.arm, "seed": r.seed, "draw": r.draw, "label": r.label,
                    "outcomes": dict(r.outcomes),
                    "latent_rms": {place(k): v for k, v in r.latent_rms.items()},
                }
                for r in self.runs
            ],
            "pairings": [
                {
                    "arm": p.arm, "seed": p.seed, "draw": p.draw, "label": p.label,
                    "response": dict(p.response), "differences": dict(p.differences),
                }
                for p in self.pairings
            ],
        }  # fmt: skip


# -- running ------------------------------------------------------------------


def _guarded(hook: Hook, valid: np.ndarray) -> Hook:
    """``hook`` with its edit held to the contract: the shape it was given, masked nodes
    left as they were, and numbers wherever a node is valid. A NaN let through here would
    surface, many steps later and far from its cause, as a NaN in a field."""

    def checked(latents: np.ndarray) -> np.ndarray:
        out = np.asarray(hook.edit(latents))
        if out.shape != latents.shape:
            raise RequestError(
                f"the edit at layer {hook.layer}, time {hook.time} returned {out.shape} "
                f"for {latents.shape}"
            )
        out = np.where(valid[:, None], out, latents)
        if not np.isfinite(out[valid]).all():
            raise RequestError(
                f"the edit at layer {hook.layer}, time {hook.time} produced values that are "
                "not finite on valid nodes"
            )
        return out

    return Hook(hook.layer, hook.time, checked)


def _reduce(
    fields: Mapping[str, np.ndarray],
    names: Sequence[str],
    grid: Grid,
    steps: int,
    outcome: Outcome,
) -> dict[str, np.ndarray]:
    """Each field reduced to one number per step, as ``outcome`` says. A valid node that
    holds NaN or infinity is a ``RequestError``, as in ``read_latents``."""
    valid = grid.valid
    weights = grid.weights()  # raises where the mask leaves no node
    nodes = outcome.nodes(grid)
    share = weights[nodes] / weights[nodes].sum()
    out = {}
    for name in names:
        if name not in fields:
            raise RequestError(
                f"the system wrote no field {name!r}; it wrote {', '.join(sorted(fields))}"
            )
        values = np.asarray(fields[name], dtype=np.float64)
        if values.shape != (steps, grid.n_nodes):
            raise RequestError(
                f"field {name!r} has shape {values.shape}, expected {(steps, grid.n_nodes)}"
            )
        bad = np.flatnonzero(~np.isfinite(values[:, valid]).all(axis=1))
        if bad.size:
            raise RequestError(
                f"field {name!r} is not finite on valid nodes at step {int(bad[0])}; a node the "
                "system cannot supply belongs in its mask, not left NaN"
            )
        chosen = values[:, nodes]
        if outcome.statistic == "mean":
            out[name] = chosen @ share
        else:
            out[name] = chosen.min(axis=1) if outcome.statistic == "min" else chosen.max(axis=1)
    return out


def _rms(difference: np.ndarray, grid: Grid) -> float:
    valid = grid.valid
    squares = (difference[valid].astype(np.float64) ** 2).mean(axis=1)
    return float(np.sqrt(squares @ grid.weights()[valid]))


def _random_unit(n_channels: int, seed: int, draw: int) -> np.ndarray:
    vector = np.random.default_rng([seed, draw]).normal(size=n_channels)
    return vector / np.linalg.norm(vector)


@dataclass(frozen=True)
class _Plan:
    """What every arm of an experiment shares, worked out and checked once, before any
    run: the system, the steer, the places recorded, and how outcomes are measured."""

    system: Intervenable
    info: LatentInfo
    grid: Grid
    basis: Decomposition
    steer: Steer
    steps: int
    seeds: tuple[int, ...]
    fields: tuple[str, ...] | None
    places: tuple[tuple[int, int], ...]
    times: tuple[int, ...]
    direction: np.ndarray
    outcome: Outcome

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.direction))

    @property
    def first(self) -> int:
        return self.times[0]

    def run(
        self, state: Any, hooks: Sequence[Hook], seed: int, *, check_finite: bool = True
    ) -> Rollout:
        rollout = self.system.run(state, self.steps, hooks, noise_seed=seed, record=self.places)
        if missing := [p for p in self.places if p not in rollout.latents]:
            raise AdapterError(f"the system did not record (layer, time) {missing}")
        valid = self.grid.valid
        for place in self.places if check_finite else ():
            if not np.isfinite(np.asarray(rollout.latents[place])[valid]).all():
                raise RequestError(
                    f"the system recorded latents that are not finite on valid nodes at layer "
                    f"{place[0]}, time {place[1]}"
                )
        return rollout

    def hooks(
        self,
        steer: Steer,
        vector: np.ndarray | None,
        control: Rollout,
        *,
        given: Mapping[int, np.ndarray] | None = None,
        applied: dict[int, np.ndarray] | None = None,
    ) -> list[Hook]:
        """The edits of one arm: the reconstruction (``vector`` None), or a ``delta`` along
        ``vector``. The delta is ``given[t]`` when given; otherwise ``steer``'s, worked out
        as ``steer.loop`` says. ``applied`` collects, per time, the delta actually used."""
        valid, layer, basis = self.grid.valid, steer.layer, self.basis
        hooks = []
        for t in self.times:
            if vector is None:
                edit = _reconstruction(basis)
            elif given is not None:
                edit = _along(given[t], vector, applied, t)
            elif steer.loop == "open":
                deltas = steer.deltas(basis, control.latents[(layer, t)])
                edit = _along(deltas, vector, applied, t)
            else:
                edit = _closed(basis, steer, vector, applied, t)
            hooks.append(_guarded(Hook(layer, t, edit), valid))
        return hooks

    def names(self, control: Rollout) -> list[str]:
        names = list(self.fields) if self.fields is not None else sorted(control.fields)
        if not names:
            raise RequestError("the system wrote no fields to measure")
        return names


def _acting(steer: Steer, grid: Grid) -> np.ndarray:
    """The valid nodes ``steer`` edits: all of them, or those of its ``nodes``."""
    acting = grid.valid.copy()
    if steer.nodes is not None:
        chosen = np.zeros(grid.n_nodes, dtype=bool)
        chosen[list(steer.nodes)] = True
        acting &= chosen
    return acting


def _check_reference(steer: Steer, grid: Grid, acting: np.ndarray) -> None:
    """A steer's reference, if it has one, is one finite value per node of this grid where
    the steer acts, read on this grid's nodes in this order."""
    if steer.reference is None:
        return
    whose = f"feature {steer.feature}'s reference"
    if steer.reference.shape != (grid.n_nodes,):
        raise RequestError(
            f"{whose} holds {steer.reference.size} value(s) for {grid.n_nodes} node(s)"
        )
    if not np.isfinite(steer.reference[acting]).all():
        raise RequestError(f"{whose} is not finite on a valid node the steer acts at")
    read_on = (steer.reference_meta or {}).get("grid")
    if read_on is not None and read_on != node_digest(grid):
        raise RequestError(
            f"{whose} was read on a grid laid out otherwise than this system's: its nodes are "
            "not this system's nodes, in this order"
        )


def _plan(
    system: Intervenable,
    basis: Decomposition,
    steer: Steer,
    *,
    steps: int,
    seeds: Sequence[int],
    fields: Sequence[str] | None,
    record_layers: Sequence[int] | None = None,
    outcome: Outcome | None = None,
    allow_unverified_basis: bool = False,
) -> _Plan:
    info, grid = system.info(), system.grid()
    layer = steer.layer
    steps, seeds = int(steps), tuple(int(s) for s in seeds)
    if steps < 1:
        raise RequestError("steps must be at least 1")
    if not seeds or len(set(seeds)) != len(seeds):
        raise RequestError("seeds must be a non-empty list of distinct integers")
    if fields is not None and not fields:
        raise RequestError("no fields were asked for")
    late = [t for t in steer.times if not 0 <= t < steps]
    if late:
        raise RequestError(f"cannot act at time(s) {late} in a run of {steps} step(s)")
    if steer.nodes is not None:
        if any(not 0 <= n < grid.n_nodes for n in steer.nodes):
            raise RequestError(f"nodes must lie in 0..{grid.n_nodes - 1}")
        if not grid.valid[list(steer.nodes)].any():
            raise RequestError("every node asked for is masked out")
    acting = _acting(steer, grid)
    _check_reference(steer, grid, acting)
    check_basis_fits(basis, info, layer, allow_unverified=allow_unverified_basis)
    direction = feature_direction(basis, steer.feature)
    recorded = sorted({layer, *(record_layers or ())})
    if unknown := [r for r in recorded if r not in {e.index for e in info.layers}]:
        raise RequestError(f"cannot record layer(s) {unknown}: not in the system")
    outcome = outcome or Outcome()
    outcome.nodes(grid)  # an empty region is refused before anything runs
    return _Plan(
        system=system, info=info, grid=grid, basis=basis, steer=steer, steps=steps,
        seeds=seeds, fields=None if fields is None else tuple(fields),
        places=tuple((r, t) for r in recorded for t in range(steps)),
        times=tuple(sorted(set(steer.times))), direction=direction, outcome=outcome,
    )  # fmt: skip


def _summarize(
    names: Sequence[str],
    pairings: Sequence[Pairing],
    seeds: Sequence[int],
    n_random: int,
    comparisons: Sequence[str],
    matched: Sequence[int],
) -> dict[str, Effect]:
    def mean_of(arm: str, name: str, **where: Any) -> float:
        picked = [
            p.response[name]
            for p in pairings
            if p.arm == arm and all(getattr(p, k) == v for k, v in where.items())
        ]
        return float(np.mean(picked))

    effects = {}
    for name in names:
        by_seed = {p.seed: p.response[name] for p in pairings if p.arm == FEATURE}
        feature = np.array([by_seed[s] for s in seeds])
        draws = np.array([mean_of(RANDOM, name, draw=j) for j in range(n_random)])
        size, magnitudes = abs(float(feature.mean())), np.abs(draws)
        spread = float(magnitudes.std(ddof=1)) if n_random > 1 else 0.0
        others = np.array([mean_of(MATCHED, name, draw=k) for k in range(len(matched))])
        effects[name] = Effect(
            field=name,
            feature_response=float(feature.mean()),
            feature_se=(
                float(feature.std(ddof=1) / np.sqrt(feature.size)) if feature.size > 1 else None
            ),
            reconstruction_response=mean_of(RECONSTRUCTION, name),
            random_responses=draws,
            effect_size=(
                (size - float(magnitudes.mean())) / spread if spread > 0.0 else None
            ),
            rank=1 + int(np.count_nonzero(magnitudes >= size)) if size > 0.0 else n_random + 1,
            p_value=(1 + int(np.count_nonzero(magnitudes >= size))) / (n_random + 1),
            comparison_responses={c: mean_of(COMPARISON, name, label=c) for c in comparisons},
            matched_features=tuple(int(j) for j in matched),
            matched_responses=others if len(matched) else None,
            matched_rank=(
                (1 + int(np.count_nonzero(np.abs(others) >= size)) if size > 0.0
                 else len(matched) + 1) if len(matched) else None
            ),
        )  # fmt: skip
    return effects


def _direction_lengths(basis: Decomposition) -> np.ndarray:
    directions = np.asarray(basis.directions(), dtype=np.float64)
    return np.linalg.norm(directions, axis=1) * _scale_of(basis)


def contributions(
    basis: Decomposition, steer: Steer, latents: Mapping[int, np.ndarray], grid: Grid
) -> np.ndarray:
    """What each feature of ``basis`` carries where ``steer`` acts: its activation's
    magnitude, area-weighted over the valid nodes the steer edits and summed over the
    times given (``latents[t]`` is the layer at time ``t``), times the length of its
    direction -- a feature's size is what it contributes. ``(n_features,)``."""
    nodes = np.flatnonzero(_acting(steer, grid))
    weights = grid.weights()[nodes]
    total = np.zeros(basis.n_features)
    for values in latents.values():
        for start in range(0, nodes.size, _BLOCK):
            block = nodes[start : start + _BLOCK]
            active = np.abs(basis.transform(values[block]))
            if not np.isfinite(active).all():
                raise RequestError(
                    "the basis returned activations that are not finite where the edit acts"
                )
            total += weights[start : start + _BLOCK] @ active
    return total * _direction_lengths(basis)


def matched_features(
    basis: Decomposition,
    feature: int,
    carried: np.ndarray,
    *,
    n: int,
    max_cosine: float = 0.3,
) -> list[int]:
    """Up to ``n`` features of ``basis`` that carry about as much as ``feature`` does
    (``carried``, from ``contributions``), closest first by the size of the log of the
    ratio, among those whose direction is no closer to the feature's than ``max_cosine``
    in magnitude and that carry anything at all. Ties go to the lower index."""
    if carried[feature] <= 0.0:
        raise RequestError(
            f"feature {feature} carries nothing where the edit acts, so there is nothing to "
            "match it on"
        )
    directions = np.asarray(basis.directions(), dtype=np.float64)
    norms = np.linalg.norm(directions, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = directions @ directions[feature] / (norms * norms[feature])
    candidates = [
        j
        for j in range(basis.n_features)
        if j != feature and carried[j] > 0.0 and abs(float(np.nan_to_num(cosine[j]))) <= max_cosine
    ]
    distance = {j: abs(math.log(carried[j] / carried[feature])) for j in candidates}
    return sorted(candidates, key=lambda j: (distance[j], j))[: max(n, 0)]


def run_steering(
    system: Intervenable,
    basis: Decomposition,
    steer: Steer,
    *,
    steps: int,
    seeds: Sequence[int] = (0,),
    n_random: int = 20,
    random_seed: int = 0,
    fields: Sequence[str] | None = None,
    record_layers: Sequence[int] | None = None,
    initial_state: Any = None,
    allow_unverified_basis: bool = False,
    outcome: Outcome | None = None,
    comparisons: Mapping[str, tuple[Decomposition, int]] | None = None,
    n_matched: int = 0,
    max_cosine: float = 0.3,
    reference_for: Callable[[int], Any] | None = None,
) -> SteeringResult:
    """Run the arms for every seed, and compare them with the control.

    The control runs first and records the latents at each place the steer acts. With
    ``steer.loop`` open, ``delta`` is worked out from those, so every arm is edited by
    the same amount at the same nodes however the runs drift apart afterwards (it is
    worked out again for each seed, whose control differs); closed, each arm works it
    out from its own latents as it goes. Within a seed all arms share the noise.
    ``record_layers`` are the layers whose latents are compared (default: the steer's
    own), at every step. ``fields`` default to every field the system writes, and
    ``outcome`` says how each is reduced to a number per step (default: global mean).

    ``comparisons`` names other directions to push along by the same ``delta``, each a
    feature of another basis read in this layer's channels (a ``Directions`` from
    ``mean_difference``, a PCA component), scaled to the feature direction's length.
    ``n_matched`` adds that many arms that apply the same steer to other features of the
    basis, chosen by ``matched_features`` on the first seed's control at the times the
    steer acts; for ``toward`` and ``excess`` each needs its own reference, which
    ``reference_for(feature)`` supplies (an array, or an array and its provenance, as
    ``feature_reference`` returns), checked against this grid as the steer's is. A matched
    arm is the same steer on another feature: its delta is that feature's own, not the
    feature arm's, so its rank says how unusual the response is among features steered
    alike, not among pushes of one size (the random and comparison arms are those).
    """
    if n_random < 1:
        raise RequestError("at least one random-direction draw is needed to say what is unusual")
    if n_matched < 0:
        raise RequestError("n_matched cannot be negative")
    if n_matched and steer.mode in REFERENCE_MODES and reference_for is None:
        raise RequestError(
            f"matched features under mode {steer.mode!r} each need a reference: pass reference_for"
        )
    plan = _plan(
        system, basis, steer, steps=steps, seeds=seeds, fields=fields,
        record_layers=record_layers, outcome=outcome,
        allow_unverified_basis=allow_unverified_basis,
    )  # fmt: skip
    grid, layer, length = plan.grid, steer.layer, plan.length
    named: dict[str, np.ndarray] = {}
    for name, (other, index) in (comparisons or {}).items():
        if other.n_channels != basis.n_channels:
            raise RequestError(
                f"comparison {name!r} reads {other.n_channels} channel(s), and this layer has "
                f"{basis.n_channels}"
            )
        try:
            check_basis_fits(other, plan.info, layer, allow_unverified=allow_unverified_basis)
        except RequestError as exc:
            raise RequestError(f"comparison {name!r}: {exc}") from exc
        vector = feature_direction(other, index)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise RequestError(f"comparison {name!r} has a direction of zero length")
        named[name] = vector / norm * length
    state = system.initial_state() if initial_state is None else initial_state

    runs: list[Run] = []
    pairings: list[Pairing] = []
    names: list[str] = []
    picked: list[int] = []
    matched_steers: list[Steer] = []
    carried: np.ndarray | None = None

    for seed in plan.seeds:
        control = plan.run(state, (), seed)
        if not names:
            names = plan.names(control)
        series = _reduce(control.fields, names, grid, plan.steps, plan.outcome)
        runs.append(Run(CONTROL, seed, None, series, {place: 0.0 for place in plan.places}))
        if n_matched and carried is None:
            acting = {t: control.latents[(layer, t)] for t in plan.times}
            carried = contributions(basis, steer, acting, grid)
            picked = matched_features(basis, steer.feature, carried, n=n_matched,
                                      max_cosine=max_cosine)  # fmt: skip
            matched_steers = []
            for j in picked:
                reference = meta = None
                if reference_for is not None and steer.reference is not None:
                    # a reference with its provenance, or the bare array
                    got = reference_for(j)
                    reference, meta = got if isinstance(got, tuple) else (got, None)
                    reference = np.asarray(reference, dtype=np.float64)
                other = replace(steer, feature=j, reference=reference, reference_meta=meta)
                _check_reference(other, grid, _acting(other, grid))
                matched_steers.append(other)
        arms: list[tuple[str, int | None, str | None, Steer, np.ndarray | None]] = [
            (RECONSTRUCTION, None, None, steer, None),
            (FEATURE, None, None, steer, plan.direction),
        ]
        for draw in range(n_random):
            vector = _random_unit(basis.n_channels, random_seed, draw) * length
            arms.append((RANDOM, draw, None, steer, vector))
        for name, vector in named.items():
            arms.append((COMPARISON, None, name, steer, vector))
        for k, other in enumerate(matched_steers):
            arms.append((MATCHED, k, f"feature {other.feature}", other,
                         feature_direction(basis, other.feature)))  # fmt: skip
        applied: dict[int, np.ndarray] = {}
        for arm, draw, label, arm_steer, vector in arms:
            # In a closed loop, what a random or comparison direction is given is what the
            # feature arm actually did, step by step: worked out from their own latents the
            # pushes would differ in size, and "the same size" would no longer hold.
            given = applied if steer.loop == "closed" and arm in (RANDOM, COMPARISON) else None
            hooks = plan.hooks(arm_steer, vector, control, given=given,
                               applied=applied if arm == FEATURE else None)  # fmt: skip
            rollout = plan.run(state, hooks, seed)
            if arm == FEATURE and set(applied) != set(plan.times):
                raise RequestError(
                    f"{plan.info.source} ran the feature's edit at time(s) {sorted(applied)}, not "
                    f"at every time asked for, {list(plan.times)}: an adapter must call each hook "
                    "once, at its layer and time"
                )
            outcomes = _reduce(rollout.fields, names, grid, plan.steps, plan.outcome)
            differences = {n: outcomes[n] - series[n] for n in names}
            latent_rms = {p: _rms(rollout.latents[p] - control.latents[p], grid)
                          for p in plan.places}  # fmt: skip
            runs.append(Run(arm, seed, draw, outcomes, latent_rms, label))
            pairings.append(
                Pairing(
                    arm, seed, draw, differences,
                    {n: float(differences[n][plan.first :].mean()) for n in names}, label,
                )
            )  # fmt: skip

    spec = {
        "steer": steer.to_dict(), "steps": plan.steps, "seeds": list(plan.seeds),
        "n_random": n_random, "random_seed": random_seed, "fields": names,
        "record_layers": sorted({p[0] for p in plan.places}), "feature_length": length,
        "outcome": plan.outcome.to_dict(),
        "comparisons": {
            name: {"feature": int(index), "basis": basis_provenance(other)}
            for name, (other, index) in (comparisons or {}).items()
        },
        "matched": None if not n_matched else {
            "n": n_matched, "max_cosine": max_cosine, "features": picked,
            "contributions": {str(j): float(carried[j]) for j in [steer.feature, *picked]},
            "references": {
                str(m.feature): None if m.reference is None else array_digest(m.reference)
                for m in matched_steers
            },
        },
    }  # fmt: skip
    return SteeringResult(
        spec=spec,
        provenance=result_provenance(plan.info, basis=basis),
        runs=tuple(runs),
        pairings=tuple(pairings),
        effects=_summarize(names, pairings, plan.seeds, n_random, list(named), picked),
    )


def _reconstruction(basis: Decomposition):
    return lambda latents: basis.reconstruct(latents).astype(latents.dtype)


def _along(deltas: np.ndarray, vector: np.ndarray, applied=None, time: int = 0):
    def edit(latents: np.ndarray) -> np.ndarray:
        if applied is not None:
            applied[time] = deltas
        return _moved(latents, deltas, vector)

    return edit


def _closed(basis: Decomposition, steer: Steer, vector: np.ndarray, applied=None, time: int = 0):
    def edit(latents: np.ndarray) -> np.ndarray:
        deltas = steer.deltas(basis, latents)
        if applied is not None:
            applied[time] = deltas
        return _moved(latents, deltas, vector)

    return edit


def _moved(latents: np.ndarray, deltas: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """``latents`` moved by ``deltas[node] * direction``, in their own dtype. A delta that
    is not a number (a reference missing where the grid is masked) is not hidden here:
    ``_guarded`` puts a masked node back as it was and refuses a valid one."""
    return latents + (deltas[:, None] * direction[None, :]).astype(latents.dtype)


# -- many initial states ------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """One initial state to steer from: ``start`` as ``Intervenable.initial_state`` takes
    it, and optionally a ``region`` that both restricts the edit to its nodes and is
    where the outcome is measured (a box around a storm). ``label`` names it."""

    start: str | int = 0
    region: AnyRegion | None = None
    label: str | None = None

    @property
    def name(self) -> str:
        return self.label if self.label is not None else str(self.start)

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "region": _region_dict(self.region), "label": self.name}


def _case_setup(case: Case, steer: Steer, outcome: Outcome, grid: Grid) -> tuple[Steer, Outcome]:
    if case.region is None:
        return steer, outcome
    nodes = case.region.nodes(grid)
    if nodes.size == 0:
        raise RequestError(f"case {case.name!r} has no valid nodes {case.region.describe()}")
    if steer.nodes is not None:
        nodes = np.intersect1d(nodes, np.asarray(steer.nodes))
        if nodes.size == 0:
            raise RequestError(f"case {case.name!r}: the steer's nodes lie outside its region")
    return (
        replace(steer, nodes=tuple(int(n) for n in nodes)),
        replace(outcome, region=case.region) if outcome.region is None else outcome,
    )


def _check_cases(cases: Sequence[Case]) -> list[Case]:
    cases = list(cases)
    if not cases:
        raise RequestError("no cases were given")
    names = [c.name for c in cases]
    if len(set(names)) != len(names):
        raise RequestError(f"case names must differ; got {names}")
    return cases


@dataclass(frozen=True)
class CaseStudy:
    """One steering experiment per case, and how the feature's response is distributed
    across them: one case is an anecdote, and the spread is the finding."""

    spec: Mapping[str, Any]
    provenance: Mapping[str, Any]
    cases: tuple[Case, ...]
    results: tuple[SteeringResult, ...]

    def responses(self, name: str) -> np.ndarray:
        """The feature's seed-mean response on field ``name``, one per case."""
        return np.array([r.effects[name].feature_response for r in self.results])

    def summary(self) -> dict[str, Any]:
        fields = list(self.results[0].effects)
        out = {}
        for name in fields:
            effects = [r.effects[name] for r in self.results]
            responses = np.array([e.feature_response for e in effects])
            out[name] = {
                "cases": [c.name for c in self.cases],
                "responses": responses.tolist(),
                "median": float(np.median(responses)),
                "positive": int(np.count_nonzero(responses > 0.0)),
                "negative": int(np.count_nonzero(responses < 0.0)),
                "ranks": [e.rank for e in effects],
                "p_values": [e.p_value for e in effects],
                "matched_ranks": [e.matched_rank for e in effects],
            }
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": dict(self.spec),
            "provenance": dict(self.provenance),
            "cases": [c.to_dict() for c in self.cases],
            "summary": self.summary(),
            "results": [r.to_dict() for r in self.results],
        }


def run_cases(
    system: Intervenable,
    basis: Decomposition,
    steer: Steer,
    cases: Sequence[Case],
    *,
    outcome: Outcome | None = None,
    **kwargs: Any,
) -> CaseStudy:
    """``run_steering`` from each case's initial state. A case's region restricts the
    steer to its nodes and, unless ``outcome`` names a region of its own, is where the
    outcome is measured. Everything else in ``kwargs`` goes to ``run_steering`` as it
    is, so every case is run the same way."""
    if "initial_state" in kwargs:
        raise RequestError("the cases say where each run starts; initial_state cannot be given")
    cases, grid = _check_cases(cases), system.grid()
    outcome = outcome or Outcome()
    results = []
    for case in cases:
        case_steer, case_outcome = _case_setup(case, steer, outcome, grid)
        results.append(
            run_steering(
                system, basis, case_steer, outcome=case_outcome,
                initial_state=system.initial_state(case.start), **kwargs,
            )
        )  # fmt: skip
    spec = {
        "steer": steer.to_dict(), "outcome": outcome.to_dict(),
        "cases": [c.to_dict() for c in cases],
    }  # fmt: skip
    return CaseStudy(
        spec=spec,
        provenance=dict(results[0].provenance),
        cases=tuple(cases),
        results=tuple(results),
    )


# -- dose curves, and choosing an amount honestly -----------------------------


def leave_one_out(scores: np.ndarray, *, lower_is_better: bool = True) -> tuple[np.ndarray, ...]:
    """For each case (row of ``scores``, ``(n_cases, n_amounts)``), the amount (column)
    that does best by the median over the *other* cases, and that case's score there.
    Ties go to the first column. Returns ``(columns, scores)``. An amount picked on the
    case it is then scored on is a hindsight optimum, not a correction anyone could
    apply; this is the choice that could have been made without seeing the case."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] < 2:
        raise RequestError("choosing leave-one-out needs at least two cases")
    picks = np.empty(scores.shape[0], dtype=np.intp)
    for case in range(scores.shape[0]):
        others = np.nanmedian(np.delete(scores, case, axis=0), axis=0)
        others = others if lower_is_better else -others
        picks[case] = int(np.nanargmin(others))
    return picks, scores[np.arange(scores.shape[0]), picks]


@dataclass(frozen=True)
class Selection:
    """How well an amount does on one field when it is chosen leave-one-out, beside
    the hindsight optimum and doing nothing. Per case: ``loo_amounts`` and
    ``loo_errors``; ``oracle_amounts`` and ``oracle_errors``, the best amount for that
    case on that case (an upper bound no forecaster could reach); ``baseline_errors``,
    the control's. ``medians`` holds the median of each over cases."""

    field: str
    loo_amounts: tuple[float, ...]
    loo_errors: np.ndarray
    oracle_amounts: tuple[float, ...]
    oracle_errors: np.ndarray
    baseline_errors: np.ndarray

    @property
    def medians(self) -> dict[str, float]:
        return {
            "loo": float(np.median(self.loo_errors)),
            "oracle": float(np.median(self.oracle_errors)),
            "baseline": float(np.median(self.baseline_errors)),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field, "loo_amounts": list(self.loo_amounts),
            "loo_errors": self.loo_errors, "oracle_amounts": list(self.oracle_amounts),
            "oracle_errors": self.oracle_errors, "baseline_errors": self.baseline_errors,
            "medians": self.medians,
        }  # fmt: skip


@dataclass(frozen=True)
class DoseCurve:
    """The feature's response at each amount, from each case, and -- given targets --
    the error at each amount and the amount chosen leave-one-out.

    ``responses`` is ``(cases, amounts, fields)``: the seed-mean paired response, as in
    ``run_steering``. ``errors`` (with targets) is the same shape: the seed-mean RMS, over
    the steps from the first edit on, of the outcome less the target; ``baseline_errors``
    ``(cases, fields)`` is the control's. ``selections[field]`` is a ``Selection``. The
    whole curve is the result, never one point of it."""

    spec: Mapping[str, Any]
    provenance: Mapping[str, Any]
    cases: tuple[str, ...]
    amounts: tuple[float, ...]
    fields: tuple[str, ...]
    responses: np.ndarray
    errors: np.ndarray | None
    baseline_errors: np.ndarray | None
    selections: Mapping[str, Selection]

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": dict(self.spec), "provenance": dict(self.provenance),
            "cases": list(self.cases), "amounts": list(self.amounts),
            "fields": list(self.fields), "responses": self.responses, "errors": self.errors,
            "baseline_errors": self.baseline_errors,
            "selections": {n: s.to_dict() for n, s in self.selections.items()},
        }  # fmt: skip


def dose_curve(
    system: Intervenable,
    basis: Decomposition,
    steer: Steer,
    amounts: Sequence[float],
    *,
    steps: int,
    cases: Sequence[Case] = (Case(),),
    seeds: Sequence[int] = (0,),
    fields: Sequence[str] | None = None,
    outcome: Outcome | None = None,
    targets: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
    allow_unverified_basis: bool = False,
) -> DoseCurve:
    """The control and the feature arm at every amount, from every case and seed.

    ``targets[case][field]`` is the series, one value per step, that the reduced outcome
    of that case should follow (an analysis, a best track). With targets, every amount is
    scored by its error, and an amount is chosen for each case leave-one-out
    (``leave_one_out``) when there are at least two cases. Random and matched arms are
    not run here: a dose curve says how the response grows, ``run_steering`` at a chosen
    amount says whether it is unusual."""
    amounts = tuple(float(a) for a in amounts)
    if not amounts:
        raise RequestError("a dose curve needs at least one amount")
    if len(set(amounts)) != len(amounts):
        raise RequestError(f"amounts must differ; got {list(amounts)}")
    cases, grid = _check_cases(cases), system.grid()
    outcome = outcome or Outcome()
    if targets is not None:
        missing = [c.name for c in cases if c.name not in targets]
        if missing:
            raise RequestError(f"no target for case(s) {missing}")
    plans = []
    for case in cases:
        case_steer, case_outcome = _case_setup(case, steer, outcome, grid)
        plans.append(
            _plan(
                system, basis, case_steer, steps=steps, seeds=seeds, fields=fields,
                outcome=case_outcome, allow_unverified_basis=allow_unverified_basis,
            )
        )  # fmt: skip
    names: list[str] = []
    responses = errors = baseline = None
    for c, (case, plan) in enumerate(zip(cases, plans, strict=True)):
        state = system.initial_state(case.start)
        for seed in plan.seeds:
            control = plan.run(state, (), seed)
            if not names:
                names = plan.names(control)
                shape = (len(cases), len(amounts), len(names))
                responses = np.zeros(shape)
                if targets is not None:
                    errors, baseline = np.zeros(shape), np.zeros((len(cases), len(names)))
            series = _reduce(control.fields, names, grid, plan.steps, plan.outcome)
            goal = None
            if targets is not None:
                goal = {n: _target(targets, case.name, n, plan.steps) for n in names}
                for m, n in enumerate(names):
                    baseline[c, m] += _error(series[n], goal[n], plan.first) / len(plan.seeds)
            for a, amount in enumerate(amounts):
                dosed = replace(plan.steer, amount=amount)
                rollout = plan.run(state, plan.hooks(dosed, plan.direction, control), seed)
                outcomes = _reduce(rollout.fields, names, grid, plan.steps, plan.outcome)
                for m, n in enumerate(names):
                    difference = outcomes[n] - series[n]
                    responses[c, a, m] += float(difference[plan.first :].mean()) / len(plan.seeds)
                    if goal is not None:
                        errors[c, a, m] += _error(outcomes[n], goal[n], plan.first) / len(
                            plan.seeds
                        )
    selections = {}
    if errors is not None and len(cases) >= 2:
        for m, n in enumerate(names):
            picks, loo = leave_one_out(errors[:, :, m])
            best = np.nanargmin(errors[:, :, m], axis=1)
            selections[n] = Selection(
                field=n,
                loo_amounts=tuple(amounts[i] for i in picks),
                loo_errors=loo,
                oracle_amounts=tuple(amounts[i] for i in best),
                oracle_errors=errors[np.arange(len(cases)), best, m],
                baseline_errors=baseline[:, m],
            )
    spec = {
        "steer": steer.to_dict(), "amounts": list(amounts), "steps": plans[0].steps,
        "seeds": list(plans[0].seeds), "fields": names, "outcome": outcome.to_dict(),
        "cases": [c.to_dict() for c in cases], "targets": targets is not None,
    }  # fmt: skip
    return DoseCurve(
        spec=spec,
        provenance=result_provenance(plans[0].info, basis=basis),
        cases=tuple(c.name for c in cases),
        amounts=amounts,
        fields=tuple(names),
        responses=responses,
        errors=errors,
        baseline_errors=baseline,
        selections=selections,
    )


def _target(
    targets: Mapping[str, Mapping[str, Sequence[float]]], case: str, name: str, steps: int
) -> np.ndarray:
    if name not in targets[case]:
        raise RequestError(f"no target for field {name!r} of case {case!r}")
    series = np.asarray(targets[case][name], dtype=np.float64)
    if series.shape != (steps,):
        raise RequestError(
            f"the target for {name!r} of case {case!r} holds {series.size} value(s), not one "
            f"per step ({steps})"
        )
    return series


def _error(series: np.ndarray, target: np.ndarray, first: int) -> float:
    gap = series[first:] - target[first:]
    keep = np.isfinite(gap)
    if not keep.any():
        raise RequestError("the target has no values at the steps from the first edit on")
    return float(np.sqrt(np.mean(gap[keep] ** 2)))
