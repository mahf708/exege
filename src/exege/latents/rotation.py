"""Does a feature move with the weather, or stay where the model put it?

Turn everything a model is given east by some degrees and run it once more. A feature
that describes the weather turns with it. A feature that stays where it was is tied to
something that did not turn: the model's own sense of place (a learned positional
embedding, the coordinates a mesh node carries, a local time computed from the grid's
longitudes), or whatever input was left out of the turn.

That last clause is the point of naming *parts*. A system that can be turned
(``Rotatable``) says which parts of its state it can turn -- the weather, the boundary
fields it is given (orography, land), the forcings -- and the experiment turns the ones
asked for. Turning everything isolates what the model holds of its own; turning the
weather alone also catches features tied to the boundary, which in a climate model are
physical and not artefacts. A feature that stays when everything is turned is the
model's; one that stays when only the weather is turned, and follows when the boundary
turns too, belongs to the boundary.

For each column (a channel, or a feature of a basis) and each shift:

- ``stays``   the correlation, over nodes, of the turned run's map with the original map
              where it was;
- ``follows`` its correlation with the original map turned by the same shift;
- ``score``   ``stays - follows``: near +1 stays put, near -1 moves with the weather;
- ``self``    the original map's correlation with itself turned: a pattern that maps onto
              itself (a zonal band) cannot say which it does, and that shift has no power
              for it.

Correlations are area-weighted, over nodes valid where they are and where they came from.
Needs the ``latents`` extra.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from exege.core.errors import RequestError
from exege.core.extras import missing_extra
from exege.latents.basis import Decomposition, result_provenance
from exege.latents.grid import Grid, longitude_partner, rotation_shifts
from exege.latents.source import check_basis_fits
from exege.latents.steering import Intervenable

try:
    import numpy as np
except ImportError as exc:
    raise missing_extra("numpy", "latents") from exc


@runtime_checkable
class Rotatable(Protocol):
    """A system whose initial state can be turned east by whole grid columns.

    An optional capability beside ``Intervenable``. ``rotation_parts`` names the parts of
    a state that can be turned (``"state"``, ``"static"``, ``"forcing"``, ...: the
    adapter's to name), and ``rotate`` returns a new state with the parts named turned
    ``columns`` columns east and the rest as they were, leaving the one given unchanged.
    What the model holds of its own -- its weights, a positional embedding, the
    coordinates of its grid -- is never part of a state, and never turns.
    """

    def rotation_parts(self) -> tuple[str, ...]: ...

    def rotate(self, state: Any, columns: int, parts: Sequence[str]) -> Any: ...


def _columns(grid: Grid, degrees: float) -> int:
    if not np.isfinite(degrees):
        raise RequestError(f"a shift is a finite number of degrees, not {degrees!r}")
    if grid.shape is None:
        raise RequestError("a rotation turns a structured grid by whole columns; this is a mesh")
    n_lon = grid.shape[1]
    columns = degrees / (360.0 / n_lon)
    if abs(columns - round(columns)) > 1e-6:
        raise RequestError(
            f"{degrees:g} degrees is not a whole number of this grid's {360.0 / n_lon:g}-degree "
            "columns"
        )
    columns = int(round(columns)) % n_lon
    if columns == 0:
        raise RequestError(f"{degrees:g} degrees turns the grid onto itself")
    return columns


def _score(
    stays: np.ndarray,
    follows: np.ndarray,
    selfsim: np.ndarray,
    spread_stayed: np.ndarray | float = 1.0,
    spread_moved: np.ndarray | float = 1.0,
) -> np.ndarray:
    """How much of the turned map is the original where it was (``a``) against the
    original turned along (``b``): the area-weighted regression of the one on the two, in
    the column's own units, ``(a - b) / (|a| + |b|)``. A column that is a share ``alpha``
    of the map that stayed and ``1 - alpha`` of the one that moved scores ``2 alpha - 1``
    whatever the two maps' correlation (the self-similarity ``s``) and their spreads over
    the nodes compared (``spread_*``, which a mask makes differ); stays less follows would
    be ``(2 alpha - 1)(1 - s)``, which reads a half-anchored column as anchored on a shift
    where ``s`` is negative."""
    with np.errstate(invalid="ignore", divide="ignore"):
        # standardized coefficients, both over 1 - s^2 (which cancels), less each map's spread
        a = (stays - selfsim * follows) / spread_stayed
        b = (follows - selfsim * stays) / spread_moved
        return (a - b) / (np.abs(a) + np.abs(b))


def _spread(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = weights / weights.sum()
    centred = values - w @ values
    return np.sqrt(w @ (centred * centred))


def _correlate(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Column-wise area-weighted Pearson correlation of ``(n, C)`` arrays; NaN where either
    column does not vary (beyond rounding of its own size)."""
    total = weights.sum()
    if total <= 0.0:
        raise RequestError("no valid node with area is left to correlate over")
    w = weights / total
    # the size a column's rounding is measured against is its own, before centring: a
    # constant column centres to rounding residue, which is not variation
    scale_a = np.maximum(np.abs(a).max(axis=0), 1e-300) if a.size else 1.0
    scale_b = np.maximum(np.abs(b).max(axis=0), 1e-300) if b.size else 1.0
    a = a - w @ a
    b = b - w @ b
    var_a, var_b = w @ (a * a), w @ (b * b)
    flat = (var_a <= (1e-12 * scale_a) ** 2) | (var_b <= (1e-12 * scale_b) ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (w @ (a * b)) / np.sqrt(var_a * var_b)
    return np.where(flat, np.nan, r)


@dataclass(frozen=True, eq=False)
class RotationTest:
    """Per start, shift and column: ``stays``, ``follows`` and ``self_similarity``,
    ``(n_starts, n_shifts, n_columns)``. A start and shift has *power* for a column when
    its original map turns neither onto itself nor onto its own opposite
    (``|self_similarity| < max_self_similarity``). ``score`` is the anchored share read
    from regressing the turned map on the two (see ``_score``: ``2 alpha - 1`` for a
    column a share ``alpha`` anchored, in -1..1), averaged over the powered ones, NaN for
    a column with none; ``n_powered`` counts
    them, and ``anchored`` marks columns whose every powered score is a number at least
    ``threshold`` -- a column that vanished when turned has no correlation, and is not
    anchored."""

    settings: dict[str, Any]
    provenance: dict[str, Any]
    columns: tuple[int, ...]
    shifts: tuple[float, ...]
    parts: tuple[str, ...]
    stays: np.ndarray
    follows: np.ndarray
    self_similarity: np.ndarray
    score: np.ndarray
    n_powered: np.ndarray
    anchored: np.ndarray

    @property
    def powered(self) -> np.ndarray:
        """Which starts and shifts have power for each column."""
        sim = self.self_similarity
        return np.isfinite(sim) & (sim < self.settings["max_self_similarity"])

    def ranked(self) -> np.ndarray:
        """Positions into ``columns``, the most anchored first; no power last."""
        return np.argsort(np.nan_to_num(-self.score, nan=np.inf), kind="stable")

    def summary(self, top: int | None = None) -> dict[str, Any]:
        order = self.ranked()
        order = order if top is None else order[: max(top, 0)]
        return {
            "settings": self.settings,
            "provenance": self.provenance,
            "columns": [
                {
                    "column": self.columns[i],
                    "score": _number(self.score[i]),
                    "n_powered": int(self.n_powered[i]),
                    "anchored": bool(self.anchored[i]),
                    "stays": _powered_mean(self.stays[:, :, i], self.powered[:, :, i]),
                    "follows": _powered_mean(self.follows[:, :, i], self.powered[:, :, i]),
                }
                for i in order
            ],
        }


def _number(value: float) -> float | None:
    return None if not np.isfinite(value) else float(value)


def _powered_mean(values: np.ndarray, powered: np.ndarray) -> float | None:
    chosen = values[powered & np.isfinite(values)]
    return float(chosen.mean()) if chosen.size else None


def rotation_test(
    system: Intervenable,
    *,
    layer: int,
    shifts: Sequence[float] | None = None,
    parts: Sequence[str] | None = None,
    basis: Decomposition | None = None,
    columns: Sequence[int] | None = None,
    starts: Sequence[str | int] = (0,),
    time: int = 0,
    max_self_similarity: float = 0.5,
    threshold: float = 0.5,
    allow_unverified_basis: bool = False,
) -> RotationTest:
    """Run ``system`` from each start as it is and turned east by each of ``shifts``
    (degrees, whole grid columns), and say for every column of ``layer`` at forward step
    ``time`` whether it stayed or followed. ``parts`` are what is turned (default: every
    part the system names). Without ``shifts``, three turns of at least 60 degrees spread
    round the circle (``rotation_shifts``): a quarter, a half and three quarters where the
    grid allows. Runs carry no noise, so the two runs differ by the turn alone.
    """
    if not isinstance(system, Rotatable):
        raise RequestError(f"{system.info().source} cannot be rotated (it is not Rotatable)")
    info, grid = system.info(), system.grid()
    known = tuple(system.rotation_parts())
    parts = known if parts is None else tuple(parts)
    if not parts:
        raise RequestError("nothing to rotate: name at least one part")
    if unknown := [p for p in parts if p not in known]:
        raise RequestError(f"no part(s) {unknown} to rotate; there are {', '.join(known)}")
    if not starts:
        raise RequestError("at least one start is needed")
    if shifts is None:
        if grid.shape is None:
            raise RequestError(
                "a rotation turns a structured grid by whole columns; this is a mesh"
            )
        spacing = 360.0 / grid.shape[1]
        shifts = [k * spacing for k in rotation_shifts(grid, 3, min_degrees=60.0)]
    if not shifts:
        raise RequestError("at least one shift is needed")
    if time < 0:
        raise RequestError("the time is a forward step, from 0")
    width = info.layer(layer).n_channels
    if basis is not None:
        check_basis_fits(basis, info, layer, allow_unverified=allow_unverified_basis)
        width = basis.n_features
    chosen = tuple(range(width)) if columns is None else tuple(int(c) for c in columns)
    if not chosen or any(not 0 <= c < width for c in chosen):
        raise RequestError(f"columns must lie in 0..{width - 1}")
    turns = [_columns(grid, float(d)) for d in shifts]

    def read(state: Any) -> np.ndarray:
        rollout = system.run(state, time + 1, (), noise_seed=None, record=[(layer, time)])
        if (layer, time) not in rollout.latents:
            raise RequestError(f"the system did not record layer {layer} at time {time}")
        latents = np.asarray(rollout.latents[(layer, time)])
        if not np.isfinite(latents[grid.valid]).all():
            raise RequestError(
                f"the system recorded latents at layer {layer}, time {time} that are not finite "
                "on valid nodes"
            )
        if basis is not None:
            return basis.transform(latents, list(chosen))
        return latents[:, list(chosen)].astype(np.float64)

    weights, valid = grid.weights(), grid.valid
    shape = (len(starts), len(turns), len(chosen))
    stays, follows, selfsim = np.full(shape, np.nan), np.full(shape, np.nan), np.full(shape, np.nan)
    spread_stayed, spread_moved = np.full(shape, np.nan), np.full(shape, np.nan)
    for s, start in enumerate(starts):
        state = system.initial_state(start)
        original = read(state)
        for k, columns_east in enumerate(turns):
            turned = read(system.rotate(state, columns_east, parts))
            source = longitude_partner(grid, -columns_east)  # node j's map came from here
            keep = valid & valid[source]
            moved = original[source]
            stays[s, k] = _correlate(turned[keep], original[keep], weights[keep])
            follows[s, k] = _correlate(turned[keep], moved[keep], weights[keep])
            selfsim[s, k] = _correlate(original[keep], moved[keep], weights[keep])
            spread_stayed[s, k] = _spread(original[keep], weights[keep])
            spread_moved[s, k] = _spread(moved[keep], weights[keep])
    scores = _score(stays, follows, selfsim, spread_stayed, spread_moved)
    powered = np.isfinite(selfsim) & (np.abs(selfsim) < max_self_similarity)
    n_powered = powered.sum(axis=(0, 1))
    counted = powered & np.isfinite(scores)
    with np.errstate(invalid="ignore"):
        score = np.where(
            counted.sum(axis=(0, 1)) > 0,
            np.where(counted, scores, 0.0).sum(axis=(0, 1)) / counted.sum(axis=(0, 1)),
            np.nan,
        )
    held = np.where(np.isfinite(scores), scores >= threshold, False)
    anchored = (n_powered > 0) & np.all(~powered | held, axis=(0, 1))
    return RotationTest(
        settings={
            "layer": layer,
            "time": time,
            "shifts": [float(d) for d in shifts],
            "columns_east": turns,
            "parts": list(parts),
            "starts": list(starts),
            "columns": "features" if basis is not None else "channels",
            "basis": None if basis is None else basis.meta.get("path"),
            "max_self_similarity": max_self_similarity,
            "threshold": threshold,
        },  # fmt: skip
        provenance=result_provenance(info, basis=basis),
        columns=chosen,
        shifts=tuple(float(d) for d in shifts),
        parts=parts,
        stays=stays,
        follows=follows,
        self_similarity=selfsim,
        score=score,
        n_powered=n_powered,
        anchored=anchored,
    )
