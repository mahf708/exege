"""Fixed features: what a column does at the same place every time.

Some channels and features follow the weather; others sit where they sit. A census
at one time cannot tell the two apart, and a profile describes what a column goes
with, not whether it ever moves. This module splits each column's variance over the
times read into the part held by its per-node time mean (the same thing at the same
place at every time) and the rest, and then asks what that fixed part is made of:

- **latitude**: one value per latitude row (only on a structured grid), the kind of
  structure a model may build from the latitude it is given, or from the sun;
- **the boundary**: the fields named as ``statics`` (land fraction, orography, a
  sea-surface temperature held fixed), each averaged over the times read;
- **the rest**: fixed structure that neither explains. A model that learned a
  positional embedding, or that leans on an input it was given and this analysis
  was not, puts it here.

What this cannot do is tell those last two apart: a fixed pattern that no named field
explains may be a learned embedding, or the imprint of an input that is not among the
statics. Rotating the model's inputs and seeing whether the column stays on the grid
is the direct test, and that needs a running model, not an archive.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from exege.core.errors import RequestError
from exege.core.extras import missing_extra
from exege.latents.basis import Decomposition, result_provenance
from exege.latents.samples import _time_labels
from exege.latents.source import LatentSource, ReferenceFields, check_basis_fits, read_latents
from exege.latents.through import _field_at

try:
    import numpy as np
except ImportError as exc:
    raise missing_extra("numpy", "latents") from exc

_BLOCK = 8192
# A static whose part beyond the rows is this small a share of its own spread repeats
# them; among the statics so normalized, a singular value this small a share of the
# largest spans nothing new.
_RANK_TOLERANCE = 1e-9


@dataclass(frozen=True, eq=False)
class Stationarity:
    """How much of each column is fixed in place, and what the fixed part is made of.

    Arrays are parallel to ``columns`` (channel or feature indices). Over the nodes
    used and the times read, area-weighted:

    - ``mean``: the column's mean; ``total_variance``: its variance about that mean;
      ``coverage``: the share of area and time it is active (above the threshold);
    - ``fixed_share``: the share of ``total_variance`` held by the per-node time mean
      (1 for a column that is the same map at every time, 0 for one whose time mean
      is the same everywhere);
    - ``zonal_share``: the share of the time-mean map's spatial variance that latitude
      rows explain (NaN on a mesh, which has no rows);
    - ``static_share``: what the statics explain beyond the rows (beyond a constant, on
      a mesh);
    - ``residual_fixed_share``: ``fixed_share`` times the share that neither explains:
      fixed structure of unknown origin.

    A column that does not vary reads NaN in every share; one that varies only in time
    (its time mean the same everywhere) has ``fixed_share`` and ``residual_fixed_share``
    0, and NaN for the shares of a map that is flat. A column the source leaves missing
    (not finite) at any node and time used reads NaN throughout, ``coverage`` included:
    a node seen at some times only would weigh its few values as fully as the rest.
    ``static_fixed_share`` is each
    static field's own ``fixed_share``: a reader should see 1 there, or the field is
    not static and averaging it over time says less than it seems to.
    """

    settings: dict[str, Any]
    provenance: dict[str, Any]
    times: tuple[str, ...]
    columns: np.ndarray
    mean: np.ndarray
    total_variance: np.ndarray
    coverage: np.ndarray
    fixed_share: np.ndarray
    zonal_share: np.ndarray
    static_share: np.ndarray
    residual_fixed_share: np.ndarray
    statics: tuple[str, ...]
    static_fixed_share: np.ndarray
    n_nodes_used: int

    def ranked(self, top: int | None = None) -> np.ndarray:
        """Positions into ``columns``, the most fixed structure of unknown origin first;
        columns that do not vary come last."""
        key = np.nan_to_num(self.residual_fixed_share, nan=-np.inf)
        order = np.argsort(key, kind="stable")[::-1]
        return order if top is None else order[: max(top, 0)]

    def summary(self, top: int | None = None) -> dict[str, Any]:
        def number(value: float) -> float | None:
            return None if not np.isfinite(value) else float(value)

        return {
            "settings": self.settings,
            "provenance": self.provenance,
            "times": list(self.times),
            "n_nodes_used": self.n_nodes_used,
            "statics": [
                {"field": name, "fixed_share": number(share)}
                for name, share in zip(self.statics, self.static_fixed_share, strict=True)
            ],
            "columns": [
                {
                    "column": int(self.columns[i]),
                    "mean": number(self.mean[i]),
                    "total_variance": number(self.total_variance[i]),
                    "coverage": number(self.coverage[i]),
                    "fixed_share": number(self.fixed_share[i]),
                    "zonal_share": number(self.zonal_share[i]),
                    "static_share": number(self.static_share[i]),
                    "residual_fixed_share": number(self.residual_fixed_share[i]),
                }
                for i in self.ranked(top)
            ],
        }


def _columns(n_columns: int, columns: Sequence[int] | None, kind: str) -> np.ndarray:
    if columns is None:
        return np.arange(n_columns)
    chosen = np.asarray(list(columns), dtype=np.int64)
    if chosen.size == 0:
        raise RequestError(f"no {kind}s were asked for")
    bad = chosen[(chosen < 0) | (chosen >= n_columns)]
    if bad.size:
        raise RequestError(f"no {kind} {int(bad[0])}; there are {n_columns}, numbered from 0")
    if np.unique(chosen).size != chosen.size:
        raise RequestError(f"a {kind} was asked for twice")
    return chosen


class _Moments:
    """Per-node sums over time of a set of columns, less one of each node's own values
    (the first read), so a large offset costs no precision. Missing values (a basis may
    return what is not finite) are counted out, node by node."""

    def __init__(self, n_nodes: int, n_columns: int) -> None:
        self.shift = np.full((n_nodes, n_columns), np.nan)
        self.count = np.zeros((n_nodes, n_columns))
        self.first = np.zeros((n_nodes, n_columns))
        self.second = np.zeros((n_nodes, n_columns))

    def add(self, values: np.ndarray) -> None:
        ok = np.isfinite(values)
        fresh = ok & np.isnan(self.shift)
        self.shift[fresh] = values[fresh]
        centered = np.where(ok, values - np.nan_to_num(self.shift), 0.0)
        self.count += ok
        self.first += centered
        self.second += centered * centered

    def node_mean(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.nan_to_num(self.shift) + self.first / self.count

    def node_variance(self) -> np.ndarray:
        """The temporal variance at each node: zero where a node was read once."""
        with np.errstate(invalid="ignore", divide="ignore"):
            inner = self.first / self.count
            return np.maximum(self.second / self.count - inner * inner, 0.0)


@dataclass(frozen=True, eq=False)
class _Design:
    """What a time-mean map is fitted on, for the nodes used. ``groups`` holds each
    node's latitude row as a run of positions (``starts`` into the nodes, which come in
    row order), None on a mesh; ``statics`` an orthonormal basis, already scaled by the
    square root of the weights, of the statics' parts that the rows (or, on a mesh, the
    constant) do not explain. Rows are fitted analytically, a weighted mean per row, so
    nothing ``(n_nodes, n_rows)`` is ever made."""

    starts: np.ndarray | None
    index: np.ndarray | None
    statics: np.ndarray

    def row_means(self, weights: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, ...]:
        """The weight of each row, and each column's weighted mean in it."""
        row_weight = np.add.reduceat(weights, self.starts)
        sums = np.add.reduceat(weights[:, None] * values, self.starts, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            means = np.where(row_weight[:, None] > 0.0, sums / row_weight[:, None], 0.0)
        return row_weight, means

    def beyond_rows(self, weights: np.ndarray, values: np.ndarray) -> np.ndarray:
        """``values`` less their weighted mean: per row, or over all nodes on a mesh."""
        if self.starts is None:
            return values - weights @ values
        return values - self.row_means(weights, values)[1][self.index]


def _design(grid, nodes: np.ndarray, weights: np.ndarray, statics: Sequence[np.ndarray]) -> _Design:
    starts = index = None
    if grid.shape is not None:
        rows = nodes // grid.shape[1]  # nodes are sorted, so each row is one run
        starts = np.flatnonzero(np.r_[True, rows[1:] != rows[:-1]])
        index = np.repeat(np.arange(starts.size), np.diff(np.r_[starts, rows.size]))
    design = _Design(starts, index, np.zeros((nodes.size, 0)))
    root = np.sqrt(weights)
    columns = []
    for values in statics:
        centered = values - weights @ values
        residual = design.beyond_rows(weights, values[:, None])[:, 0]
        spread = np.sqrt(weights @ (centered * centered))
        left = np.sqrt(weights @ (residual * residual))
        # Normalized one by one, so a static in small units (a mixing ratio near 1e-11)
        # weighs as much as one in large units (a geopotential near 5e4).
        if spread > 0.0 and left > _RANK_TOLERANCE * spread:
            columns.append(root * residual / left)
    if columns:
        u, singular, _ = np.linalg.svd(np.stack(columns, axis=1), full_matrices=False)
        design = _Design(starts, index, u[:, singular > _RANK_TOLERANCE * singular.max()])
    return design


def _decompose(
    moments: _Moments, weights: np.ndarray, design: _Design, n_times: int
) -> dict[str, np.ndarray]:
    """Shares for every column of one chunk. ``weights`` sum to one over the nodes used."""
    complete = (moments.count == n_times).all(axis=0)  # read at every node and time
    mean_map = np.where(moments.count > 0, moments.node_mean(), 0.0)
    overall = weights @ mean_map
    anomaly = mean_map - overall
    fixed = weights @ (anomaly * anomaly)
    temporal = weights @ np.where(moments.count > 0, moments.node_variance(), 0.0)
    # a time-mean map that is flat to rounding is flat: the per-node means of a column that
    # is the same everywhere differ by rounding only, which another machine rounds otherwise
    scale = np.abs(mean_map).max(axis=0) if mean_map.size else np.zeros(fixed.shape)
    fixed = np.where(fixed <= (1e-12 * scale) ** 2, 0.0, fixed)
    total = fixed + temporal
    if design.starts is not None:
        row_weight, row_mean = design.row_means(weights, anomaly)
        zonal_part = row_weight @ (row_mean * row_mean)
    else:
        zonal_part = np.zeros(anomaly.shape[1])
    residual = design.beyond_rows(weights, anomaly)
    coefficients = design.statics.T @ (np.sqrt(weights)[:, None] * residual)
    static_part = (coefficients * coefficients).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        fixed_share = np.where(total > 0.0, fixed / total, np.nan)
        explained = np.where(
            fixed > 0.0, np.clip((zonal_part + static_part) / fixed, 0.0, 1.0), np.nan
        )
        if design.starts is not None:
            zonal = np.where(fixed > 0.0, np.clip(zonal_part / fixed, 0.0, 1.0), np.nan)
            zonal = np.minimum(zonal, explained)
        else:
            zonal = np.full(anomaly.shape[1], np.nan)
    static = np.maximum(explained - np.nan_to_num(zonal), 0.0)
    # A map that is flat has nothing fixed, and nothing of it unexplained.
    residual_share = np.where(fixed > 0.0, fixed_share * (1.0 - explained), 0.0)
    keep = np.where(complete & np.isfinite(fixed_share), 1.0, np.nan)
    present = np.where(complete, 1.0, np.nan)
    return {
        "mean": overall * present,
        "total_variance": total * present,
        "fixed_share": fixed_share * keep,
        "zonal_share": zonal * keep,
        "static_share": static * keep,
        "residual_fixed_share": residual_share * keep,
    }


def feature_stationarity(
    source: LatentSource,
    *,
    layer: int,
    basis: Decomposition | None = None,
    columns: Sequence[int] | None = None,
    times: Sequence[str | int] | None = None,
    statics: Sequence[str] = (),
    threshold: float = 0.0,
    chunk: int = 256,
    allow_unverified_basis: bool = False,
) -> Stationarity:
    """How much of each channel -- or each of a ``basis``'s features -- is fixed in
    place over ``times`` (all, by default), and how much of that latitude and the named
    ``statics`` explain. See ``Stationarity`` for what each number is.

    Columns are taken ``chunk`` at a time, reading every time once per chunk, so memory
    is a few ``(n_nodes, chunk)`` arrays however wide the basis is. A ``topk``
    dictionary encodes every feature to find a node's top ``k``, so each chunk pays for
    the whole encoder: a wide chunk is cheaper in time, a narrow one in memory. With
    ``statics``, only nodes where every one of them has a value are used, for every
    number; a static is a field of the source's ``ReferenceFields``, averaged over the
    times read. Nodes of no area (a pole row, on some grids) carry no weight and are
    left out. At least two times are needed: one time has no time mean to speak of.
    """
    info, grid = source.info(), source.grid()
    width = info.layer(layer).n_channels
    if basis is not None:
        check_basis_fits(basis, info, layer, allow_unverified=allow_unverified_basis)
    kind = "feature" if basis is not None else "channel"
    chosen = _columns(basis.n_features if basis is not None else width, columns, kind)
    if chunk < 1:
        raise RequestError("chunk must be at least 1")
    statics = tuple(statics)
    if len(set(statics)) != len(statics):
        raise RequestError("a static field was named twice")
    if statics:
        if not isinstance(source, ReferenceFields):
            raise RequestError(f"{info.source} keeps no physical fields to use as statics")
        known = source.field_names()
        if unknown := [s for s in statics if s not in known]:
            raise RequestError(
                f"no field(s) {unknown} in {info.source}; fields are {', '.join(known)}"
            )
    labels = _time_labels(source, times)
    if len(labels) < 2:
        raise RequestError(
            "at least two times are needed: at one, every column is its own time mean"
        )

    # The statics first: they decide which nodes every number is over.
    used = grid.valid & (grid.weights() > 0.0)  # raises where the mask leaves no node
    static_values, static_shares = [], []
    for name in statics:
        moments = _Moments(grid.n_nodes, 1)
        for label in labels:
            moments.add(_field_at(source, name, label)[:, None])
        static_values.append(moments.node_mean()[:, 0])
        used &= (moments.count[:, 0] == len(labels)) & np.isfinite(static_values[-1])
        static_shares.append(moments)
    nodes = np.flatnonzero(used)
    if nodes.size < 2:
        raise RequestError(
            "fewer than two valid nodes of any area"
            + (" where every static has a value" if statics else "")
        )
    weights = grid.weights()[nodes]
    weights = weights / weights.sum()

    design = _design(grid, nodes, weights, [values[nodes] for values in static_values])
    flat = _Design(None, None, np.zeros((nodes.size, 0)))

    static_fixed = (
        np.array(
            [
                _decompose(_restricted(m, nodes), weights, flat, len(labels))["fixed_share"][0]
                for m in static_shares
            ]
        )
        if statics
        else np.zeros(0)
    )

    keys = ("mean", "total_variance", "fixed_share", "zonal_share", "static_share",
            "residual_fixed_share")  # fmt: skip
    out = {key: np.full(chosen.size, np.nan) for key in keys}
    coverage = np.full(chosen.size, np.nan)
    for start in range(0, chosen.size, chunk):
        part = chosen[start : start + chunk]
        moments = _Moments(nodes.size, part.size)
        active = np.zeros(part.size)
        for label in labels:
            if basis is not None:
                values = np.empty((nodes.size, part.size))
                latents = read_latents(source, label, layer, nodes=nodes)
                for at in range(0, nodes.size, _BLOCK):
                    values[at : at + _BLOCK] = basis.transform(
                        latents[at : at + _BLOCK], features=part
                    )
                del latents
            else:
                values = read_latents(source, label, layer, channels=part, nodes=nodes)
                values = values.astype(np.float64)
            moments.add(values)
            on = np.isfinite(values) & (values > threshold)
            active += weights @ on
        decomposed = _decompose(moments, weights, design, len(labels))
        for key, value in decomposed.items():
            out[key][start : start + part.size] = value
        present = (moments.count == len(labels)).all(axis=0)
        coverage[start : start + part.size] = np.where(present, active / len(labels), np.nan)

    return Stationarity(
        settings={
            "layer": layer,
            "columns": f"{kind}s",
            "basis": None if basis is None else basis.meta.get("path"),
            "statics": list(statics),
            "threshold": threshold,
            "chunk": chunk,
            "allow_unverified_basis": allow_unverified_basis,
        },
        provenance=result_provenance(info, basis=basis),
        times=tuple(labels),
        columns=chosen,
        coverage=coverage,
        statics=statics,
        static_fixed_share=static_fixed,
        n_nodes_used=int(nodes.size),
        **out,
    )


def _restricted(moments: _Moments, nodes: np.ndarray) -> _Moments:
    """The same sums at only ``nodes``."""
    out = _Moments(nodes.size, moments.shift.shape[1])
    out.shift, out.count = moments.shift[nodes], moments.count[nodes]
    out.first, out.second = moments.first[nodes], moments.second[nodes]
    return out
