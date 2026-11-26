"""The gap between a run driven by reference inputs and one left to run free.

An emulator rolled out on its own feeds itself its own output. The same network fed
the reference state at every step (teacher forcing) sees what it was trained on. Set
column for column, the two say which channels or features free running weakens or
strengthens, and whether that grows with lead time: where a deterministic emulator,
trained to the mean, lets what it carries fade.

What a gap says, and does not: a column that is weaker in the free run is where the
emulator's own states drive the network differently from the reference states. That
is not by itself an error -- a free run is supposed to wander from the reference --
and a gap in a feature is a gap in what the network computes, not in the weather. The
teacher-forced archive is written by the exporter, outside exege; this module only
sets the two archives against each other, which must be one network on one grid.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from exege import __version__
from exege.core.errors import RequestError
from exege.core.extras import missing_extra
from exege.latents.analysis import AnyRegion
from exege.latents.basis import Decomposition, basis_provenance
from exege.latents.source import (
    LatentSource,
    check_basis_fits,
    check_comparable,
    read_latents,
    shared_grid,
)
from exege.latents.stationary import _columns
from exege.latents.through import _common_times, _elapsed

try:
    import numpy as np
except ImportError as exc:
    raise missing_extra("numpy", "latents") from exc

_BLOCK = 8192
REFERENCE, RUN = "reference", "run"


@dataclass(frozen=True, eq=False)
class FeatureGap:
    """Each column in a reference archive (teacher forced, say) and in a run (free).

    Per time, ``(n_times, n_columns)`` for each side: ``*_mean`` is the area-weighted
    mean activation, ``*_coverage`` the share of the area active (above the
    threshold), ``*_strength`` the mean where active (NaN where it never is). At each
    time a column is read over the nodes where *both* sides have a value for it, so the
    two are always set against each other over one area. Pooled over the times, every
    time counting equally, ``mean``, ``coverage``, ``strength`` and ``rms`` (the root
    mean square: the size of what a column carries, whatever its sign) are
    ``(2, n_columns)``, the reference first.

    ``ratio_mean`` and ``ratio_strength`` are the run's over the reference's (NaN where
    the reference's is zero), ``difference`` the run's mean less the reference's, and
    ``deficit`` one less the ratio of the run's root mean square to the reference's:
    0.5 where the run carries half as much, negative where it carries more. It is read
    from magnitudes, not the mean, so a channel that is negative, or that swings about
    zero, and is damped reads as weakened, as a feature does. ``reference_rms`` is
    ``rms[0]``. ``elapsed_seconds`` is the reference's, where its calendar can say.
    """

    settings: dict[str, Any]
    provenance: dict[str, Any]
    times: tuple[str, ...]
    elapsed_seconds: tuple[float, ...] | None
    columns: np.ndarray
    reference_mean: np.ndarray
    run_mean: np.ndarray
    reference_coverage: np.ndarray
    run_coverage: np.ndarray
    reference_strength: np.ndarray
    run_strength: np.ndarray
    mean: np.ndarray
    coverage: np.ndarray
    strength: np.ndarray
    reference_rms: np.ndarray
    rms: np.ndarray
    ratio_mean: np.ndarray
    ratio_strength: np.ndarray
    difference: np.ndarray
    deficit: np.ndarray

    def ranked(self, top: int | None = None) -> np.ndarray:
        """Positions into ``columns``, the one free running weakens most first; columns
        with no deficit to speak of (silent in the reference) come last."""
        order = np.argsort(np.nan_to_num(self.deficit, nan=-np.inf), kind="stable")[::-1]
        return order if top is None else order[: max(top, 0)]

    def summary(self, top: int | None = None) -> dict[str, Any]:
        def number(value: float) -> float | None:
            return None if not np.isfinite(value) else float(value)

        def series(values: np.ndarray, i: int) -> list[float | None]:
            return [number(v) for v in values[:, i]]

        return {
            "settings": self.settings,
            "provenance": self.provenance,
            "times": list(self.times),
            "elapsed_seconds": None if self.elapsed_seconds is None else list(self.elapsed_seconds),
            "columns": [
                {
                    "column": int(self.columns[i]),
                    "deficit": number(self.deficit[i]),
                    "ratio_mean": number(self.ratio_mean[i]),
                    "ratio_strength": number(self.ratio_strength[i]),
                    "difference": number(self.difference[i]),
                    REFERENCE: {
                        "mean": number(self.mean[0, i]),
                        "coverage": number(self.coverage[0, i]),
                        "strength": number(self.strength[0, i]),
                        "rms": number(self.rms[0, i]),
                        "mean_by_time": series(self.reference_mean, i),
                    },
                    RUN: {
                        "mean": number(self.mean[1, i]),
                        "coverage": number(self.coverage[1, i]),
                        "strength": number(self.strength[1, i]),
                        "rms": number(self.rms[1, i]),
                        "mean_by_time": series(self.run_mean, i),
                    },
                }
                for i in self.ranked(top)
            ],
        }


def _values(
    source: LatentSource,
    label: str,
    layer: int,
    nodes: np.ndarray,
    columns: np.ndarray,
    basis: Decomposition | None,
):
    """The columns at one time, a block of nodes at a time: a wide basis is never held
    whole across the nodes, only the latents it reads."""
    if basis is not None:
        latents = read_latents(source, label, layer, nodes=nodes)
    else:
        latents = read_latents(source, label, layer, channels=columns, nodes=nodes)
    for start in range(0, nodes.size, _BLOCK):
        block = latents[start : start + _BLOCK]
        if basis is not None:
            yield basis.transform(block, features=columns)
        else:
            yield block.astype(np.float64)


def _one_time(
    sources: tuple[LatentSource, LatentSource],
    label: str,
    layer: int,
    nodes: np.ndarray,
    weights: np.ndarray,
    columns: np.ndarray,
    basis: Decomposition | None,
    threshold: float,
) -> np.ndarray:
    """Area-weighted sums at one time, ``(2, 5, n_columns)``: weight read, total, square,
    active weight, active total, for each side, over the nodes where both have a value."""
    out = np.zeros((2, 5, columns.size))
    blocks = zip(*(_values(s, label, layer, nodes, columns, basis) for s in sources), strict=True)
    for start, pair in zip(range(0, nodes.size, _BLOCK), blocks, strict=True):
        w = weights[start : start + _BLOCK]
        ok = np.isfinite(pair[0]) & np.isfinite(pair[1])
        for side, values in enumerate(pair):
            values = np.where(ok, values, 0.0)
            on = ok & (values > threshold)
            out[side] += (
                w @ ok, w @ values, w @ (values * values), w @ on,
                w @ np.where(on, values, 0.0),
            )  # fmt: skip
    return out


def feature_gap(
    reference: LatentSource,
    run: LatentSource,
    *,
    layer: int,
    basis: Decomposition | None = None,
    columns: Sequence[int] | None = None,
    times: Sequence[str | int] | None = None,
    region: AnyRegion | None = None,
    threshold: float = 0.0,
    across_models: bool = False,
    allow_unverified: bool = False,
    allow_unverified_basis: bool = False,
) -> FeatureGap:
    """Every channel -- or every one of a ``basis``'s features -- in ``reference`` (a
    teacher-forced archive, say) against ``run`` (a free one), at the times both keep
    (``times``: all the reference keeps that the run does too, by default), over the
    nodes valid in both, or those of a ``region``.

    The two must be one network on one grid (``check_comparable``, which says what
    ``across_models`` and ``allow_unverified`` lift), and a basis must fit both. See
    ``FeatureGap`` for what is measured.
    """
    check_comparable(
        reference, run, layer=layer, across_models=across_models, allow_unverified=allow_unverified
    )
    infos = (reference.info(), run.info())
    width = infos[0].layer(layer).n_channels
    if basis is not None:
        for info in infos:
            check_basis_fits(basis, info, layer, allow_unverified=allow_unverified_basis)
    kind = "feature" if basis is not None else "channel"
    chosen = _columns(basis.n_features if basis is not None else width, columns, kind)
    labels = _common_times(reference, run, times)
    grid = shared_grid(reference, run)
    area = grid.weights()  # raises where the shared mask leaves no node
    nodes = np.flatnonzero(grid.valid) if region is None else region.nodes(grid)
    nodes = nodes[area[nodes] > 0.0]  # a node of no area says nothing
    if nodes.size == 0:
        where = "" if region is None else f" {region.describe()}"
        raise RequestError(f"no node of any area valid in both runs{where}; widen the region")
    weights = area[nodes]
    weights = weights / weights.sum()

    shape = (2, len(labels), chosen.size)
    mean, coverage, strength = (
        np.full(shape, np.nan),
        np.full(shape, np.nan),
        np.full(shape, np.nan),
    )
    # Pooled: each time's sums divided by the area read then, so every time counts
    # equally whatever was missing at it; ``times`` counts the times with any area.
    sums, counted = np.zeros((2, 4, chosen.size)), np.zeros(chosen.size)
    for t, label in enumerate(labels):
        read, total, square, area_on, active_total = _one_time(
            (reference, run), label, layer, nodes, weights, chosen, basis, threshold
        ).transpose(1, 0, 2)
        seen = read[0] > 0.0  # the same area on both sides, by construction
        with np.errstate(invalid="ignore", divide="ignore"):
            mean[:, t] = np.where(seen, total / read, np.nan)
            coverage[:, t] = np.where(seen, area_on / read, np.nan)
            strength[:, t] = np.where(area_on > 0.0, active_total / area_on, np.nan)
            share = np.where(seen, 1.0 / read, 0.0)
        sums += np.stack([total, square, area_on, active_total], axis=1) * share[:, None, :]
        counted += seen

    total, square, area_on, active_total = sums.transpose(1, 0, 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        times_seen = np.where(counted > 0.0, counted, np.nan)
        pooled_mean = total / times_seen
        pooled_coverage = area_on / times_seen
        pooled_strength = np.where(area_on > 0.0, active_total / area_on, np.nan)
        rms = np.sqrt(square / times_seen)
        ratio_mean = np.where(pooled_mean[0] != 0.0, pooled_mean[1] / pooled_mean[0], np.nan)
        ratio_strength = np.where(
            pooled_strength[0] != 0.0, pooled_strength[1] / pooled_strength[0], np.nan
        )
        difference = pooled_mean[1] - pooled_mean[0]
        deficit = np.where(rms[0] > 0.0, 1.0 - rms[1] / rms[0], np.nan)

    provenance = {
        REFERENCE: infos[0].provenance(),
        RUN: infos[1].provenance(),
        "exege": __version__,
    }
    if basis is not None:
        provenance["basis"] = basis_provenance(basis)
    return FeatureGap(
        settings={
            "layer": layer,
            "columns": f"{kind}s",
            "basis": None if basis is None else basis.meta.get("path"),
            "region": None if region is None else asdict(region),
            "threshold": threshold,
            "across_models": across_models,
            "allow_unverified": allow_unverified,
            "allow_unverified_basis": allow_unverified_basis,
            "n_nodes_compared": int(nodes.size),
        },
        provenance=provenance,
        times=tuple(labels),
        elapsed_seconds=_elapsed(reference, labels),
        columns=chosen,
        reference_mean=mean[0],
        run_mean=mean[1],
        reference_coverage=coverage[0],
        run_coverage=coverage[1],
        reference_strength=strength[0],
        run_strength=strength[1],
        mean=pooled_mean,
        coverage=pooled_coverage,
        strength=pooled_strength,
        reference_rms=rms[0],
        rms=rms,
        ratio_mean=ratio_mean,
        ratio_strength=ratio_strength,
        difference=difference,
        deficit=deficit,
    )
