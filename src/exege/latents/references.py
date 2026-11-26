"""What a steer is measured against, or pushed toward, read from an archive.

A steering experiment edits a running model, but what it needs to know about "normal"
comes from a recording. Two things here:

- ``feature_reference``: one feature's activation at every node, averaged over chosen
  times -- its climatology over many times (a mean-ablation pulls a feature toward
  it), its value at one other time (a resample-ablation), or its level on quiet days
  of the same season (what ``Steer(mode="excess")`` scales the excess above).
- ``mean_difference``: the direction, in a layer's channels, from the mean latents at
  one set of times to the mean at another, over a region: the simplest direction that
  separates an event from its absence, with no dictionary involved. As a
  ``Directions`` basis it is a comparison arm of ``run_steering``, which asks whether a
  dictionary's feature does anything this plainer direction does not.

Both carry what they were read from, so a result built on them says so. Needs the
``latents`` extra.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from exege.core.errors import RequestError
from exege.core.extras import missing_extra
from exege.latents.analysis import AnyRegion
from exege.latents.basis import Decomposition, Directions, basis_provenance
from exege.latents.grid import node_digest
from exege.latents.samples import _time_labels
from exege.latents.source import LatentSource, check_basis_fits, read_latents

try:
    import numpy as np
except ImportError as exc:
    raise missing_extra("numpy", "latents") from exc

_BLOCK = 8192


def feature_reference(
    source: LatentSource,
    basis: Decomposition,
    *,
    layer: int,
    feature: int,
    times: Sequence[str | int],
    allow_unverified_basis: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """One feature's activation at each node, averaged over ``times``: ``(n_nodes,)``
    float64, NaN where the grid is masked. Returns it with a description of what it was
    read from (the source's provenance, the layer, the feature, the times and the basis),
    which ``Steer.reference_meta`` takes as it is.

    The description carries the grid's ``node_digest``: a steer refuses a reference read
    on nodes laid out otherwise than the system's, however many there are.

    One time is a resample: the feature as it was then. Many times are a climatology.
    Choosing them -- the same season in other years, days with no storm about -- is the
    caller's, and the reason it is written down."""
    info, grid = source.info(), source.grid()
    if not times:
        raise RequestError("a reference needs at least one time to read")
    check_basis_fits(basis, info, layer, allow_unverified=allow_unverified_basis)
    if not 0 <= feature < basis.n_features:
        raise RequestError(f"feature {feature} is not in 0..{basis.n_features - 1}")
    labels = _time_labels(source, times)
    nodes = np.flatnonzero(grid.valid)
    total = np.zeros(nodes.size)
    for label in labels:
        latents = read_latents(source, label, layer, nodes=nodes)
        for start in range(0, nodes.size, _BLOCK):
            block = latents[start : start + _BLOCK]
            total[start : start + _BLOCK] += basis.transform(block, [feature])[:, 0]
        del latents
    reference = np.full(grid.n_nodes, np.nan)
    reference[nodes] = total / len(labels)
    meta = {
        "source": info.provenance(), "layer": layer, "feature": feature,
        "times": list(labels), "basis": basis_provenance(basis), "grid": node_digest(grid),
    }  # fmt: skip
    return reference, meta


def _regional_mean(
    source: LatentSource, labels: Sequence[str], layer: int, nodes: np.ndarray, weights
) -> np.ndarray:
    total = None
    for label in labels:
        latents = read_latents(source, label, layer, nodes=nodes)
        mean = weights @ latents.astype(np.float64)
        total = mean if total is None else total + mean
    return total / len(labels)


def mean_difference(
    source: LatentSource,
    *,
    layer: int,
    event_times: Sequence[str | int],
    quiet_times: Sequence[str | int],
    region: AnyRegion | None = None,
    label: str = "mean difference",
) -> Directions:
    """The unit direction from the area-weighted mean latents at ``quiet_times`` to the
    mean at ``event_times``, over the valid nodes of ``region`` (default: the globe), as
    a one-direction ``Directions`` basis whose mean is the quiet one.

    It records where it was fitted, like any basis, so a steering experiment checks it
    belongs to the layer it is used in. The two sets of times must not overlap: a time
    in both would be a difference with itself."""
    info, grid = source.info(), source.grid()
    width = info.layer(layer).n_channels
    events, quiet = _time_labels(source, event_times), _time_labels(source, quiet_times)
    if shared := sorted(set(events) & set(quiet)):
        raise RequestError(f"time(s) {shared} are both event and quiet times")
    nodes = np.flatnonzero(grid.valid) if region is None else region.nodes(grid)
    if nodes.size == 0:
        where = "on the grid" if region is None else region.describe()
        raise RequestError(f"no valid nodes {where}; widen the region")
    weights = grid.weights()[nodes]
    weights = weights / weights.sum()
    event_mean = _regional_mean(source, events, layer, nodes, weights)
    quiet_mean = _regional_mean(source, quiet, layer, nodes, weights)
    difference = event_mean - quiet_mean
    norm = float(np.linalg.norm(difference))
    if norm == 0.0:
        raise RequestError("the event and quiet means are the same: there is no direction")
    return Directions(
        mean=quiet_mean.astype(np.float64),
        directions_=(difference / norm)[None, :].astype(np.float64),
        meta={
            "labels": [label],
            "length": norm,
            "fitted_on": {
                "provenance": info.provenance(),
                "layer": layer,
                "network_layer": info.layer(layer).position,
                "times": [*events, *quiet],
                "event_times": list(events),
                "quiet_times": list(quiet),
                "region": None if region is None else asdict(region),
                "n_channels": width,
            },
        },
    )
