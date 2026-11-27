"""Which field a feature goes with, measured against chance.

``feature_profile`` describes one feature by every field, and ``rank_by_field`` one field
by every feature. Neither says when a number is large enough to mean something, and a
fixed bar cannot: the mean of a field over the nodes where a feature fires has a noise
scale set by how many nodes those are, so one bar is strict for a feature that covers
half the world and lax for one that covers a few hundred nodes -- the compact features,
which are the ones worth steering. And a label taken as the field with the largest
association goes vacuous as fields are added: something always comes first.

``associate`` measures every association against its own null. The field is read again
at the node a whole number of columns east of every node -- the map turned west by that
many columns -- many times over: a rigid rotation
keeps the feature's footprint exactly (its size, its shape, its latitudes) and keeps the
field's own spatial structure, and only the east-west alignment of the two is broken. The
spread of the association over those rotations is the noise it has to clear, whatever
the footprint. From there:

- each (feature, field) has a z-score against its null and a two-sided p-value;
- Benjamini-Hochberg holds the false-discovery rate across the whole grid of them;
- a feature is labeled with its strongest field only if that association is significant
  *and* beats the runner-up by more than the same margin does, at the 95th percentile,
  when each rotation in turn is treated as if it were the data. Two fields that both
  follow a feature (moisture and ascent, say, which go together) leave it ``ambiguous``
  rather than decided by a coin toss;
- the whole labelling is run once more with one rotation (the one nearest half way
  round) as the data, where no association is real, against rotations *of that
  rotation* by the same shifts, less those that would come back near the data -- the
  exact procedure, on an input with nothing in it --
  and the share it labels is reported: it should sit near ``q`` or below, and if it does
  not, nothing else here is readable. (A narrow band of shifts, from a large
  ``min_shift_degrees``, is the usual way to fail it: the null then barely varies.)

**What the null cannot see.** It tests east-west alignment and nothing else. A feature
and a field that are both organised by latitude alone -- a polar feature and a polar
field -- keep their alignment under every rotation, so the association reads null even
when it is real. The raw association (``statistic``) is reported beside the z-score for
that reason: the first includes what latitude explains, the second only what it does not.
It needs a structured grid whose longitudes go evenly all the way round; a mesh is refused.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from exege.core.errors import RequestError
from exege.core.extras import missing_extra
from exege.latents.basis import Decomposition, result_provenance
from exege.latents.grid import _east_step, longitude_partner, rotation_shifts
from exege.latents.samples import _time_labels
from exege.latents.source import LatentSource, ReferenceFields, check_basis_fits, read_latents
from exege.latents.through import _field_at

try:
    import numpy as np
except ImportError as exc:
    raise missing_extra("numpy", "latents") from exc

_ORBIT_CHUNK = 128  # columns per FFT pass of the orbit null
_BLOCK = 2048  # nodes encoded at a time: a block of a 4096-feature dictionary is 32 MB
GAP_PERCENTILE = 95.0
NULLS = ("rotations", "orbit")
MIN_ROTATIONS = 4  # a null with a spread once any one draw is left out for the gap bar


def two_sided_p(z: np.ndarray) -> np.ndarray:
    """The two-sided Gaussian tail of a z-score, NaN where the z-score is."""
    z = np.asarray(z, dtype=np.float64)
    out = np.full(z.shape, np.nan)
    ok = np.isfinite(z)
    out[ok] = [math.erfc(abs(v) / math.sqrt(2.0)) for v in z[ok]]
    return out


def benjamini_hochberg(p: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Which p-values are discoveries at false-discovery rate ``q`` (Benjamini and
    Hochberg 1995), over every finite one; a NaN is never a discovery and does not count
    toward the number of tests."""
    p = np.asarray(p, dtype=np.float64)
    keep = np.zeros(p.shape, dtype=bool)
    flat = p.ravel()
    finite = np.flatnonzero(np.isfinite(flat))
    if finite.size == 0:
        return keep
    order = finite[np.argsort(flat[finite], kind="stable")]
    bar = q * np.arange(1, order.size + 1) / order.size
    below = np.flatnonzero(flat[order] <= bar)
    if below.size:
        keep.ravel()[order[: below[-1] + 1]] = True
    return keep


@dataclass(frozen=True, eq=False)
class _Labels:
    z: np.ndarray
    p: np.ndarray
    significant: np.ndarray
    top: np.ndarray  # field index of the largest |z|, -1 where none is finite
    gap: np.ndarray
    gap_bar: np.ndarray
    labeled: np.ndarray
    any_significant: np.ndarray


@contextmanager
def _quiet():
    """Silence numpy's warnings about empty slices and zero spreads: a column that is
    never active has no statistic, and reads NaN on purpose."""
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


def _standardized(observed: np.ndarray, null: np.ndarray) -> np.ndarray:
    with _quiet():
        mean, spread = np.nanmean(null, axis=-1), np.nanstd(null, axis=-1, ddof=1)
        # A null that reproduces the data to rounding (a field that depends on latitude
        # alone) has no spread to measure against, only rounding: no z-score.
        floor = 1e-9 * np.maximum(1.0, np.abs(mean))
        return np.where(spread > floor, (observed - mean) / spread, np.nan)


def _top_and_gap(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per row: the column of the largest ``|z|`` (-1 if none is finite), and how far it
    stands above the second (above zero, for a row with one finite column)."""
    size = np.where(np.isfinite(z), np.abs(z), -np.inf)
    order = np.argsort(-size, axis=1, kind="stable")
    rows = np.arange(z.shape[0])
    first = size[rows, order[:, 0]]
    second = size[rows, order[:, 1]] if z.shape[1] > 1 else np.zeros(z.shape[0])
    second = np.where(np.isfinite(second), second, 0.0)
    top = np.where(np.isfinite(first), order[:, 0], -1)
    gap = np.where(np.isfinite(first), first - second, np.nan)
    return top, gap


def _leave_one_out_gaps(null: np.ndarray) -> np.ndarray:
    """The runner-up gap with each rotation in turn as if it were the data, standardized
    against the other rotations only: a draw measured against a null that holds it is
    measured partly against itself, and reads too small. ``(columns, rotations)``.

    The leave-one-out mean and spread come from the sums over every draw less the one
    left out, so nothing of the size of the null is copied per rotation."""
    finite = np.isfinite(null)
    with _quiet():
        centre = np.nanmean(null, axis=-1)
    centre = np.where(np.isfinite(centre), centre, 0.0)
    x = np.where(finite, null - centre[..., None], 0.0)  # centred: no cancellation
    count = finite.sum(axis=-1).astype(np.float64)
    first, second = x.sum(axis=-1), (x * x).sum(axis=-1)
    gaps = np.full((null.shape[0], null.shape[-1]), np.nan)
    for r in range(null.shape[-1]):
        here, valid = x[..., r], finite[..., r]
        others = count - valid
        with _quiet():
            mean = (first - here) / others
            var = (second - here * here - others * mean * mean) / (others - 1.0)
            spread = np.sqrt(np.maximum(var, 0.0))
            floor = 1e-9 * np.maximum(1.0, np.abs(mean + centre))
            z = np.where(valid & (others >= 2) & (spread > floor), (here - mean) / spread, np.nan)
        gaps[:, r] = _top_and_gap(z)[1]
    return gaps


def _empirical_p(observed: np.ndarray, null: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A rank p-value over the data and its finite rotations: centre on the mean of all
    ``n + 1``, and count those at least as far from it as the data is (the data
    included), over ``n + 1``. Distances within rounding of each other are ties. NaN
    where the data is, or where no rotation is finite; also returns ``n``."""
    finite = np.isfinite(null)
    n = finite.sum(axis=-1)
    with _quiet():
        centre = np.nanmean(np.concatenate([observed[..., None], null], axis=-1), axis=-1)
        here = np.abs(observed - centre)
        there = np.abs(null - centre[..., None])
    tie = 1e-9 * np.maximum(1.0, np.abs(np.nan_to_num(centre)))
    beyond = finite & (there >= (here - tie)[..., None])
    p = (1.0 + beyond.sum(axis=-1)) / (n + 1.0)
    return np.where(np.isfinite(observed) & (n >= 1), p, np.nan), n


def _label(observed: np.ndarray, null: np.ndarray, q: float) -> _Labels:
    """Label every row of ``observed`` (columns x fields) against ``null`` (the same,
    with one more axis of rotations)."""
    z = _standardized(observed, null)
    p = two_sided_p(z)
    significant = benjamini_hochberg(p, q)
    top, gap = _top_and_gap(z)
    gaps = _leave_one_out_gaps(null)
    with _quiet():
        finite = np.isfinite(gaps).any(axis=1)
        bar = np.full(observed.shape[0], np.nan)
        if finite.any():
            bar[finite] = np.nanpercentile(gaps[finite], GAP_PERCENTILE, axis=1)
    rows = np.arange(observed.shape[0])
    top_significant = np.where(top >= 0, significant[rows, np.maximum(top, 0)], False)
    labeled = top_significant & np.isfinite(bar) & (gap > bar)
    return _Labels(z, p, significant, top, gap, bar, labeled, significant.any(axis=1))


def _moments(null: np.ndarray) -> tuple[float, float]:
    """Median ``|skew|`` and median excess kurtosis of the null, over every (column,
    field) whose null varies: how far the Gaussian p-value can be trusted."""
    with _quiet():
        centred = null - np.nanmean(null, axis=-1, keepdims=True)
        var = np.nanmean(centred**2, axis=-1)
        skew = np.nanmean(centred**3, axis=-1) / var**1.5
        kurt = np.nanmean(centred**4, axis=-1) / var**2 - 3.0
    ok = np.isfinite(skew) & np.isfinite(kurt) & (var > 0.0)
    if not ok.any():
        return float("nan"), float("nan")
    return float(np.median(np.abs(skew[ok]))), float(np.median(kurt[ok]))


@dataclass(frozen=True, eq=False)
class Association:
    """Every (column, field) association of a layer, measured against a rotation null.

    ``statistic[c, m]`` is the area-weighted mean of field ``m``, standardized at each
    time, over the node-times where column ``c`` is active; ``null_mean`` and
    ``null_sd`` are those of the same statistic over the ``shifts`` (the field read that
    many columns east of every node: the map turned west), and ``z`` is the statistic
    against them. ``p`` is two-sided Gaussian on ``z``, and ``p_empirical`` a rank: the
    share of the ``n + 1`` (the data and its ``n_null`` finite rotations) at least as far
    from their common mean as the data is, which never goes below ``1 / (n + 1)``.
    ``significant``
    holds the Benjamini-Hochberg discoveries at ``q`` across the whole table.

    ``label[c]`` is a field's name, ``"ambiguous"`` (something is significant but no
    field stands clear of the runner-up by ``gap_bar``), or ``"none"``.
    ``calibration_label_rate`` is the share of live columns the same procedure labels
    when one rotation is taken as the data, against its own rotations by the same shifts
    (``calibration_shift`` is that rotation, and ``calibration_rotations`` how many of its
    own rotations stay ``min_shift_degrees`` from the data: fewer than the main null's);
    ``null_shape`` is the median ``|skew|`` and
    excess kurtosis of the null, for how far a Gaussian p-value can be trusted.

    With ``settings["null"] == "orbit"``, ``feature_p`` is each column's ``p_orbit``, and
    ``label``, the top field and ``gap`` come from the orbit test (``gap`` in units of
    the orbit's own spread, not of ``z``); ``gap_bar`` does not apply and is NaN, and
    ``z``, ``p`` and ``significant`` are still the rotations null's, for the size and
    sign of each association. The calibration is the orbit test with the shift half way
    round as the data: averaged over every shift it labels at most ``q`` by construction,
    so it checks the computation, not whether every alignment is as likely as the data's.
    """

    settings: dict[str, Any]
    provenance: dict[str, Any]
    columns: tuple[int, ...]
    fields: tuple[str, ...]
    shifts: tuple[int, ...]
    coverage: np.ndarray
    statistic: np.ndarray
    null_mean: np.ndarray
    null_sd: np.ndarray
    z: np.ndarray
    p: np.ndarray
    p_empirical: np.ndarray
    n_null: np.ndarray
    significant: np.ndarray
    label: tuple[str, ...]
    gap: np.ndarray
    gap_bar: np.ndarray
    calibration_label_rate: float
    calibration_shift: int
    calibration_rotations: int
    null_shape: dict[str, float]
    feature_p: np.ndarray | None = None

    @property
    def n_labeled(self) -> int:
        return sum(1 for name in self.label if name not in ("ambiguous", "none"))

    def order(self) -> np.ndarray:
        """Row positions: labeled first, then ambiguous, then none; within each, by the
        largest ``|z|`` of the row."""
        rank = {"ambiguous": 1, "none": 2}
        size = np.nan_to_num(np.nanmax(np.where(np.isfinite(self.z), np.abs(self.z), -1), 1))
        keys = [(rank.get(name, 0), -float(size[i])) for i, name in enumerate(self.label)]
        return np.array(sorted(range(len(keys)), key=keys.__getitem__), dtype=np.intp)

    def summary(self, top: int | None = None) -> dict[str, Any]:
        def num(value: float) -> float | None:
            return float(value) if np.isfinite(value) else None

        rows = self.order()
        rows = rows if top is None else rows[: max(top, 0)]
        out = []
        for i in rows:
            z_row = np.where(np.isfinite(self.z[i]), np.abs(self.z[i]), -np.inf)
            best = int(np.argmax(z_row)) if np.isfinite(z_row).any() else None
            out.append(
                {
                    "column": self.columns[i],
                    "label": self.label[i],
                    "field": None if best is None else self.fields[best],
                    "z": None if best is None else num(self.z[i, best]),
                    "p": None if best is None else num(self.p[i, best]),
                    "gap": num(self.gap[i]),
                    "gap_bar": num(self.gap_bar[i]),
                    "p_orbit": None if self.feature_p is None else num(self.feature_p[i]),
                    "coverage": num(self.coverage[i]),
                    "by_field": {
                        name: {
                            "statistic": num(self.statistic[i, m]),
                            "z": num(self.z[i, m]),
                            "p": num(self.p[i, m]),
                            "p_empirical": num(self.p_empirical[i, m]),
                            "n_null": int(self.n_null[i, m]),
                            "significant": bool(self.significant[i, m]),
                        }
                        for m, name in enumerate(self.fields)
                    },
                }
            )
        return {
            "settings": self.settings,
            "provenance": self.provenance,
            "shifts": list(self.shifts),
            "n_labeled": self.n_labeled,
            "calibration_label_rate": num(self.calibration_label_rate),
            "calibration_shift": self.calibration_shift,
            "calibration_rotations": self.calibration_rotations,
            "null_shape": {k: num(v) for k, v in self.null_shape.items()},
            "columns": out,
        }


def calibration_shifts(
    shifts: Sequence[int], n_lon: int, min_degrees: float
) -> tuple[int, list[int]]:
    """The rotation the calibration takes as its data (the one nearest half way round),
    and the absolute shifts of its null: it turned again by each of ``shifts``, less any
    that would come back within ``min_degrees`` of the real alignment (shift 0), which
    would be the data itself and make the calibration look cleaner than it is."""
    spacing = 360.0 / n_lon
    held = shifts[int(np.argmin([abs(k * spacing - 180.0) for k in shifts]))]
    again = []
    for k in shifts:
        at = (held + k) % n_lon
        if at != 0 and min(at, n_lon - at) * spacing >= min_degrees - 1e-9:
            again.append(at)
    if len(again) < MIN_ROTATIONS:
        raise RequestError(
            f"only {len(again)} rotation(s) of the calibration stay {min_degrees:g} degrees "
            f"from the data; at least {MIN_ROTATIONS} are needed (more rotations, or a smaller "
            "--min-shift-degrees)"
        )
    return held, again


def _orbit_sums(
    grid, on_weight: np.ndarray, standard: np.ndarray, total: np.ndarray, weight: np.ndarray
) -> None:
    """Add one time's sums at every shift of the orbit to ``total`` and ``weight``
    (``(columns, fields * n_lon)``, shift fastest), as the partner reads would.

    Along a latitude row a shift is a circular cross-correlation, so each row's sums at
    every shift come from one real FFT of the active weights (``on_weight``, columns by
    nodes) and of the standardized fields (missing as zero, and their presence), and a
    product per frequency summed over the rows. Columns go a chunk at a time."""
    n_lat, n_lon = grid.shape
    _, sign = _east_step(grid)
    present = np.isfinite(standard)
    values = np.where(present, standard, 0.0).reshape(-1, n_lat, n_lon)
    f_values = np.fft.rfft(values, axis=-1).transpose(2, 1, 0)  # (freq, rows, fields)
    f_present = np.fft.rfft(present.reshape(-1, n_lat, n_lon).astype(np.float64), axis=-1)
    f_present = f_present.transpose(2, 1, 0)
    # node i's partner at shift k is (column + sign * k) mod n_lon: the correlation's lag
    lags = (sign * np.arange(n_lon)) % n_lon
    for start in range(0, on_weight.shape[0], _ORBIT_CHUNK):
        part = on_weight[start : start + _ORBIT_CHUNK].astype(np.float64)
        f_on = np.conj(np.fft.rfft(part.reshape(-1, n_lat, n_lon), axis=-1)).transpose(2, 0, 1)
        for spectrum, into in ((f_values, total), (f_present, weight)):
            product = np.matmul(f_on, spectrum)  # (freq, columns, fields)
            lagged = np.fft.irfft(product.transpose(1, 2, 0), n=n_lon, axis=-1)
            into[start : start + part.shape[0]] += lagged[..., lags].reshape(part.shape[0], -1)


def _orbit_test(statistic: np.ndarray, q: float, min_degrees: float, centre: int = 0) -> dict:
    """Each column's best field at shift ``centre`` against the whole orbit of shifts.

    ``statistic`` is ``(columns, fields, n_lon)``, one value per shift. Each pair is
    standardized over its orbit (so no shift is special), and at every shift its size is
    read against the shifts at least ``min_degrees`` away from it -- the same neighbourhood
    left out at every shift, the data's included. A column's severity at a shift is its
    largest such size plus that field's lead over the runner-up, and its ``p`` the share of
    the ``n_lon`` shifts whose severity is at least the one at ``centre``: under a null
    where every alignment round the circle is as likely as the data's, ``p <= q`` happens
    with probability at most ``q``, for each column. A column is labeled with its best
    field at ``p <= q`` and a positive lead. Pairs missing at any shift, or flat over the
    orbit, take no part."""
    n = statistic.shape[-1]
    valid = np.isfinite(statistic).all(axis=-1)
    x = np.where(valid[..., None], statistic, 0.0)
    mean, spread = x.mean(axis=-1), x.std(axis=-1)
    valid &= spread > 1e-9 * np.maximum(1.0, np.abs(mean))
    with _quiet():
        z = np.where(valid[..., None], (x - mean[..., None]) / spread[..., None], 0.0)
    shift = np.arange(n)
    far = (np.minimum(shift, n - shift) * 360.0 / n >= min_degrees - 1e-9).astype(np.float64)
    if far.sum() < 2:
        raise RequestError(f"fewer than two shifts lie {min_degrees:g} degrees from each other")
    kernel = np.fft.rfft(far)

    def around(a):  # the mean over the far shifts of each shift (far is symmetric)
        return np.fft.irfft(np.fft.rfft(a, axis=-1) * kernel, n=n, axis=-1) / far.sum()

    centre_mean = around(z)
    var = np.maximum(around(z * z) - centre_mean**2, 0.0)
    with _quiet():
        size = np.abs(z - centre_mean) / np.sqrt(np.maximum(var, 1e-24))
    size = np.where(valid[..., None], size, -np.inf)  # (columns, fields, shifts)
    order = np.argsort(-size, axis=1, kind="stable")
    first = np.take_along_axis(size, order[:, :1], axis=1)[:, 0]
    second = (
        np.take_along_axis(size, order[:, 1:2], axis=1)[:, 0]
        if size.shape[1] > 1
        else np.zeros_like(first)
    )
    second = np.where(np.isfinite(second), np.maximum(second, 0.0), 0.0)
    live = np.isfinite(first)
    gap = np.where(live, first - second, np.nan)
    severity = np.where(live, first + np.nan_to_num(gap), -np.inf)  # (columns, shifts)
    here = severity[:, centre]
    tie = 1e-9 * np.maximum(1.0, np.abs(np.where(np.isfinite(here), here, 0.0)))
    p = (severity >= (here - tie)[:, None]).sum(axis=1) / n
    p = np.where(live[:, centre], p, np.nan)
    gap_here = gap[:, centre]
    labeled = live[:, centre] & (p <= q) & (gap_here > 1e-9)
    top = np.where(live[:, centre], order[:, 0, centre], -1)
    return {"p": p, "label": labeled, "top": top, "gap": gap_here}


def associate(
    source: LatentSource,
    *,
    layer: int,
    basis: Decomposition | None = None,
    columns: Sequence[int] | None = None,
    fields: Sequence[str] | None = None,
    times: Sequence[str | int] | None = None,
    threshold: float = 0.0,
    n_rotations: int = 64,
    min_shift_degrees: float = 30.0,
    q: float = 0.05,
    lead: int = 0,
    allow_unverified_basis: bool = False,
    null: str = "rotations",
) -> Association:
    """Label the channels of a layer -- or the features of a ``basis``, or the
    ``columns`` named -- by the physical field each goes with, against a rotation null.

    Pooled over ``times`` (all, by default), on every valid node of the grid. "Active"
    is above ``threshold``. ``n_rotations`` shifts of at least ``min_shift_degrees`` are
    used (fewer if the grid has fewer), and the false-discovery rate is held at ``q``.
    ``lead`` sets the columns against each field that many reference times later, as
    ``feature_profile`` does. See the module's docstring for what the null tests and
    what it cannot see.
    """
    info, grid = source.info(), source.grid()
    if not isinstance(source, ReferenceFields):
        raise RequestError(f"{info.source} keeps no physical fields beside its latents")
    width = info.layer(layer).n_channels
    if basis is not None:
        check_basis_fits(basis, info, layer, allow_unverified=allow_unverified_basis)
        width = basis.n_features
    kind = "feature" if basis is not None else "channel"
    chosen = list(range(width)) if columns is None else [int(c) for c in columns]
    if not chosen:
        raise RequestError(f"no {kind}s were asked for")
    if bad := [c for c in chosen if not 0 <= c < width]:
        raise RequestError(f"layer {layer} has no {kind} {bad[0]}; there are {width}")
    if len(set(chosen)) != len(chosen):
        raise RequestError(f"a {kind} is named twice")
    known = source.field_names()
    names = tuple(known if fields is None else fields)
    if unknown := [n for n in names if n not in known]:
        raise RequestError(f"no field(s) {unknown} in {info.source}; fields are {', '.join(known)}")
    if not names:
        raise RequestError(f"{info.source} keeps no physical fields to set the {kind}s against")
    if not 0.0 < q < 1.0:
        raise RequestError(f"q is a false-discovery rate, between 0 and 1; not {q:g}")
    if null not in NULLS:
        raise RequestError(f"null must be one of {', '.join(NULLS)}; not {null!r}")
    null_kind, orbit = null, null == "orbit"  # ``null`` names the draws further down
    shifts = rotation_shifts(grid, n_rotations, min_shift_degrees)
    if len(shifts) < MIN_ROTATIONS:
        raise RequestError(
            f"{len(shifts)} rotation(s) allowed; at least {MIN_ROTATIONS} are needed for a null "
            "with a spread and a calibration (more rotations, or a smaller --min-shift-degrees)"
        )
    labels = _time_labels(source, times)
    # A map read at the partners of shift k is the field turned k columns west under a
    # footprint that stayed where it was. The calibration takes the rotation nearest half
    # way round as its data, and needs that rotation turned by every shift in turn.
    n_lon = grid.shape[1]
    if orbit and n_lon * q < 1.0:
        raise RequestError(
            f"the orbit has {n_lon} shift(s), so p_orbit is never below 1/{n_lon} = "
            f"{1.0 / n_lon:.3g}, more than q = {q:g}: nothing could be labeled (a larger q, or "
            "null='rotations')"
        )
    held, again = calibration_shifts(shifts, n_lon, min_shift_degrees)
    needed = list(range(n_lon)) if orbit else sorted({0, *shifts, *again})
    where = {k: i for i, k in enumerate(needed)}
    # the orbit reads every shift through FFTs along each row, not through partner indices
    partners = None if orbit else np.stack([longitude_partner(grid, k) for k in needed])
    n_cols, n_fields, n_reads = len(chosen), len(names), len(needed)

    nodes = np.flatnonzero(grid.valid)
    weights = grid.weights()
    total = np.zeros((n_cols, n_fields * n_reads))
    weight = np.zeros((n_cols, n_fields * n_reads))
    active_area = np.zeros(n_cols)
    finite_area = np.zeros(n_cols)
    for label in labels:
        on_full = np.zeros((n_cols, grid.n_nodes), dtype=np.float32) if orbit else None
        standard = np.full((n_fields, grid.n_nodes), np.nan)
        for m, name in enumerate(names):
            values = np.asarray(_field_at(source, name, label, lead), dtype=np.float64)
            ok = grid.valid & np.isfinite(values)
            w = weights[ok]
            if w.sum() <= 0.0:
                continue  # nothing of this field at this time: it adds nothing
            mean = float(w @ values[ok]) / float(w.sum())
            spread = math.sqrt(max(float(w @ (values[ok] - mean) ** 2) / float(w.sum()), 0.0))
            # A field the same everywhere has a spread of rounding alone, which would
            # standardize to noise of unit size: it carries nothing at this time.
            if spread > 1e-9 * float(np.abs(values[ok]).max()):
                standard[m, ok] = (values[ok] - mean) / spread
        if basis is None:
            latents = read_latents(source, label, layer, channels=chosen, nodes=nodes)
        else:
            latents = read_latents(source, label, layer, nodes=nodes)
        for start in range(0, nodes.size, _BLOCK):
            block = nodes[start : start + _BLOCK]
            here = latents[start : start + _BLOCK]
            values = (
                here.astype(np.float64)
                if basis is None
                else np.asarray(basis.transform(here, features=chosen), dtype=np.float64)
            )
            read = np.isfinite(values)
            on = read & (values > threshold)
            w = weights[block]
            finite_area += w @ read
            active_area += w @ on
            on_weight = (on * w[:, None]).astype(np.float64)  # (block, columns)
            if orbit:
                on_full[:, block] = on_weight.T
                continue
            # (block, fields, reads): every field at every rotation, for these nodes only
            rotated = standard[:, partners[:, block]].transpose(2, 0, 1)
            present = np.isfinite(rotated)
            rotated = np.where(present, rotated, 0.0).reshape(block.size, -1)
            total += on_weight.T @ rotated
            weight += on_weight.T @ present.reshape(block.size, -1).astype(np.float64)
        del latents
        if orbit:
            _orbit_sums(grid, on_full, standard, total, weight)

    with _quiet():
        statistic = np.where(weight > 0.0, total / weight, np.nan).reshape(
            n_cols, n_fields, n_reads
        )
        coverage = np.where(finite_area > 0.0, active_area / finite_area, np.nan)
    observed = statistic[..., where[0]]
    null = statistic[..., [where[k] for k in shifts]]
    main = _label(observed, null, q)
    feature_p = None
    if orbit:
        tested = _orbit_test(statistic, q, min_shift_degrees, centre=0)
        no_bar = np.full_like(main.gap_bar, np.nan)  # the orbit's lead needs no bar
        main = _Labels(main.z, main.p, main.significant, tested["top"], tested["gap"],
                       no_bar, tested["label"], tested["label"])  # fmt: skip
        feature_p = tested["p"]
    with _quiet():
        null_mean = np.nanmean(null, axis=-1)
        null_sd = np.nanstd(null, axis=-1, ddof=1)
    p_empirical, n_null = _empirical_p(observed, null)

    # The calibration: the exact procedure, with a rotation as the data and its own
    # rotations by the same shifts as the null.
    if orbit:  # the same test with the rotation half way round as the data
        fake_labeled = _orbit_test(statistic, q, min_shift_degrees, centre=held)["label"]
    else:
        fake = _label(statistic[..., where[held]], statistic[..., [where[k] for k in again]], q)
        fake_labeled = fake.labeled
    live = np.isfinite(observed).any(axis=1) & (np.nan_to_num(coverage) > 0.0)
    calibration = float(fake_labeled[live].mean()) if live.any() else float("nan")
    skew, kurtosis = _moments(null)

    label = tuple(
        names[main.top[i]]
        if main.labeled[i]
        else ("ambiguous" if main.any_significant[i] else "none")
        for i in range(n_cols)
    )
    return Association(
        settings={
            "layer": layer,
            "columns": "features" if basis is not None else "channels",
            "basis": None if basis is None else basis.meta.get("path"),
            "times": list(labels),
            "threshold": threshold,
            "n_rotations": n_rotations,
            "min_shift_degrees": min_shift_degrees,
            "q": q,
            "lead": lead,
            "allow_unverified_basis": allow_unverified_basis,
            "null": null_kind,
        },
        provenance=result_provenance(info, basis=basis),
        columns=tuple(chosen),
        fields=names,
        shifts=tuple(int(k) for k in shifts),
        coverage=coverage,
        statistic=observed,
        null_mean=null_mean,
        null_sd=null_sd,
        z=main.z,
        p=main.p,
        p_empirical=p_empirical,
        n_null=n_null,
        significant=main.significant,
        label=label,
        gap=main.gap,
        gap_bar=main.gap_bar,
        calibration_label_rate=calibration,
        calibration_shift=int(held),
        feature_p=feature_p,
        calibration_rotations=n_lon if orbit else len(again),
        null_shape={"median_abs_skew": skew, "median_excess_kurtosis": kurtosis},
    )
