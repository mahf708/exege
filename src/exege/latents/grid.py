"""Nodes on a sphere: coordinates, validity, area weights, regions.

Shared by every diagnostic here, whether it looks at a model's outputs or at its
internals. Two rules are built in rather than left to callers, because both fail
silently when forgotten:

- **Area-weight.** Nodes of a latitude-longitude grid crowd toward the poles, so
  an unweighted mean over them is simply wrong.
- **Respect the mask.** Ocean quantities are undefined over land (about 31% of
  points in the E3SMv3 configuration); a plain mean returns NaN or, worse, a
  plausible biased number.

Nodes are always flat, ``(n_nodes,)``. A structured grid additionally records its
``shape``, so maps can be drawn and exact cell areas computed; a mesh (GraphCast's
icosahedron, say) records none and works with everything else unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

from exege.core.errors import RequestError
from exege.core.extras import missing_extra

try:
    import numpy as np
except ImportError as exc:
    raise missing_extra("numpy", "latents") from exc

EARTH_RADIUS_KM = 6371.0
_BLOCK = 8192  # nodes per step of a blocked reduction
_SAME_DEGREES = 1e-6  # coordinates closer than this are one row, or one column


def cell_area_weights(lat_1d: np.ndarray) -> np.ndarray:
    """Relative area of each latitude band of a structured grid.

    Band edges sit midway between neighboring latitudes and at the poles, and a
    band's area goes as the difference in ``sin(lat)`` across it, in either
    latitude order. These are the exact cell areas of a regular grid, and unlike
    ``cos(lat)`` they do not give a row that sits on a pole zero weight.

    On a Gaussian grid they are not the quadrature weights, but measured against
    them (n=180) every row agrees to better than 0.1% except the two polar ones,
    which are 6% off while holding 1e-5 of the sphere; means of smooth fields
    agree to 1e-5, where not weighting at all is wrong by 1e-1. A source that
    knows its true weights should pass them as ``Grid.area`` instead.
    """
    lat = np.asarray(lat_1d, dtype=np.float64)
    if lat.ndim != 1 or lat.size < 2:
        raise ValueError("need at least two latitudes to form bands")
    ends = (-90.0, 90.0) if lat[0] < lat[-1] else (90.0, -90.0)
    edges = np.concatenate(([ends[0]], (lat[:-1] + lat[1:]) / 2.0, [ends[1]]))
    return np.abs(np.diff(np.sin(np.radians(edges))))


def great_circle_km(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float) -> np.ndarray:
    """Haversine distance from every node to one point, in kilometres."""
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    lat0_r, lon0_r = np.radians(lat0), np.radians(lon0)
    a = (
        np.sin((lat_r - lat0_r) / 2.0) ** 2
        + np.cos(lat0_r) * np.cos(lat_r) * np.sin((lon_r - lon0_r) / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def small_circle(
    lat0: float, lon0: float, radius_km: float, n: int = 181
) -> tuple[np.ndarray, np.ndarray]:
    """The outline of a region: ``n`` points ``radius_km`` from a center, as
    ``(lat, lon)`` with longitude in -180..180. It is a circle on the sphere, so
    on a map it flattens toward the poles and may cross the dateline."""
    bearing = np.linspace(0.0, 2.0 * np.pi, n)
    delta = radius_km / EARTH_RADIUS_KM
    lat0_r, lon0_r = np.radians(lat0), np.radians(lon0)
    sin_lat = np.sin(lat0_r) * np.cos(delta) + np.cos(lat0_r) * np.sin(delta) * np.cos(bearing)
    lat = np.arcsin(np.clip(sin_lat, -1.0, 1.0))
    lon = lon0_r + np.arctan2(
        np.sin(bearing) * np.sin(delta) * np.cos(lat0_r), np.cos(delta) - np.sin(lat0_r) * sin_lat
    )
    return np.degrees(lat), (np.degrees(lon) + 180.0) % 360.0 - 180.0


def _east_step(grid: Grid) -> tuple[int, int]:
    """``(n_lon, +1 or -1)``: how many columns a row has, and which way along a row is
    east. Refused unless the grid is structured and its columns sit at evenly spaced
    longitudes that go all the way round."""
    if grid.shape is None:
        raise RequestError(
            "a longitude shift needs a structured (lat, lon) grid; this one is a mesh, whose "
            "nodes do not map onto each other under a rotation"
        )
    n_lon = grid.shape[1]
    if n_lon < 2:
        raise RequestError("a grid of one longitude cannot be shifted")
    lon = np.asarray(grid.lon, dtype=np.float64).reshape(grid.shape)[0]
    spacing = 360.0 / n_lon
    steps = np.diff(np.concatenate([lon, lon[:1]])) % 360.0
    for sign, step in ((1, spacing), (-1, 360.0 - spacing)):
        if np.allclose(steps, step, atol=1e-6 * 360.0):
            return n_lon, sign
    raise RequestError(
        f"a longitude shift needs {n_lon} evenly spaced longitudes all the way round "
        f"({spacing:g} degrees apart); this grid's are not"
    )


def longitude_partner(grid: Grid, columns: int) -> np.ndarray:
    """For every node, the index of the node ``columns`` columns east of it in the same
    row (negative: west), wrapping round the dateline.

    ``values[longitude_partner(grid, k)]`` is a map whose node ``i`` holds what sat ``k``
    columns east of ``i``: the map moved ``k`` columns *west*. On a structured grid with
    evenly spaced longitudes that cover the circle the mapping is exact, a permutation of
    the nodes that keeps every latitude; anything else -- a mesh, a regional grid -- is a
    ``RequestError``, since there a rotation does not take nodes onto nodes.
    """
    n_lon, sign = _east_step(grid)
    n_lat = grid.shape[0]
    rows = np.repeat(np.arange(n_lat), n_lon)
    cols = np.tile(np.arange(n_lon), n_lat)
    return rows * n_lon + (cols + sign * int(columns)) % n_lon


def rotation_shifts(grid: Grid, n: int, min_degrees: float = 30.0) -> tuple[int, ...]:
    """Up to ``n`` distinct eastward shifts, in columns, for a rotation null.

    Every shift moves the grid at least ``min_degrees`` of longitude away from where it
    was, either way round: a shift of one column leaves a smooth field almost where it
    was, and a null drawn from such shifts reproduces the alignment it is meant to break.
    Among those allowed, ``n`` are taken evenly spaced (all, when there are fewer).
    """
    n_lon, _ = _east_step(grid)
    if n < 1:
        raise RequestError(f"at least one rotation is needed, not {n}")
    if not 0.0 <= min_degrees <= 180.0:
        raise RequestError(f"min_degrees must lie in 0..180, not {min_degrees:g}")
    spacing = 360.0 / n_lon
    allowed = [k for k in range(1, n_lon) if min(k, n_lon - k) * spacing >= min_degrees - 1e-9]
    if not allowed:
        raise RequestError(
            f"no shift of this grid's {n_lon} columns moves it {min_degrees:g} degrees or more"
        )
    if len(allowed) <= n:
        return tuple(allowed)
    picks = np.unique(np.round(np.linspace(0, len(allowed) - 1, n)).astype(int))
    return tuple(allowed[i] for i in picks)


@dataclass(frozen=True, eq=False)
class Grid:
    """Where a set of nodes is, and which of them mean anything.

    ``lon`` may follow either convention (0..360 or -180..180); distances do not
    care. ``mask`` is True where a node carries information, and ``area`` is an
    explicit per-node area for meshes whose cells are far from uniform.
    """

    lat: np.ndarray
    lon: np.ndarray
    shape: tuple[int, int] | None = None
    mask: np.ndarray | None = None
    area: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = self.lat.shape
        if self.lat.ndim != 1 or self.lon.shape != n:
            raise ValueError(f"lat and lon must be flat and alike, got {n} and {self.lon.shape}")
        if self.shape is not None:
            if self.shape[0] * self.shape[1] != n[0]:
                raise ValueError(f"grid shape {self.shape} does not hold {n[0]} nodes")
            # Band areas and maps both read rows as latitudes. Nodes stored the
            # other way round reshape without complaint and weight wrongly.
            lat, lon = self.lat.reshape(self.shape), self.lon.reshape(self.shape)
            along_rows = float((lat.max(axis=1) - lat.min(axis=1)).max())
            along_columns = float((lon.max(axis=0) - lon.min(axis=0)).max())
            if max(along_rows, along_columns) > _SAME_DEGREES:
                raise ValueError(
                    f"grid shape {self.shape} must be (n_lat, n_lon) in C order, with latitude "
                    "constant along a row and longitude along a column; these nodes are not "
                    "(transposed, or a mesh that should have no shape)"
                )
        for name in ("mask", "area"):
            extra = getattr(self, name)
            if extra is not None and extra.shape != n:
                raise ValueError(f"{name} has shape {extra.shape}, expected {n}")

    @property
    def n_nodes(self) -> int:
        return int(self.lat.size)

    @property
    def valid(self) -> np.ndarray:
        """Boolean per node; all True when there is no mask."""
        return np.ones(self.n_nodes, dtype=bool) if self.mask is None else self.mask.astype(bool)

    def weights(self) -> np.ndarray:
        """Per-node area weights: zero where invalid, summing to one.

        Explicit ``area`` if the grid has it, exact cell areas if it is
        structured, and otherwise uniform -- right for a quasi-uniform mesh and
        the honest default when nothing better is known.
        """
        if self.area is not None:
            w = np.asarray(self.area, dtype=np.float64).copy()
        elif self.shape is not None:
            rows = cell_area_weights(self.lat.reshape(self.shape)[:, 0])
            w = np.repeat(rows / self.shape[1], self.shape[1])
        else:
            w = np.ones(self.n_nodes, dtype=np.float64)
        w[~self.valid] = 0.0
        total = w.sum()
        if total <= 0.0:
            raise RequestError("no valid nodes: the mask leaves every node out")
        return w / total

    def distance_km(self, lat: float, lon: float) -> np.ndarray:
        return great_circle_km(self.lat, self.lon, lat, lon)

    def within(self, lat: float, lon: float, radius_km: float) -> np.ndarray:
        """Indices of the valid nodes within ``radius_km`` of a point."""
        return np.flatnonzero((self.distance_km(lat, lon) <= radius_km) & self.valid)

    def nearest(self, lat: float, lon: float) -> int:
        """Index of the valid node closest to a point."""
        distance = np.where(self.valid, self.distance_km(lat, lon), np.inf)
        return int(np.argmin(distance))

    def mean(self, values: np.ndarray) -> np.ndarray:
        """Area-weighted mean over valid nodes, along the node axis (the first).
        Invalid nodes are ignored whatever they hold, NaN included.

        Accumulated in float64 a block of nodes at a time, so the precision does
        not depend on the input's dtype and a large float32 layer is never
        copied whole.
        """
        values = np.asarray(values)
        if values.shape[:1] != (self.n_nodes,):
            raise ValueError(f"expected {self.n_nodes} nodes first, got shape {values.shape}")
        w = self.weights()
        total = np.zeros(values.shape[1:], dtype=np.float64)
        for start in range(0, self.n_nodes, _BLOCK):
            keep = np.flatnonzero(w[start : start + _BLOCK] > 0.0) + start
            if keep.size:
                block = values[keep].astype(np.float64)
                total += np.tensordot(w[keep], block, axes=(0, 0))
        return total

    def to_map(self, values: np.ndarray) -> np.ndarray:
        """Per-node values as a ``(n_lat, n_lon)`` float map, NaN where invalid."""
        if self.shape is None:
            raise ValueError("this grid is a mesh; it has no (lat, lon) shape to map onto")
        out = np.array(values, dtype=np.float64).reshape(self.shape)
        out[~self.valid.reshape(self.shape)] = np.nan
        return out
