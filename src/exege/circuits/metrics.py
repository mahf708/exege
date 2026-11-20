"""What a circuit is a circuit *for*: a scalar of one forward step's output.

A classifier has logits; an emulator has fields. A metric here is any function of the
fields a step wrote, ``{name: (batch, n_nodes)} -> float``. A ``LinearMetric`` is a weighted
sum over nodes, averaged over the batch: the area-weighted mean of a field over a region is
one, and it is the kind of metric whose gradient a system can be asked for
(``EdgeGradients``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from exege.core.errors import RequestError
from exege.core.extras import missing_extra
from exege.latents.grid import Grid

try:
    import numpy as np
except ImportError as exc:
    raise missing_extra("numpy", "circuits") from exc


@dataclass(frozen=True)
class LinearMetric:
    """``mean_b sum_f sum_n weights[f][n] * out[f][b, n]``.

    NaN in the output is read as 0 where the weight is 0 (a masked node), and is an error
    where it is not.
    """

    weights: Mapping[str, np.ndarray]
    label: str = ""
    settings: Mapping[str, Any] = field(default_factory=dict)

    def __call__(self, out: Mapping[str, np.ndarray]) -> float:
        total = 0.0
        for name, w in self.weights.items():
            if name not in out:
                raise RequestError(f"the metric reads {name!r}, which the step did not write")
            x = np.asarray(out[name], dtype=np.float64)
            x = x.reshape(-1, x.shape[-1])
            used = w != 0
            if not np.isfinite(x[:, used]).all():
                raise RequestError(
                    f"{name} is not finite where {self.label or 'the metric'} reads it"
                )
            total += float((x[:, used] @ w[used]).mean())
        return total

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "linear",
            "label": self.label,
            "fields": sorted(self.weights),
            **dict(self.settings),
        }


def field_mean(
    grid: Grid, name: str, where: np.ndarray | None = None, label: str = ""
) -> LinearMetric:
    """The area-weighted mean of field ``name`` over the valid nodes in ``where`` -- node
    indices (``grid.within(50, -30, 1500)``) or a boolean per node -- or over all of them."""
    keep = grid.valid.copy()
    if where is not None:
        w = np.asarray(where)
        if w.dtype == bool:
            if w.shape != (grid.n_nodes,):
                raise RequestError(f"a boolean where must be ({grid.n_nodes},), got {w.shape}")
            keep &= w
        else:
            if w.size and (w.min() < 0 or w.max() >= grid.n_nodes):
                raise RequestError(f"node indices must lie in 0..{grid.n_nodes - 1}")
            chosen = np.zeros(grid.n_nodes, dtype=bool)
            chosen[w.astype(int)] = True
            keep &= chosen
    if not keep.any():
        raise RequestError("the region holds no valid node")
    w = np.where(keep, grid.weights(), 0.0)
    w = w / w.sum()
    return LinearMetric(
        {name: w}, label or f"mean {name}", {"field": name, "n_nodes": int(keep.sum())}
    )


def recovery(value: float, clean: float, corrupt: float) -> float:
    """How much of the clean-vs-corrupt difference a patched value keeps: 1 is all of it
    (the clean run's value), 0 none (the source's). NaN when the two runs agree."""
    gap = clean - corrupt
    return float("nan") if gap == 0 else (value - corrupt) / gap
