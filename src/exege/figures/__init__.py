"""figures -- reusable figures, independent of any web framework.

    from exege.figures import map_figure

    fig = map_figure(source.grid(), result.similarity, region=region, title="similarity")
    fig.savefig("similarity.png")

Needs the ``figures`` extra (matplotlib, and cartopy for coastlines).
"""

from __future__ import annotations

from exege.figures.bars import profile_figure, response_figure
from exege.figures.grids import hovmoller_figure, layer_time_figure
from exege.figures.maps import have_coastlines, map_figure, to_png, why_no_coastlines
from exege.figures.series import series_figure
from exege.figures.training import loss_figure

__all__ = [
    "have_coastlines",
    "hovmoller_figure",
    "layer_time_figure",
    "loss_figure",
    "map_figure",
    "profile_figure",
    "response_figure",
    "series_figure",
    "to_png",
    "why_no_coastlines",
]
