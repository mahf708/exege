Figures API
===========

Needs the ``figures`` extra (matplotlib, and cartopy for coastlines). Every function returns
a plain ``matplotlib.figure.Figure`` and never touches ``pyplot``.

.. autofunction:: exege.figures.map_figure

.. autofunction:: exege.figures.series_figure

.. autofunction:: exege.figures.hovmoller_figure

.. autofunction:: exege.figures.layer_time_figure

.. autofunction:: exege.figures.profile_figure

.. autofunction:: exege.figures.response_figure

Helpers
-------

.. autofunction:: exege.figures.to_png

.. autofunction:: exege.figures.have_coastlines

.. autofunction:: exege.figures.why_no_coastlines
