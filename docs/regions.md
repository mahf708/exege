# Analysing a region

Rank channels by their peak absolute activation within 1500 km of a point in the
equatorial Pacific, and fit three principal components there:

```console
$ exege latents region latents/atmosphere --lat 5 --lon -140 --radius-km 1500 \
    --centered --top 6 --pcs 3
572 node(s) at layer 8, time 0425-01-03T18:00:00

RANK  CHANNEL  PEAK_ABS
1     45       2.021
2     107      1.626
3     108      1.577
4     248      1.545
5     224      1.352
6     217      1.314

PC0   25.0%  45(+0.21)  109(-0.18)  193(-0.16)  330(-0.16)  56(-0.15)  224(-0.14)

PC1   14.4%  108(+0.19)  287(+0.15)  45(-0.14)  135(-0.14)  148(-0.14)  356(+0.14)

PC2   10.6%  326(+0.21)  351(-0.19)  179(+0.19)  336(+0.15)  349(-0.14)  344(+0.14)
```

This takes 0.3 s and peaks near 275 MB resident (`/usr/bin/time -l`) on an Apple M1 Max,
for an archive whose layers total 7.6 GB: one layer at one time is ever in memory, and
everything that can be refused — no such layer, an empty region, more components than the
region's nodes can carry — is refused before the first read. `--json` emits the settings,
provenance and results, which is enough to rerun an analysis and to check that the rerun
agrees.

## Options

| Option | Meaning |
| --- | --- |
| `--time` | a time label, or a position (`0`, `-1`) |
| `--layer` | the layer similarity and the features are computed at; the last by default |
| `--rank-layer` | the layer channels are ranked at; the last by default — what the network ends up emphasizing. It must be as wide as `--layer`: a channel is followed from one to the other by its index, which only means something along a residual stream |
| `--centered` | remove each channel's area-weighted global mean first |
| `--pin` | list a channel first whatever it scores, to follow it across layers |
| `--reference` | what "the region" is as one vector: the `nearest` node to its center, or its area-weighted `mean` |
| `--pcs`, `--features` | how many features to map: principal components fitted in the region or, with `--basis`, the features of that basis which respond most strongly there |
| `--basis` | a [basis file](bases.md): a global PCA, a sparse autoencoder |

Every option is on the [command-line reference](cli.rst) too.

## From Python

```python
from exege.latents import Region, analyze_region, open_source

source = open_source("latents/atmosphere")
result = analyze_region(
    source,
    time="0425-01-03T18:00:00",
    layer=8,
    region=Region(lat=5, lon=-140, radius_km=1500),
    centered=True,
    n_components=3,
)

result.ranking.channels  # which channels respond in the region
result.similarity  # per node: where else the model looks like this
result.feature_info  # per feature: its label, size, and the channels that weigh most
result.summary()  # settings + provenance + results, JSON-ready

grid = source.grid()
first_pc = grid.to_map(result.scores[:, 0])  # (n_lat, n_lon), NaN where invalid
```

To draw any of it, `exege.figures.map_figure(grid, values, region=...)` returns a
matplotlib figure, and [`exege app`](app.md) puts the whole routine behind
widgets.

The pieces are plain functions over `(n_nodes, n_channels)` arrays — `rank_channels`,
`cosine_similarity`, `fit_pca`, `correlate_field` — for when the routine above is not the
question being asked. See the [API reference](api/latents.rst).
