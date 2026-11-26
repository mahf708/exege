# Fixed and positional features

Some channels follow the weather; others do the same thing at the same place every
time. A [census](features.md#census-and-profile) at one time cannot tell the two apart. A
model that has learned where it is — from the latitude it is given, from the land, or from
a positional embedding of its own — shows it here first.

## Splitting what is fixed from what moves

`exege latents stationary` reads every time of a layer and splits each channel's variance
(area-weighted, over the nodes and times read) into the part held by its per-node time
mean and the rest. That share is `FIXED`: 1 for a channel that is the same map at every
time, near 0 for one whose time mean is the same everywhere. The fixed map is then fitted,
by weighted least squares, on one value per latitude row (`ZONAL`) and on any fields named
with `--static` (`STATIC`, beyond what the rows already explain). What neither explains,
as a share of the whole variance, is `UNEXPLAINED`, and columns are ranked by it.

```console
$ exege latents toy scratch/fix/control
$ exege latents stationary scratch/fix/control --top 6
channels of layer 3 over 8 time(s) and 1152 node(s), by fixed structure that latitude does not explain

CHANNEL  COVERAGE  FIXED  ZONAL  STATIC  UNEXPLAINED
9        0.995     0.623  0.511  0.000   0.305
5        0.238     0.262  0.268  0.000   0.192
14       0.005     0.621  0.755  0.000   0.152
15       0.569     0.973  0.846  0.000   0.150
0        0.590     0.409  0.703  0.000   0.122
11       0.478     0.892  0.878  0.000   0.109
```

That is the [toy emulator](quickstart.md), so these are numbers about the command. Channel
15 is almost entirely fixed (0.973), and latitude explains most of that: the toy hands
its network the sine and cosine of latitude. Channel 9 is fixed by less but explained by
latitude by less too, leaving 0.305 of its variance as a map that stays put and is not a
latitude profile. Channel 5 is the planted storm, which drifts east: its fixed part is
small, and what there is of it is the storm's track, which a time mean over eight steps
cannot tell from a place.

`COVERAGE` is the share of area and time a column is active over (above `--threshold`):
for a sparse basis, read the shares of a feature that fires over 0.5% of the world
with care.

## Statics

A fixed map that a boundary field explains is the model doing what it was told: a
channel that tracks land fraction is not a mystery. Name such fields with `--static` and
their part moves from `UNEXPLAINED` to `STATIC`:

```console
$ exege latents stationary scratch/fix/control --static sst --top 6
channels of layer 3 over 8 time(s) and 1062 node(s), by fixed structure that latitude and sst do not explain

static sst: fixed share 1.000

CHANNEL  COVERAGE  FIXED  ZONAL  STATIC  UNEXPLAINED
5        0.222     0.266  0.265  0.000   0.196
0        0.600     0.433  0.786  0.000   0.093
14       0.006     0.513  0.931  0.000   0.035
10       0.449     0.882  0.967  0.000   0.029
6        0.003     0.774  0.975  0.000   0.020
9        1.000     0.395  0.962  0.000   0.015
```

The toy keeps no land field, and its sea-surface temperature is a function of latitude
alone, so it explains nothing the rows did not (`STATIC` 0). What changed is the nodes:
with a static, only nodes where every static has a value are used, for every number, and
`sst` is missing over the continent. Channel 9 falls from 0.305 to 0.015 unexplained
because its fixed structure *was* the continent. A land-fraction field, had the toy kept
one, would have said so without leaving the continent out. Each static's own fixed share
is printed so that a field that is not static after all shows itself; a static is
averaged over the times read.

Statics are weighed alike whatever their units: each is first stripped of what the rows
already explain, then scaled to unit size, before they are fitted together. A mixing
ratio near 1e-11 beside a geopotential near 5e4 explains as much as it explains, and a
static that is a function of latitude alone (as `sst` is here) adds nothing and drops
out. Rows are fitted as one weighted mean per row, so no `(n_nodes, n_rows)` matrix is
ever made, at 0.25 degrees or anywhere else.

## What it cannot say

`UNEXPLAINED` is fixed structure of unknown origin, and that is all. A learned positional
embedding puts structure there, and so does any input the model was given and this
analysis was not — the orography, a land mask, the local time of day. The direct test is
to rotate the model's inputs in longitude and see whether a column stays on the grid or
moves with them, which needs a running model: that is [the rotation test](rotation.md).
On a mesh (no latitude rows) `ZONAL` is undefined, and only the statics, beyond a
constant, are fitted.

A column that varies only in time -- the same value everywhere at each time -- has
`FIXED` and `UNEXPLAINED` 0 and no zonal or static share: a flat map has nothing to
explain. A column a basis leaves missing (not finite) at any node and time used reads
nothing at all, coverage included, since a node seen at a few times would weigh those few
as fully as the rest. Nodes of no area are left out, and at least two times are needed.

Columns are taken `--chunk` at a time (256 by default), each chunk reading every time
once, so memory is a few `(n_nodes, chunk)` arrays however wide the basis is; a
4,096-feature dictionary is sixteen passes over the archive. A `topk` dictionary encodes
every feature to find each node's top `k`, so every pass pays for the whole encoder: a
wider chunk costs memory and saves time.

## From Python

```python
from exege.latents import feature_stationarity, load_basis, open_source

source = open_source("scratch/fix/control")
result = feature_stationarity(source, layer=3, statics=["sst"])
result.columns[result.ranked(top=5)]  # the most fixed structure of unknown origin
result.fixed_share, result.zonal_share, result.static_share, result.residual_fixed_share
result.summary(top=10)  # what --json prints

result = feature_stationarity(
    source, layer=3, basis=load_basis("sae.npz"), columns=[12, 40], chunk=64
)
```
