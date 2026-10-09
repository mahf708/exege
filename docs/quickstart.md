# Quickstart

This page needs no model and no data: exege ships a toy emulator that writes an archive of
latents in a few seconds. It needs the `latents` extra (see [Installation](installing.md)).

## Make an archive

The toy is an MLP with a residual stream on a small Gaussian grid, in numpy. It is shaped
like a real archive where that matters to a reader: kept steps with a gap between them,
fields that begin one step before the latents, a continent the mask has to come from, a
calendar without leap days. One channel, 5, is planted to follow a storm that drifts east,
so an analysis has a right answer to find.

```console
$ exege latents toy scratch/toy/control
wrote scratch/toy/control; try `exege latents info scratch/toy/control --mask-variable sst`
$ exege latents info scratch/toy/control --mask-variable sst
source                 scratch/toy/control
model                  exege-toy
component              atmosphere
checkpoint             seed-0
calendar               noleap
timestep_s             21600
grid                   24x48, 1152 nodes, 1062 valid
times                  8: 0424-02-27T06:00:00 .. 0424-03-02T00:00:00
experiment.seed        0
experiment.kept_steps  [1, 2, 3, 4, 9, 10, 11, 12]
experiment.exege       0.6.0

layers
LAYER  CHANNELS  LABEL
0      16        encoder output
1      16        block 1 output
2      16        block 2 output
3      16        block 3 output

3 reference field(s); see `latents fields`
```

`--mask-variable sst` says that nodes where sea-surface temperature is missing are land,
and mean nothing: 90 of the 1,152 are left out of everything that follows
([masks](archives.md#masks)).

## Find what responds in a region

The storm starts near 10°N, 138°W. Rank the channels of the last layer by how strongly
they respond within 1500 km of it, and fit two principal components there:

```console
$ exege latents region scratch/toy/control --mask-variable sst --lat 10 --lon -138 \
    --radius-km 1500 --centered --top 5 --pcs 2
9 node(s) at layer 3, time 0424-02-27T06:00:00

RANK  CHANNEL  PEAK_ABS
1     5        3.903
2     10       0.7902
3     7        0.7257
4     12       0.6814
5     0        0.6489

PC0   98.9%  5(+0.92)  10(-0.20)  0(+0.20)  6(+0.13)  7(+0.12)  12(-0.10)

PC1    0.9%  8(+0.48)  11(-0.46)  4(+0.42)  3(+0.27)  15(+0.26)  10(+0.26)
```

Channel 5 answers five times more strongly than any other, and is almost all of the first
component. [Analysing a region](regions.md) explains each option.

## Follow it through time

The region stays where it is; the storm does not:

```console
$ exege latents series scratch/toy/control --mask-variable sst --lat 10 --lon -138 \
    --radius-km 1500 --channel 5
TIME                 HOURS  5
0424-02-27T06:00:00  0      2.735
0424-02-27T12:00:00  6      1.545
0424-02-27T18:00:00  12     0.2753
0424-02-28T00:00:00  18     0.0128
0424-03-01T06:00:00  48     0
0424-03-01T12:00:00  54     0
0424-03-01T18:00:00  60     0
0424-03-02T00:00:00  66     0
```

`HOURS` comes from the archive's own calendar, so the gap between the fourth and fifth
kept steps shows. See [Through time](time.md).

## Ask which channel follows a field

The archive keeps physical fields beside its latents. Which channel follows the
precipitation the model writes?

```console
$ exege latents fields scratch/toy/control --mask-variable sst --field precipitation \
    --top 3 --lead 1
channels of layer 3 against precipitation at 0424-02-27T06:00:00, +1 time(s) later

RANK  CHANNEL  CORRELATION
1     5        +0.672
2     0        +0.469
3     6        +0.388
```

The planted channel again. `--lead 1` sets the latents against what their forward pass
produced rather than what it read; [Finding features](features.md) explains why that
matters.

## The same from Python

The command line is a thin client of the API; anything it can do, a notebook can.

```python
from exege.latents import Region, analyze_region, open_source

source = open_source("scratch/toy/control", mask_variable="sst")
result = analyze_region(
    source,
    time=0,
    layer=3,
    region=Region(lat=10, lon=-138, radius_km=1500),
    centered=True,
    n_components=2,
)
result.ranking.channels[:3]  # array([ 5, 10,  7])
source.grid().to_map(result.scores[:, 0])  # the first component as a (24, 48) map
```

## Next steps

- Point the same commands at a real archive: [Reading an archive](archives.md).
- Train a dictionary whose features are easier to name than channels:
  [Sparse autoencoders](nn.md).
- Look at all of it in a browser: [The web app](app.md).
