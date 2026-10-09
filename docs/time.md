# Through time

`analyze_region` is a snapshot. The commands on this page follow something from one
physics step to the next, and through the layers of the network.

## A region's series

```console
$ exege latents series latents/atmosphere --lat 5 --lon -140 --radius-km 1500 \
    --channel 45 --channel 107
TIME                 HOURS  45        107
0425-01-03T18:00:00  0      0.04793   4.694
0425-01-04T00:00:00  6      0.1133    4.762
...
0425-01-05T12:00:00  42     0.1648    4.633
0425-01-07T00:00:00  78     0.1314    4.9
```

`HOURS` comes from the archive's own calendar, by hand: positions are not lead times. This
exporter kept two runs of steps a day and a half apart, and a plot against position would
hide that. Uncentered, only the region's nodes are read — a series over every time of the
7.6 GB archive is a few MB — and `--basis … --feature N` follows a feature instead.

## Storylines

A **storyline** asks, for one physical field, how closely each layer follows it at each
time: the best absolute correlation any channel reaches (or any feature, for layers given
a basis). Bright at the first layer means the field comes in with the inputs; bright only
deep in the network means the network builds it.

```console
$ exege latents storyline latents/atmosphere --field surface_precipitation_rate
$ exege latents storyline latents/atmosphere --field surface_precipitation_rate \
    --bases 'bases/sae_L{layer:02d}.npz'
```

A time at which the field has no values — a diagnostic output, before the model's first
step — is left empty rather than refused.

### Inputs, outputs and lead times

Latents at a time belong to the forward pass that *starts* there: it reads the state at that
time and writes the next. A field at the same time is what the pass read — right for an
input such as sunlight — but for an output it is the *previous* pass's, which this one never
saw. `--lead 1` (`lead=1` in Python) sets the latents against the field one reference time
later, what the pass itself produced; `fields`, `storyline` and `profile` all take it.
Precipitation, the best SAE feature per layer:

```console
$ exege latents storyline latents/atmosphere --field surface_precipitation_rate \
    --layer 0 --layer 4 --layer 6 --layer 8 --time 4 --time 7 --time 19 \
    --bases 'bases/sae_L{layer:02d}.npz' --lead 1
best |r| of any feature (basis) with surface_precipitation_rate, by layer and time

TIME                 LAYER 0  LAYER 4  LAYER 6  LAYER 8
2015-01-04T12:00:00  0.51     0.464    0.755    0.86
2015-01-05T06:00:00  0.47     0.411    0.773    0.86
2015-01-08T06:00:00  0.492    0.46     0.746    0.842
```

Without `--lead` the last column reads 0.65, 0.52 and 0.55: the rain the network builds deep
down is set against rain it did not build. The reference axis keeps every forward step, so a
lead is exact even where latents were kept for fewer, and the last latent time's output is
there too. Which fields are inputs a manifest does not say; the exporter that wrote it does.

## Hovmöller diagrams

A **Hovmöller diagram** averages one quantity over a latitude band, per longitude and
time: anything that travels draws tilted stripes, whose slope is its speed. The quantity
is a field, one channel of a layer, or one feature of a basis.

```console
$ exege latents hovmoller latents/atmosphere --lat-min 40 --lat-max 60 --field V_3
$ exege latents hovmoller latents/atmosphere --lat-min 40 --lat-max 60 --layer 8 --channel 45 \
    --out hovmoller.npz
```

The command prints the diagram in coarse longitude bins; `--out` keeps it whole, and
`exege.figures.hovmoller_figure` draws it. `exege.figures.layer_time_figure` draws a
storyline, or a [`--growth` table](comparing.md): anything that is layers against time.

## From Python

```python
from exege.latents import field_storyline, hovmoller, load_basis, open_source

source = open_source("latents/atmosphere")
basis = load_basis("sae8.npz")
story = field_storyline(source, field="SOLIN", bases={8: basis})
story.best  # (n_times, n_layers): the best |r| of any channel, or of layer 8's features
band = hovmoller(source, lat_min=40, lat_max=60, layer=8, channel=45)
band.values  # (n_times, n_lon), with band.lon
```
