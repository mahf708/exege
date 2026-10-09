# Finding features

## Correlating with a field

An archive that keeps physical fields beside its latents (`reference.nc`) can say which
channels, or which features of a basis, track one of them:

```console
$ exege latents fields latents/atmosphere          # the 62 fields this archive keeps
$ exege latents fields latents/atmosphere --field surface_precipitation_rate --top 3
channels of layer 8 against surface_precipitation_rate at 0425-01-03T18:00:00

RANK  CHANNEL  CORRELATION
1     45       -0.588
2     248      -0.540
3     224      +0.509
$ exege latents fields latents/atmosphere --field surface_precipitation_rate --top 3 \
    --basis sae8.npz
features of layer 8 against surface_precipitation_rate at 0425-01-03T18:00:00

RANK  FEATURE  CORRELATION
1     676      +0.966
2     48       +0.490
3     360      +0.469
```

No single channel of this layer follows precipitation better than |r| = 0.59; one feature
of a [sparse autoencoder trained in eleven seconds](nn.md) follows it at 0.97 — the
same feature 676 that answered most strongly [in the equatorial Pacific](bases.md#using-a-basis),
built mostly from the same channel 45. Correlation is area-weighted over valid nodes,
leaves out nodes where the field is missing, is taken at one time, and says nothing about
cause: it is where an expedition starts, and a [steering experiment](steering.md) is
where it ends. That 0.97 is in-sample: the dictionary was fitted on these times. How a
dictionary does on times it was not fitted to is [its own page](evaluation.md).

For an output of the model, such as precipitation, set the latents against the field the
pass *wrote* with `--lead 1`; [Through time](time.md#inputs-outputs-and-lead-times) explains why.

## Census and profile

A correlation asks about one field you already had in mind, and one number stands for a
whole map: a feature that is on over sunlit land and off everywhere else correlates only
modestly with sunlight, although that is exactly what it is. Two commands start from the
features instead.

A **census** lists every channel of a layer at one time, or every feature of a basis, with
how much of the area it is active over, its mean, its mean where active (*strength*), and
where it peaks. It is a catalog to browse:

```console
$ exege latents census latents/atmosphere --time 4 --layer 4 --basis bases/sae_L04.npz --top 3
features of layer 4 at 2015-01-04T12:00:00, by coverage

FEATURE  COVERAGE  MEAN   STRENGTH  PEAK  AT
982      0.239     2      8.38      24.5  79, 244
121      0.189     0.456  2.41      6.86  8, 134
379      0.177     1.31   7.4       24.6  -85, 288
```

A **profile** takes one of them and sets every physical field the archive keeps where it
is active against where it is not, as a difference in units of each field's own spread, so
fields of any unit share one axis. Here is the feature that followed sunlight at layer 4
with |r| = 0.42 only:

```console
$ exege latents profile latents/atmosphere --layer 4 --basis bases/sae_L04.npz --feature 883 \
    --time 1 --time 6 --time 11 --time 16
feature 883 of layer 4: active over 4.3% of the area and 4 time(s)

FIELD                       EFFECT  ACTIVE     INACTIVE
SOLIN                       +1.66   994.4      323.3
LANDFRAC                    +1.17   0.7735     0.272
OCNFRAC                     -1.06   0.2253     0.6942
T_1                         -0.68   198.4      205.7
TS_input                    +0.65   297.4      286.5
```

Sunlit land: a sharper description than the correlation gave. Pool times that differ in hour
as well as day. Every fourth time of a 6-hourly run is the same hour each day, and a profile
of those alone describes the feature at that hour only: over five 18Z times, this one comes
out as land whose sensible heat flux is high.

"Active" means above `--threshold`, zero by default: *firing*, for a sparse autoencoder,
whose activations are mostly exactly zero; *positive*, for a channel or a PCA score, where
another threshold may say more. Both commands read the whole grid or, with `--lat`,
`--lon` and `--radius-km`, a region: the day side only, say. A profile says what a feature
goes with, not what it does.

## From Python

```python
from exege.latents import feature_census, feature_profile, load_basis, open_source

source = open_source("latents/atmosphere")
basis = load_basis("bases/sae_L04.npz")
census = feature_census(source, time=4, layer=4, basis=basis)
census.ranked("coverage", top=20)  # or "mean", "strength", "peak"
profile = feature_profile(source, layer=4, column=883, basis=basis, times=range(1, 20, 5))
profile.fields, profile.effect  # largest effect first; exege.figures.profile_figure draws it
```
