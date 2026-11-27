# Associations against chance

A [profile](features.md#census-and-profile) says what fields a feature goes with, and a
[correlation](features.md#correlating-with-a-field) which features go with a field.
Neither says when a number is large enough to mean something, and no fixed bar can.

## Why a fixed bar goes vacuous

Take the mean of a standardized field over the nodes where a feature is active. Its
noise is set by how many nodes that is: a feature over half the world averages away
almost everything, and one over a few hundred nodes does not. A bar of, say, 0.15 is then
strict for the broad features and lax for the compact ones, which are the ones worth
steering. Labeling a feature with whichever field it goes with most strongly makes it
worse: something always comes first, so the more fields are asked about, the more
features are labeled, until every feature is. On the [toy archive](quickstart.md) below,
taking the largest association labels every channel of the last layer that is ever
positive, 15 of 16 -- channel 2 among them, which is positive everywhere and so goes with
nothing at all. Measured against chance, 11 are labeled, 3 are ambiguous and 2 are not.

## The rotation null

`exege latents associate` measures each association against its own null. The field is
read again at the node a whole number of columns east of every node -- the map turned
west by that many columns -- many times over (64 by default, each at least 30 degrees
from where it was, either way round). A rigid rotation
keeps the feature's footprint exactly -- its size, shape and latitudes -- and keeps the
field's own spatial structure. Only the east-west alignment of the two is broken. The
spread of the association over those rotations is the noise it has to clear, whatever
the footprint.

From there:

- each (feature, field) has a **z-score** against its null, and a two-sided Gaussian
  p-value; beside it, an empirical one, a rank: the share of the data and its `n`
  rotations that sit at least as far from the mean of all of them as the data does. It
  never goes below one in `n + 1`, and a rotation that reads nothing (a missing field
  where it lands) is left out of `n`;
- **Benjamini-Hochberg** holds the false-discovery rate at `--q` across the whole table
  of features and fields;
- a feature is **labeled** with its strongest field only if that association is a
  discovery *and* stands above the runner-up by more than the same margin does, at the
  95th percentile, when each rotation in turn is treated as if it were the data. Two
  fields that go together (moisture and ascent, say) leave a feature `ambiguous` rather
  than settled by whichever came out a hair ahead;
- the whole labelling is run once more with the rotation nearest half way round as the
  data, where no association is real, against rotations *of that rotation* by the same
  shifts: the same procedure, on an input with nothing in it. The share of features it
  labels is the **calibration**: it should sit near `--q` or below, and if it does not,
  nothing else in the table is readable. The usual way to fail it is a narrow band of
  shifts (a large `--min-shift-degrees`) under a field of one or two waves: every
  rotation then sees nearly the same thing, the null barely varies, and on a planted
  case of that shape the procedure labels 28 of 36 features and the calibration 78% of
  them. The null's median skew and excess kurtosis are printed beside it,
  for how far the Gaussian p-value can be trusted.

```console
$ exege latents toy scratch/toy/control
$ exege latents associate scratch/toy/control --top 6
channels of layer 3 against 3 field(s), 8 time(s), 41 rotation(s): 11 of 16 labeled at q = 0.05
calibration: 0.0% labeled with a rotation as the data against 34 of its own rotations (q = 0.05); null |skew| 0.54, excess kurtosis -0.41

CHANNEL  LABEL          FIELD          Z       P        GAP    BAR  COVERAGE
6        precipitation  precipitation  +284.7  0        284.1  1.5  0.003
14       precipitation  precipitation  +157.8  0        153.2  1.9  0.005
12       precipitation  precipitation  -10.2   2.3e-24  6.9    1.6  0.850
1        precipitation  precipitation  +5.6    2e-08    3.0    1.2  0.834
3        temperature    temperature    +5.5    4.7e-08  4.9    1.4  0.809
8        sst            sst            -5.1    3.8e-07  2.0    1.0  0.428
```

The toy grid has 48 columns of 7.5 degrees, so only 41 rotations clear 30 degrees, and
those are all used. `GAP` is how far the label's `|z|` stands above the runner-up's,
`BAR` what it had to clear. Channel 12 is positive over 85% of the world, and where it is,
precipitation is lower than its rotations say it would be by chance: a z of -10 is a
label too, of the opposite sign. "Active" means above `--threshold`, as for a profile. The
planted storm channel, 5, is positive almost everywhere and only strongly so under the
storm; asked about where it exceeds 1, it is labeled as it should be:

```console
$ exege latents associate scratch/toy/control --threshold 1 --channel 5
```

which gives a z of +65.4 for precipitation, standing 64.7 above the runner-up against a
bar of 1.7. A profile takes the same null with `--rotations`:

```console
$ exege latents profile scratch/toy/control --channel 5 --threshold 1 --rotations 32
channel 5 of layer 3: active over 1.9% of the area and 8 time(s)

against 32 rotation(s) of each field

FIELD          EFFECT  ACTIVE  INACTIVE  NULL_Z  P
precipitation  +6.48   2.137   0.01376   +58.0   0
sst            +1.07   26.73   17.67     +0.1    0.91
temperature    +0.85   306.7   295.5     -0.4    0.7
```

An effect of +1.07 standard deviations in sea-surface temperature looked like something
in the profile alone. Against its rotations it is nothing: the storm sits at 10 N, where
the sea is warm at every longitude.

## The orbit null

`--null orbit` asks a different question from the default rotations. It reads the
statistic at every shift of the grid, all `n_lon` of them, and tests each column as a
whole. Within each pair (column, field), the statistic is standardized over its orbit.
At every shift, its size is taken against the shifts at least `--min-shift-degrees` away
from that shift; the same neighbourhood is left out at every shift, the data's included.
A column's severity is its best field's size plus that field's lead over the runner-up.
Its `p_orbit` is the share of the `n_lon` shifts whose severity is at least the data's.

There is no Gaussian tail, and no shift is special. If every alignment round the circle
is as likely as the data's, a column gets `p_orbit <= q` with probability at most `q`.
So `q` is now each column's chance of a false label, not a false-discovery rate across
the table. A column is labeled with its best field at `p_orbit <= q` and a positive lead.
The sums at every shift come from one FFT along each latitude row (a shift is a circular
cross-correlation), so the whole orbit costs little more than 64 rotations.

The calibration runs the same test with the shift half way round as the data. Unlike the
rotations null's, it cannot fail: each column's `p_orbit` ranks one shift of its orbit
against all of them, so over every shift a column is labeled at no more than a share `q`
of them, whatever the data. It reads near `q` or below by construction, and checks the
computation, not the null. What can fail is the assumption that every alignment is as
likely as the data's; the evidence on that is below, from fields with nothing in common
and from weather weeks apart, not from the calibration column.

In the table the orbit prints `P_ORBIT` and `LEAD` (the best field's lead over the
runner-up, in units of the orbit's own spread: there is no bar to clear), and `Z` from
the rotations null for the size and sign of the association. A grid of fewer than `1/q`
columns is refused: its smallest `p_orbit`, one over the number of columns, is above `q`.

On ACE's latents (20 times, 39 fields), compared with the default (the orbit's
calibration is the computation check above, not evidence like the rotations'):

| layer | null | labeled | calibration |
|---|---|---|---|
| 8 | rotations | 537 of 1024 | 10.0% |
| 8 | orbit | 586 of 1024 | 1.7% |
| 0 | rotations | 365 of 1024 | 12.2% |
| 0 | orbit | 326 of 1024 | 4.1% |

Layer-8 feature 242 is still labeled with precipitation (`p_orbit` = 3/360). Layer-0
feature 142, which follows incoming sunlight, is still not labeled with it: its best
`SOLIN` size is that of a field of one wave, and 10-m wind wins, at `p_orbit` = 0.05.
On fields with nothing in common (random east-west waves up to wavenumber 1, 4 or 30),
the orbit labels 3–5% of columns at `q` = 0.05.

The guarantee holds only as far as every alignment is as likely as the data's. Fixed
geography breaks that: a land-sea pattern in a field lines up with a feature tied to land
at shift 0 and nowhere else, so the association is real and the orbit labels it. That is
an association with geography, not with weather. To ask about weather alone, give the
command anomaly fields (each node's time mean removed). On ACE's 30-day archive (block 4,
seven weather fields, 24 times), set against the weather 24 days later:

| fields | null | labeled at lag 0 | labeled at 24 days |
|---|---|---|---|
| as they are | rotations | 364 | 140 |
| as they are | orbit | 300 | 114 |
| anomalies | rotations | 392 | 83 |
| anomalies | orbit | 382 | 77 |

Most of what the raw fields keep 24 days on goes with sensible heat flux, 2-m temperature
and the trade winds, the fields with fixed geography, and removing each node's mean takes
it away. What is left, 7–8%, includes weather slow enough to last 24 days, and the
anomalies are about a 30-day mean, so this is close to a null but not one.

## What the null cannot see

It tests east-west alignment and nothing else. A feature and a field that are both
organised by latitude alone -- a polar feature and a polar field, an equatorial band and
the trade winds -- keep their alignment under every rotation, so the association reads
null even when it is real (when the null reproduces the data to rounding, the z-score is
left out altogether). The raw association, `statistic` in the result, is reported beside
the z-score for that reason: the first includes what latitude explains, the second only
what it does not. The sea-surface temperature above is that case.

The opposite case is a field with one large east-west wave, such as incoming sunlight,
whose day side is one hemisphere. Turning it moves the day side through every phase, so
the null ranges from as aligned as the data to the opposite, and its spread is about as
large as the association itself. A z-score cannot then get much past 2 however tight the
association is, and the label goes to a smaller-scale field instead. On ACE's encoder
output, SAE feature 142 follows incoming sunlight (spatial correlation 0.80). Its
statistic for `SOLIN` beats all 64 rotations (`p_empirical` = 1/65), yet z = 1.9, so it
is labeled by 10-m wind at z = -5. Read `p_empirical` beside the z-score for fields of
the largest scales. With 64 rotations it cannot fall below 1/65, so it never passes the
false-discovery cut across a whole layer by itself.

Neighbouring rotations of a smooth field are close to one another, so the null has fewer
independent draws than rotations. A z-score far out in the tail (the cut across
thousands of pairs needs about 5) extrapolates the Gaussian from them, and that is
anti-conservative for large-scale fields. The calibration line is the check: when it
reads well above `--q`, treat the labels as a ranking, not as discoveries.

It needs a structured grid whose longitudes are evenly spaced all the way round, where a
rotation takes nodes exactly onto nodes; a mesh is refused. Masked nodes are left out,
and so is a node wherever the rotated field is missing there or lands on a masked node,
rotation by rotation. A field the same everywhere at a time carries nothing at that time. Times
are pooled with equal weight; neighbouring times are not independent samples, which the
rotations do not repair, so pool times far enough apart to differ.

`--out FILE` writes every column, with every field's statistic, z-score, both p-values
and the number of rotations behind the empirical one, and `--json` the same for the
columns printed.

## From Python

```python
from exege.latents import associate, feature_profile, load_basis, open_source

source = open_source("scratch/toy/control")
result = associate(source, layer=3, n_rotations=64, q=0.05)
result.label  # a field's name, "ambiguous" or "none", per column
result.z, result.p  # (columns, fields)
result.calibration_label_rate
result.summary(top=20)  # labeled first, by |z|

profile = feature_profile(source, layer=3, column=5, threshold=1.0, rotations=32)
profile.null_z, profile.p_value

basis = load_basis("bases/sae_L04.npz")  # or the features of a dictionary
associate(source, layer=4, basis=basis, columns=[883, 982])
```
