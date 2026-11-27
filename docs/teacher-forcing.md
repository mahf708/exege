# Free running against teacher forcing

An emulator rolled out on its own feeds itself its own output. Fed the reference state at
every step instead (teacher forcing), the same network sees what it was trained on. Set
side by side, the two say which channels or features a free run weakens, and whether the
weakening grows with lead time: a deterministic emulator trained to the mean is expected
to let what it carries fade, and this is where to see it happen inside the network.

## Two archives of one network

`exege latents gap REFERENCE RUN` reads both at every time they both keep, over the nodes
valid in both -- and, at each time, for each column, over the nodes where both have a
value, so the two sides are always read over one area. The two must be one network on
one grid; that is checked as for [comparing runs](comparing.md). For every channel (or
feature, with `--basis`) it reports the area-weighted root mean square (`RMS`: the size of
what the column carries, whatever its sign), the mean, the share of the area it is active
over (`COVER`) and, from those, the run's mean and strength (mean where active) as ratios
of the reference's. Columns are ordered by `DEFICIT`, one less the ratio of the run's RMS
to the reference's: 0.5 where the run carries half as much, negative where it carries
more. It is read from magnitudes, so a channel that is negative, or that swings about
zero, and is damped reads as weakened, as a feature does. Pooled numbers count every time
equally.

```{admonition} The teacher-forced archive comes from the exporter
:class: note

exege sets two archives against each other; it does not run the model. Writing a
teacher-forced archive (the reference state fed in at every step, the latents kept) is
an exporter's option, and the [roadmap](roadmap.md#latent-diagnostics) has it.
```

Until an exporter writes one, the [toy emulator](quickstart.md) can stand in, with a run
pushed by a known amount playing the free run. That shows what the command reports, and
nothing about free running:

```console
$ exege latents toy scratch/fix/control
$ exege latents toy scratch/fix/damped --steer 1:2:-2
$ exege latents gap scratch/fix/control scratch/fix/damped --top 5
channels of layer 3 over 8 time(s) and 1152 node(s), the run's largest deficit first

CHANNEL  DEFICIT  REF_RMS  RUN_RMS  MEAN_RATIO  STRENGTH_RATIO  REF_MEAN  RUN_MEAN  REF_COVER  RUN_COVER
2        0.4      5        3        0.6         0.6             5         3         1.000      1.000
6        0.2928   0.5321   0.3763   0.6856      1.388           -0.5154   -0.3533   0.003      0.009
13       0.2273   0.3766   0.291    0.7637      -               -0.3719   -0.284    0.000      0.000
7        0.2162   0.2887   0.2263   0.481       1.382           -0.2005   -0.09644  0.155      0.381
14       0.1772   0.2709   0.2229   0.7909      0.5852          -0.2525   -0.1997   0.005      0.020
```

The pushed run took two from the offset channel, 2, at the first block, and the residual
stream carries it to the last: it sits at 3 instead of 5, so it carries 0.6 of what it
did, and its deficit is 0.4. Channels 6, 13, 7 and 14 moved because blocks read
channel 2; all four are negative on average, and each carries less, which the deficit
says although their means *rose*. The ratios show why `DEFICIT` leads: a ratio of two
means near zero, or of strengths over a handful of active nodes, is a large number about
very little. For a sparse basis, whose activations are not negative, the ratios are the
natural reading.

A push the other way says the opposite: taking one from the storm channel, 5
(`--steer 1:5:-1`), moves its mean from 0.05 to −0.95, so it carries *more* (RMS 0.33 to
1), and it comes last, with a deficit of −2.0.

## Through lead time

`--json` keeps every time, `mean_by_time` for each side, so a gap that opens with lead time
shows as one, where the pooled numbers above would average it away. `elapsed_seconds` is
the reference's, under its calendar. Times are paired by label: `--time` names times in
REFERENCE, and one the run does not keep is refused.

## What a gap says

A column weaker in the free run is where the emulator's own states drive the network
differently from the reference states. That is not by itself an error — a free run is
meant to depart from the reference, and a stochastic model's should — and it is a gap in
what the network computes, not in the weather. It points at what to steer, or what to look
for in the fields, not at a fix.

A deficit needs a noise floor before it means anything. Run `gap` between two runs that
differ only in their noise seed and read the deficits there first. On ACE's last block
over five days, four seed pairs give SAE features active over 1% or more of the area a
median |deficit| of 0.017, a 95th percentile of 0.05–0.06, and a largest of 0.15–0.24, out
of 763 features. Suppressing one feature (layer-8 feature 242, pushed by -3 at every node
and step) gives it a deficit of 0.36, first in the list, against 0.002–0.033 for it under
the seeds alone. A feature of a sparse dictionary that fires over a handful of nodes reads
a deficit near one, or far below zero, from noise alone: on one seed pair, one active over
0.07% of the area topped the list at 0.41. Such features are ranked after the rest
(`--min-coverage`, 1% by default), though their numbers are still reported. The floor is
for sparse features only; a channel or a PCA component swings both ways, and its
coverage means less. Repeated `--time`s are refused, since a time named twice would
count twice.

## From Python

```python
from exege.latents import feature_gap, load_basis, open_source

reference = open_source("latents/teacher-forced")
run = open_source("latents/free")
gap = feature_gap(reference, run, layer=8, basis=load_basis("sae8.npz"))
gap.columns[gap.ranked(top=10)]  # what free running weakens most
gap.reference_mean, gap.run_mean  # (n_times, n_columns): does the gap grow?
gap.ratio_strength, gap.deficit  # deficit: 1 - run RMS / reference RMS
gap.rms  # (2, n_columns): reference, run
```
