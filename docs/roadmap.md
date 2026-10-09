# Roadmap

What is planned, gathered from every part of the package. Within a section, items are
roughly in the order we mean to take them.

## Latent diagnostics

- Bias and time-mean maps, spectra, zonal means (a `FieldSource` beside `LatentSource`, on
  the same `latents.grid`)
- A GraphCast mesh adapter, including the *translator* of Tempest et al. (2026), which
  puts an intermediate processor step in the basis of the last one
- The activation exporter as an adapter of its own, behind a framework extra — with
  the hooks that *write* a layer, for steering along a basis file's direction, and an
  `experiment` block (the SamudrACE exporter takes `--seed` and does not record it)
- Ensemble members and per-layer grids in `LatentSource`
- Differences and series in a basis's features between two runs
- How redundant a basis is: the pairwise correlation of its features' activations over
  time, the measure Cheon (2026) reports beside explained variance
- A probe for a labeled phenomenon on features against one on channels (MacMillan &
  Ouellette 2025 find a tropical-cyclone feature a probe on neurons cannot)

## Sparse autoencoders

- The B-spline autoencoder as Cheon (2026) has it, replacing `bspline`, beside the
  `relu` baseline it is measured against, with that paper's table: explained variance,
  features alive and dead, mean L1 norm, redundancy between features. Two things its
  text leaves open need an answer first: what a spline does outside its knots, and
  whether the knot vector is extended past the measured range
- An auxiliary loss that revives dead features (MacMillan & Ouellette 2025, after
  [Gao et al. 2024](https://arxiv.org/abs/2406.04093)): what has not fired in a long while
  is made to rebuild the residual
- Steering a feature as MacMillan & Ouellette do it: keep the autoencoder's
  reconstruction error, scale one feature's activation, add the error back and let the
  model run on. The [toy emulator](quickstart.md) can do this in numpy today; a real model
  needs a hook in its exporter
- A cross-layer transcoder (several decoders on one encoder), and tracing a
  feature to its antecedents in an earlier layer, as Cheon (2026) does by correlation
- Features compared across seeds of the ablation campaign

## Evaluation

- Block-bootstrapped intervals on the held-out numbers: eight held-out times is a
  point estimate, not a distribution
- Splits that buffer by elapsed time (`LatentInfo.elapsed_seconds`) rather than by
  position, for archives that keep irregular steps
- The same evaluation for a cross-layer transcoder, when there is one

## Steering

- A real adapter, in the model's environment
- Fitted dictionaries on the toy system, to show a learned feature that is not planted

## Experiment records

- Check a record against the files it names (`bases`, the archive's commit) and say
  which have changed
- Records of the other commands (`diff`, `growth`), if anyone comes back to them

## The web app

- A reference-field panel beside the latent maps (needs `FieldSource`)
- Click on a map to move the region
- A run against its control (`latents diff`), and two archives side by side
- Two records side by side, and a record against the files it names
- A PDF report of a session
