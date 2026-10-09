# Comparing runs

A perturbed or steered run is set against its control node for node and channel for
channel, so the two must be one network on one grid; that is checked, not assumed. Two
archives that declare different models or checkpoints are refused — channel 42 of another
seed is not channel 42 of this one, whatever number it carries — unless `--across-models`
says the index does carry over (a fine-tune of the same weights); comparing networks
trained apart needs a different tool altogether. The layer is matched by its place in the
network (`network_layer` in the manifest, else its index), so a run whose layer 0 is
network layer 2 is not set against one whose layer 0 is network layer 8 (when both declare
one), and nothing lifts that refusal. Where either run leaves out its model, component,
checkpoint or layer placement the comparison cannot be verified, and is refused until you
pass `--allow-unverified-sources` (`allow_unverified=True` from Python; the result's
settings record that you did). Only nodes valid in *all* the runs compared — control,
experiment and any noise run — are weighed: a node one of them masks holds whatever it
holds there.

```console
$ exege latents diff latents/control latents/steered --layer 8
$ exege latents diff latents/control latents/steered --growth
```

The first ranks channels by the area-weighted RMS of `steered − control` at one time —
which channels the change reached. `--growth` follows its size through every layer and
every time the two share, relative to the control's own spread across the globe.

With no model to hand, the [toy emulator](quickstart.md) makes a steered twin of its
control:

```console
$ exege latents toy scratch/toy/control
$ exege latents toy scratch/toy/steered --steer 2:7:3     # +3 on channel 7 of layer 2, every step
$ exege latents info scratch/toy/steered --mask-variable sst
model                  exege-toy
calendar               noleap
grid                   24x48, 1152 nodes, 1062 valid
times                  8: 0424-02-27T06:00:00 .. 0424-03-02T00:00:00
experiment.steer       {'layer': 2, 'channel': 7, 'by': 3.0}
...
$ exege latents diff scratch/toy/control scratch/toy/steered --mask-variable sst --growth
```

## Measuring against noise

A stochastic model — one that draws noise at every step — makes a node-for-node
comparison mean something only when both runs drew the *same* noise: give the exporter one
seed for the control and every experiment. Even then, a perturbation grows as the runs
drift apart, and some of what `--growth` shows after a few steps is that drift. The
yardstick is a third run: the control again, with another seed.

```console
$ exege latents diff latents/control latents/steered --growth --noise latents/control-seed1
```

The second table it prints is the experiment's difference as a multiple of the one a new
noise draw makes, layer by layer and time by time: above 1, the steer moved the
layer more than chance does.

## Saying what a run did

For two archives to be told apart at all, the exporter has to say what it did. Anything
under `experiment` in the [manifest](archive-format.md) — a seed, a perturbed input, a
steered channel — is shown by `latents info` and carried into the provenance of every
result, as is the way the archive was read (a mask variable).

## From Python

```python
from exege.latents import difference, difference_growth, open_source

control, steered = open_source("latents/control"), open_source("latents/steered")
difference(control, steered, time=-1, layer=8).ranking.channels  # what the change reached
difference_growth(control, steered).relative  # (n_times, n_layers)

growth = difference_growth(control, steered, noise=open_source("latents/control-seed1"))
growth.signal_to_noise  # (n_times, n_layers); growth.noise_rms beside it
```
