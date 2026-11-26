# Mediation

A [steering experiment](steering.md) says that changing a feature changes a field. It does
not say how. Did the change pass through some other feature on its way -- one at a later
layer, or the same one at later steps -- or around it? Mediation asks, for one other
feature at a time.

## The design

Three runs from one initial state, under one noise seed, beside the control:

| Run | What it is | What it measures |
|---|---|---|
| total | the steer, as `run_steering` applies it | the whole effect |
| direct | the steer, with the mediator **held** at the value it took in the control, at every node it is held at and every step from the first edit on | the part of the effect that does not pass through the mediator |
| hold only | the mediator held, and no steer | nothing, if holding works: it must equal the control |

The *indirect* effect is the total less the direct, and the *share* is the indirect over
the total. A mediator at the steer's own layer and time is held *after* the steer acts, so
holding the steered feature itself takes back whatever of the edit that feature can see.

The hold-only run is the check that matters before any share is read. Holding a feature
at the value it already has should change nothing, and in a deterministic system it must
change nothing *exactly*. `exactness` is the largest difference found between that run and
the control, over every field and recorded latent. A value that is not zero means the hold
is not a no-op -- a forward pass that is not deterministic, an encode that rounds -- and
every share computed with it carries that error. The command prints it on its first line.

`mediator_change` is how far the steer alone moved the mediator's activation, as an
area-weighted RMS over the nodes it is held at and the steps from the first edit on. Zero
means the steer never reached it there, and there is nothing for it to mediate.

## How a feature is held

Moving the layer by `d` along a feature's direction moves its activation by `d` only when
its encoder and decoder agree -- a tied dictionary, a unit direction. An untied dictionary
moves it by some other amount, a token-normalized one by an amount that depends on the
node, and a ReLU stops moving it at all once it reaches zero, where every further move
reaches the same zero. So the move is found per node: two secant steps from
`target - now`, exact at once where encoder and decoder agree and wherever nothing needs
moving (which is what keeps the hold-only run exact to the bit); then, for nodes still off
their target or whose path bends on the way, a bisection for the *nearest* move that
reaches it -- the smallest change that holds the feature, not any change that does.
`hold_error` is the largest miss left, over the direct runs: a target no move along the
direction can reach (a ReLU feature asked to rise from zero where the push does not reach)
shows there instead of passing silently.

## A planted answer

In the toy system an add on channel 0 of layer 0 reaches moisture only through channel 1 of
layer 1, which the mixer moves it to. Holding that channel, read through a basis that is
every channel as a direction (`Directions` with the identity), takes the whole effect away;
holding channel 2 takes none of it:

```console
$ exege latents mediate --adapter toy-dynamics --adapter-option masked=3 \
    --basis scratch/steer/planted0.npz --layer 0 --feature 0 --amount 1.5 --time 1 --steps 5 \
    --seeds 0,1 --mediator-layer 1 --mediator-feature 1 \
    --mediator-basis scratch/steer/channels1.npz --field moisture
add 1.5 on feature 0 of layer 0, through feature 1 of layer 1; the steer moved the mediator by 0.75 (RMS); holding it alone changed the run by 0, and the hold missed its target by at most 5.96e-08

FIELD     TOTAL   DIRECT      INDIRECT  SHARE
moisture  0.7031  -3.818e-09  0.7031    100.0%
$ ... --mediator-feature 2
add 1.5 on feature 0 of layer 0, through feature 2 of layer 1; the steer moved the mediator by 0 (RMS); holding it alone changed the run by 0, and the hold missed its target by at most 0

FIELD     TOTAL   DIRECT  INDIRECT  SHARE
moisture  0.7031  0.7031  0         0.0%
```

## A sparse feature holds only what it encodes

Hold the same channel through the planted *dictionary*, whose feature is the channel after
a ReLU, and the share falls to 58%:

```console
$ ... --mediator-feature 1 --mediator-basis scratch/steer/planted.npz
add 1.5 on feature 0 of layer 0, through feature 1 of layer 1; the steer moved the mediator by 0.5122 (RMS); holding it alone changed the run by 0, and the hold missed its target by at most 5.96e-08

FIELD     TOTAL   DIRECT  INDIRECT  SHARE
moisture  0.7031  0.2955  0.4076    58.0%
```

Nothing went wrong. Holding a feature holds its *activation*, and where the channel sits
below zero the activation is zero both before and after the steer: the hold has nothing to
undo there, and the part of the change the feature cannot see passes on regardless. That
is a fact about the dictionary, and the reason a share read through a sparse feature is a
lower bound on what the path through that *channel* carries, not a measure of it.

The command ends, like `steer`, with a `reproduce:` line that pins the steered basis and,
when it is another file, the mediator's (`--mediator-basis-sha256`).

## What it does not say

- One mediator at a time. Two paths that each carry the whole effect -- redundancy --
  read as zero share each.
- Holding is a forward-pass substitution, like every edit here: a mediator held far from
  where it would have gone may send the model somewhere it has never been. The size of
  `mediator_change` says how far that is.
- The share is about this steer, from this state, at this amount. A different amount, or a
  closed loop, can route the effect differently; run it again to know.

## From Python

```python
from exege.adapters.toy_dynamics import ToyDynamics
from exege.latents import Mediator, Steer, load_basis, run_mediation

system = ToyDynamics(masked=3)
result = run_mediation(
    system,
    load_basis("scratch/steer/planted0.npz"),
    Steer(layer=0, feature=0, amount=1.5, times=(1,)),
    Mediator(layer=1, feature=1, basis=load_basis("scratch/steer/channels1.npz")),
    steps=5,
    seeds=(0, 1),
)
result.exactness  # 0.0: holding alone changes nothing
result.effects["moisture"].share  # 1.0
```
