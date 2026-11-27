# Rotating the inputs

[Fixed structure](stationary.md) in a feature -- the same thing at the same place, whatever
the weather -- can come from three places: a learned positional embedding, the grid's own
coordinates or a local time computed from them, or a boundary field the model is given
(orography, land, sea-surface temperature). An archive alone cannot tell them apart. A
running model can: turn what it is given east and run it again.

## The experiment

A system that can be turned is `Rotatable`: beside `Intervenable` it names the parts of its
state it can turn (`rotation_parts`: the weather, the statics, the forcings, as the adapter
sees them) and turns the ones asked for by whole grid columns (`rotate`). What the model
holds of its own -- its weights, its positional embedding, its grid -- is never part of a
state and never turns.

`rotation_test` runs the system once as it is and once turned, without noise, records a
layer, and asks of each column (a channel, or a feature of a basis):

- **stays**: how well the turned run's map correlates with the original map where it was;
- **follows**: how well it correlates with the original map turned by the same shift;
- **score**: the turned map regressed on both, `a` times the original map where it was
  plus `b` times the original turned along (area-weighted, in the column's own units), as
  `(max(a, 0) - max(b, 0)) / max(|a| + |b|, 1)`: +1 for a column that stayed, -1 for one
  that moved with the weather, `2 alpha - 1` for one that is a share `alpha` of each, and
  near 0 for one the two maps do not explain (noise, or minus the moved map), which is
  not anchored.

A pattern that maps onto itself under a shift -- a zonal band, anything with the same
symmetry as the turn -- cannot say which it did. The original map's correlation with itself
turned is its *self-similarity*, `s`, and a shift where `|s|` is at or above
`--max-self-similarity` (0.5) has no power for the column and is left out of its score. A
pattern that turns into its own *opposite* (`s` near -1: a wavenumber-two wave turned a
quarter of the circle) is no better: a mixture of the two maps is then a multiple of one,
and only its sign is left to read. The regression keeps the score from depending on `s`
in between; stays less follows would be `(2 alpha - 1)(1 - s)`, so a half-anchored column
read on a shift with negative `s` would pass as anchored.

A column is **anchored** when every shift with power gives a score of at least
`--threshold` (0.5: three quarters anchored); one that vanishes when turned, and so
correlates with nothing, is not.
STAYS and FOLLOWS are averaged over the shifts with power, like the score. Correlations are
area-weighted, over nodes valid both where they are and where they came from. The default
shifts are three turns of at least 60 degrees spread round the circle.

## Parts are the point

What is turned decides what "stays" means:

| Turned | A column that stays is tied to |
|---|---|
| everything | what the model holds of its own: a positional embedding, the grid's coordinates |
| the weather alone | that, or a boundary field that was left where it was |
| the boundaries alone | anything but the boundaries |

In a climate model a column tied to the land or the orography is physical, not an
artefact; one tied to the grid is the model's. A test that turns the weather and leaves the
statics, or that turns the inputs but not a local time worked out from the grid's
longitudes, mixes the two, and its "stays on the grid" is half of each. Name the parts.

## A planted answer

The toy system with `positional` and `static` set adds a fixed pattern to channel 3 of layer
0 that never moves, and a boundary field to channel 2 that moves only with the `static`
part. Turning everything leaves only channel 3 anchored; turning the weather alone leaves
channel 2 behind as well:

```console
$ exege latents rotate --adapter toy-dynamics --adapter-option positional=5 \
    --adapter-option static=5 --adapter-option n_lat=8 --adapter-option n_lon=12 \
    --layer 0 --start 0 --start 1
channels of layer 0 at time 0, turned 60, 180, 300 degrees east; parts turned: state, static

CHANNEL  SCORE  STAYS  FOLLOWS  POWERED  ANCHORED
3        +0.98  +0.98  -0.05    6        yes
0        -1.00  -0.03  +1.00    6        -
1        -1.00  +0.06  +1.00    6        -
2        -1.00  +0.06  +1.00    6        -
$ ... --part state
channels of layer 0 at time 0, turned 60, 180, 300 degrees east; parts turned: state

CHANNEL  SCORE  STAYS  FOLLOWS  POWERED  ANCHORED
3        +0.98  +0.98  -0.05    6        yes
2        +0.95  +0.97  +0.07    6        yes
0        -1.00  -0.03  +1.00    6        -
1        -1.00  +0.06  +1.00    6        -
```

Channels 0 to 2 follow the weather exactly (-1.00); channel 3's +0.98 is the fixed pattern
less the little weather it also carries.

## What it does not say

- Whether an anchored column *matters*. That is a [steering](steering.md) question: steer
  it toward its mean (`--mode toward --amount 1`, a mean-ablation) rather than to zero,
  which takes a feature that is nearly constant somewhere it has never been.
- Anything on a mesh, which a turn does not take node to node; or on a grid whose
  longitudes are uneven or do not close the circle. Both are refused.
- Anything about a system whose adapter cannot turn its state: the protocol is the
  adapter's to implement, and a real one is on the [roadmap](roadmap.md#steering).

## From Python

```python
from exege.adapters.toy_dynamics import ToyDynamics
from exege.latents import rotation_test

system = ToyDynamics(n_lat=8, n_lon=12, positional=5.0, static=5.0)
result = rotation_test(system, layer=0, parts=["state"], starts=(0, 1))
result.anchored  # channels 2 and 3
result.summary(top=3)
```
