# Steering

Everything in `exege` so far *reads*: an adapter hands over activations a model produced,
and the package computes on them. A steering experiment asks a different question, "what
does the model do if this feature is changed?", and answering it means an adapter
**writes into** the model. This page describes the experiment, the adapter protocol that
makes it possible, and what is measured.

```{admonition} Status
:class: note

Implemented: the protocol (`Intervenable`, `Hook`, `Steer`), the runner
(`run_steering`) with its six arms, reference modes, open and closed loops, outcomes in a
region, many cases (`run_cases`) and dose curves with an amount chosen leave-one-out
(`dose_curve`); the `exege latents steer`, `dose` and `direction` commands; and a toy
system to test them against. Planned: a real adapter, a view and a figure in the app, and fitted
dictionaries on the toy system; see the [roadmap](roadmap.md#steering).
```

## What a steer is

A steer edits the model's latents at one layer, along a feature's direction, and
then lets the forward pass continue from the edited value. Three things name it:

- **where**: a layer, and the forward steps (times) at which it acts, optionally on a
  subset of nodes;
- **along what**: a feature of a `latents.Decomposition`, i.e. a direction in that layer's
  channel space (`Dictionary.decoder[f]` in the layer's units; a PCA component);
- **by how much**: `add` an amount of the direction, `scale` the feature's own activation
  by a factor, `clamp` it to a value, or move it `toward` a reference or scale its `excess`
  above one ([below](#modes-and-references)). Every mode reduces to a per-node change
  `delta` in the feature's activation, and the layer moves by `delta * direction`.

The edit is made on the model's own latents (`h' = h + delta * direction`), not on a
reconstruction of them, so whatever the dictionary fails to explain is left as it was. That
is what makes the arms below separable.

## The adapter contract

The adapter contract is otherwise a set of read protocols (`LatentSource`,
`ReferenceFields`): the adapter is passive, and a result is a function of files that exist.
A steer is a *run*: the adapter owns the model, and the experiment owns what
happens to the latents at a point inside it. Three consequences:

1. **A new protocol, not a new method on `LatentSource`.** Reading a recorded archive is
   cheap, repeatable and safe; running a model is none of those, and most sources (an
   archive on a hub) cannot do it. One adapter object may implement both, and callers ask
   with `isinstance`, as everywhere else.
2. **The model's side stays in the model's environment.** The protocol hands the adapter
   a function to call and takes arrays back. No torch, no framework object crosses it, so
   the code that designs an experiment still runs on numpy alone.
3. **Determinism becomes the adapter's promise.** A recorded archive is the same every
   time it is read. A run is the same only if the adapter makes its noise a function of a
   seed it is given, which the paired design below needs.

The protocol is `latents.Intervenable`:

```python
class Intervenable(Protocol):
    def info(self) -> LatentInfo: ...
    def grid(self) -> Grid: ...
    def initial_state(self, start: str | int = 0) -> Any: ...
    def run(
        self,
        initial_state,
        steps: int,
        hooks: Sequence[Hook] = (),
        *,
        noise_seed: int | None = None,
        record: Sequence[tuple[int, int]] = (),
    ) -> Rollout: ...


@dataclass(frozen=True)
class Hook:
    layer: int
    time: int  # the forward step, from 0
    edit: Callable[[np.ndarray], np.ndarray]  # (n_nodes, n_channels) -> same shape
```

A hook at `(layer, time)` receives the latent tensor the forward pass produced there and
returns what it continues with. `Rollout` holds the physical fields each step wrote
(`(steps, n_nodes)` per name, step `t` being what the pass starting at `t` produced, the
`lead=1` convention of `ReferenceFields`) and the latents at the `record`ed
`(layer, time)` pairs, taken *after* any hook there. The initial state is opaque to exege:
it is whatever the adapter's `initial_state` returned.

## The arms

Each is one run from the same initial state, with the same noise.

| Arm | The run | What it isolates |
|---|---|---|
| control | no hooks | the baseline every difference is taken against |
| reconstruction-only | at the hook point, `h` is replaced by `decode(encode(h))` | the effect of passing through the dictionary at all: its reconstruction error, with no feature touched |
| feature | `h' = h + delta * direction` for the chosen feature | the effect of the feature, plus nothing the dictionary dropped |
| random direction | the same `delta`, along a random unit direction scaled to the feature's direction length; repeated draws | what an edit of this size does to the model in a direction that means nothing |
| comparison | the same `delta`, along a named direction from another basis (`--compare`), scaled the same way | whether the feature does anything a plainer direction does not: a [difference of means](#comparison-and-matched-arms), a principal component |
| matched | the same steer applied to another feature of the basis that carries about as much where the edit acts (`--matched N`) | what a real, in-distribution feature of this size does; a random direction may push the model somewhere it has never been |

The reconstruction arm is not subtracted from the feature arm: the feature edit is applied
to `h`, so the feature arm's difference from control *is* the feature's effect. The
reconstruction arm says how large the effect of simply using the dictionary would be, so a
reader can see whether a claimed feature effect is bigger than that. The random arm is the
yardstick for "bigger than an arbitrary push of the same size": the feature's response is
reported as an effect size and an empirical rank within the random draws.

## Paired noise

A stochastic model run twice from one state differs for reasons that have nothing to do
with the edit. All arms of one repetition therefore share a `noise_seed`, so the
difference against control contains the edit and nothing else; with no steer it is
exactly zero. Seeds are then repeated (`--seeds`) to get an uncertainty: the paired
difference is computed per seed, and the spread across seeds says how far to trust the
mean. The random draws are fixed across seeds, so a draw is one direction tested under
every noise. Pairing is only as good as the adapter's promise that its noise does not
depend on what the hooks did (the noise is drawn from the seed, not from the state).


## Modes and references

`add`, `scale` and `clamp` need nothing but the amount. Two modes measure the activation
against a **reference**, one value per node:

| Mode | The activation becomes | Amount that changes nothing |
|---|---|---|
| `toward` | `f + amount * (reference - f)`: 1 sets it to the reference | 0 |
| `excess` | `f + (amount - 1) * max(f - reference, 0)`: only what lies above the reference is scaled, so 0 caps it there and 2 doubles the excess | 1 |

What the reference *is* decides what the experiment asks. The feature's mean over many
times makes `toward 1` a mean-ablation, the usual remedy for zero-ablation taking a
feature to a value it never has. Its value at one other time makes it a resample-ablation.
Its level on quiet days of the same season makes `excess 0` "this feature as it is on a
day without the storm", and `excess 2` "twice the storm's share of it". A reference is one
value per *node*, so it must be read on the same nodes in the same order: it carries the
grid's `node_digest`, and a steer refuses one read on a grid laid out otherwise, however
many nodes it has.
`feature_reference` reads one from an archive of the same network, averaging the
feature over the times named, and the steer's spec records those times and the
reference's content hash. On the command line, `--reference-from ARCHIVE` with
`--reference-time T`, repeated.

## Open and closed loops

A steer acting at several steps can work out its change two ways (`--loop`):

- **open** (the default): from the control's latents. Every arm of a seed changes the
  same nodes by the same amounts however the runs drift apart: a forcing applied from
  outside, and what the paired design above assumes.
- **closed**: from the latents the steered run itself produced, at the moment of the edit.
  An edit at step 2 reads what the edit at step 1 did. This is how steering inside a
  model's own forward pass behaves, and it compounds: amplifying a feature that feeds
  itself amplifies what the last amplification left.

They agree for `add`, and for any system whose latents do not read its state back. In a
closed loop the random and comparison arms are given the changes the feature arm actually
made, step by step: worked out from their own latents, pushes along other directions would
grow differently, and "an edit of the same size" would stop being true. (Matched arms are
steers of their own, and close their own loop.) The toy system with `feedback` set reads
its state back, and scaling the planted feature by 2 at three steps gives a response half
as large again in a closed loop:

```console
$ exege latents steer --adapter toy-dynamics --adapter-option masked=3 \
    --adapter-option feedback=0.4 --basis scratch/steer/planted.npz --layer 1 --feature 0 \
    --mode scale --amount 2 --time 1 --time 2 --time 3 --steps 6 --seeds 0,1,2 \
    --random-draws 10 --field temperature --loop open
FIELD        RESPONSE  +-    RECON_ONLY  RANDOM_|RESP|  EFFECT_SIZE  RANK  P
temperature  10        0.05  9.807       2.158          5.62         1/11  0.091
$ ... --loop closed
FIELD        RESPONSE  +-     RECON_ONLY  RANDOM_|RESP|  EFFECT_SIZE  RANK  P
temperature  14.74     0.066  9.807       3.181          5.62         1/11  0.091
```

Neither is the right one: they are different experiments, and a dose curve read off one
says nothing about the other. A response that runs away in a closed loop and not an open
one is the loop's doing, not the feature's. (The reconstruction arm is large here for the
same reason: with feedback, the channel the dictionary drops is dropped at every step,
and the loss compounds.)

## Comparison and matched arms

A random direction answers "is this bigger than an arbitrary push?". Two more arms ask
sharper questions.

**Comparison** arms push along a direction from another basis by the same `delta`, scaled
to the feature's length. The plainest is a **difference of means**: the mean latents at
times the event is present less the mean at times it is absent, over a region.
`exege latents direction` writes one as a basis of one direction (kind `directions`),
which `--compare NAME=FILE` reads:

```console
$ exege latents toy scratch/dir/control
$ exege latents direction scratch/dir/control --layer 3 --event-time 0 --event-time 1 \
    --quiet-time 6 --quiet-time 7 --lat 10 --lon -140 --radius-km 2500 --label storm \
    --out scratch/dir/storm.npz
wrote scratch/dir/storm.npz: the direction from 2 quiet to 2 event time(s) at layer 3, length 1.27; --basis-sha256 75923395126e67f085f80715d406f1e0245d40e6f90fc19346b288e5266ee4e9
```

If the difference of means does as much as the dictionary's feature, the dictionary has
found a direction, not a mechanism anyone needed a dictionary for.

**Matched** arms apply the same steer -- same mode, amount, times, nodes and loop -- to
other features of the same basis. They are chosen on the first seed's control, at the
times the steer acts: what each feature *carries* there (its activation's magnitude,
area-weighted over the nodes the steer edits, times its direction's length) is set
against the feature's, and the closest by the log of the ratio are taken, among those that
carry anything and whose direction is no closer to the feature's than `--max-cosine`
(0.3). Matching on what a feature carries where the edit acts, and not on how often it
fires across the globe, is the point: a feature that fires as often but is nearly silent
inside the storm receives almost no edit under `scale` or `excess`, and its small effect
then says nothing about the feature's. Under `toward` and `excess` each matched feature
takes its own reference, read the same way. The feature's `MATCHED_RANK` is its place, by
magnitude, among itself and them.

On the toy, with a comparison along the fourth channel, which the planted dictionary does
not read (`Directions` with one unit row):

```console
$ exege latents steer --adapter toy-dynamics --adapter-option masked=3 \
    --basis scratch/steer/planted.npz --layer 1 --feature 0 --mode scale --amount 3 \
    --time 1 --steps 5 --seeds 0,1,2 --random-draws 20 --matched 2 \
    --compare ch3=scratch/steer/along3.npz
scale 3 on feature 0 of layer 1 at time(s) 1 (open loop); 5 step(s), seed(s) 0, 1, 2, 20 random direction(s), 2 matched feature(s)

FIELD        RESPONSE  +-      RECON_ONLY  RANDOM_|RESP|  EFFECT_SIZE  RANK   P      MATCHED_RANK  CH3
moisture     0         0       0.3107      0.08638        -1.83        21/21  1      3/3           0
pressure     0         0       0.1266      0.1239         -2.07        21/21  1      3/3           0.1025
temperature  0.4099    0.0045  0.3335      0.1375         2.52         1/21   0.048  1/3           0

matched features: 1, 2
```

The two matched features reach moisture and pressure, which the planted one does not, so
on temperature it ranks first among them and on the others last: a response of zero ties
with anything and wins nothing, so it takes the bottom place, among the random draws as
among the matched features. The comparison along channel 3 reaches pressure, at half
weight, and nothing else. A comparison basis is checked against the layer like the steered
one: a direction fitted on another layer or network is refused.

## Where an outcome is measured

By default a field is reduced to its area-weighted global mean at each step. An edit
that acts in one place is diluted to nothing by a global mean, so `Outcome` (`--outcome`)
also takes the `min` or `max` over the valid nodes, and a region to measure in. On the
command line `--lat`, `--lon` and `--radius-km` name one region that both restricts the
edit to its nodes and is where the outcome is measured: the minimum pressure within
1,500 km of a storm, say.

## Cases

One initial state is an anecdote. `run_cases` runs the whole experiment from each of
several (`--start`, repeated), a region per case if they differ, and sets the feature's
responses side by side: their median, how many were positive and negative, how many ranked
first among the random draws. Each case keeps its own result, so the spread is there to
read, not averaged away.

```console
$ exege latents steer --adapter toy-dynamics --adapter-option masked=3 \
    --basis scratch/steer/planted.npz --layer 1 --feature 0 --amount 1.5 --time 1 --steps 5 \
    --start 0 --start 1 --start 2 --lat 40 --lon 60 --radius-km 4000 --outcome max \
    --field temperature --random-draws 10
...
across cases
FIELD        MEDIAN  POSITIVE  NEGATIVE  RANK_1  CASES
temperature  1.406   3         0         3       3
```

## Dose curves, and choosing an amount

How the response grows with the amount is a curve, and the curve is the result:
`dose_curve` (`exege latents dose --amounts`) runs the control and the feature arm at every
amount, from every case. Given **targets** -- for each case, the series its outcome should
have followed (an analysis, a best track) -- every amount is scored by its error, the RMS
over the steps from the first edit on.

Picking the amount that does best on a case and reporting how well it does *on that case*
is a hindsight optimum: no forecaster could have made that choice. With two or more cases,
`leave_one_out` picks, for each case, the amount that does best by the median over the
*other* cases, and reports that case's error there, beside the hindsight optimum and the
unsteered control. A correction is judged by the first number. On the toy, with targets
made by adding 2 to the planted feature in each case's control:

```console
$ exege latents dose --adapter toy-dynamics --adapter-option masked=3 \
    --basis scratch/steer/planted.npz --layer 1 --feature 0 --mode add --amounts 0,1,2,3 \
    --time 1 --steps 5 --start 0 --start 1 --start 2 --field temperature \
    --targets scratch/steer/targets.json
...
temperature: amount chosen leave-one-out
CASE  LOO_AMOUNT  LOO_ERROR  ORACLE_AMOUNT  ORACLE_ERROR  CONTROL_ERROR
0     2           2.9e-09    2              2.9e-09       2.305
1     2           1.15e-08   2              1.15e-08      2.305
2     2           2.105e-09  2              2.105e-09     2.305
median error: leave-one-out 2.9e-09, hindsight 2.9e-09, control 2.305
```

A planted answer agrees with itself, so here the three coincide. Where the best amount
swings from case to case, they part, and the leave-one-out error can be worse than doing
nothing at all. That is the finding, not a failure of the method. Random and matched arms
are not run inside a dose curve: it says how the response grows, and `steer` at a chosen
amount says whether it is unusual.

## What is measured

- **Latent differences**: per arm, the area-weighted RMS of `latents - control latents`
  (over nodes and channels) at every recorded `(layer, time)`, so the spread of an edit
  through the layers and steps is visible, not just its outcome.
- **Physical outcomes**: the fields the model writes (`Rollout.fields`), reduced to an
  area-weighted, mask-aware global mean per step. The *response* of an arm is the mean of
  its paired difference over the steps from the first edit on. The feature's `|response|`
  is set against the `|response|` of each random draw.

Nonfinite latents or fields on a valid node, a hook that changes the shape or returns a
nonfinite value, and empty selections (no seeds, no fields, no steps, no draws) are
`RequestError`s, as for every other read in the package. An edit never changes a masked
node.

`run_steering` returns a `SteeringResult` holding every `Run` (arm, seed,
draw, outcome series, latent RMS), every `Pairing` against the control of its seed, and
an `Effect` per field: the feature's signed response and its standard error across
seeds, the reconstruction arm's, the random draws', and the feature's `effect_size`,
`rank` (1 is a larger magnitude than every draw) and `p_value` among them.
The edit's size is worked out once per seed from the control's latents, so every arm of a
seed changes the same nodes by the same amounts however the runs drift apart; `scale` and
`clamp` are therefore relative to the control, not to the arm's own activations.

## Provenance

A result carries `result_provenance(info, basis=...)` of the system it ran (source,
commit, options, the basis's content hash) and the full specification of the experiment,
so a printed reproduction command pins what moves: `exege latents steer` ends with a
`reproduce:` line that carries the adapter and its options, `--basis-sha256`, and every
setting, numbers as they were given rather than rounded. Everything else it reads is pinned
by content too: each `--compare` file by `--compare-sha256 NAME=HASH`, a reference read
from an archive by `--reference-sha256` (the hash of the per-node values, so a reference
read from other times, or from a changed archive, is refused), and a dose curve's
`--targets` file by `--targets-sha256`. The matched features' references are named in the
spec by the same kind of hash, and together by `--matched-references-sha256`, which a run
whose matched references came out otherwise refuses. A result saved with `save_result` is the same record as JSON.

## Example

The synthetic system is linear and its latents do not read its state back, so the answer is
known: adding `a` to channel 0 of layer 1 moves temperature by `gain * a` on that step and by
`decay` times as much on each step after, and nothing else. The planted dictionary reads
three of its four channels, so it has a reconstruction error too. The numbers below are
from running exactly this (`masked=3` leaves three nodes with NaN fields, like land).

```console
$ python -c "
from exege.adapters.toy_dynamics import ToyDynamics
from exege.latents import save_basis
save_basis('scratch/steer/planted.npz', ToyDynamics(masked=3).planted_dictionary())"
$ exege latents steer --adapter toy-dynamics --adapter-option masked=3 \
    --basis scratch/steer/planted.npz --layer 1 --feature 0 --amount 1.5 --time 1 \
    --steps 5 --seeds 0,1,2 --random-draws 20
add 1.5 on feature 0 of layer 1 at time(s) 1; 5 step(s), seed(s) 0, 1, 2, 20 random direction(s)

FIELD        RESPONSE  +-       RECON_ONLY  RANDOM_|RESP|  EFFECT_SIZE  RANK   PERCENTILE
moisture     0         0        0.3107      0.2964         -1.83        21/21  0
pressure     0         0        0.1266      0.4251         -2.07        21/21  0
temperature  1.406     1.2e-09  0.3335      0.4719         2.52         1/21   100

reproduce: exege latents steer --adapter toy-dynamics --adapter-option masked=3 --basis scratch/steer/planted.npz --basis-sha256 72ac49085d99b99ce37574a829a851708590fb99c1501497f907d09db4440960 --layer 1 --feature 0 --mode add --amount 1.5 --time 1 --steps 5 --seeds 0,1,2 --random-draws 20 --random-seed 0
```

- **response** is the mean over steps 1 to 4 of the paired temperature difference:
  `1.5 * 2.0 * (1 + 1/2 + 1/4 + 1/8) / 4 = 1.406`, to rounding. Moisture and pressure do
  not move at all. The standard error across the three noise seeds is `1e-9`: the paired
  difference holds the edit and nothing else.
- **random** directions of the same length move temperature by `gain * 1.5 * r[0]` on the
  first step with `|r[0]| < 1`, so none matches the feature (rank 1 of 21), while on
  moisture and pressure, which the feature does not reach, it is the weakest of the draws
  (rank 21 of 21): an effect size is only meaningful against the field it is about.
- **recon_only** is what substituting `decode(encode(h))` does with nothing edited. Here
  it is a quarter of the feature's response (0.33 against 1.41), which is the reason the
  arm exists: a feature effect near that size would not be distinguishable from the
  dictionary being lossy.

`scale` and `clamp` change the activation the feature has, so their size depends on the
noise and the standard error is no longer zero:

```console
$ exege latents steer --adapter toy-dynamics --adapter-option masked=3 \
    --basis scratch/steer/planted.npz --layer 1 --feature 0 --mode scale --amount 3 \
    --time 1 --steps 5 --seeds 0,1,2 --random-draws 20 --field temperature
FIELD        RESPONSE  +-      RECON_ONLY  RANDOM_|RESP|  EFFECT_SIZE  RANK  PERCENTILE
temperature  0.4099    0.0045  0.3335      0.1375         2.52         1/21  100
```

`--record FILE` also writes a [record](records.md) of this: its spec, each field's response
and every random draw, the basis by hash and the `reproduce:` command, in 4.8 KB where `--out`
is 110 KB. `exege app --record FILE` draws each field's response against the draws.

## Non-goals

- No real model is run, loaded or downloaded by `exege`; a real adapter is a later piece
  of work and lives in its own environment.
- No search over features, and no claim that a feature "causes" anything beyond what the
  arms show; the numbers are evidence for a reader to weigh. A dose curve reports every
  amount it ran, and the only amount it *chooses* is chosen leave-one-out.
- No steer on several features at once beyond holding one in place: the runner varies one
  feature, and matched and comparison arms each push one other.
- No gradients: an edit is a forward-pass substitution.
- No plotting here; figures and the app may draw a result later.

## From Python

```python
from exege.adapters.toy_dynamics import ToyDynamics
from exege.latents import Steer, load_basis, run_steering, save_result

system = ToyDynamics(masked=3)
basis = load_basis("scratch/steer/planted.npz")
result = run_steering(
    system, basis, Steer(layer=1, feature=0, amount=1.5, times=(1,)), steps=5, seeds=(0, 1, 2)
)
result.effects["temperature"].rank  # 1: larger than every random direction
save_result("steer.json", result)
```

The arms and modes above, and many cases and a dose curve:

```python
from exege.latents import (
    Box, Case, Outcome, dose_curve, feature_reference, mean_difference, open_source,
    run_cases,
)

archive = open_source("latents/atmosphere")  # an archive of the same network
reference, meta = feature_reference(archive, basis, layer=8, feature=676, times=quiet_days)
steer = Steer(layer=8, feature=676, mode="excess", amount=2.0, times=(0, 1, 2, 3),
              loop="closed", reference=reference, reference_meta=meta)
dom = mean_difference(archive, layer=8, event_times=storm_days, quiet_times=quiet_days,
                      region=Box(0, 30, -100, -60))
storm = Box(lat_min=15, lat_max=35, lon_min=-100, lon_max=-75)
study = run_cases(
    system, basis, steer, [Case(start=t, region=storm) for t in starts],
    outcome=Outcome(statistic="min"), steps=16, comparisons={"dom": (dom, 0)},
    n_matched=3, reference_for=lambda j: feature_reference(
        archive, basis, layer=8, feature=j, times=quiet_days)[0],
)
study.summary()["mean_sea_level_pressure"]["median"]
curve = dose_curve(system, basis, steer, [0, 1, 1.5, 2, 3], steps=16, cases=cases,
                   targets=analyses)
curve.selections["mean_sea_level_pressure"].medians  # loo, oracle, baseline
```
