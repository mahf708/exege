# Physical laws

A model that predicts the atmosphere well may or may not have learned the laws that govern
it, and may or may not use them. `exege.laws` asks both questions the way
[LAWFUL](https://arxiv.org/abs/2607.28672) does, for any system that can be
[patched](circuits.md#the-adapter-contract).

- **Measure.** Plant a physical counterfactual on the model's own states, a small wave say,
  take a step, and read the law's observable off the response. Do it across a family of
  perturbations, and score the model against what the law predicts there.
- **Attribute.** Find the heads and MLPs that carry that agreement. The agreement must
  survive when the perturbation reaches only those components (sufficiency), and fail when
  it reaches everything else (necessity).

## Waves

Two laws ship, both over a family of zonal wavenumbers `s`. Each plants a wave in a
latitude band and reads its angular frequency from the phase advance of the model's
response, projected on `band(lat) * exp(i s lon)`.

| Law | Predicts | For |
| --- | --- | --- |
| `ZonalAdvection(field, speed)` | `omega = k U` | a pattern carried by a uniform flow; the toy's planted law |
| `Rossby(u, v, observe_field)` | `omega = k U_bar - beta k / (k^2 + l^2)` | barotropic Rossby waves on the state's own jet |

The Rossby wave is non-divergent, with winds from a streamfunction, and optionally balanced
geostrophically through surface pressure. By default it is scored on the *intrinsic*
frequency, `omega - k U_bar = -beta k / K^2`. That is the part the beta effect alone is
responsible for; otherwise the jet's Doppler shift dominates the score.

## The score

The consistency over the family is LAWFUL's eq. 10: one minus the squared misfit between
the model and the law, over the law's own spread across the family. It is 1 when the model
follows the law exactly, 0 when it is no better than the law's mean, and negative when it
is worse. The law must vary across the family for it to mean anything, which is why the
wave laws are scored on frequency and not on phase speed.

With `score="correlation"` (`--score correlation`), M is instead the correlation of the
model's values with the law's across the family: LAWFUL's similarity-based choice of the
scoring function. It asks whether the model follows the law's *shape* whatever its offset
and scale, which is the more useful question when the bridge has a free choice in it, such
as which level's wind steers the wave.

The model's response is its *tendency* response, `[step(x') - x'] - [step(x) - x]`. A
time-stepper carries any perturbation forward through its skip connection, and that is not
physics any component did.

## On the toy

`toy-transformer` advects its tracer with one head, `a1.h0`, at half a cell per step and
at every wavenumber, so its law is `ZonalAdvection` with that speed:

```python
import numpy as np
from exege.circuits import open_patchable
from exege.laws import Envelope, ZonalAdvection, deviations, measure, search

toy = open_patchable(None, "toy-transformer", n_lat=3, n_lon=24)
base = toy.states([0, 1, 2])
law = ZonalAdvection(
    "tracer",
    speed=np.deg2rad(15) / 2,
    wavenumbers=(1, 2, 3, 4, 5, 6),
    band=Envelope(0, 40),
    dt=1.0,
    radius=1.0,
)
measure(toy, base, law).M  # 1.0
d = deviations(toy, base, law, ["a1.h0"], controls=[toy.states([5, 6, 7])])
d.d_suf, d.d_nec  # (0.0, 5.2): sufficient and necessary
search(toy, base, law).circuits[-1].components  # ('a1.h0',)
```

## Circuits that carry a law

`search` is LAWFUL's incremental greedy search. Taking tau in increasing order, it adds the
component that most lowers the sufficiency deviation until the set is *tau-consistent*:
`d_suf <= (1 - tau) M0` and `d_nec >= tau M0`. It then drops whatever minimality does not
need. The circuits are nested.

Components are every sender but the input embedding, since patching the embedding would
route the whole perturbation. A run with components patched keeps the base state as its
input and takes those components' outputs from a run on the perturbed state.

Choosing the next component costs one patched sweep of the family per candidate, and
necessity is measured only for the circuits tested. With 56 heads and MLPs that adds up, so
search over whole blocks first: `search(..., groups=blocks(system))` (`--by-block`) has 8
candidates. Then search single components inside the blocks it finds
(`candidates=[...]`).

Ask for controls by passing other states as `controls=`. Their perturbed runs supply the
patched components instead (LAWFUL's source-randomized control). A circuit that its
controls also recover is not evidence of anything.

## On an emulator

```python
from exege.laws import Rossby, measure

law = Rossby(u=("U_2", "U_3", "U_4"), v=("V_2", "V_3", "V_4"), observe_field="V_3", ps="PS")
system = open_patchable("best.pt", "vit-interp")  # an adapter in the model's package
result = measure(system, system.states([0, 40, 80, 120]), law)
result.M, result.rows()
```

Each alpha costs one step on the perturbed state (and one patched step per
component set in a search), on the whole batch.
