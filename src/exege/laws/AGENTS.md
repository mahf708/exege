# laws — whether a model obeys a law, and which of its parts do

LAWFUL (Chen, Parker & Arora 2026) on any `circuits.Patchable` system.

| Module | Holds |
|---|---|
| `law.py` | the `Law` contract, `consistency` (LAWFUL eq. 10), and waves on a sphere: `Envelope`, `projection`, `band_mean`, `phase_rate` |
| `waves.py` | `ZonalAdvection` (omega = k U) and `Rossby` (barotropic dispersion, intrinsic by default) |
| `lawful.py` | `measure`, `deviations` (sufficiency, necessity, source-randomized controls), `search` (tau-consistent circuits) |

## Rules that must not be lost

- **A law is physics in numpy.** It names fields, plants a counterfactual on the model's own
  states, reads an observable from the response, and predicts it. It never imports a
  model; the model is a `Patchable`, reached through the registry, and every procedure is
  tested on `toy-transformer`, whose law (omega = k U, carried by `a1.h0`) is planted.
- **The response is a tendency response.** `[step(x') - x'] - [step(x) - x]` for the model,
  `step_C(x) - step(x)` for a patched run: a time-stepper's skip carries any perturbation
  forward, and that is no component's doing. Do not score next-state differences.
- **The input embedding is not a component.** Patching it routes the whole perturbation;
  `components()` leaves it out and `measure` refuses it.
- **The law must vary over the family.** Eq. 10 divides by the law's own spread; a family
  over which the law is constant (phase *speed* of advection, say) cannot be scored — use
  the frequency. `consistency` refuses rather than divides by zero.
- **Doppler shifting is not the beta effect.** `Rossby` scores the intrinsic frequency by
  default; with `intrinsic=False` the jet's `k U` dominates and M says little about beta.
- **Controls come with circuits.** A component set is reported with source-randomized
  controls when they are asked for (`controls=`); a circuit whose control recovers M as
  well is not evidence of anything.
- Third-party imports are capped at numpy and click by `tests/test_purity.py`.
