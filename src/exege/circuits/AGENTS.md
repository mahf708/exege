# circuits — which paths a behaviour runs along

Edge-level circuits in a system that can be patched, after Vi-CD (Zukowska et al. 2026),
ACDC (Conmy et al. 2023) and EAP (Syed et al. 2024).

| Module | Holds |
|---|---|
| `graph.py` | `Graph` (senders, receivers, edges, masks); Vi-CD's reduced `residual_graph` |
| `contract.py` | `Patchable` (edge and component patching), the optional `EdgeGradients`, `open_patchable` |
| `metrics.py` | a behaviour as a scalar of a step's fields: `LinearMetric`, `field_mean`, `recovery` |
| `discover.py` | `attribute` (exact or EAP), `prune`, `faithfulness_curve`, `jaccard`, `edge_frequency`, `core` |

## Rules that must not be lost

- **The model stays behind `Patchable`.** Nothing here imports a framework; states are
  opaque and only numpy crosses. A system is reached through the registry, and every
  algorithm is tested on `adapters/toy_transformer.py`, whose circuits are planted.
- **Both ends are exact.** An all-ones `EdgePatch` is the plain step and all-zeros the
  source's; no components in a `ComponentPatch` is the plain step and every sender the
  source's. The contract tests ask this of every adapter: a patching scheme that does not
  meet it is wrong, however plausible its circuits.
- **A recovery is never reported without its gap.** A `Circuit`, an `Attribution` and a
  `FaithfulnessCurve` all keep the base and source metric values; when the two agree there
  is nothing to explain and `prune` refuses.
- **Masks are senders × receivers, 1 = the base's value.** Entries off the graph mean
  nothing and are zeroed by `Graph.check_mask`. Greedy pruning walks receivers from the
  output back, and each receiver's senders from the latest back.
- **Area-weight metrics.** `field_mean` uses `Grid.weights()`; a metric reads NaN only where
  its weight is zero.
- **EAP is first order.** It is offered only with a `LinearMetric` (whose gradient an
  adapter can be asked for) and is a ranking, not a replacement for patching: check a
  circuit it suggests with `patched` or `faithfulness_curve`.
- No user interface and no file formats here; third-party imports are capped at numpy and
  click by `tests/test_purity.py`.
