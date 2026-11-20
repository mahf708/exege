# Circuits

A circuit is the part of a network a behavior runs through: a subset of the edges along
which components (the input embedding, attention heads, MLPs) write into the residual
stream and other components read it. `exege.circuits` finds circuits the way
[Vi-CD](https://arxiv.org/abs/2604.14477) does for vision transformers. It patches one edge
at a time from a counterfactual run and keeps the edges whose loss moves the behavior.

This page uses `toy-transformer`, a two-block network shipped for tests and docs whose
circuits are planted. A real model plugs in the same way, through an adapter that
implements [`Patchable`](#the-adapter-contract).

## The graph

Senders write into the residual stream and receivers read it. In Vi-CD's reduced graph, the
heads of a block are separate senders but share one receiver, the block's attention input,
because their queries, keys and values all read the same normalized stream:

```python
from exege.circuits import residual_graph

g = residual_graph(depth=8, heads=6)
len(g.senders), len(g.receivers), g.n_edges  # (57, 17, 513)
g.upstream("m0")  # embed and block 0's six heads
```

## A behavior and its counterfactual

An emulator has no logits. The behavior is a scalar of what one step wrote: the
area-weighted mean of a field over a region, say. The counterfactual is the same step on a
*source* state with the evidence removed. Vi-CD inpaints the object out of an image; for an
emulator the source might be the same day with a feature taken out, another day, or the
climatology.

```python
from exege.circuits import field_mean, open_patchable

system = open_patchable(None, "toy-transformer")
grid = system.grid()
base, source = system.states([0, 1, 2]), system.states([3, 4, 5])
metric = field_mean(grid, "tracer", where=grid.within(0, 90, 4000))
```

## Finding a circuit

`prune` is Vi-CD's greedy pass, in ACDC's order. It visits receivers from the output
backwards and patches each edge to the source. An edge stays patched when the metric moves
by less than `tau` of the gap between the base and the source:

```python
from exege.circuits import prune

circuit = prune(system, base, source, metric, tau=0.01)
circuit.edges()  # ['embed->a1', 'embed->out', 'a1.h0->out']
circuit.recovery  # 1.0: the circuit alone gives the base run's value
circuit.fraction  # 3 of 21 edges
```

The toy's tracer is advected by one head, `a1.h0`, which reads the tracer from the
embedding and attends to each node's western neighbor. Those three edges are exactly what
its next tracer depends on, and pruning finds them and nothing else.

## Scoring edges

`attribute` gives every edge a score: the metric's change when that edge alone carries the
source's value. It is computed exactly, with one step per edge (`method="patch"`). It can
also be computed to first order from one gradient (`method="eap"`, edge attribution
patching), which needs a `LinearMetric` and an adapter that implements `EdgeGradients`. On
the toy's tracer path, which is linear, the two agree to round-off. `faithfulness_curve`
keeps the top-k edges by a score and patches the rest, giving the sparsity-against-
faithfulness comparison of Vi-CD's first experiment.

## Comparing circuits

Circuits for different regions, seasons or seeds can be compared with `jaccard`.
`edge_frequency` gives the share of circuits each edge appears in, and `core` keeps the
edges in at least 90% of them. That is Vi-CD's stable core.

## The adapter contract

A system whose circuits can be found implements `Patchable`. Its states are opaque to
exege, and only numpy crosses the protocol:

| Method | Returns |
| --- | --- |
| `graph()` | the `Graph` of senders, receivers and edges |
| `grid()` | where the fields live |
| `field_names()` | what a step writes |
| `states(starts)` | an opaque batch of initial states |
| `fields(state)` | a state's fields, `(batch, n_nodes)` per name |
| `perturb(state, delta)` | a new state with `delta` added to its fields |
| `step(state, patch=None)` | what one forward call wrote, `(batch, n_nodes)` per name |

A patch is either an `EdgePatch(source, mask)` or a `ComponentPatch(source, components)`.

- **`EdgePatch`.** Every receiver reads `sum_s source_out[s] + mask[s, r] *
  (base_out[s] - source_out[s])`. A mask of ones is exactly the plain step, and a mask of
  zeros is exactly the source's.
- **`ComponentPatch`.** The named senders write what they wrote on the source. This is what
  [LAWFUL](https://arxiv.org/abs/2607.28672) patches.

Both ends of each are checked in `tests/test_adapter_contracts.py`, together with what
cannot be had. `EdgeGradients` is optional: the derivative of a linear metric with respect
to the mask, at ones.

## From Python

```python
from exege.circuits import attribute, core, faithfulness_curve, prune

scores = attribute(system, base, source, metric)  # EAP when it can be had
curve = faithfulness_curve(system, base, source, metric, scores, sizes=[1, 3, 10])
found = [prune(system, system.states([i]), system.states([i + 8]), metric) for i in range(4)]
grid_core = core(found)  # in at least 90% of them
```
