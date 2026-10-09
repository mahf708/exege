# exege

*exege* is a light, framework-agnostic package for understanding and evaluating
scientific machine-learning models. It takes its name from the Greek stem *exēgē-*,
associated with explanation and interpretation.

It reads the activations recorded from inside a model — an emulator of the atmosphere or
the ocean, say — and asks what its internal channels respond to: which ones light up over
a region, where else the model looks the same, what the main patterns are, how they evolve
from one step to the next, and what a perturbation did to them. On top of that it trains
sparse autoencoders whose features are easier to name than channels, scores them on data
they never saw, and steers a model along one of them to see what it does.

```{admonition} Research tool
:class: note

exege is early. What these pages describe works; expect it to change.
```

## What is in it

- **`exege.latents`** — [latent diagnostics](archives.md): regions, bases, time series,
  comparisons between runs, and the fields a feature goes with. Also
  [steering](steering.md) and [experiment records](records.md).
- **`exege.nn`** — [sparse autoencoders](nn.md) trained on those latents, and their
  [held-out evaluation](evaluation.md).
- **`exege.figures`** — maps and figures, with no web framework in them.
- **`exege.app`** — [a local web app](app.md) over all of the above.

Every API returns objects and prints nothing; the [command line](cli.rst) is one client of
it, a notebook another. New to it? [Install it](installing.md), then follow the
[quickstart](quickstart.md), which needs no model and no data.

## How to cite

exege reimplements and builds on the work of others; [Background](background.md) lists the
papers behind it. Cite them if you use it.

## Table of contents

```{toctree}
:maxdepth: 1

installing
quickstart
```

```{toctree}
:caption: Latent diagnostics
:maxdepth: 1

archives
regions
bases
time
comparing
features
provenance
```

```{toctree}
:caption: Sparse autoencoders
:maxdepth: 1

nn
evaluation
```

```{toctree}
:caption: Experiments and the app
:maxdepth: 1

steering
records
app
```

```{toctree}
:caption: Extending exege
:maxdepth: 1

design
adapters
```

```{toctree}
:caption: Reference
:maxdepth: 1

cli
archive-format
api/latents
api/nn
api/figures
api/adapters
```

```{toctree}
:caption: About
:maxdepth: 1

background
roadmap
```
