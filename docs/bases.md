# Bases: PCA and dictionaries

A PCA fitted in the region is one way to turn channels into features. A PCA fitted over
the whole globe and every time is another, and a [sparse autoencoder](nn.md) a third.
They differ in how they are found and agree in what is done with them afterwards, so all
of them are a `Decomposition` — `transform`, `directions`, `describe` — and every analysis,
the CLI and the [web app](app.md) take one wherever they take another.

## Fitting a global PCA

A basis is fitted once and used many times, so it has a file: one `.npz` of plain arrays
and a JSON record of how it was made.

```console
$ exege latents pca latents/atmosphere --components 32 --out pca8.npz
wrote pca8.npz: 32 component(s) of layer 8 over 17 time(s), 69.1% of the variance; --basis-sha256 <sha256 of pca8.npz>
```

The line ends with the file's content hash, in the form a later command takes it
([provenance](provenance.md)).

That is an area-weighted PCA over all 1.1 million node-times of the layer, from moments
accumulated a block at a time: 2.9 s, and the sums are 384 × 384 however many times there
are. It is the baseline a learned dictionary has to beat — on this layer, 32 components
hold 69.1% of the variance, and [a top-32 sparse autoencoder](nn.md) 82.1%.

## Using a basis

```console
$ exege latents region latents/atmosphere --lat 5 --lon -140 --radius-km 1500 \
    --centered --top 6 --features 3 --basis sae8.npz
...
F676  peak 27.5  45(-0.18)  107(+0.14)  351(-0.13)  124(+0.13)  129(+0.13)  326(+0.13)
F48   peak 12.3  124(+0.16)  326(-0.15)  108(+0.14)  104(+0.14)  380(+0.13)  196(+0.13)
F500  peak 11.5  280(+0.17)  351(-0.16)  332(+0.16)  211(+0.15)  22(+0.14)  114(-0.14)
```

A basis is given the raw latents whatever `--centered` says: it carries the standardization
it was fitted with, and centering twice is simply wrong. Only the features asked for are
computed, so a map of three features out of 1,024 does not cost the other 1,021. They are
ranked by what each *contributes* in the region — its activation times the length of its
direction — because a dictionary is free to trade one for the other.

````{admonition} An index is not an identity
:class: warning

Every layer of this model is 384 channels wide, and so is every seed of a campaign, so
a basis *fits* anywhere and means something in one place. Its file records the
network and layer it was fitted on, and it is refused anywhere else:

```console
$ exege latents region latents/atmosphere --lat 5 --lon -140 --layer 4 --rank-layer 4 \
    --features 3 --basis sae8.npz
Error: the basis (sae8.npz) was not fitted here: layer 4 against the layer 8 it was fitted on
```

A basis that does not say where it was fitted — one made with `fit_pca` from plain
arrays, say — is refused too, until you pass `--allow-unverified-basis`
(`allow_unverified_basis=True` from Python, on `analyze_region`, `region_series` and
`rank_by_field` alike); the choice is recorded in the result's settings. To put a basis
to a layer it says it was *not* fitted on, strip what it says first,
`dataclasses.replace(basis, meta={})`, and then allow it: along a residual stream, a
dictionary from one layer can be a fair question to put to the next.
````

```{admonition} The way back to the model
:class: tip

The same file is the hand-off to a steering experiment. The model's environment
needs nothing but numpy to read it — `np.load("sae8.npz")["decoder"][676]` is the
direction feature 676 writes, and `["components"][0]` the first principal component —
and the file's `record` says which archive, layer and times it was fitted on.
```

## Reusing a region's PCA

The PCA `analyze_region(..., n_components=2)` fits is a basis like any other: save
`result.pca` with `save_basis`, and the file keeps the layer, time, region and source it
was fitted on, so it is checked like the rest when it is used again through `basis=` or
`--basis`. It takes raw latents even when the analysis was `centered=True`, since it
carries its own mean.

## Importing a dictionary trained elsewhere

A dictionary need not come from [`exege nn`](nn.md). The one MacMillan & Ouellette
(2025) trained on GraphCast's processor is public, and others like it are laid out the
same way: an encoder `W_enc` (features × channels), a decoder `W_dec` (channels ×
features, unit-length columns), a pre-bias `b_pre`, and a `k`. What they share is
**token normalization**: each node is standardized on its own, its channels less their
mean, divided by the length of what is left,

```
xn    = (x - mean(x)) / |x - mean(x)|          per node, over its channels
code  = top_k( relu((xn - b_pre) @ W_enc.T) )
recon = code @ W_dec.T + b_pre                  in the layer's raw units
```

and that is undone by nothing on the way out. A node's own length cannot be put back
by one mean and one scale, so the decoder writes the layer's **raw** units directly,
and the pre-bias is taken off in normalized units and added back in raw ones. exege
reads it as a `Dictionary` with `input_norm="token"` (mean zero, scale one): an
activation is invariant to rescaling or shifting a node as a whole, and a decoder row is
a direction in raw units, which is what a [steer](steering.md) moves along.

`exege latents import-sae` writes one as a basis file. The arrays are named by
`--encoder`, `--decoder` and `--pre-bias` (the defaults above); the decoder may be either
way round, and only a square one needs `--decoder-layout`. Here it is on a random
dictionary of the [toy emulator](quickstart.md)'s width, written with numpy:

```console
$ exege latents toy scratch/sae/toy
$ python -c "
import numpy as np
rng = np.random.default_rng(0)
W_enc = rng.normal(size=(64, 16)).astype('float32')
W_dec = rng.normal(size=(16, 64)).astype('float32')
W_dec /= np.linalg.norm(W_dec, axis=0)
np.savez('scratch/sae/published.npz', W_enc=W_enc, W_dec=W_dec, b_pre=np.zeros(16, 'float32'))"
$ exege latents import-sae scratch/sae/published.npz --k 4 --like scratch/sae/toy --layer 3 \
    --out scratch/sae/imported.npz
wrote scratch/sae/imported.npz: 64 feature(s) over 16 channel(s), top 4, token-normalized; --basis-sha256 11cf08a1d92cade60b0f81014269c08b9df676ee38224a542b56ec6ae384cba8
$ exege latents census scratch/sae/toy --layer 3 --basis scratch/sae/imported.npz --top 3
features of layer 3 at 0424-02-27T06:00:00, by coverage

FEATURE  COVERAGE  MEAN  STRENGTH  PEAK  AT
14       0.988     1.82  1.84      2.04  -77, -86
47       0.979     1.42  1.45      2.35  11, -139
22       0.760     1.13  1.49      1.67  -4, -139
```

A file like this says nothing about which network it was trained on, and
[an index is not an identity](#using-a-basis). `--like SOURCE --layer N` records the
identity of an archive of that network at that layer, so the basis is accepted there and
refused anywhere else:

```console
$ exege latents census scratch/sae/toy --layer 2 --basis scratch/sae/imported.npz
Error: the basis (scratch/sae/imported.npz) was not fitted here: layer 2 against the network layer 3 it was fitted on
```

Without `--like`, the basis records no identity and every command needs
`--allow-unverified-basis` to use it. Either way it records no training times, so an
[evaluation](evaluation.md) marks it `fitted_on: unknown`, and takes its explained
variance about the mean of the times it measures (there is no training mean to take it
about). The file read and its sha256 are kept under `meta["imported_from"]`.

## From Python

```python
from exege.latents import (
    Region,
    accumulate_moments,
    analyze_region,
    load_basis,
    open_source,
    pca_from_moments,
    region_series,
    save_basis,
)

source = open_source("latents/atmosphere")
region = Region(lat=5, lon=-140, radius_km=1500)
moments = accumulate_moments(source, layer=8)  # area-weighted, over every time
save_basis("pca8.npz", pca_from_moments(moments, 32))  # records where it was fitted

basis = load_basis("sae8.npz")
result = analyze_region(source, time=0, layer=8, region=region, n_components=3, basis=basis)
series = region_series(source, layer=8, region=region, basis=basis, features=[676])
series.elapsed_seconds  # under the archive's calendar; None when it cannot say

import numpy as np
from exege.latents import token_normalized_dictionary

with np.load("published.npz") as arrays:  # a file you read: exege.latents takes arrays
    imported = token_normalized_dictionary(
        arrays["W_enc"],
        arrays["W_dec"],
        arrays["b_pre"],
        k=32,
        meta={"fitted_on": basis.meta["fitted_on"]},  # where it belongs, if you know
    )
save_basis("imported.npz", imported)
```
