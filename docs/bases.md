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
```
