# exege

Tools for understanding and evaluating scientific machine-learning models: what an
emulator holds inside, sparse autoencoders trained on it, figures, and a local app over
them. The name comes from the Greek stem *exēgē-*, associated with explanation and
interpretation.

> **Research tool.** `exege` is early. What is described here works; expect it to change.

## Install

The base install pulls only Click. Anything heavier sits behind an extra named after the
subpackage that needs it:

```console
$ uv pip install 'exege[latents]'     # or: pip install 'exege[latents]'
```

| Extra | Pulls | Gets you |
| --- | --- | --- |
| `latents` | numpy, xarray, netCDF4 | `exege.latents`: latent diagnostics on a grid |
| `figures` | matplotlib, cartopy | `exege.figures`: maps and series figures (brings `latents`) |
| `app` | streamlit | `exege app`: a local web app (brings `figures`) |
| `nn` | torch | `exege.nn` and `exege nn`: sparse autoencoders (brings `latents`) |
| `hf` | huggingface_hub | archives read from a Hugging Face repository, `hf://datasets/...` (brings `latents`) |

A missing extra says so, with the command that fits how `exege` was installed.

## Try it, with no model and no data

A toy emulator writes a latent archive with nothing but numpy; everything else reads it
like any other:

```console
$ exege latents toy scratch/toy/control
$ exege latents info scratch/toy/control --mask-variable sst
$ exege latents region scratch/toy/control --lat 10 --lon -114 --time 2 --centered --pcs 2
$ exege latents fields scratch/toy/control --field precipitation --top 3
$ exege nn sae scratch/toy/control --features 64 --k 4 --out scratch/toy/sae.npz   # needs exege[nn]
$ exege app --latents scratch/toy                                                  # needs exege[app]
```

The CLI is a thin client of the Python API; anything it can do, a notebook can:

```python
from exege.latents import Region, analyze_region, open_source

source = open_source("scratch/toy/control", mask_variable="sst")
result = analyze_region(
    source, time=2, layer=3, region=Region(lat=10, lon=-114, radius_km=1500), centered=True
)
result.ranking.channels  # the channels that respond most strongly there
```

## More

- Documentation: <https://mahf708.github.io/exege/>
- Source and issues: <https://github.com/mahf708/exege>

BSD-3-Clause. `exege.latents`, `exege.figures` and `exege.app` grew out of
the [latent space visualiser for weather models](https://github.com/ktempestuous/latent_space_visualiser_weather_models)
(Tempest, Beylich & Craig 2026, arXiv:2604.20467, doi:10.1007/978-3-032-29915-4_10); see
`NOTICE`. The sparse autoencoders follow MacMillan & Ouellette (2025, arXiv:2512.24440);
the B-spline autoencoder of Cheon (2026, arXiv:2605.17493) is what `exege.nn` is
heading for and does not implement yet.
