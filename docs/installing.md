# Installation

exege is published on PyPI twice, with the same code:
[exege](https://pypi.org/project/exege/) is the full install, every extra below but `nn`,
and [exege-core](https://pypi.org/project/exege-core/) is the light one, with the extras
you ask for.

```console
$ uv pip install exege                    # everything but torch
$ uv pip install 'exege[nn]'              # and torch
$ uv pip install 'exege-core[latents]'    # Click, and what exege.latents needs
$ uv pip install 'exege-core[latents] @ git+https://github.com/mahf708/exege'   # what main holds and no release does yet
```

On conda-forge, `exege` is everything, torch included.

## Extras

`exege-core` alone pulls only Click. Anything heavier sits behind an extra named after the
subpackage that needs it:

| Extra | Pulls | Gets you |
| --- | --- | --- |
| `latents` | numpy, xarray, netCDF4 | `exege.latents` |
| `figures` | matplotlib, cartopy | `exege.figures`: maps and figures (brings `latents`) |
| `app` | streamlit | [`exege app`](app.md) (brings `figures`) |
| `nn` | torch | [`exege.nn`](nn.md) and `exege nn` |
| `hf` | huggingface_hub | [archives read from a Hugging Face repository](archives.md#from-a-hugging-face-repository) (brings `latents`) |

Torch is large, and whether it should be a CPU or a CUDA build is the machine's business,
so no install brings it unless asked. A missing extra says so, with the command that fits
how this exege was installed:

```console
$ exege latents info latents/atmosphere
Error: numpy is not installed; it comes with the 'latents' extra: uv pip install -e '/path/to/exege[latents]'  (in that checkout: `uv sync --extra latents`)
```

## From a checkout

```console
$ git clone https://github.com/mahf708/exege
$ cd exege
$ uv sync
$ uv run exege --help
```

In a checkout, `uv sync` (or the first `uv run`) installs every extra but `nn`, plus
pytest and ruff. Ask for torch with `uv sync --extra nn`.

````{admonition} The uv cache on NERSC
:class: tip

Keep the cache off `$HOME`:

```console
$ export UV_CACHE_DIR="$PSCRATCH/.cache/uv"
```
````

## Two environments

Recording activations needs the model's own environment — torch, the framework, a
checkpoint, usually a pinned Python. Studying them needs numpy. The
[latent archive](archive-format.md) is the hand-off between the two, so neither side
installs the other's stack: install exege where you study the latents, not where the
model runs.
