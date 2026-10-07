# aigroup

Docs, scripts, examples, and prototypes for E3SM AI efforts.

This repo hosts two peers:

- **`docs/`** — the guide site, published at <https://e3sm-project.github.io/aigroup>
- **`src/exege/`** — `exege`, a light Python package for working with AI campaigns: so far,
  diagnostics of an emulator's latent space (`latents`), sparse autoencoders trained on
  it (`nn`), figures (`figures`), and a local web app over them (`app`)

## Install

```console
$ uv sync
$ uv run exege --help
```

In a checkout, `uv sync` (or the first `uv run`) installs every extra but torch, plus
pytest and ruff; `uv sync --extra nn` adds torch. To use it from another project,
install it from [PyPI](https://pypi.org/project/exege/), asking for the extras you need:

```console
$ uv pip install 'exege[latents]'
$ uv pip install 'exege[latents] @ git+https://github.com/E3SM-Project/aigroup'   # what main holds and no release does yet
```

The base install pulls only Click. Anything heavier sits behind an extra named after the
subpackage that needs it (`latents`, `figures`, `nn`, `app`), so the core
stays nimble.

## Use

```console
$ exege latents toy scratch/toy/control        # no model to hand? make an archive with numpy
$ exege latents info scratch/toy/control --mask-variable sst
$ exege latents info /path/to/latents/atmosphere
$ exege latents region /path/to/latents/atmosphere --lat 5 --lon -140 --centered --pcs 3
$ exege nn sae /path/to/latents/atmosphere --out sae.npz      # needs `uv sync --extra nn`
$ exege latents region /path/to/latents/atmosphere --lat 5 --lon -140 --features 3 --basis sae.npz
$ exege app --latents /path/to/latents/   # the same, in a local web app
```

## Develop

```console
$ uv run ruff check && uv run ruff format --check
$ uv run pytest
$ uv run --group docs mkdocs build --strict   # MKDOCS_SOCIAL=false without cairo
```

See `AGENTS.md` for how the pieces fit together, and the `AGENTS.md` in each
subdirectory for that directory's rules.
