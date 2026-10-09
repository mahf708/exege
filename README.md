# exege

Tools for understanding and evaluating scientific machine-learning models. `exege` takes
its name from the Greek stem *exēgē-*, associated with explanation and interpretation.

So far: diagnostics of an emulator's latent space (`latents`), sparse autoencoders trained
on it (`nn`), figures (`figures`), and a local web app over them (`app`). Documentation is
at <https://exege.readthedocs.io>.

## Install

```console
$ uv sync
$ uv run exege --help
```

In a checkout, `uv sync` (or the first `uv run`) installs every extra but torch, plus
pytest and ruff; `uv sync --extra nn` adds torch. To use it from another project,
install it from PyPI: [exege](https://pypi.org/project/exege/) is everything but torch,
[exege-core](https://pypi.org/project/exege-core/) the same code with only the extras you
ask for.

```console
$ uv pip install exege                    # everything but torch; 'exege[nn]' adds it
$ uv pip install 'exege-core[latents]'    # Click, and what exege.latents needs
$ uv pip install 'exege-core[latents] @ git+https://github.com/mahf708/exege'   # what main holds and no release does yet
```

`exege-core` alone pulls only Click. Anything heavier sits behind an extra named after
the subpackage that needs it (`latents`, `figures`, `nn`, `app`, `hf`), so the core stays
nimble. `exege` lives in `packages/exege`: no code, only that dependency.

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
$ uv run --group docs sphinx-build -M html docs docs/_build -W   # docs/_build/html/index.html
```

See `AGENTS.md` for how the pieces fit together, and the `AGENTS.md` in each
subdirectory for that directory's rules.

`exege` was started as `xaig` in [E3SM-Project/aigroup](https://github.com/E3SM-Project/aigroup),
which keeps the E3SM AI guides; its history came with it.
