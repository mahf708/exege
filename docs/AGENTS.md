# docs

Sphinx, with the Furo theme and pages in Markdown (MyST), built on Read the Docs at
<https://exege.readthedocs.io> from `.readthedocs.yaml`: HTML, plus a PDF, an EPUB and a
zipped HTML to download. Read the Docs builds with warnings as errors, and so must you —
broken links and docstrings that are not valid reST fail the build:

```console
$ uv run --group docs sphinx-build -M html docs docs/_build -W
```

Versions: `latest` is `main`, `stable` the newest release, and each minor release has one
version, built from a branch named after it (`0.7`) that the release workflow moves to
each `vX.Y.Z` tag. Read the Docs activates those branches by an automation rule.

`-M epub`, `-M singlehtml` and `-M latexpdf` (needs a TeX install with xelatex) build the
other formats locally.

## Layout

The sections are the `toctree`s at the bottom of `index.md`, laid out after nanobind's:

| Section | Holds |
| --- | --- |
| (none) | installing, a quickstart on the toy emulator, background and citations, the roadmap |
| Latent diagnostics | one page per task: reading an archive, a region, bases, through time, comparing runs, finding features, provenance |
| Sparse autoencoders, Experiments, Tools | `nn`, evaluation, steering, records, the app |
| Extending exege | design, writing an adapter |
| Reference | the CLI (sphinx-click), the archive format, and the API (autodoc, under `api/`) |

A page does one task. Planned work goes in `roadmap.md`, not at the foot of a page, and
installation in `installing.md`.

## Conventions

- Shell blocks are fenced as `console`, with `$` prompts; the copy button drops them.
- Python examples go under a `## From Python` heading at the end of a page.
- Headings and admonition titles are plain sentence case, without code in a page title.
  An admonition is a MyST fence: ```` ```{admonition} The uv cache on NERSC ```` then
  `:class: tip` on the next line.
- Cross-links are relative (`regions.md`, `time.md#storylines`); headings down to
  `####` have anchors.
- Quote real measured numbers and real NERSC paths rather than genericizing them.

## The API reference

`api/*.rst` list what each subpackage exports, by hand, with `autofunction` and `autoclass`. They are reST, not MyST: autodoc writes reST, which a Markdown page would print as text.
A name added to `exege.latents._LAZY` (or any public export) is added there too. torch is
mocked (`autodoc_mock_imports` in `conf.py`); everything else the reference imports is
installed with `.[figures]`.

## Adding a page

Add the file, then add it to one of the `toctree`s at the bottom of `index.md` by hand —
a page in none of them fails the build. Pages are written by hand, and quote real output:
rerun the commands they show when the code behind them changes.
