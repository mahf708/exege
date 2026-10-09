# docs

Sphinx, with the Furo theme and pages in Markdown (MyST), built on Read the Docs at
<https://exege.readthedocs.io> from `.readthedocs.yaml`: HTML, plus a PDF, an EPUB and a
zipped HTML to download. Read the Docs builds with warnings as errors, and so must you —
broken links fail the build:

```console
$ uv run --group docs sphinx-build -M html docs docs/_build -W
```

`-M epub`, `-M singlehtml` and `-M latexpdf` (needs a TeX install with xelatex) build the
other formats locally.

## Conventions

- Shell blocks are fenced as `console`, with `$` prompts; the copy button drops them.
- Admonitions are MyST fences with lowercase titles and a class:
  ```` ```{admonition} uv cache ```` then `:class: tip` on the next line.
- Cross-links are relative (`latents.md`, `latents.md#python-api`); headings down to
  `####` have anchors.
- Quote real measured numbers and real NERSC paths rather than genericizing them.

## Adding a page

Add the file, then add it to one of the `toctree`s at the bottom of `index.md` by hand —
a page in none of them fails the build. Pages are written by hand, and quote real output:
rerun the commands they show when the code behind them changes.
