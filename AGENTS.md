# exege

The `exege` Python package (`src/exege/`), its tests, and its documentation site
(`docs/`, Sphinx, built on Read the Docs).

## Rules

- Work on `user/topic` branches; merge to `main` via PR. Short lowercase imperative
  commit subjects.
- `uv run --group docs sphinx-build -M html docs docs/_build -W` must pass. New pages
  must be added to a `toctree` in `docs/index.md` by hand.
- `ruff check`, `ruff format --check` and `pytest` must pass. All run in
  `.github/workflows/ci.yml`, on three tiers: a base install (Click only), a full
  one, and one with torch. Tests that need numpy skip on the first, and those that need
  torch on the first two.
- `uv` is the tool of record: `uv sync` once, then `uv run …`. A checkout gets every
  extra but torch, plus pytest and ruff, by default (the `dev` and `full` dependency
  groups); `uv sync --extra nn` adds torch. An installed `exege-core` stays on the base
  tier; `exege` (`packages/exege`, no code) is the full install.
  ACE itself pins Python 3.11.
- Ship in ~1000-line increments. Each increment leaves the repo working and useful.
- `scratch/` is ignored by git: write throwaway output there
  (`exege latents toy scratch/toy/control`), never beside the code.

## Where to look

| Path | AGENTS.md covers |
|---|---|
| `src/exege/` | package architecture, who may import whom, how to extend it |
| `src/exege/core/` | the purity contract |
| `src/exege/adapters/` | the factory contract; writing a new adapter |
| `src/exege/{latents,nn,figures,app}/` | each subpackage's scope, rules and non-goals |
| `tests/` | fixture rules |
| `docs/` | prose and nav conventions |
