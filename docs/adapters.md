# Writing an adapter

An adapter is one class implementing one or more protocols, constructed as
`Adapter(source, **options)`. For latents the protocol is `LatentSource`: three methods —
`info()`, `grid()` and `load(time, layer, channels, nodes)` — plus, optionally,
`ReferenceFields` (`field_names()`, `field(name, time)`) for the physical fields kept
beside them. A model that can be run forward with hooks implements
[`Intervenable`](steering.md#the-adapter-contract) as well.

## Registering one

Register it, from this repository or any other package:

```toml
[project.entry-points."exege.adapters"]
myframework = "mypkg.adapter:MyAdapter"
```

```console
$ uv sync   # entry points are read from installed metadata
$ exege latents info /path/to/export --adapter myframework
```

```python
from exege.latents import open_source

source = open_source("/path/to/export", adapter="myframework")
```

Meshes need no special handling: without a `grid_shape` everything works except
`to_map`, and weights are uniform unless the adapter supplies `area`.

## Step by step

1. **The module** goes in `exege/adapters/` (or any package): a class whose first positional
   parameter is the source and whose options are keyword arguments. An option it does not
   declare is refused by name for it, by the registry. A source that is broken is an
   `AdapterError`; a request it cannot meet is a `RequestError`, never an `IndexError`.
2. **A writer**, `write(path, **contents)` with the signature of `write_archive`, is
   optional and worth having: the toy emulator writes through it
   (`exege latents toy OUT --adapter myframework`), so the reader is tested against it.
3. **The entry point**, as above, and `uv sync`.
4. **The contract tests.** Add one line to `LATENT_CASES` in
   `tests/test_adapter_contracts.py` (and to `INTERVENABLE_CASES` if the adapter can be run
   forward with hooks), and every test there is asked of it; `src/exege/adapters/AGENTS.md`
   lists what they ask. The suite fails until a registered adapter is under contract.

## A second layout, as an example

The repository ships two layouts, and the second exists to show that this is all there is.
`latent-archive` keeps `(n_times, n_nodes, n_channels)` in one file per layer; `bundle-dir`
keeps `(n_channels, n_lat, n_lon)` in one file per level *and* time, with its own words in
its JSON manifest (`system`, `clock`, `levels`). Nothing that reads latents was edited to
add it:

```console
$ exege latents toy scratch/bundle --adapter bundle-dir
$ exege latents info scratch/bundle --adapter bundle-dir
source                 scratch/bundle
model                  exege-toy
component              atmosphere
checkpoint             seed-0
calendar               noleap
timestep_s             21600
grid                   24x48, 1152 nodes, 1152 valid
times                  8: 0424-02-27T06:00:00 .. 0424-03-02T00:00:00
…
```

A bundle keeps its mask in `coords.npz` (`ocean`, true where the model says something). The
toy's land is known only to its `sst` field, so a toy bundle has no mask, and says
`1152 valid`; `--mask-variable` is an archive's option and a bundle refuses it, naming the
one it has (`unmasked`).
