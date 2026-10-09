# Reading an archive

`exege.latents` asks what a model's internal channels respond to, with the science of the
[latent space visualiser](background.md) lifted out of its app so that a notebook, a
batch job and the CLI all run the same code. It reads a [latent archive](archive-format.md): a directory of activations an exporter
wrote from inside the model, one per model component. It needs the `latents` extra.

## What is in an archive

```console
$ exege latents info latents/atmosphere
source      latents/atmosphere
model       SamudrACE-E3SMv3
component   atmosphere
checkpoint  SamudrACE-E3SMv3.tar
calendar    noleap
timestep_s  21600
grid        180x360, 64800 nodes, 64800 valid
times       17: 0425-01-03T18:00:00 .. 0425-01-09T00:00:00

layers
LAYER  CHANNELS  LABEL
0      384       encoder output (input to block 1)
1      384       Fourier block 1 output
...
8      384       Fourier block 8 output
```

## Masks

A grid's mask travels in `grid.npz` when the archive carries one. For an archive that
does not, name a variable of its reference file that is missing exactly where nodes mean
nothing — `sst` for the ocean:

```console
$ exege latents info latents/ocean --mask-variable sst
...
grid        180x360, 64800 nodes, 44892 valid
```

That is 30.7% of points over land, excluded from everything that follows.

### Non-finite values and empty selections

Both are refused everywhere with a one-line error that says what and where, never with a
NaN that turns up three steps later.

- **A valid node whose activations are NaN or infinite.** What a masked node holds is
  never looked at, but a node the mask says is valid has to hold numbers. Every analysis
  and the training loop read latents through `read_latents`, which raises
  `layer 8 at 0425-01-03T18:00:00 holds 12 valid node(s) with activations that are not finite
  (channel 41, 77) in latents/atmosphere; a node the source cannot supply belongs in its
  mask, not left NaN`. The remedy is the mask (`--mask-variable`, or `mask` in `grid.npz`).
  Fields are different: a reference field is NaN where it is missing, and stays so.
- **An empty selection.** No times, a region or box with no valid node in it, or a mask
  (or the runs compared) that leaves no node: each is a `RequestError` naming what was
  empty (`no times selected`, `no valid nodes within 1 km of (0, 0)`), not an empty result.

## From Python

```python
from exege.latents import iter_batches, open_source

source = open_source("latents/atmosphere")
source.info().layers  # what was recorded, without loading any of it
nodes = source.grid().within(5, -140, 1500)
source.load(0, 8, nodes=nodes)  # one region of one layer, and nothing else

for batch in iter_batches(source, layer=8, batch_size=4096):  # to train on
    ...  # float32 (4096, 384): valid nodes only, drawn in proportion to area
```

`source.load(time, layer, channels=..., nodes=...)` reads only what it is asked for.

```{admonition} Weight by area
:class: warning

A 1° grid has as many nodes in its last row as on the equator, covering 1/115 of the
area, and a third of an ocean model's nodes are land. `iter_batches` draws nodes by
area and never where the grid is invalid, so a plain mean over a batch is already the
area-weighted loss. Anything trained on `source.load(...)` directly should do the same.
```

What a request cannot have — a layer that is not there, an empty region, a basis for a
different width — is a `RequestError` (an `ExegeError` and a `ValueError`), raised before
anything is read; any other exception is a bug and keeps its traceback.

## From a Hugging Face repository

An archive kept in a Hugging Face dataset repository opens in place, with the `hf`
extra:

```console
$ uv pip install 'exege-core[hf]'
$ exege latents info hf://datasets/<owner>/<repo>/<folder>
```

Nothing is downloaded until something needs it, and then one file at a time, into the
Hugging Face cache: opening an archive fetches its manifest, a map its grid, a layer its
one `step_XX.npy`. A notebook that looks at one layer of a nine-layer archive downloads
one layer. Every file comes from the one commit the repository was at when the archive was
opened: a revision you name (`open_source(url, revision="v1")`, `--revision v1`) is
resolved to its commit then, and so is the default, and
[both are recorded](provenance.md). `source.file("bases/sae_L08.npz")`
fetches any other file kept in the folder, and says None when there is none. A private or
gated repository reads the token `huggingface_hub` finds (`HF_TOKEN`, or `hf auth login`).
