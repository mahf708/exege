# The latent archive format

A directory per model component. Any exporter that writes this layout can be read by the
`latent-archive` adapter; the SamudrACE one is the visualiser's
`scripts/extract_samudrace_latents.py`.

| File | Content |
| --- | --- |
| `manifest.json` | times, layers and provenance (below) |
| `grid.npz` | `lat`, `lon` per node, flat. Optional: `grid_shape` `(n_lat, n_lon)` for a structured grid in C order (absent for a mesh), `mask` (true where a node means something), `area` (per-node area, for meshes with uneven cells) |
| `step_XX.npy` | `(n_times, n_nodes, n_channels)`, any float dtype (float16 halves the disk), one file per layer, read memory-mapped |
| `reference.nc` | optional: physical fields on the same grid |
| `bases/` | optional: basis files fitted on this archive (`exege latents pca`, `exege nn sae`), under any names. `source.files("bases")` lists them and `source.file(name)` reads one; the [app](app.md) offers every one that fits the layer shown |

## The manifest

```json
{
  "model": "SamudrACE-E3SMv3",
  "component": "atmosphere",
  "checkpoint": "SamudrACE-E3SMv3.tar",
  "calendar": "noleap",
  "timestep_seconds": 21600,
  "n_nodes": 64800,
  "latent_times": ["0425-01-03T18:00:00", "..."],
  "steps": [
    {"index": 0, "label": "encoder output", "file": "step_00.npy", "n_channels": 384}
  ],
  "extra_steps": [],
  "reference_file": "reference.nc",
  "reference_times": ["0425-01-03T12:00:00", "..."],
  "experiment": {"seed": 0, "steer": {"layer": 4, "basis": "sae4.npz", "feature": 12, "by": 3.0}}
}
```

`n_nodes`, `latent_times` and `steps` (each with `index`, `file`, `n_channels`) are
required; the rest is provenance, carried into every result.

- **`network_layer`.** A step may say where it sits in the network, `"network_layer": 8`,
  for an archive that keeps some layers and not others (a 30-day run of layers 0, 2, 4, 6
  and 8 stores layer 8 at index 4): a basis is matched to a layer by that place, so one
  fitted on layer 8 of a run that kept them all fits index 4 here, and one fitted on
  layer 4 does not. Without it, the index is the place.
- **`experiment`** is free-form: whatever distinguishes this run from a plain one.
- **`reference_times`** labels the reference file's time axis, which usually holds the
  state each forward call started from as well; without it the file is taken to share the
  latents' times.
- **Times are labels**, kept as text: emulators run on calendars (no-leap, year 425) that
  the usual datetime types cannot hold.
- **`grid_shape`** is `(n_lat, n_lon)` in C order — latitude constant along a row — which
  the reader checks, because nodes stored the other way round reshape without complaint
  and weight wrongly.
- **`extra_steps`** are layers recorded on a coarser grid than `grid.npz` describes — the
  inner levels of a U-Net — and are listed but not loadable.

A file whose shape contradicts the manifest is refused rather than misread.

To read a different layout, [write an adapter](adapters.md).
