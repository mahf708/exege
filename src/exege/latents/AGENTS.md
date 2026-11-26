# latents — what an emulator holds inside

Diagnostics of a model's latent space: what its internal channels respond to, on a grid.

| Module | Holds |
|---|---|
| `grid.py` | nodes on a sphere: masks, area weights, regions, maps |
| `source.py` | the contract: `LatentSource`, the optional `ReferenceFields`, `LatentInfo` |
| `toy.py` | a toy emulator in numpy, so an archive can be made with no model and no data |
| `basis.py` | `Decomposition`: `PCA`, a sparse `Dictionary` (also one trained elsewhere and token-normalized), and the basis file |
| `analysis.py` | one region at one time: ranking, similarity, a decomposition |
| `samples.py` | many times at once: moments, a global PCA, batches to train on |
| `through.py` | through time and between runs: series, differences, field correlation |
| `evaluate.py` | a frozen basis on held-out times: time-block splits, fidelity and sparsity, dead and redundant features, stability across seeds, PCA against a dictionary |
| `steering.py` | steering: the `Intervenable` contract, six arms (control, reconstruction, feature, random direction, comparison, matched) under paired noise, reference modes, open and closed loops, outcomes in a region, many cases, dose curves with an amount chosen leave-one-out |
| `references.py` | what a steer is measured against, read from an archive: a feature's reference level, a difference-of-means direction |
| `record.py` | the experiment record: a versioned, compact JSON file of one evaluation or steering run, its loader, and the writers the commands call |
| `features.py` | a feature without a field in mind: a census of a layer, one feature's profile |
| `calibrate.py` | which field each feature goes with, against a longitude-rotation null: z-scores, Benjamini-Hochberg, labels with a runner-up bar, and their calibration on null input |
| `stationary.py` | what is fixed in place: each column's variance split into its per-node time mean and the rest, the fixed part fitted on latitude rows and static fields |
| `gap.py` | a free run against a teacher-forced one of the same network: each column's mean, coverage and strength in both, time by time |
| `cli.py` | `exege latents …`, a thin client of the above |
| `__init__.py` | the public names, re-exported lazily (see the first rule) |

## Rules that must not be lost

- **`__init__.py` imports nothing heavy.** `exege --help` imports `cli.py`, and with it
  this package, on a base install. The public names (`from exege.latents import Region`)
  are resolved on first use through `_LAZY`; a new public name goes there, in `__all__`
  and under `TYPE_CHECKING`.
- **Area-weight everything.** Unweighted means on a lat-lon grid are simply wrong. On
  real SamudrACE latents, unweighted centering moved one channel's global mean by 40% of
  its standard deviation and changed the top-5 ranking.
- **Be NaN- and mask-aware.** Ocean channels are undefined over land (30.7% of points in
  the E3SMv3 configuration); a plain `mean` silently returns NaN or a biased number.
  Invalid nodes are left out of regions and read NaN in every map.
- **No user interface and no foreign file formats here.** Functions take arrays or a
  source and return structured results carrying their settings and provenance. Reading
  what a framework wrote is an adapter's job (`LatentSource`); drawing is a client's.
  Third-party imports are capped at numpy and click by `tests/test_purity.py`. The two
  files this package itself reads and writes are its own: the basis `.npz` (`basis.py`),
  plain arrays and a JSON record, because a basis is fitted once — often by `nn`,
  with torch — and used many times, including from the model's environment, with numpy
  alone; and the experiment record (`record.py`), plain JSON that a client opens without
  rerunning anything, versioned and validated on the way in. Nothing else is read or
  written as a file format here.
- **An index is not an identity.** Channel 42 of one trained network is not channel 42
  of another, and every layer of a model is as wide as the next. Whatever lines two
  things up by index checks who they are first, never only that widths match:
  `check_comparable` for two runs, `check_basis_fits` for a basis against a layer. Their
  docstrings say what is checked and what `allow_unverified` and `across_models` lift.
- **Provenance pins what moves.** A result's provenance names the commit a versioned source
  was opened at (`LatentInfo.revision`: a branch, a tag or the default resolved, a full
  commit taken as given) and, through `result_provenance`, the content hash of every basis used. A new
  result builds its provenance with `result_provenance(info, basis=...)`, never by hand. A
  basis file's `sha256` is checked on load and is over its arrays, scalars and
  `fitted_on`; a file without one is `unhashed` and needs `allow_unverified_basis`.
  Whatever a client prints to reproduce a result pins both (`--revision`,
  `--basis-sha256`).
- **A held-out number is of a frozen basis on times it was not fitted on.** Splits are
  contiguous time blocks with a buffer, never nodes or adjacent steps, and are written into
  the result as labels. Normalization, PCA and dictionary are fitted on `split.train` only;
  `evaluate_basis` refuses a basis whose `fitted_on.times` includes a held-out time, and
  marks one that says nothing `unknown`. A result is built with `result_provenance`; this
  package does not train (seeds and sweeps are `nn`'s, handed in as `Dictionary`s).
- **A steer is compared with its control, under one noise, and against chance.**
  Every arm of a seed shares the control's initial state and `noise_seed`. In an open loop
  (the default) the edit's size is fixed from the control's latents; in a closed loop the
  feature arm works it out from its own, and the random and comparison arms are given
  exactly what it did, step by step. Either way a paired difference holds the edit and
  nothing else (exactly zero when the edit is), and every arm against which the feature is
  ranked was pushed by the same amount. The feature's response is reported against random
  directions of the same length and size, never alone, beside what merely passing through
  the dictionary does (the reconstruction arm), and, when asked, against matched features
  and named comparison directions; a tie never ranks as a win. Anything per node that a
  steer reads from elsewhere (a reference) is checked to lie on the system's nodes
  (`node_digest`), and anything a reproduction reads is pinned by content. Edits and fields are held to
  the same finiteness rule as `read_latents`. Nothing here imports a model: a system is an
  `Intervenable`, reached through the registry, and tested on `adapters/toy_dynamics.py`,
  whose answers are planted. See `docs/steering.md`.
- **A record is a contract, so it is versioned and validated.** `record.py` holds the one
  file other things read (the app today): provenance with the commit and the bases by
  hash, settings, the split as time labels, the numbers, and the command that reproduces
  it. `load_record` refuses an unknown version, format or kind with a `RequestError` and
  checks what a view relies on; a writer may add keys within a version, and removing or
  renaming one is `VERSION + 1`. It keeps results and not runs (every pairing stays with
  `SteeringResult.to_dict`). The command is built by the `cli` that ran it, from what that
  command was asked, and tested by running it again (`tests/test_latent_record.py`). It
  lives here and not in core because every writer and reader sits on `latents`.
- **A feature's size is what it contributes.** Activation times the length of its
  direction: a dictionary may trade one for the other, so rank and compare by the
  product. `nn` keeps directions at unit length so that the two agree.
- **A method is a value.** Anything that turns channels into features is a
  `Decomposition`; routines take one as `basis=` rather than growing an argument per
  method. A basis is handed *raw* latents: its standardization is its own and travels
  with it, so centering an analysis must not center its input twice.
- **Area and mask hold for training data too.** `iter_batches` draws nodes by area and
  never where the grid is invalid, so a plain mean over a batch is the area-weighted
  loss. Do not train on `source.load()` directly.
- **Read latents through `read_latents`.** It is `source.load` plus one rule: a valid node
  holding NaN or infinity is a `RequestError` naming layer, time and channel, never a NaN
  carried into a mean. Masked nodes may hold anything.
- **Refuse before reading.** Whatever can be wrong with a request is checked before the
  first 100 MB is loaded, and raised as `RequestError`.
- **Positions are not lead times.** An exporter keeps the forward calls it was asked to.
  Anything plotted through time uses `LatentInfo.elapsed_seconds()`, and falls back to
  positions knowingly when that is None.
- **Mind the memory.** One layer of a 1-degree, 384-channel model is 100 MB; nine layers
  at one time is 0.9 GB. Ask a source for the nodes and channels you need, keep a layer
  in its own precision, and never make a second copy of one.
  `exege latents region` on the real atmosphere archive peaks near 275 MB
  resident (`/usr/bin/time -l`), a quarter of it the archive's own mapped pages; the
  first draft took 650.
- **Deterministic results.** No dependence on node order; PCA signs are fixed; batches
  are a function of their seed.
- Plotting requires the `figures` extra. Do not import matplotlib at package import time.
- May not import `nn` (`nn` imports this, never the reverse).
