# Provenance

A result that cannot be traced to its inputs cannot be rerun, and a branch name or a file
path is not an input: both move. Everything a result was computed from is therefore
pinned by content, and written into the `provenance` of its `--json` and `summary()`.

## Archives are pinned by commit

Opening `hf://…` at the default branch, a branch or a tag resolves it to a commit SHA
through the hub; a full 40-character commit is taken as given, so a pinned rerun works
offline from a warm cache. What was asked and what it became are both kept
(`LatentInfo.revision`), and the files are fetched by the commit, so a tag moved halfway
through a session changes nothing already open:

```console
$ exege latents region hf://datasets/<owner>/<repo>/control --revision v1 --json …
"revision": {"requested": "v1", "commit": "5b0e…"}
```

A comparison (`diff` and `diff --growth`) says it for each side, under `control`,
`experiment` and `noise`. `--revision` applies to every `hf://` source of a command and is
passed over for a local one; a command with no `hf://` source refuses it. From Python, give
each `open_source` its own. A local archive has no revision and its provenance says none.

## Bases are pinned by hash

`save_basis` writes a `sha256` into the file's record: over the kind, every array (name,
dtype, shape and bytes), the scalars, and what the basis says it was fitted on — which
includes the commit of the archive it was fitted from. `load_basis` recomputes it, and a
mismatch is a `RequestError`. A result that used a basis carries
`{"path", "sha256", "status"}` under `provenance.basis` (and `provenance.bases`, by layer,
for a storyline):

```console
$ exege latents region scratch/prov/control --time 0 --lat 7.5 --lon 45 --radius-km 2500 \
    --features 2 --basis scratch/prov/pca3.npz --json
  "basis": {
    "path": "scratch/prov/pca3.npz",
    "sha256": "0843ad5199ac1109f52de93e3f362b28ba53dfb2a653789a8cbc7d526be008eb",
    "status": "verified"
  }
```

`status` is `verified` for a file that carried its hash and matched it, `computed` for a
basis fitted in the session, and `unhashed` for a file written before hashes existed.
Those still load and can be pinned, but are not vouched for: they need
`--allow-unverified-basis` (`allow_unverified_basis=True`), like a basis of unverified
identity.

## Reproducing a result

The [app](app.md)'s *Reproduce* tab and the commands above give `--revision <commit>`
(never the tag) and `--basis-sha256 <hash>`; the Python gives
`open_source(path, revision=<commit>)` and `load_basis(path, sha256=<hash>)`. Either refuses
a basis whose content is not the one named, and `--basis-sha256` without `--basis` is
refused.

```{admonition} Known limitation
:class: note

The per-layer `--bases` of `storyline` are hashed into the result but cannot be pinned by
a flag; pin those from Python with `load_basis(path, sha256=...)`.
```
