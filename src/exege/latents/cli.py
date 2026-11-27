"""``exege latents``, the command half: a thin client of the ``exege.latents`` APIs.

This module must import on a base install (``exege --help`` lists every command),
so numpy-backed modules are imported inside the commands that use them. What a
command cannot do it learns as a ``RequestError``, which the top-level command
prints as one line.
"""

from __future__ import annotations

import functools
import json
import shlex
from pathlib import Path

import click

from exege import _render

_adapter_option = click.option(
    "--adapter", default="latent-archive", show_default=True, help="How SOURCE is read."
)


def _remember(key: str):
    """An option no command takes as a parameter: it is kept on the click context, where
    ``open_for_cli`` and ``_basis`` find it. Pinning a source or a basis is a way of
    reading one, so it need not widen the signature of every command."""

    def callback(ctx, param, value):
        if value is not None:
            ctx.meta[f"exege.{key}"] = value
        return value

    return callback


def _remembered(key: str):
    return click.get_current_context().meta.get(f"exege.{key}")


_revision_click_option = click.option(
    "--revision",
    expose_value=False,
    callback=_remember("revision"),
    help="For hf:// sources: the branch, tag or commit to read. Resolved to a commit when "
    "opened (a full 40-character commit is taken as it is), and recorded; applies to every "
    "hf:// source of the command.",
)


def revision_option(command):
    """``--revision``, refused by a command none of whose sources is on the hub: it would
    be silently ignored, and a reproduction that believed it was pinned would not be."""

    @functools.wraps(command)
    def checked(*args, **kwargs):
        if _remembered("revision") and not any(
            isinstance(value, str) and value.startswith("hf://") for value in kwargs.values()
        ):
            from exege.core.errors import RequestError

            raise RequestError(
                "--revision applies to hf:// sources, and none of this command's sources is one"
            )
        return command(*args, **kwargs)

    return _revision_click_option(checked)


_mask_click_option = click.option(
    "--mask-variable",
    help="Reference-file variable that is missing where nodes mean nothing (e.g. sst).",
)
_basis_click_option = click.option(
    "--basis",
    "basis_path",
    type=click.Path(dir_okay=False),
    help="A basis file (from `latents pca` or `exege nn sae`) whose features to use.",
)
_basis_sha_option = click.option(
    "--basis-sha256",
    expose_value=False,
    callback=_remember("basis_sha256"),
    metavar="HASH",
    help="Refuse the --basis file unless its content has this sha256 (reproduction commands "
    "give it).",
)


_record_option = click.option(
    "--record", "record_path", type=click.Path(dir_okay=False),
    help="Also write a compact experiment record (JSON): provenance, settings, results and "
    "the command that reproduces it, which `exege app` opens.",
)  # fmt: skip


def _mask_option(command):
    """``--mask-variable``, and ``--revision`` beside it: every command that opens a source."""
    return revision_option(_mask_click_option(command))


def _basis_option(command):
    return _basis_sha_option(_basis_click_option(command))


_unverified_option = click.option(
    "--allow-unverified-basis",
    is_flag=True,
    help="Allow a basis with incomplete model/layer identity; known mismatches still fail.",
)
_lead_option = click.option(
    "--lead",
    type=int,
    default=0,
    show_default=True,
    help="Set the latents against the field this many times later: 1 for what that forward "
    "pass produced (an output), 0 for what it read (an input).",
)
_json_option = click.option(
    "--json", "as_json", is_flag=True, help="Emit settings, provenance and results."
)


def open_for_cli(source: str, adapter: str, mask_variable: str | None):
    """Open SOURCE as the command line asks, ``--revision`` included."""
    from exege.latents import open_source

    options = {"mask_variable": mask_variable} if mask_variable else {}
    revision = _remembered("revision")
    if revision and source.startswith("hf://"):  # a local source has no revision to pin
        options["revision"] = revision
    return open_source(source, adapter=adapter, **options)


def _basis(path: str | None):
    sha256 = _remembered("basis_sha256")
    if path is None:
        if sha256 is not None:
            from exege.core.errors import RequestError

            raise RequestError("--basis-sha256 pins the --basis file, and no --basis is given")
        return None
    from exege.latents import load_basis

    return load_basis(path, sha256=sha256)


def _time(text: str) -> str | int:
    from exege.latents import parse_time

    return parse_time(text)


def _region_options(command):
    for option in (
        click.option("--radius-km", type=float, default=1000.0, show_default=True),
        click.option("--lon", type=float, required=True),
        click.option("--lat", type=float, required=True),
    ):
        command = option(command)
    return command


@click.group(name="latents")
def latents() -> None:
    """Explore activations recorded from inside a model."""


@latents.command("info")
@click.argument("source", type=click.Path())
@_adapter_option
@_mask_option
def info_cmd(source, adapter, mask_variable) -> None:
    """Describe what SOURCE holds, without loading it."""
    from exege.latents import ReferenceFields

    opened = open_for_cli(source, adapter, mask_variable)
    info, grid = opened.info(), opened.grid()
    shape = "x".join(str(n) for n in grid.shape) if grid.shape else "mesh"
    provenance = info.provenance()
    experiment = provenance.pop("experiment", {})
    provenance.pop("options", None)
    revision = provenance.pop("revision", {})
    if revision:
        provenance["requested_revision"] = revision["requested"] or "(default)"
        provenance["commit"] = revision["commit"]
    click.echo(
        _render.pairs(
            {
                **provenance,
                "calendar": info.calendar,
                "timestep_s": info.timestep_seconds,
                "grid": f"{shape}, {grid.n_nodes} nodes, {int(grid.valid.sum())} valid",
                "times": f"{len(info.times)}: {info.times[0]} .. {info.times[-1]}",
                **{f"experiment.{key}": value for key, value in experiment.items()},
            }
        )
    )
    click.echo("\nlayers")
    rows = [{"layer": x.index, "channels": x.n_channels, "label": x.label} for x in info.layers]
    click.echo(_render.table(rows))
    if info.off_grid_layers:
        click.echo(f"\n{len(info.off_grid_layers)} more layer(s) on coarser grids, not loadable")
    if isinstance(opened, ReferenceFields) and opened.field_names():
        click.echo(f"\n{len(opened.field_names())} reference field(s); see `latents fields`")


@latents.command("toy")
@click.argument("out", type=click.Path(file_okay=False))
@click.option("--steps", type=int, default=12, show_default=True, help="Steps to roll forward.")
@click.option("--keep", default="1-4,9-12", show_default=True, help="Steps whose latents are kept.")
@click.option("--seed", type=int, default=0, show_default=True)
@click.option(
    "--steer",
    metavar="LAYER:CHANNEL:AMOUNT",
    help="Add AMOUNT to one channel of one layer at every step; a twin without it is its control.",
)
@click.option("--overwrite", is_flag=True, help="Replace OUT if it holds anything.")
@_adapter_option
def toy_cmd(out, steps, keep, seed, steer, overwrite, adapter) -> None:
    """Run a toy emulator and write its latents to OUT, with nothing but numpy.

    An MLP with a residual stream on a small Gaussian grid: no checkpoint, no
    data, a few seconds. What comes out is read like any other archive.
    """
    from exege.core.errors import RequestError
    from exege.latents.toy import parse_steps, write_toy

    pushed = None
    if steer:
        try:
            layer, channel, amount = steer.split(":")
            pushed = (int(layer), int(channel), float(amount))
        except ValueError as exc:
            raise RequestError(f"--steer is LAYER:CHANNEL:AMOUNT, not {steer!r}") from exc
    path = write_toy(
        out,
        adapter=adapter,
        overwrite=overwrite,
        n_steps=steps,
        keep=parse_steps(keep),
        seed=seed,
        steer=pushed,
    )
    click.echo(f"wrote {path}; try `exege latents info {path} --mask-variable sst`")


@latents.command("region")
@click.argument("source", type=click.Path())
@_adapter_option
@_mask_option
@click.option("--time", "time", default="0", show_default=True, help="Time label, or position.")
@click.option("--layer", type=int, help="Layer to analyze.  [default: the last]")
@click.option("--rank-layer", type=int, help="Layer to rank channels at.  [default: the last]")
@_region_options
@click.option("--top", type=int, default=15, show_default=True, help="Channels to rank.")
@click.option("--pin", "pinned", type=int, multiple=True, help="Channel to list first; repeatable.")
@click.option("--centered", is_flag=True, help="Remove each channel's global mean first.")
@click.option("--reference", type=click.Choice(["nearest", "mean"]), default="nearest")
@click.option(
    "--pcs",
    "--features",
    "n_components",
    type=int,
    default=0,
    help="Features to map: principal components fitted in the region, or with --basis "
    "the features of it that respond most there.",
)
@_basis_option
@_unverified_option
@_json_option
def region_cmd(
    source, adapter, mask_variable, time, layer, as_json, lat, lon, radius_km, basis_path, **kw
):
    """Rank the channels that respond in a region; optionally map a decomposition."""
    from exege.latents import Region, analyze_region

    opened = open_for_cli(source, adapter, mask_variable)
    result = analyze_region(
        opened,
        time=_time(time),
        layer=opened.info().last_layer if layer is None else layer,
        region=Region(lat, lon, radius_km),
        basis=_basis(basis_path),
        **kw,
    )
    summary = result.summary()
    if as_json:
        click.echo(json.dumps(summary, indent=2))
        return
    s = summary["settings"]
    click.echo(f"{summary['n_region_nodes']} node(s) at layer {s['layer']}, time {s['time']}\n")
    rows = [
        {"rank": i, "channel": r["channel"], "peak_abs": f"{r['peak_abs']:.4g}"}
        for i, r in enumerate(summary["ranking"], start=1)
    ]
    click.echo(_render.table(rows))
    for feature in summary.get("features", []):
        loadings = "  ".join(
            f"{x['channel']}({x['loading']:+.2f})" for x in feature["top_loadings"]
        )
        if "peak_abs" in feature:  # a given basis: how strongly it responds here
            size = f"peak {feature['peak_abs']:.3g}"
        else:
            size = f"{100 * feature['explained_variance_ratio']:5.1f}%"
        click.echo(f"\n{feature['label']}  {size}  {loadings}")


@latents.command("series")
@click.argument("source", type=click.Path())
@_adapter_option
@_mask_option
@click.option("--layer", type=int, help="Layer to follow.  [default: the last]")
@_region_options
@click.option("--channel", "channels", type=int, multiple=True, help="Repeatable.")
@click.option("--feature", "features", type=int, multiple=True, help="With --basis; repeatable.")
@click.option("--centered", is_flag=True, help="Remove each channel's global mean at each time.")
@_basis_option
@_unverified_option
@_json_option
def series_cmd(
    source, adapter, mask_variable, layer, lat, lon, radius_km, channels, features, centered,
    basis_path, allow_unverified_basis, as_json,
):  # fmt: skip
    """Follow a region's mean response through every time of SOURCE."""
    from exege.latents import Region, region_series

    if not channels and not (basis_path and features):
        raise click.UsageError("name what to follow: --channel N, or --basis FILE --feature N")
    opened = open_for_cli(source, adapter, mask_variable)
    result = region_series(
        opened,
        layer=opened.info().last_layer if layer is None else layer,
        region=Region(lat, lon, radius_km),
        channels=channels or None,
        basis=_basis(basis_path),
        features=features or None,
        centered=centered,
        allow_unverified_basis=allow_unverified_basis,
    )
    if as_json:
        click.echo(json.dumps(result.summary(), indent=2))
        return
    hours = result.elapsed_seconds
    rows = [
        {
            "time": label,
            "hours": "" if hours is None else f"{hours[i] / 3600:g}",
            **{str(c): f"{v:.4g}" for c, v in zip(result.columns, result.values[i], strict=True)},
        }
        for i, label in enumerate(result.times)
    ]
    click.echo(_render.table(rows))


@latents.command("pca")
@click.argument("source", type=click.Path())
@_adapter_option
@_mask_option
@click.option("--layer", type=int, help="Layer to decompose.  [default: the last]")
@click.option("--components", type=int, default=16, show_default=True)
@click.option("--time", "times", multiple=True, help="Time label or position; repeatable.")
@click.option("--out", type=click.Path(dir_okay=False), required=True, help="Basis file to write.")
def pca_cmd(source, adapter, mask_variable, layer, components, times, out) -> None:
    """Fit a global, area-weighted PCA over every node and time; write a basis.

    The baseline any learned dictionary has to beat, and usable wherever one is:
    `latents region --basis`, `latents series --basis`.
    """
    from exege.latents import accumulate_moments, basis_hash, pca_from_moments, save_basis

    opened = open_for_cli(source, adapter, mask_variable)
    layer = opened.info().last_layer if layer is None else layer
    moments = accumulate_moments(opened, layer=layer, times=[_time(t) for t in times] or None)
    basis = pca_from_moments(moments, components)
    save_basis(out, basis)
    explained = 100 * float(basis.explained_variance_ratio.sum())
    click.echo(
        f"wrote {out}: {components} component(s) of layer {layer} over "
        f"{len(moments.times)} time(s), {explained:.1f}% of the variance; "
        f"--basis-sha256 {basis_hash(basis)}"
    )


@latents.command("import-sae")
@click.argument("path", type=click.Path(dir_okay=False))
@click.option("--out", type=click.Path(dir_okay=False), required=True, help="Basis file to write.")
@click.option("--k", "k", type=int, required=True, help="Features kept active at each node.")
@click.option("--encoder", "encoder_name", default="W_enc", show_default=True,
              help="Array holding the encoder, (features, channels).")  # fmt: skip
@click.option("--decoder", "decoder_name", default="W_dec", show_default=True,
              help="Array holding the decoder, either way round.")  # fmt: skip
@click.option("--pre-bias", "pre_bias_name", default="b_pre", show_default=True,
              help="Array holding the pre-bias, (channels,).")  # fmt: skip
@click.option("--decoder-layout", type=click.Choice(["features-first", "channels-first"]),
              help="Which way round the decoder is; needed only when it is square.")  # fmt: skip
@click.option("--like", "like", type=click.Path(),
              help="An archive of the network the dictionary was trained on: its identity "
              "at --layer is recorded, so the basis is accepted there and refused "
              "elsewhere.")  # fmt: skip
@click.option("--layer", type=int, help="With --like: the layer the dictionary reads.")
@_adapter_option
@_mask_option
def import_sae_cmd(
    path, out, k, encoder_name, decoder_name, pre_bias_name, decoder_layout, like, layer,
    adapter, mask_variable,
):  # fmt: skip
    """Write a token-normalized top-k dictionary trained elsewhere as a basis.

    PATH is an .npz holding an encoder, a decoder and a pre-bias, for a dictionary that
    centers each node's channels and divides them by their length before encoding, and
    is trained to rebuild that normalized node (the public GraphCast dictionary is one).
    """
    import hashlib

    import numpy as np

    from exege.core.errors import RequestError
    from exege.latents import basis_hash, save_basis, token_normalized_dictionary

    if (like is None) != (layer is None):
        raise click.UsageError("give --like SOURCE and --layer N together, or neither")
    # The foreign file is read here, by the client, and never by exege.latents, whose
    # API takes arrays: the only files that package reads are its own.
    file = Path(path)
    if not file.is_file():
        raise RequestError(f"no file: {path}")
    wanted = (encoder_name, decoder_name, pre_bias_name)
    try:
        with np.load(file, allow_pickle=False) as stored:
            names = list(stored.files)
            arrays = {n: stored[n] for n in wanted if n in names}
    except (OSError, ValueError) as exc:
        raise RequestError(f"{path} is not a readable .npz ({exc!r})") from exc
    if missing := [n for n in wanted if n not in arrays]:
        raise RequestError(
            f"{path} holds no array(s) {', '.join(missing)}; it holds {', '.join(names)}"
        )
    imported = {"path": str(file), "sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
    meta: dict = {"imported_from": imported}
    if like is not None:
        info = open_for_cli(like, adapter, mask_variable).info()
        meta["fitted_on"] = {
            "provenance": info.provenance(),
            "layer": layer,
            "network_layer": info.layer(layer).position,
        }
    basis = token_normalized_dictionary(
        arrays[encoder_name], arrays[decoder_name], arrays[pre_bias_name], k,
        decoder_layout=decoder_layout, meta=meta,
    )  # fmt: skip
    if like is not None and basis.n_channels != info.layer(layer).n_channels:
        raise RequestError(
            f"the dictionary reads {basis.n_channels} channel(s), and layer {layer} of {like} "
            f"has {info.layer(layer).n_channels}"
        )
    save_basis(out, basis)
    click.echo(
        f"wrote {out}: {basis.n_features} feature(s) over {basis.n_channels} channel(s), "
        f"top {k}, token-normalized; --basis-sha256 {basis_hash(basis)}"
    )
    if like is None:
        click.echo(
            "it records no network or layer: give --like SOURCE --layer N to say which, or "
            "pass --allow-unverified-basis wherever it is used"
        )


@latents.command("diff")
@click.argument("control", type=click.Path())
@click.argument("experiment", type=click.Path())
@_adapter_option
@_mask_option
@click.option("--time", "time", default="-1", show_default=True, help="Time label, or position.")
@click.option("--layer", type=int, help="Layer to compare.  [default: the last]")
@click.option("--top", type=int, default=15, show_default=True, help="Channels to rank.")
@click.option(
    "--growth", is_flag=True, help="Instead: how large the difference is, by layer and time."
)
@click.option(
    "--across-models",
    is_flag=True,
    help="Compare although the two declare different networks. A channel's index means "
    "nothing between checkpoints trained apart; this is for when it does (a fine-tune).",
)
@click.option(
    "--allow-unverified-sources",
    is_flag=True,
    help="Compare although a model, component, checkpoint or network layer is undeclared by "
    "either run; known mismatches still fail.",
)
@click.option(
    "--noise",
    type=click.Path(),
    help="With --growth: the control rerun with another seed. Adds the difference that "
    "noise alone makes, and the experiment's difference as a multiple of it.",
)
@_json_option
def diff_cmd(
    control, experiment, adapter, mask_variable, time, layer, top, growth, across_models,
    allow_unverified_sources, noise, as_json,
):  # fmt: skip
    """Set a perturbed or steered run against its CONTROL, node for node."""
    from exege.latents import difference, difference_growth

    if noise is not None and not growth:
        raise click.UsageError("--noise goes with --growth")
    a = open_for_cli(control, adapter, mask_variable)
    b = open_for_cli(experiment, adapter, mask_variable)
    if growth:
        grown = difference_growth(
            a,
            b,
            layers=None if layer is None else [layer],
            across_models=across_models,
            allow_unverified=allow_unverified_sources,
            noise=None if noise is None else open_for_cli(noise, adapter, mask_variable),
        )
        if as_json:
            click.echo(json.dumps(grown.summary(), indent=2))
            return
        names = [f"layer {x}" for x in grown.layers]
        click.echo("RMS difference, relative to the control's own spread\n")
        click.echo(_layer_table(grown.times, names, grown.relative))
        if grown.noise_rms is not None:
            click.echo("\nRMS difference, as a multiple of what the noise run's makes\n")
            click.echo(_layer_table(grown.times, names, grown.signal_to_noise))
        return
    result = difference(
        a,
        b,
        time=_time(time),
        layer=a.info().last_layer if layer is None else layer,
        top=top,
        across_models=across_models,
        allow_unverified=allow_unverified_sources,
    )
    summary = result.summary()
    if as_json:
        click.echo(json.dumps(summary, indent=2))
        return
    s = summary["settings"]
    click.echo(f"layer {s['layer']}, time {s['time']}: total RMS {summary['total_rms']:.4g}\n")
    rows = [
        {"rank": i, "channel": r["channel"], "rms": f"{r['rms']:.4g}"}
        for i, r in enumerate(summary["ranking"], start=1)
    ]
    click.echo(_render.table(rows))


def _layer_table(times, names, values) -> str:
    rows = [
        {"time": label, **{n: f"{v:.3g}" for n, v in zip(names, row, strict=True)}}
        for label, row in zip(times, values, strict=True)
    ]
    return _render.table(rows)


@latents.command("storyline")
@click.argument("source", type=click.Path())
@_adapter_option
@_mask_option
@click.option("--field", required=True, help="The physical field to follow.")
@click.option(
    "--layer", "layers", type=int, multiple=True, help="Layer to read; repeatable.  [default: all]"
)
@click.option(
    "--bases",
    help="Basis file per layer, as a template with {layer}: bases/sae_L{layer:02d}.npz. "
    "Layers without a file are read by their channels.",
)
@click.option("--time", "times", multiple=True, help="Time label or position; repeatable.")
@_lead_option
@_unverified_option
@_json_option
def storyline_cmd(
    source, adapter, mask_variable, field, layers, bases, times, lead, allow_unverified_basis,
    as_json,
):  # fmt: skip
    """Where a physical field lives in the network, time by time.

    The best absolute correlation any channel (or feature) of each layer reaches.
    """
    from pathlib import Path

    from exege.latents import field_storyline

    opened = open_for_cli(source, adapter, mask_variable)
    chosen = list(layers) or [x.index for x in opened.info().layers]
    found = {}
    if bases:
        for layer in chosen:
            path = Path(bases.format(layer=layer))
            if path.is_file():
                found[layer] = _basis(str(path))
        if not found:
            raise click.UsageError(f"no basis file matches {bases!r} for layers {chosen}")
    result = field_storyline(
        opened,
        field=field,
        layers=chosen,
        times=[_time(t) for t in times] or None,
        bases=found,
        allow_unverified_basis=allow_unverified_basis,
        lead=lead,
    )
    if as_json:
        click.echo(json.dumps(result.summary(), indent=2))
        return
    kind = "feature (basis)" if found else "channel"
    click.echo(f"best |r| of any {kind} with {field}, by layer and time\n")
    click.echo(_layer_table(result.times, [f"layer {x}" for x in result.layers], result.best))


@latents.command("hovmoller")
@click.argument("source", type=click.Path())
@_adapter_option
@_mask_option
@click.option("--lat-min", type=float, required=True)
@click.option("--lat-max", type=float, required=True)
@click.option("--field", help="A physical field kept beside the latents.")
@click.option("--layer", type=int, help="With --channel, or with --basis and --feature.")
@click.option("--channel", type=int)
@_basis_option
@click.option("--feature", type=int)
@click.option("--bins", type=int, default=12, show_default=True, help="Longitude bins printed.")
@click.option("--out", type=click.Path(dir_okay=False), help="Also write the full diagram (.npz).")
@_unverified_option
@_json_option
def hovmoller_cmd(
    source, adapter, mask_variable, lat_min, lat_max, field, layer, channel, basis_path, feature,
    bins, out, allow_unverified_basis, as_json,
):  # fmt: skip
    """A quantity along a latitude band, longitude against time: traveling things tilt."""
    import numpy as np

    from exege.latents import hovmoller

    opened = open_for_cli(source, adapter, mask_variable)
    result = hovmoller(
        opened,
        lat_min=lat_min,
        lat_max=lat_max,
        field=field,
        layer=layer,
        channel=channel,
        basis=_basis(basis_path),
        feature=feature,
        allow_unverified_basis=allow_unverified_basis,
    )
    if out:
        np.savez(out, values=result.values, lon=result.lon, times=np.array(result.times))
    if as_json:
        click.echo(json.dumps(result.summary(), indent=2))
        return
    lon = np.mod(result.lon, 360.0)
    edges = np.linspace(0.0, 360.0, max(bins, 1) + 1)
    which = np.clip(np.digitize(lon, edges) - 1, 0, len(edges) - 2)
    names = [f"{edges[k]:.0f}E" for k in range(len(edges) - 1)]
    with np.errstate(invalid="ignore"):
        binned = np.stack(
            [np.nanmean(result.values[:, which == k], axis=1) for k in range(len(edges) - 1)],
            axis=1,
        )
    what = field or (
        f"layer {layer} channel {channel}" if basis_path is None else f"feature {feature}"
    )
    click.echo(
        f"{what}, {lat_min} to {lat_max} degrees, mean per {360 / (len(edges) - 1):.0f} degrees\n"
    )
    click.echo(_layer_table(result.times, names, binned))


@latents.command("fields")
@click.argument("source", type=click.Path())
@_adapter_option
@_mask_option
@click.option("--field", help="Rank channels by correlation with this field.  [default: list them]")
@click.option("--time", "time", default="0", show_default=True, help="Time label, or position.")
@click.option("--layer", type=int, help="Layer to correlate.  [default: the last]")
@click.option("--top", type=int, default=15, show_default=True)
@_basis_option
@_lead_option
@_unverified_option
@_json_option
def fields_cmd(
    source, adapter, mask_variable, field, time, layer, top, basis_path, lead,
    allow_unverified_basis, as_json,
):  # fmt: skip
    """Which channels (or features) track a physical field kept beside the latents."""
    from exege.latents import ReferenceFields, rank_by_field

    opened = open_for_cli(source, adapter, mask_variable)
    if field is None:
        names = opened.field_names() if isinstance(opened, ReferenceFields) else ()
        click.echo("\n".join(names) if names else "no reference fields")
        return
    result = rank_by_field(
        opened,
        time=_time(time),
        layer=opened.info().last_layer if layer is None else layer,
        field=field,
        top=top,
        basis=_basis(basis_path),
        allow_unverified_basis=allow_unverified_basis,
        lead=lead,
    )
    summary = result.summary()
    if as_json:
        click.echo(json.dumps(summary, indent=2))
        return
    s = summary["settings"]
    later = f", {lead:+d} time(s) later" if lead else ""
    click.echo(f"{s['columns']} of layer {s['layer']} against {field} at {s['time']}{later}\n")
    rows = [
        {"rank": i, s["columns"][:-1]: r["column"], "correlation": f"{r['correlation']:+.3f}"}
        for i, r in enumerate(summary["ranking"], start=1)
    ]
    click.echo(_render.table(rows))


def _optional_region(lat, lon, radius_km):
    from exege.latents import Region

    if (lat is None) != (lon is None):
        raise click.UsageError("give both --lat and --lon, or neither")
    return None if lat is None else Region(lat, lon, radius_km)


_optional_region_options = [
    click.option("--lat", type=float, help="With --lon: read only a region around this point."),
    click.option("--lon", type=float),
    click.option("--radius-km", type=float, default=2000.0, show_default=True),
]
_threshold_option = click.option(
    "--threshold",
    type=float,
    default=0.0,
    show_default=True,
    help="Active means above this: firing, for a sparse basis; positive, for channels.",
)


def _with(options):
    def apply(command):
        for option in reversed(options):
            command = option(command)
        return command

    return apply


@latents.command("census")
@click.argument("source", type=click.Path())
@_adapter_option
@_mask_option
@click.option("--time", "time", default="0", show_default=True, help="Time label, or position.")
@click.option("--layer", type=int, help="Layer to survey.  [default: the last]")
@_basis_option
@click.option(
    "--by",
    type=click.Choice(["coverage", "mean", "strength", "peak"]),
    default="coverage",
    show_default=True,
    help="Order: share of the area active, mean, mean where active, or largest value.",
)
@click.option("--top", type=int, default=20, show_default=True)
@_with(_optional_region_options)
@_threshold_option
@_unverified_option
@_json_option
def census_cmd(
    source, adapter, mask_variable, time, layer, basis_path, by, top, lat, lon, radius_km,
    threshold, allow_unverified_basis, as_json,
):  # fmt: skip
    """Every channel (or feature) of a layer: how much of the world it is active over."""
    from exege.latents import feature_census

    opened = open_for_cli(source, adapter, mask_variable)
    result = feature_census(
        opened,
        time=_time(time),
        layer=opened.info().last_layer if layer is None else layer,
        basis=_basis(basis_path),
        region=_optional_region(lat, lon, radius_km),
        threshold=threshold,
        allow_unverified_basis=allow_unverified_basis,
    )
    summary = result.summary(by=by, top=top)
    if as_json:
        click.echo(json.dumps(summary, indent=2))
        return
    s = summary["settings"]
    click.echo(f"{s['columns']} of layer {s['layer']} at {s['time']}, by {by}\n")
    rows = [
        {
            s["columns"][:-1]: c["column"],
            "coverage": f"{c['coverage']:.3f}",
            "mean": f"{c['mean']:.3g}",
            "strength": f"{c['strength']:.3g}",
            "peak": f"{c['peak']:.3g}",
            "at": f"{c['peak_lat']:.0f}, {c['peak_lon']:.0f}",
        }
        for c in summary["columns"]
    ]
    click.echo(_render.table(rows))


@latents.command("profile")
@click.argument("source", type=click.Path())
@_adapter_option
@_mask_option
@click.option("--layer", type=int, help="Layer to read.  [default: the last]")
@click.option("--channel", type=int, help="The channel to profile.")
@_basis_option
@click.option("--feature", type=int, help="With --basis: the feature to profile.")
@click.option("--time", "times", multiple=True, help="Time label or position; repeatable.")
@click.option("--field", "fields", multiple=True, help="Repeatable.  [default: every field]")
@_with(_optional_region_options)
@_threshold_option
@_lead_option
@click.option(
    "--rotations", type=int, default=0, show_default=True,
    help="Also set each effect against this many longitude rotations of the field (a null "
    "that keeps the footprint and breaks its east-west alignment).",
)  # fmt: skip
@_unverified_option
@_json_option
def profile_cmd(
    source, adapter, mask_variable, layer, channel, basis_path, feature, times, fields, lat, lon,
    radius_km, threshold, lead, rotations, allow_unverified_basis, as_json,
):  # fmt: skip
    """Every physical field where one channel (or feature) is active, against where it is not."""
    from exege.latents import feature_profile

    by_channel = channel is not None and basis_path is None and feature is None
    by_feature = channel is None and basis_path is not None and feature is not None
    if not (by_channel or by_feature):
        raise click.UsageError("name one: --channel N, or --basis FILE --feature N")
    opened = open_for_cli(source, adapter, mask_variable)
    result = feature_profile(
        opened,
        layer=opened.info().last_layer if layer is None else layer,
        column=feature if channel is None else channel,
        basis=_basis(basis_path),
        times=[_time(t) for t in times] or None,
        fields=fields or None,
        region=_optional_region(lat, lon, radius_km),
        threshold=threshold,
        allow_unverified_basis=allow_unverified_basis,
        lead=lead,
        rotations=rotations,
    )
    if as_json:
        click.echo(json.dumps(result.summary(), indent=2))
        return
    s = result.settings
    click.echo(
        f"{s['columns'][:-1]} {s['column']} of layer {s['layer']}: active over "
        f"{result.coverage:.1%} of the area and {len(result.times)} time(s)\n"
    )
    rows = [
        {"field": n, "effect": f"{e:+.2f}", "active": f"{a:.4g}", "inactive": f"{i:.4g}"}
        for n, e, a, i in zip(
            result.fields, result.effect, result.active_mean, result.inactive_mean, strict=True
        )
    ]
    if result.null_z is not None:
        for row, z, p in zip(rows, result.null_z, result.p_value, strict=True):
            row["null_z"] = "-" if z != z else f"{z:+.1f}"
            row["p"] = "-" if p != p else f"{p:.2g}"
        click.echo(f"against {len(s['shifts'])} rotation(s) of each field\n")
    click.echo(_render.table(rows))


def _number(value, spec: str) -> str:
    return "-" if value is None or value != value else format(value, spec)


@latents.command("associate")
@click.argument("source", type=click.Path())
@_adapter_option
@_mask_option
@click.option("--layer", type=int, help="Layer to read.  [default: the last]")
@_basis_option
@click.option("--feature", "features", type=int, multiple=True,
              help="With --basis: a feature to label; repeatable.  [default: all]")  # fmt: skip
@click.option("--channel", "channels", type=int, multiple=True,
              help="Without --basis: a channel to label; repeatable.  [default: all]")  # fmt: skip
@click.option("--field", "fields", multiple=True, help="Repeatable.  [default: every field]")
@click.option("--time", "times", multiple=True, help="Time label or position; repeatable.")
@click.option("--rotations", type=int, default=64, show_default=True,
              help="Longitude rotations that make the null.")  # fmt: skip
@click.option("--min-shift-degrees", type=float, default=30.0, show_default=True,
              help="No rotation moves the map less than this, either way round.")  # fmt: skip
@click.option("--q", type=float, default=0.05, show_default=True,
              help="False-discovery rate, across every column and field (with --null "
              "orbit: each column's chance of a false label).")  # fmt: skip
@click.option("--null", "null", type=click.Choice(["rotations", "orbit"]), default="rotations",
              show_default=True,
              help="rotations: a Gaussian z against --rotations shifts, Benjamini-Hochberg "
              "across the table. orbit: every shift of the grid, and each column's best field "
              "ranked against the whole orbit, with no Gaussian tail.")  # fmt: skip
@_threshold_option
@_lead_option
@click.option("--top", type=int, default=20, show_default=True, help="Columns to print.")
@click.option("--out", type=click.Path(dir_okay=False), help="Also write every column (JSON).")
@_unverified_option
@_json_option
def associate_cmd(
    source, adapter, mask_variable, layer, basis_path, features, channels, fields, times,
    rotations, min_shift_degrees, q, null, threshold, lead, top, out, allow_unverified_basis,
    as_json,
):  # fmt: skip
    """Label each channel (or feature) by the field it goes with, against chance.

    Each association is measured against the same one with the field turned east by
    whole columns, many times: the footprint stays exactly as it is and only its
    east-west alignment with the field is broken. Labels need a discovery at --q and a
    clear lead over the runner-up; the same rule run on a rotation, where nothing is
    real, says how often it labels by chance. Associations organised by latitude alone
    are invisible to this null.
    """
    from exege.latents import associate

    if features and channels:
        raise click.UsageError("--feature goes with --basis, --channel without; not both")
    if features and basis_path is None:
        raise click.UsageError("--feature needs --basis FILE")
    if channels and basis_path is not None:
        raise click.UsageError("with --basis, name features with --feature")
    opened = open_for_cli(source, adapter, mask_variable)
    result = associate(
        opened,
        layer=opened.info().last_layer if layer is None else layer,
        basis=_basis(basis_path),
        columns=list(features or channels) or None,
        fields=list(fields) or None,
        times=[_time(t) for t in times] or None,
        threshold=threshold,
        n_rotations=rotations,
        min_shift_degrees=min_shift_degrees,
        q=q,
        lead=lead,
        allow_unverified_basis=allow_unverified_basis,
        null=null,
    )
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(result.summary(), indent=2) + "\n")
    summary = result.summary(top=top)
    if as_json:
        click.echo(json.dumps(summary, indent=2))
        return
    s, shape = result.settings, result.null_shape
    click.echo(
        f"{s['columns']} of layer {s['layer']} against {len(result.fields)} field(s), "
        f"{len(s['times'])} time(s), "
        + (
            f"the whole orbit of {result.calibration_rotations} shifts: "
            if null == "orbit"
            else f"{len(result.shifts)} rotation(s): "
        )
        + f"{result.n_labeled} of {len(result.columns)} labeled at q = {q:g}"
    )
    if null == "orbit":
        click.echo(
            f"calibration: {_percent(result.calibration_label_rate)} labeled with the shift half "
            f"way round as the data (q = {q:g}; at most about q by construction, so a check of "
            "the computation, not of the null)\n"
        )
    else:
        click.echo(
            f"calibration: {_percent(result.calibration_label_rate)} labeled with a rotation as "
            f"the data against {result.calibration_rotations} of its own rotations (q = {q:g}); "
            f"null |skew| {_number(shape['median_abs_skew'], '.2f')}, excess "
            f"kurtosis {_number(shape['median_excess_kurtosis'], '+.2f')}\n"
        )
    # with the orbit, the label, field and lead are the orbit test's (the lead in its own
    # units, with no bar to clear); z is the rotations null's, for size and sign
    rows = [
        {
            s["columns"][:-1]: c["column"], "label": c["label"], "field": c["field"] or "-",
            "z": _number(c["z"], "+.1f"),
            **({"p_orbit": _number(c["p_orbit"], ".3g"), "lead": _number(c["gap"], ".1f")}
               if null == "orbit"
               else {"p": _number(c["p"], ".2g"), "gap": _number(c["gap"], ".1f"),
                     "bar": _number(c["gap_bar"], ".1f")}),
            "coverage": _number(c["coverage"], ".3f"),
        }
        for c in summary["columns"]
    ]  # fmt: skip
    click.echo(_render.table(rows))


def _share(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _picked(channels, features, basis_path) -> tuple[list[int] | None, str]:
    """``--channel`` or ``--feature`` (with ``--basis``), never both: the columns asked
    for, or None for all of them."""
    if channels and features:
        raise click.UsageError("name --channel or --feature (with --basis), not both")
    if features and basis_path is None:
        raise click.UsageError("--feature needs --basis")
    if channels and basis_path is not None:
        raise click.UsageError("--channel reads the layer itself; drop --basis, or use --feature")
    return list(channels or features) or None, "feature" if basis_path else "channel"


@latents.command("stationary")
@click.argument("source", type=click.Path())
@_adapter_option
@_mask_option
@click.option("--layer", type=int, help="Layer to read.  [default: the last]")
@_basis_option
@click.option("--channel", "channels", type=int, multiple=True, help="Repeatable.  [default: all]")
@click.option("--feature", "features", type=int, multiple=True,
              help="With --basis; repeatable.  [default: all]")  # fmt: skip
@click.option("--time", "times", multiple=True, help="Time label or position; repeatable.")
@click.option("--static", "statics", multiple=True,
              help="A kept field that does not change (land fraction, orography): fixed "
              "structure it explains is not counted as unexplained; repeatable.")  # fmt: skip
@_threshold_option
@click.option("--chunk", type=int, default=256, show_default=True,
              help="Columns taken at a time; each chunk reads every time once.")  # fmt: skip
@click.option("--top", type=int, default=20, show_default=True)
@_unverified_option
@_json_option
def stationary_cmd(
    source, adapter, mask_variable, layer, basis_path, channels, features, times, statics,
    threshold, chunk, top, allow_unverified_basis, as_json,
):  # fmt: skip
    """How much of each channel (or feature) sits in the same place at every time.

    The share of a column's variance held by its per-node time mean is split into what
    latitude rows explain, what the --static fields explain, and the rest: fixed
    structure of unknown origin, where a learned positional embedding would show.
    """
    from exege.latents import feature_stationarity

    columns, kind = _picked(channels, features, basis_path)
    opened = open_for_cli(source, adapter, mask_variable)
    result = feature_stationarity(
        opened,
        layer=opened.info().last_layer if layer is None else layer,
        basis=_basis(basis_path),
        columns=columns,
        times=[_time(t) for t in times] or None,
        statics=statics,
        threshold=threshold,
        chunk=chunk,
        allow_unverified_basis=allow_unverified_basis,
    )
    summary = result.summary(top=top)
    if as_json:
        click.echo(json.dumps(summary, indent=2))
        return
    s = summary["settings"]
    click.echo(
        f"{s['columns']} of layer {s['layer']} over {len(result.times)} time(s) and "
        f"{result.n_nodes_used} node(s), by fixed structure that latitude"
        + (f" and {', '.join(result.statics)} do" if result.statics else " does")
        + " not explain\n"
    )
    for item in summary["statics"]:
        click.echo(f"static {item['field']}: fixed share {_share(item['fixed_share'])}")
    if summary["statics"]:
        click.echo()
    rows = [
        {
            kind: c["column"], "coverage": _share(c["coverage"]), "fixed": _share(c["fixed_share"]),
            "zonal": _share(c["zonal_share"]), "static": _share(c["static_share"]),
            "unexplained": _share(c["residual_fixed_share"]),
            "fixed_split": _share(c["fixed_share_split"]),
            "unexplained_split": _share(c["residual_fixed_share_split"]),
        }
        for c in summary["columns"]
    ]  # fmt: skip
    click.echo(_render.table(rows))


@latents.command("gap")
@click.argument("reference", type=click.Path())
@click.argument("run", type=click.Path())
@_adapter_option
@_mask_option
@click.option("--layer", type=int, help="Layer to compare.  [default: the last]")
@_basis_option
@click.option("--channel", "channels", type=int, multiple=True, help="Repeatable.  [default: all]")
@click.option("--feature", "features", type=int, multiple=True,
              help="With --basis; repeatable.  [default: all]")  # fmt: skip
@click.option("--time", "times", multiple=True,
              help="Time label or position (in REFERENCE); repeatable.  [default: every time "
              "both keep]")  # fmt: skip
@_with(_optional_region_options)
@_threshold_option
@click.option("--min-coverage", type=float, default=0.01, show_default=True,
              help="With --basis: features active over less of the REFERENCE's area and time "
              "are ranked last; their deficit is noise.")  # fmt: skip
@click.option("--top", type=int, default=15, show_default=True)
@click.option(
    "--allow-unverified-sources",
    is_flag=True,
    help="Compare although a model, component, checkpoint or network layer is undeclared by "
    "either run; known mismatches still fail.",
)
@_unverified_option
@_json_option
def gap_cmd(
    reference, run, adapter, mask_variable, layer, basis_path, channels, features, times, lat,
    lon, radius_km, threshold, min_coverage, top, allow_unverified_sources,
    allow_unverified_basis, as_json,
):  # fmt: skip
    """Set a free RUN against a REFERENCE run of the same network (teacher forced, say).

    Every channel (or feature) is read at the times both keep: its mean, the share of the
    area it is active over and its strength where active, in each, and which ones the RUN
    weakens most: the deficit is one less the ratio of the RUN's root mean square to the
    REFERENCE's, so 0.5 is half the size, whatever the column's sign.
    """
    from exege.latents import feature_gap

    columns, kind = _picked(channels, features, basis_path)
    a = open_for_cli(reference, adapter, mask_variable)
    b = open_for_cli(run, adapter, mask_variable)
    result = feature_gap(
        a,
        b,
        layer=a.info().last_layer if layer is None else layer,
        basis=_basis(basis_path),
        columns=columns,
        times=[_time(t) for t in times] or None,
        region=_optional_region(lat, lon, radius_km),
        threshold=threshold,
        min_coverage=min_coverage,
        allow_unverified=allow_unverified_sources,
        allow_unverified_basis=allow_unverified_basis,
    )
    summary = result.summary(top=top)
    if as_json:
        click.echo(json.dumps(summary, indent=2))
        return
    s = summary["settings"]
    click.echo(
        f"{s['columns']} of layer {s['layer']} over {len(result.times)} time(s) and "
        f"{s['n_nodes_compared']} node(s), the run's largest deficit first\n"
    )

    def g(value):
        return "-" if value is None else f"{value:.4g}"

    rows = [
        {
            kind: c["column"], "deficit": g(c["deficit"]), "ref_rms": g(c["reference"]["rms"]),
            "run_rms": g(c["run"]["rms"]), "mean_ratio": g(c["ratio_mean"]),
            "strength_ratio": g(c["ratio_strength"]), "ref_mean": g(c["reference"]["mean"]),
            "run_mean": g(c["run"]["mean"]), "ref_cover": _share(c["reference"]["coverage"]),
            "run_cover": _share(c["run"]["coverage"]),
        }
        for c in summary["columns"]
    ]  # fmt: skip
    click.echo(_render.table(rows))


def _ints(text: str) -> list[int]:
    try:
        return [int(part) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise click.BadParameter(f"{text!r} is not a comma-separated list of integers") from exc


def _percent(value: float | None) -> str:
    return "-" if value is None or value != value else f"{100 * value:.1f}%"


@latents.command("evaluate")
@click.argument("source", type=click.Path())
@_adapter_option
@_mask_option
@click.option("--layer", type=int, help="Layer the bases were fitted to.  [default: the last]")
@click.option("--blocks", type=int, default=5, show_default=True, help="Contiguous time blocks.")
@click.option(
    "--test-block", "test_blocks", type=int, multiple=True,
    help="Block to hold out; repeatable.  [default: the last]",
)  # fmt: skip
@click.option(
    "--gap", type=int, default=1, show_default=True,
    help="Training times dropped beside each held-out block.",
)  # fmt: skip
@click.option("--split-only", is_flag=True, help="Print the split (to fit on) and stop.")
@click.option(
    "--basis", "basis_paths", type=click.Path(dir_okay=False), multiple=True,
    help="A basis file fitted on the training times (see --split-only); repeatable.",
)  # fmt: skip
@click.option(
    "--basis-sha256", "basis_hashes", multiple=True, metavar="HASH",
    help="Refuse the --basis file in the same position unless its content has this sha256; "
    "give one per --basis, or none (the printed reproduce line gives them).",
)  # fmt: skip
@click.option(
    "--pca", "pca_text", metavar="K,K,...",
    help="Fit a PCA on the training times at these ranks, and set the bases against it.",
)  # fmt: skip
@click.option("--stability", is_flag=True, help="Match the features of the bases to each other.")
@click.option("--recur-above", type=float, default=0.9, show_default=True,
              help="Cosine at which a matched feature recurs.")  # fmt: skip
@click.option("--active-above", type=float, default=0.0, show_default=True,
              help="An activation above this magnitude is active.")  # fmt: skip
@click.option("--duplicate-above", type=float, default=0.95, show_default=True,
              help="Cosine at which two decoder directions are near-duplicates.")  # fmt: skip
@click.option("--out", type=click.Path(dir_okay=False), help="Also write the results (JSON).")
@_record_option
@_unverified_option
@_json_option
def evaluate_cmd(
    source, adapter, mask_variable, layer, blocks, test_blocks, gap, split_only, basis_paths,
    basis_hashes, pca_text, stability, recur_above, active_above, duplicate_above, out,
    record_path,
    allow_unverified_basis, as_json,
):  # fmt: skip
    """Score frozen bases on times they were not fitted on.

    Times are cut into contiguous blocks, the last held out by default, with a buffer.
    Each --basis (fit it with `--time` on the training times: `--split-only` lists
    them) is scored on both sides: explained variance, active features, dead and
    near-duplicate features. --pca sets them against a PCA fitted here, on the
    training times only; --stability matches the bases to each other.
    """
    from exege.core.errors import RequestError
    from exege.latents import (
        Dictionary,
        basis_hash,
        evaluate_basis,
        evaluation_record,
        fidelity_curve,
        load_basis,
        result_provenance,
        save_record,
        seed_stability,
        split_time_blocks,
    )
    from exege.latents.evaluate import jsonable

    if basis_hashes and len(basis_hashes) != len(basis_paths):
        raise RequestError(
            f"{len(basis_hashes)} --basis-sha256 for {len(basis_paths)} --basis: "
            "give one per --basis, in the same order, or none"
        )
    if record_path and split_only:
        raise click.UsageError("--record keeps results, and --split-only computes none")
    opened = open_for_cli(source, adapter, mask_variable)
    info = opened.info()
    layer = info.last_layer if layer is None else layer
    split = split_time_blocks(
        info.times, n_blocks=blocks, test_blocks=tuple(test_blocks) or (-1,), gap=gap
    )
    payload: dict = {
        "layer": layer,
        "split": split.to_dict(),
        "provenance": result_provenance(info),
    }
    bases = [
        load_basis(path, sha256=basis_hashes[i] if basis_hashes else None)
        for i, path in enumerate(basis_paths)
    ]
    ranks = _ints(pca_text) if pca_text else []
    if not split_only:
        if not bases and not ranks:
            raise click.UsageError(
                "give --basis FILE (repeatable) or --pca K,K,..., or --split-only"
            )
        common = {"layer": layer, "active_above": active_above,
                  "allow_unverified_basis": allow_unverified_basis}  # fmt: skip
        evaluations = [
            evaluate_basis(b, opened, split, duplicate_threshold=duplicate_above, **common)
            for b in bases
        ]
        payload["evaluations"] = [e.to_dict() for e in evaluations]
        curve = matching = None
        if ranks:
            autoencoders = [
                (b, Path(p).stem)
                for b, p in zip(bases, basis_paths, strict=True)
                if isinstance(b, Dictionary)
            ]
            curve = fidelity_curve(
                opened, split, pca_components=ranks, duplicate_threshold=duplicate_above,
                dictionaries=[b for b, _ in autoencoders], labels=[n for _, n in autoencoders],
                **common,
            )  # fmt: skip
            payload["curve"] = curve.to_dict()
        if stability:
            matching = seed_stability(
                bases, threshold=recur_above, allow_unverified=allow_unverified_basis
            )
            payload["stability"] = matching.to_dict()
        settings = {
            "adapter": adapter, "mask_variable": mask_variable, "layer": layer,
            "blocks": blocks, "test_blocks": list(test_blocks) or [-1], "gap": gap,
            "bases": [str(p) for p in basis_paths],
            "basis_sha256": [basis_hash(b) for b in bases], "pca": ranks, "stability": stability,
            "recur_above": recur_above, "active_above": active_above,
            "duplicate_above": duplicate_above,
            "allow_unverified_basis": allow_unverified_basis,
        }  # fmt: skip
        command = _evaluate_command(source, info, settings)
    payload = jsonable(payload)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(payload, indent=2) + "\n")
    if record_path:  # refused with --split-only, so everything below was computed
        save_record(
            record_path,
            evaluation_record(
                info, split, evaluations=evaluations, curve=curve, stability=matching,
                bases=bases, settings=settings, command=command,
            ),
        )  # fmt: skip
    if as_json:
        click.echo(json.dumps(payload, indent=2))
        return
    click.echo(
        f"layer {layer}: fit on {len(split.train)} time(s), hold out {len(split.test)} "
        f"({', '.join(split.test)}), {len(split.buffer)} in the buffer"
    )
    if split_only:
        click.echo("\ntrain  " + "\n       ".join(split.train))
        return
    rows = [
        {
            "basis": Path(p).name, "features": e["n_features"], "fitted": e["fitted_on"],
            "ev_train": _percent(e["train"]["explained_variance"]),
            "ev_test": _percent(e["test"]["explained_variance"]),
            "active_train": f"{e['train']['mean_active_features']:.2f}",
            "active_test": f"{e['test']['mean_active_features']:.2f}",
            "dead_test": _percent(e["test"]["dead_fraction"]),
            "duplicates": e["redundancy"]["n_near_duplicates"],
        }
        for p, e in zip(basis_paths, payload["evaluations"], strict=True)
    ]  # fmt: skip
    if rows:
        click.echo("\n" + _render.table(rows))
    if "curve" in payload:
        rows = [
            {
                "point": p["label"], "active": f"{p['mean_active_features']:.2f}",
                "ev_test": _percent(p["explained_variance"]),
                "pca_there": _percent(p["pca_at_same_sparsity"]),
            }
            for p in payload["curve"]["points"]
        ]  # fmt: skip
        click.echo("\nheld out, against mean active features\n" + _render.table(rows))
    if "stability" in payload:
        s = payload["stability"]
        click.echo(
            f"\nstability of {s['n_dictionaries']} bases: {_percent(s['fraction_recurring'])} of "
            f"matched features at cosine >= {s['threshold']:g} "
            f"(median {s['matched_similarity']['median']:.3f}; random directions "
            f"{s['chance_similarity']['median']:.3f})"
        )
    click.echo(f"\nreproduce: {command}")


def _evaluate_command(source: str, info, settings: dict) -> str:
    """The shell command that redoes an evaluation. It pins the archive's commit, which
    an ``hf://`` source resolves, and each basis by the content hash of the file read."""
    s = settings
    words = ["exege", "latents", "evaluate", source, "--adapter", s["adapter"]]
    if s["mask_variable"]:
        words += ["--mask-variable", s["mask_variable"]]
    if info.commit:
        words += ["--revision", info.commit]
    words += ["--layer", s["layer"], "--blocks", s["blocks"], "--gap", s["gap"]]
    for block in s["test_blocks"]:
        words += ["--test-block", block]
    for path, sha256 in zip(s["bases"], s["basis_sha256"], strict=True):
        words += ["--basis", path, "--basis-sha256", sha256]
    if s["pca"]:
        words += ["--pca", ",".join(map(str, s["pca"]))]
    if s["stability"]:
        words += ["--stability", "--recur-above", f"{s['recur_above']:g}"]
    words += ["--active-above", f"{s['active_above']:g}"]
    words += ["--duplicate-above", f"{s['duplicate_above']:g}"]
    if s["allow_unverified_basis"]:
        words.append("--allow-unverified-basis")
    return shlex.join(str(w) for w in words)


def _steer_command(source, settings: dict, spec: dict, basis_sha256: str, command="steer") -> str:
    """The shell command that redoes a steering experiment (or a dose curve), the basis
    pinned by content."""
    s, steer = settings, spec["steer"]
    words = ["exege", "latents", command, *([source] if source else [])]
    words += ["--adapter", s["adapter"]]
    for key, value in s["adapter_options"].items():
        words += ["--adapter-option", f"{key}={json.dumps(value)}"]
    words += ["--basis", s["basis"], "--basis-sha256", basis_sha256]
    words += ["--layer", steer["layer"], "--feature", steer["feature"], "--mode", steer["mode"]]
    if command in ("steer", "mediate"):
        words += ["--amount", f"{steer['amount']!r}"]
    else:
        words += ["--amounts", ",".join(f"{a!r}" for a in spec["amounts"])]
    if steer.get("loop", "open") != "open":
        words += ["--loop", steer["loop"]]
    for t in steer["times"]:
        words += ["--time", t]
    words += ["--steps", spec["steps"], "--seeds", ",".join(map(str, spec["seeds"]))]
    if command == "steer":
        words += ["--random-draws", spec["n_random"], "--random-seed", spec["random_seed"]]
    if s.get("reference_from"):
        words += ["--reference-from", s["reference_from"]]
        words += ["--reference-adapter", s["reference_adapter"]]
        for t in s["reference_times"]:
            words += ["--reference-time", t]
        words += ["--reference-sha256", spec["steer"]["reference"]["sha256"]]
    for name, (path, feature) in s.get("compare", {}).items():
        words += ["--compare", f"{name}={path}:{feature}"]
        words += ["--compare-sha256", f"{name}={s['compare_sha256'][name]}"]
    if s.get("matched"):
        words += ["--matched", s["matched"], "--max-cosine", repr(s["max_cosine"])]
        if s.get("matched_references_sha256"):
            words += ["--matched-references-sha256", s["matched_references_sha256"]]
    if s.get("outcome", "mean") != "mean":
        words += ["--outcome", s["outcome"]]
    if s.get("region"):
        r = s["region"]
        words += ["--lat", repr(r["lat"]), "--lon", repr(r["lon"])]
        words += ["--radius-km", repr(r["radius_km"])]
    for start in s.get("starts", []):
        words += ["--start", start]
    if s.get("targets"):
        words += ["--targets", s["targets"], "--targets-sha256", s["targets_sha256"]]
    for name in s["fields"]:
        words += ["--field", name]
    for layer in s.get("record_layers", []):
        words += ["--record-layer", layer]
    if s["allow_unverified_basis"]:
        words.append("--allow-unverified-basis")
    return shlex.join(str(w) for w in words)


def _adapter_options(pairs: tuple[str, ...]) -> dict:
    """``KEY=VALUE`` pairs as adapter options; a value is JSON where it can be (``3``,
    ``true``, ``0.5``) and text otherwise."""
    options = {}
    for pair in pairs:
        key, equals, value = pair.partition("=")
        if not equals or not key:
            raise click.BadParameter(f"{pair!r} is not KEY=VALUE", param_hint="--adapter-option")
        try:
            options[key] = json.loads(value)
        except ValueError:
            options[key] = value
    return options


def _named_hashes(pairs: tuple[str, ...]) -> dict:
    hashes = {}
    for pair in pairs:
        name, equals, value = pair.partition("=")
        if not equals or not name or not value:
            raise click.BadParameter(f"{pair!r} is not NAME=HASH", param_hint="--compare-sha256")
        hashes[name] = value
    return hashes


def _comparisons(pairs: tuple[str, ...], pins: tuple[str, ...] = ()) -> tuple[dict, dict, dict]:
    """``NAME=BASIS[:FEATURE]`` pairs as comparison directions, as they were given, and the
    content hash of each; ``pins`` (``NAME=HASH``) refuses a file whose content differs."""
    from exege.core.errors import RequestError
    from exege.latents import load_basis

    hashes = _named_hashes(pins)
    loaded, given = {}, {}
    for pair in pairs:
        name, equals, rest = pair.partition("=")
        if not equals or not name or not rest:
            raise click.BadParameter(
                f"{pair!r} is not NAME=BASIS[:FEATURE]", param_hint="--compare"
            )
        path, colon, feature = rest.rpartition(":")
        if not colon or not feature.isdigit():
            path, feature = rest, "0"
        loaded[name] = (load_basis(path, sha256=hashes.get(name)), int(feature))
        given[name] = (path, int(feature))
    if unknown := sorted(set(hashes) - set(loaded)):
        raise RequestError(f"--compare-sha256 names no --compare: {', '.join(unknown)}")
    return loaded, given, {name: basis.meta["sha256"] for name, (basis, _) in loaded.items()}


def _same_place(archive_info, system_info, layer: int) -> int:
    """The archive's index of the layer at the network place of ``system_info``'s
    ``layer``."""
    from exege.core.errors import RequestError
    from exege.latents.source import differing_identity

    if apart := differing_identity(archive_info.identity(), system_info.identity()):
        raise RequestError(
            f"{archive_info.source} is not the network {system_info.source} runs "
            f"({'; '.join(apart)}): a reference read there means nothing here"
        )
    place = system_info.layer(layer).position
    found = [e.index for e in archive_info.layers if e.position == place]
    if not found:
        kept = ", ".join(str(e.position) for e in archive_info.layers)
        raise RequestError(
            f"{archive_info.source} keeps no layer at network place {place} (layer {layer} of "
            f"{system_info.source}); it keeps network layers {kept}"
        )
    return found[0]


def _steer_parts(
    source, adapter, adapter_options, basis_path, layer, feature, mode, amount, times, loop,
    reference_from, reference_adapter, reference_times, lat, lon, radius_km, outcome_stat,
    starts, allow_unverified_basis, reference_sha256=None,
):  # fmt: skip
    """What `steer` and `dose` share: the system, the basis, the steer, its reference, the
    outcome and the cases, from the options as given."""
    from exege.latents import Case, Outcome, Steer, feature_reference, open_intervenable

    basis = _basis(basis_path)
    if basis is None:
        raise click.UsageError("--basis FILE is required")
    if mode in ("toward", "excess") and not reference_from:
        raise click.UsageError(f"--mode {mode} needs --reference-from SOURCE --reference-time T")
    if reference_from and mode not in ("toward", "excess"):
        raise click.UsageError("--reference-from is for --mode toward or excess")
    if reference_from and not reference_times:
        raise click.UsageError("--reference-from needs at least one --reference-time")
    options = _adapter_options(adapter_options)
    system = open_intervenable(source, adapter, **options)
    reference_for = None
    reference = meta = None
    if reference_from:
        archive = open_for_cli(reference_from, reference_adapter, None)
        labels = [_time(t) for t in reference_times]
        # --layer is the system's; an archive that keeps some layers numbers them its own
        # way, so the reference is read at the archive's layer in the same network place
        archive_layer = _same_place(archive.info(), system.info(), layer)

        def reference_for(j):  # the reference and where it came from, checked as the steer's
            return feature_reference(
                archive, basis, layer=archive_layer, feature=j, times=labels,
                allow_unverified_basis=allow_unverified_basis,
            )  # fmt: skip

        reference, meta = feature_reference(
            archive, basis, layer=archive_layer, feature=feature, times=labels,
            allow_unverified_basis=allow_unverified_basis,
        )  # fmt: skip
        from exege.latents.steering import array_digest

        if reference_sha256 and array_digest(reference) != reference_sha256:
            from exege.core.errors import RequestError

            raise RequestError(
                f"the reference read from {reference_from} has content sha256 "
                f"{array_digest(reference)[:12]}, not the {reference_sha256[:12]} pinned"
            )
    steer = Steer(
        layer=layer, feature=feature, mode=mode, amount=amount, times=tuple(times) or (0,),
        loop=loop, reference=reference, reference_meta=meta,
    )  # fmt: skip
    region = _optional_region(lat, lon, radius_km)
    cases = [Case(start=_time(s), region=region) for s in starts] or (
        [Case(start=0, region=region)] if region is not None else []
    )
    return system, basis, steer, Outcome(statistic=outcome_stat), cases, reference_for, options


def _steer_options(command):
    """The options `steer` and `dose` share."""
    for option in reversed(
        [
            click.option("--adapter", required=True,
                         help="An adapter that can be run (Intervenable)."),
            click.option("--adapter-option", "adapter_options", multiple=True, metavar="KEY=VALUE",
                         help="An option of the adapter; repeatable."),
            _basis_click_option,
            _basis_sha_option,
            click.option("--layer", type=int, required=True,
                         help="Layer the basis reads; the edit acts here."),
            click.option("--feature", type=int, required=True,
                         help="The feature of the basis to change."),
            click.option("--mode", type=click.Choice(["add", "scale", "clamp", "toward", "excess"]),
                         default="add", show_default=True,
                         help="What the amount does to the activation: add it, scale by it, "
                         "clamp to it, move that share of the way to the reference, or scale the "
                         "excess above the reference by it."),
            click.option("--loop", type=click.Choice(["open", "closed"]), default="open",
                         show_default=True,
                         help="Work the change out from the control's latents (open) or from "
                         "the steered run's own (closed)."),
            click.option("--time", "times", type=int, multiple=True,
                         help="Forward step to act at; repeatable.  [default: 0]"),
            click.option("--steps", type=int, required=True, help="Forward steps to run."),
            click.option("--seeds", "seeds_text", default="0", show_default=True,
                         metavar="N,N,...",
                         help="Noise seeds; every arm of a seed shares its noise."),
            click.option("--reference-from", help="An archive of the same network to read the "
                         "reference from, for --mode toward or excess."),
            click.option("--reference-adapter", default="latent-archive", show_default=True,
                         help="How --reference-from is read."),
            click.option("--reference-sha256", metavar="HASH",
                         help="Refuse the reference unless its content has this sha256."),
            click.option("--reference-time", "reference_times", multiple=True,
                         help="A time of --reference-from to average the feature over; "
                         "repeatable."),
            click.option("--lat", type=float,
                         help="With --lon: edit only, and measure only, within --radius-km "
                         "of here."),
            click.option("--lon", type=float),
            click.option("--radius-km", type=float, default=1500.0, show_default=True),
            click.option("--outcome", "outcome_stat", type=click.Choice(["mean", "min", "max"]),
                         default="mean", show_default=True,
                         help="How a field is reduced to one number per step."),
            click.option("--start", "starts", multiple=True,
                         help="An initial state to start from; repeatable, one case each."),
            click.option("--field", "fields", multiple=True,
                         help="A physical field to measure; repeatable.  [default: all]"),
        ]
    ):  # fmt: skip
        command = option(command)
    return command


@latents.command("steer")
@click.argument("source", required=False)
@_steer_options
@click.option("--amount", type=float, required=True, help="What the mode does with it.")
@click.option("--random-draws", type=int, default=20, show_default=True,
              help="Random directions of the same size, to set the feature against.")  # fmt: skip
@click.option("--random-seed", type=int, default=0, show_default=True)
@click.option("--compare", "compare", multiple=True, metavar="NAME=BASIS[:FEATURE]",
              help="Also push along this basis's feature (0 by default) by the same amount, "
              "scaled to the feature's length: a difference of means, a PCA component; "
              "repeatable.")  # fmt: skip
@click.option(
    "--compare-sha256",
    "compare_sha256",
    multiple=True,
    metavar="NAME=HASH",
    help="Refuse that --compare file unless its content has this sha256.",
)
@click.option("--matched", type=int, default=0, show_default=True,
              help="Also apply the steer to this many other features that carry about as much "
              "where it acts.")  # fmt: skip
@click.option(
    "--matched-references-sha256",
    metavar="HASH",
    help="Refuse the run unless the matched features' references, together, have this "
    "content hash (reproduce lines give it).",
)
@click.option(
    "--max-cosine",
    type=float,
    default=0.3,
    show_default=True,
    help="A matched feature's direction is at most this close to the feature's.",
)
@click.option("--record-layer", "record_layers", type=int, multiple=True,
              help="Also compare the latents of this layer; repeatable.")  # fmt: skip
@click.option("--out", type=click.Path(dir_okay=False), help="Also write the result (JSON).")
@_record_option
@_unverified_option
@_json_option
def steer_cmd(
    source, adapter, adapter_options, basis_path, layer, feature, mode, loop, times, steps,
    seeds_text, reference_from, reference_adapter, reference_times, lat, lon, radius_km,
    outcome_stat, starts, fields, amount, random_draws, random_seed, compare, matched,
    max_cosine, record_layers, out, record_path, allow_unverified_basis, as_json,
    reference_sha256, compare_sha256, matched_references_sha256,
):  # fmt: skip
    """Change a feature inside a running system and set what happens against chance.

    The arms share one initial state and one noise seed per seed: a control, one with the
    latents replaced by the dictionary's reconstruction, one with the feature changed,
    --random-draws with a random direction of the same length changed by the same amount,
    one per --compare direction, and --matched other features given the same steer. Each
    field's response to the feature is reported against the others'. With --start (or a
    region), it is run from each initial state and the responses are set side by side.
    """
    from exege.latents import run_cases, run_steering, save_record, steering_record
    from exege.latents.evaluate import jsonable, save_result

    system, basis, steer, outcome, cases, reference_for, options = _steer_parts(
        source, adapter, adapter_options, basis_path, layer, feature, mode, amount, times, loop,
        reference_from, reference_adapter, reference_times, lat, lon, radius_km, outcome_stat,
        starts, allow_unverified_basis, reference_sha256,
    )  # fmt: skip
    if record_path and len(cases) > 1:
        raise click.UsageError("--record keeps one run; with several --start, use --out")
    comparisons, given, compare_hashes = _comparisons(compare, compare_sha256)
    common = dict(
        steps=steps, seeds=_ints(seeds_text), n_random=random_draws, random_seed=random_seed,
        fields=list(fields) or None, record_layers=list(record_layers) or None,
        allow_unverified_basis=allow_unverified_basis, comparisons=comparisons,
        n_matched=matched, max_cosine=max_cosine, reference_for=reference_for,
    )  # fmt: skip
    if cases:
        study = run_cases(system, basis, steer, cases, outcome=outcome, **common)
        result = study.results[0]
    else:
        study, result = None, run_steering(system, basis, steer, outcome=outcome, **common)
    spec = result.spec
    matched_hash = None
    if matched and reference_for is not None:
        import hashlib

        from exege.core.errors import RequestError

        named = [r.spec["matched"]["references"] for r in (study.results if study else [result])]
        matched_hash = hashlib.sha256(json.dumps(named, sort_keys=True).encode()).hexdigest()
        if matched_references_sha256 and matched_references_sha256 != matched_hash:
            raise RequestError(
                f"the matched features' references have content sha256 {matched_hash[:12]}, "
                f"not the {matched_references_sha256[:12]} pinned"
            )
    settings = {
        "matched_references_sha256": matched_hash,
        "adapter": adapter, "adapter_options": options, "basis": basis_path,
        "fields": list(fields), "record_layers": list(record_layers),
        "allow_unverified_basis": allow_unverified_basis, "reference_from": reference_from,
        "reference_adapter": reference_adapter, "reference_times": list(reference_times),
        "compare": given, "compare_sha256": compare_hashes, "matched": matched,
        "max_cosine": max_cosine,
        "outcome": outcome_stat, "starts": list(starts),
        "region": None if lat is None else {"lat": lat, "lon": lon, "radius_km": radius_km},
    }  # fmt: skip
    command = _steer_command(
        source, settings, {**spec, "steer": steer.to_dict()}, result.provenance["basis"]["sha256"]
    )
    if out:
        save_result(out, study if study is not None else result)
    if record_path:
        save_record(record_path, steering_record(result, settings=settings, command=command))
    if as_json:
        click.echo(json.dumps(jsonable((study or result).to_dict()), indent=2))
        return
    click.echo(
        f"{mode} {amount:g} on feature {feature} of layer {layer} at time(s) "
        f"{', '.join(map(str, spec['steer']['times']))} ({steer.loop} loop); {steps} step(s), "
        f"seed(s) {', '.join(map(str, spec['seeds']))}, {spec['n_random']} random direction(s)"
        + (f", {matched} matched feature(s)" if matched else "")
    )
    for case, one in zip(study.cases if study else [None], study.results if study else [result],
                         strict=True):  # fmt: skip
        if case is not None:
            click.echo(f"\ncase {case.name}")
        rows = []
        for name, e in one.effects.items():
            row = {
                "field": name, "response": f"{e.feature_response:.4g}",
                "+-": "-" if e.feature_se is None else f"{e.feature_se:.2g}",
                "recon_only": f"{e.reconstruction_response:.4g}",
                "random_|resp|": f"{float(abs(e.random_responses).mean()):.4g}",
                "effect_size": "-" if e.effect_size is None else f"{e.effect_size:.2f}",
                "rank": f"{e.rank}/{e.random_responses.size + 1}", "p": f"{e.p_value:.2g}",
            }  # fmt: skip
            if e.matched_rank is not None:
                row["matched_rank"] = f"{e.matched_rank}/{len(e.matched_features) + 1}"
            for label, value in e.comparison_responses.items():
                row[label] = f"{value:.4g}"
            rows.append(row)
        click.echo("\n" + _render.table(rows))
    if study is not None and len(study.cases) > 1:
        rows = [
            {
                "field": name, "median": f"{s['median']:.4g}", "positive": s["positive"],
                "negative": s["negative"],
                "rank_1": sum(1 for r in s["ranks"] if r == 1),
                "cases": len(s["cases"]),
            }
            for name, s in study.summary().items()
        ]  # fmt: skip
        click.echo("\nacross cases\n" + _render.table(rows))
    if spec.get("matched"):
        click.echo(f"\nmatched features: {', '.join(map(str, spec['matched']['features']))}")
    click.echo(f"\nreproduce: {command}")


@latents.command("dose")
@click.argument("source", required=False)
@_steer_options
@click.option(
    "--amounts",
    "amounts_text",
    required=True,
    metavar="A,A,...",
    help="The amounts to run, the identity of the mode among them if you want it.",
)
@click.option(
    "--targets",
    type=click.Path(dir_okay=False, exists=True),
    help="JSON of {case: {field: [one value per step]}}: what each case's outcome "
    "should follow. With two or more cases, an amount is chosen leave-one-out.",
)
@click.option(
    "--targets-sha256",
    metavar="HASH",
    help="Refuse the --targets file unless its content has this sha256.",
)
@click.option("--out", type=click.Path(dir_okay=False), help="Also write the result (JSON).")
@_unverified_option
@_json_option
def dose_cmd(
    source, adapter, adapter_options, basis_path, layer, feature, mode, loop, times, steps,
    seeds_text, reference_from, reference_adapter, reference_times, lat, lon, radius_km,
    outcome_stat, starts, fields, amounts_text, targets, out, allow_unverified_basis, as_json,
    reference_sha256, targets_sha256,
):  # fmt: skip
    """The feature's response at every amount, from every case: the whole curve.

    With --targets, every amount is scored by its error against the target, and for each
    case the amount the *other* cases favour is reported beside the hindsight optimum and
    the unsteered control, so a correction can be judged by what could have been chosen.
    """
    from exege.latents import Case, dose_curve
    from exege.latents.evaluate import jsonable, save_result

    try:
        amounts = [float(a) for a in amounts_text.split(",") if a.strip()]
    except ValueError as exc:
        raise click.BadParameter(f"{amounts_text!r} is not a list of numbers") from exc
    system, basis, steer, outcome, cases, _, options = _steer_parts(
        source, adapter, adapter_options, basis_path, layer, feature, mode, amounts[0] if amounts
        else 1.0, times, loop, reference_from, reference_adapter, reference_times, lat, lon,
        radius_km, outcome_stat, starts, allow_unverified_basis, reference_sha256,
    )  # fmt: skip
    goals = targets_hash = None
    if targets:
        import hashlib

        raw = Path(targets).read_bytes()
        targets_hash = hashlib.sha256(raw).hexdigest()
        if targets_sha256 and targets_sha256 != targets_hash:
            from exege.core.errors import RequestError

            raise RequestError(
                f"{targets} has content sha256 {targets_hash[:12]}, not the "
                f"{targets_sha256[:12]} pinned"
            )
        goals = json.loads(raw)
    curve = dose_curve(
        system, basis, steer, amounts, steps=steps, cases=cases or [Case()],
        seeds=_ints(seeds_text), fields=list(fields) or None, outcome=outcome, targets=goals,
        allow_unverified_basis=allow_unverified_basis,
    )  # fmt: skip
    settings = {
        "adapter": adapter, "adapter_options": options, "basis": basis_path,
        "fields": list(fields), "allow_unverified_basis": allow_unverified_basis,
        "reference_from": reference_from, "reference_adapter": reference_adapter,
        "reference_times": list(reference_times), "outcome": outcome_stat,
        "starts": list(starts), "targets": targets, "targets_sha256": targets_hash,
        "region": None if lat is None else {"lat": lat, "lon": lon, "radius_km": radius_km},
    }  # fmt: skip
    command = _steer_command(
        source, settings, curve.spec, curve.provenance["basis"]["sha256"], command="dose"
    )
    if out:
        save_result(out, curve)
    if as_json:
        click.echo(json.dumps(jsonable(curve.to_dict()), indent=2))
        return
    click.echo(
        f"{mode} on feature {feature} of layer {layer} at time(s) "
        f"{', '.join(map(str, steer.times))} ({loop} loop), {len(curve.cases)} case(s)"
    )
    for m, name in enumerate(curve.fields):
        rows = [
            {"case": case, **{f"{a:g}": f"{curve.responses[c, k, m]:.4g}"
                              for k, a in enumerate(curve.amounts)}}
            for c, case in enumerate(curve.cases)
        ]  # fmt: skip
        click.echo(f"\n{name}: response at each amount\n" + _render.table(rows))
        if curve.errors is not None:
            rows = [
                {"case": case, "control": f"{curve.baseline_errors[c, m]:.4g}",
                 **{f"{a:g}": f"{curve.errors[c, k, m]:.4g}" for k, a in enumerate(curve.amounts)}}
                for c, case in enumerate(curve.cases)
            ]  # fmt: skip
            click.echo(f"\n{name}: error against the target\n" + _render.table(rows))
        if name in curve.selections:
            s = curve.selections[name]
            rows = [
                {"case": case, "loo_amount": f"{s.loo_amounts[c]:g}",
                 "loo_error": f"{s.loo_errors[c]:.4g}",
                 "oracle_amount": f"{s.oracle_amounts[c]:g}",
                 "oracle_error": f"{s.oracle_errors[c]:.4g}",
                 "control_error": f"{s.baseline_errors[c]:.4g}"}
                for c, case in enumerate(curve.cases)
            ]  # fmt: skip
            med = s.medians
            click.echo(
                f"\n{name}: amount chosen leave-one-out\n"
                + _render.table(rows)
                + f"\nmedian error: leave-one-out {med['loo']:.4g}, hindsight {med['oracle']:.4g}"
                f", control {med['baseline']:.4g}"
            )
    click.echo(f"\nreproduce: {command}")


@latents.command("direction")
@click.argument("source", type=click.Path())
@_adapter_option
@_mask_option
@click.option("--layer", type=int, help="Layer to read.  [default: the last]")
@click.option("--event-time", "event_times", multiple=True, required=True,
              help="A time the event is present; repeatable.")  # fmt: skip
@click.option("--quiet-time", "quiet_times", multiple=True, required=True,
              help="A time it is absent; repeatable.")  # fmt: skip
@_with(_optional_region_options)
@click.option("--label", default="mean difference", show_default=True)
@click.option("--out", type=click.Path(dir_okay=False), required=True, help="Basis file to write.")
def direction_cmd(
    source, adapter, mask_variable, layer, event_times, quiet_times, lat, lon, radius_km, label,
    out,
):  # fmt: skip
    """The direction from the mean latents at quiet times to the mean at event times.

    Written as a basis of one direction, which `steer --compare NAME=FILE` pushes along: does
    a dictionary's feature do anything this plainer direction does not?
    """
    from exege.latents import mean_difference, save_basis

    opened = open_for_cli(source, adapter, mask_variable)
    layer = opened.info().last_layer if layer is None else layer
    direction = mean_difference(
        opened, layer=layer, event_times=[_time(t) for t in event_times],
        quiet_times=[_time(t) for t in quiet_times],
        region=_optional_region(lat, lon, radius_km), label=label,
    )  # fmt: skip
    path = save_basis(out, direction)
    from exege.latents import load_basis

    click.echo(
        f"wrote {path}: the direction from {len(quiet_times)} quiet to {len(event_times)} event "
        f"time(s) at layer {layer}, length {direction.meta['length']:.4g}; "
        f"--basis-sha256 {load_basis(path).meta['sha256']}"
    )


@latents.command("rotate")
@click.argument("source", required=False)
@click.option("--adapter", required=True, help="An adapter that can be run and rotated.")
@click.option("--adapter-option", "adapter_options", multiple=True, metavar="KEY=VALUE",
              help="An option of the adapter; repeatable.")  # fmt: skip
@click.option("--layer", type=int, required=True, help="Layer whose columns to read.")
@_basis_option
@click.option("--channel", "channels", type=int, multiple=True, help="Repeatable.  [default: all]")
@click.option("--feature", "features", type=int, multiple=True,
              help="With --basis; repeatable.  [default: all]")  # fmt: skip
@click.option("--shift", "shifts", type=float, multiple=True,
              help="Degrees east to turn the input by, whole grid columns; repeatable.  "
              "[default: three turns of at least 60 degrees]")  # fmt: skip
@click.option(
    "--part",
    "parts",
    multiple=True,
    help="A part of the state to turn; repeatable.  [default: every part]",
)
@click.option("--start", "starts", multiple=True, help="Initial state; repeatable.  [default: 0]")
@click.option("--time", type=int, default=0, show_default=True, help="Forward step to read.")
@click.option(
    "--max-self-similarity",
    type=float,
    default=0.5,
    show_default=True,
    help="A shift that maps a column's pattern onto itself, or onto its opposite, this well "
    "(in absolute value) has no power.",
)
@click.option("--threshold", type=float, default=0.5, show_default=True,
              help="Anchored: every powered score at least this.")  # fmt: skip
@click.option("--top", type=int, default=15, show_default=True)
@_unverified_option
@_json_option
def rotate_cmd(
    source, adapter, adapter_options, layer, basis_path, channels, features, shifts, parts,
    starts, time, max_self_similarity, threshold, top, allow_unverified_basis, as_json,
):  # fmt: skip
    """Turn what a system is given east, and see which columns turn with it.

    A column that stays where it was when everything is turned is tied to the model's own
    sense of place; one that stays when only some parts are turned is tied to the rest.
    """
    from exege.latents import open_intervenable, rotation_test

    basis = _basis(basis_path)
    if features and basis is None:
        raise click.UsageError("--feature needs --basis")
    if channels and basis is not None:
        raise click.UsageError("--channel reads channels; with --basis, name --feature")
    system = open_intervenable(source, adapter, **_adapter_options(adapter_options))
    result = rotation_test(
        system, layer=layer, shifts=list(shifts) or None,
        parts=list(parts) or None, basis=basis, columns=list(features or channels) or None,
        starts=[_time(s) for s in starts] or (0,), time=time,
        max_self_similarity=max_self_similarity, threshold=threshold,
        allow_unverified_basis=allow_unverified_basis,
    )  # fmt: skip
    summary = result.summary(top=top)
    if as_json:
        from exege.latents.evaluate import jsonable

        click.echo(json.dumps(jsonable(summary), indent=2))
        return
    s = summary["settings"]
    click.echo(
        f"{s['columns']} of layer {layer} at time {time}, turned "
        f"{', '.join(f'{d:g}' for d in s['shifts'])} degrees east; parts turned: "
        f"{', '.join(s['parts'])}\n"
    )
    rows = [
        {
            s["columns"][:-1]: c["column"],
            "score": "-" if c["score"] is None else f"{c['score']:+.2f}",
            "stays": "-" if c["stays"] is None else f"{c['stays']:+.2f}",
            "follows": "-" if c["follows"] is None else f"{c['follows']:+.2f}",
            "powered": c["n_powered"], "anchored": "yes" if c["anchored"] else "",
        }
        for c in summary["columns"]
    ]  # fmt: skip
    click.echo(_render.table(rows))


@latents.command("mediate")
@click.argument("source", required=False)
@_steer_options
@click.option("--amount", type=float, required=True, help="What the mode does with it.")
@click.option("--mediator-layer", type=int, required=True, help="Layer the mediator is read at.")
@click.option("--mediator-feature", type=int, required=True, help="The feature to hold.")
@click.option("--mediator-basis", type=click.Path(dir_okay=False),
              help="The mediator's basis.  [default: --basis]")  # fmt: skip
@click.option(
    "--mediator-basis-sha256",
    metavar="HASH",
    help="Refuse the --mediator-basis file unless its content has this sha256.",
)
@click.option("--out", type=click.Path(dir_okay=False), help="Also write the result (JSON).")
@_unverified_option
@_json_option
def mediate_cmd(
    source, adapter, adapter_options, basis_path, layer, feature, mode, loop, times, steps,
    seeds_text, reference_from, reference_adapter, reference_times, lat, lon, radius_km,
    outcome_stat, starts, fields, amount, mediator_layer, mediator_feature, mediator_basis,
    out, allow_unverified_basis, as_json, reference_sha256, mediator_basis_sha256,
):  # fmt: skip
    """Does a steer act through another feature? Total, direct and indirect effects.

    The steer is run as it is, and again with the mediator held at its control value from
    the first edit on; what the hold takes away went through the mediator. A third run
    holds the mediator alone and must equal the control: its largest difference is printed.
    """
    from exege.latents import Mediator, load_basis, run_mediation
    from exege.latents.evaluate import jsonable, save_result

    if starts and len(starts) > 1:
        raise click.UsageError("mediate runs from one start")
    system, basis, steer, outcome, cases, _, _ = _steer_parts(
        source, adapter, adapter_options, basis_path, layer, feature, mode, amount, times, loop,
        reference_from, reference_adapter, reference_times, lat, lon, radius_km, outcome_stat,
        starts, allow_unverified_basis, reference_sha256,
    )  # fmt: skip
    if cases and cases[0].region is not None:
        from exege.latents.steering import _case_setup

        steer, outcome = _case_setup(cases[0], steer, outcome, system.grid())
    mediator = Mediator(
        layer=mediator_layer, feature=mediator_feature,
        basis=load_basis(mediator_basis, sha256=mediator_basis_sha256) if mediator_basis
        else basis,
    )  # fmt: skip
    result = run_mediation(
        system, basis, steer, mediator, steps=steps, seeds=_ints(seeds_text),
        fields=list(fields) or None, outcome=outcome,
        initial_state=system.initial_state(cases[0].start) if cases else None,
        allow_unverified_basis=allow_unverified_basis,
    )  # fmt: skip
    settings = {
        "adapter": adapter, "adapter_options": _adapter_options(adapter_options),
        "basis": basis_path, "fields": list(fields),
        "allow_unverified_basis": allow_unverified_basis, "reference_from": reference_from,
        "reference_adapter": reference_adapter, "reference_times": list(reference_times),
        "outcome": outcome_stat, "starts": list(starts),
        "region": None if lat is None else {"lat": lat, "lon": lon, "radius_km": radius_km},
    }  # fmt: skip
    command = _steer_command(
        source, settings, result.spec, result.provenance["basis"]["sha256"], command="mediate"
    )
    extra = ["--mediator-layer", str(mediator_layer), "--mediator-feature", str(mediator_feature)]
    if mediator_basis:
        extra += ["--mediator-basis", mediator_basis,
                  "--mediator-basis-sha256", mediator.basis.meta["sha256"]]  # fmt: skip
    command = command + " " + shlex.join(extra)
    if out:
        save_result(out, result)
    if as_json:
        click.echo(json.dumps(jsonable({**result.to_dict(), "command": command}), indent=2))
        return
    click.echo(
        f"{mode} {amount:g} on feature {feature} of layer {layer}, through feature "
        f"{mediator_feature} of layer {mediator_layer}; the steer moved the mediator by "
        f"{result.mediator_change:.4g} (RMS); holding it alone changed the run by "
        f"{result.exactness:.3g}, and the hold missed its target by at most "
        f"{result.hold_error:.3g} ({result.hold_rms:.3g} RMS)\n"
    )
    if not result.held:
        click.echo(
            "the hold missed by a tenth or more of what the steer moved the mediator: the "
            "direct run did not hold it, and the share below says little\n"
        )
    rows = [
        {
            "field": name, "total": f"{e.total:.4g}", "direct": f"{e.direct:.4g}",
            "indirect": f"{e.indirect:.4g}",
            "share": "-" if e.share is None else f"{e.share:.1%}",
        }
        for name, e in result.effects.items()
    ]  # fmt: skip
    click.echo(_render.table(rows))
    click.echo(f"\nreproduce: {command}")


__all__ = ["latents"]
