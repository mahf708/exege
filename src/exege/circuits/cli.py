"""``exege circuits``: find and score circuits from the command line.

A thin client of ``exege.circuits``; it imports on a base install, and numpy-backed
modules load inside the commands.

    exege circuits prune --adapter toy-transformer --field tracer --lat 0 --lon 90 \\
        --radius-km 4000 --base 0,1,2 --source 3,4,5
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from exege import _render


def parse_options(pairs: tuple[str, ...]) -> dict[str, Any]:
    """``key=value`` pairs for an adapter; values are read as JSON when they parse
    (numbers, true/false, lists), else kept as text."""
    out: dict[str, Any] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise click.BadParameter(f"expected key=value, got {pair!r}", param_hint="--option")
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def parse_starts(text: str) -> list[str | int]:
    """``0,40,80`` or ``0-3`` or time labels, comma separated."""
    starts: list[str | int] = []
    for part in text.split(","):
        part = part.strip()
        if part.isdigit():
            starts.append(int(part))
        elif "-" in part and all(p.isdigit() for p in part.split("-", 1)):
            lo, hi = (int(p) for p in part.split("-", 1))
            starts.extend(range(lo, hi + 1))
        elif part:
            starts.append(part)
    if not starts:
        raise click.BadParameter("no starts", param_hint="--base")
    return starts


def system_options(command):
    for option in reversed(
        [
            click.option("--source", "source", help="What the adapter opens (a checkpoint, say)."),
            click.option("--adapter", required=True, help="A Patchable adapter, by name."),
            click.option("--option", "options", multiple=True, help="key=value for the adapter."),
        ]
    ):
        command = option(command)
    return command


def open_system(source, adapter, options):
    from exege.circuits.contract import open_patchable

    return open_patchable(source, adapter, **parse_options(options))


def write_json(path: str | None, payload: dict[str, Any]) -> None:
    if path:
        Path(path).write_text(json.dumps(payload, indent=1, default=float))
        click.echo(f"wrote {path}")


def _metric(system, field, lat, lon, radius_km):
    from exege.circuits.metrics import field_mean

    grid = system.grid()
    where = None if lat is None else grid.within(lat, lon, radius_km)
    return field_mean(grid, field, where=where)


def _metric_options(command):
    for option in reversed(
        [
            click.option("--field", required=True, help="The field whose mean is the behaviour."),
            click.option("--lat", type=float, help="Region center; without it, the whole globe."),
            click.option("--lon", type=float, default=0.0, show_default=True),
            click.option("--radius-km", type=float, default=1500.0, show_default=True),
            click.option("--base", "base", default="0", show_default=True, help="Base starts."),
            click.option(
                "--source-starts",
                "source_starts",
                required=True,
                help="Counterfactual starts, as many as --base.",
            ),
            click.option("--out", "out", type=click.Path(dir_okay=False), help="Write JSON here."),
        ]
    ):
        command = option(command)
    return command


@click.group(name="circuits")
def circuits() -> None:
    """Which edges between heads and MLPs a behaviour runs through (Vi-CD)."""


@circuits.command("graph")
@system_options
def graph_cmd(source, adapter, options) -> None:
    """The system's graph: senders, receivers, edges."""
    g = open_system(source, adapter, options).graph()
    click.echo(
        _render.pairs(
            {"senders": len(g.senders), "receivers": len(g.receivers), "edges": g.n_edges}
        )
    )
    click.echo("senders:   " + " ".join(g.senders))
    click.echo("receivers: " + " ".join(g.receivers))


@circuits.command("prune")
@system_options
@_metric_options
@click.option(
    "--tau",
    type=float,
    default=0.01,
    show_default=True,
    help="Drop an edge when the metric moves less than this share of the gap.",
)
def prune_cmd(
    source, adapter, options, field, lat, lon, radius_km, base, source_starts, out, tau
) -> None:
    """Greedy pruning by activation patching: the circuit of a regional field mean."""
    from exege.circuits.discover import prune

    system = open_system(source, adapter, options)
    metric = _metric(system, field, lat, lon, radius_km)
    c = prune(
        system,
        system.states(parse_starts(base)),
        system.states(parse_starts(source_starts)),
        metric,
        tau=tau,
    )
    click.echo(
        _render.pairs(
            {
                "edges": f"{c.n_edges} of {c.graph.n_edges}",
                "recovery": f"{c.recovery:.4f}",
                "clean": c.clean,
                "corrupt": c.corrupt,
                "circuit": c.value,
            }
        )
    )
    for name in c.edges():
        click.echo(f"  {name}")
    write_json(out, c.to_dict())


@circuits.command("attribute")
@system_options
@_metric_options
@click.option(
    "--method", type=click.Choice(["auto", "patch", "eap"]), default="auto", show_default=True
)
@click.option("--top", type=int, default=20, show_default=True)
def attribute_cmd(
    source, adapter, options, field, lat, lon, radius_km, base, source_starts, out, method, top
) -> None:
    """Score every edge: the metric's change when it alone carries the counterfactual."""
    from exege.circuits.discover import attribute

    system = open_system(source, adapter, options)
    metric = _metric(system, field, lat, lon, radius_km)
    a = attribute(
        system,
        system.states(parse_starts(base)),
        system.states(parse_starts(source_starts)),
        metric,
        method=method,
    )
    click.echo(f"method {a.method}; clean {a.clean:.6g}, corrupt {a.corrupt:.6g}")
    click.echo(_render.table([{"edge": e, "score": s} for e, s in a.ranked(top)]))
    write_json(out, a.to_dict())
