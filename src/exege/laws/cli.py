"""``exege laws``: score a model against a physical law, and find what carries it.

A thin client of ``exege.laws``; it imports on a base install.

    exege laws rossby --adapter vit-interp --source best.pt --u U_3 --v V_3 \\
        --observe V_3 --starts 0,40,80,120 --search
"""

from __future__ import annotations

import click

from exege import _render
from exege.circuits.cli import open_system, parse_starts, system_options, write_json


def _wavenumbers(text: str) -> tuple[int, ...]:
    return tuple(int(s) for s in parse_starts(text))


def _names(text: str) -> tuple[str, ...]:
    return tuple(n.strip() for n in text.split(",") if n.strip())


def _report(system, base, law, do_search, taus, controls, out, score, by_block) -> None:
    from exege.laws.lawful import blocks, measure, search

    r = measure(system, base, law, score=score)
    meaning = (
        "1: follows the law; 0: no better than its mean"
        if score == "eq10"
        else ("correlation with the law across the family")
    )
    click.echo(f"{law.name}: M = {r.M:.4f}  ({meaning})")
    click.echo(_render.table(r.rows()))
    payload = {"measure": r.to_dict()}
    if do_search:
        ctl = [system.states(parse_starts(c)) for c in controls]
        s = search(
            system,
            base,
            law,
            taus=[float(t) for t in taus.split(",")],
            controls=ctl,
            score=score,
            groups=blocks(system) if by_block else None,
        )
        rows = [
            {
                "tau": c.tau,
                "size": len(c.components),
                "d_suf": c.deviations.d_suf,
                "d_nec": c.deviations.d_nec,
                "components": " ".join(c.units or c.components),
            }
            for c in s.circuits
        ]
        click.echo(_render.table(rows) if rows else "no tau-consistent circuit")
        payload["search"] = s.to_dict()
    write_json(out, payload)


def _common(command):
    for option in reversed(
        [
            click.option("--starts", default="0", show_default=True, help="Base states."),
            click.option("--phi0", type=float, default=45.0, show_default=True),
            click.option("--sigma", type=float, default=10.0, show_default=True),
            click.option(
                "--search",
                "do_search",
                is_flag=True,
                help="Also search for tau-consistent circuits.",
            ),
            click.option("--taus", default="0.3,0.5,0.7,0.9", show_default=True),
            click.option(
                "--control",
                "controls",
                multiple=True,
                help="Other starts for a source-randomized control; repeatable.",
            ),
            click.option("--out", type=click.Path(dir_okay=False), help="Write JSON here."),
            click.option(
                "--by-block",
                "by_block",
                is_flag=True,
                help="Search over whole blocks (8 candidates) instead of single heads and MLPs.",
            ),
            click.option(
                "--score",
                type=click.Choice(["eq10", "correlation"]),
                default="eq10",
                show_default=True,
                help="eq10: LAWFUL's misfit score; correlation: "
                "whether the model follows the law's shape across the family.",
            ),
        ]
    ):
        command = option(command)
    return command


@click.group(name="laws")
def laws() -> None:
    """Whether a model obeys a physical law, and which components carry it (LAWFUL)."""


@laws.command("rossby")
@system_options
@_common
@click.option("--u", "u", required=True, help="Zonal wind fields to plant in, comma separated.")
@click.option("--v", "v", required=True, help="Meridional wind fields, in the same order.")
@click.option("--observe", required=True, help="The meridional wind field to read.")
@click.option("--ps", help="Surface pressure field, for a geostrophically balanced wave.")
@click.option(
    "--u-bar",
    "u_bar",
    help="Zonal wind whose band mean is the flow the law "
    "Doppler-shifts by.  [default: the u at the observed level]",
)
@click.option("--wavenumbers", default="2-14", show_default=True)
@click.option("--amplitude", type=float, default=1.0, show_default=True, help="Peak v', m/s.")
@click.option("--dt", type=float, default=21600.0, show_default=True, help="Seconds per step.")
@click.option("--doppler", is_flag=True, help="Score the full frequency, not the intrinsic.")
def rossby_cmd(
    source,
    adapter,
    options,
    starts,
    phi0,
    sigma,
    do_search,
    taus,
    controls,
    out,
    score,
    by_block,
    u,
    v,
    observe,
    ps,
    u_bar,
    wavenumbers,
    amplitude,
    dt,
    doppler,
) -> None:
    """Barotropic Rossby waves: omega - k U = -beta k / (k^2 + l^2), over wavenumbers."""
    from exege.laws.law import Envelope
    from exege.laws.waves import Rossby

    system = open_system(source, adapter, options)
    law = Rossby(
        u=_names(u),
        v=_names(v),
        observe_field=observe,
        ps=ps,
        u_bar=u_bar,
        wavenumbers=_wavenumbers(wavenumbers),
        band=Envelope(phi0, sigma),
        amplitude=amplitude,
        intrinsic=not doppler,
        dt=dt,
    )
    _report(
        system,
        system.states(parse_starts(starts)),
        law,
        do_search,
        taus,
        controls,
        out,
        score,
        by_block,
    )


@laws.command("advection")
@system_options
@_common
@click.option("--field", required=True, help="The field a wave is planted in and read from.")
@click.option("--speed", type=float, required=True, help="Zonal speed at phi0, m/s.")
@click.option("--wavenumbers", default="1-6", show_default=True)
@click.option("--dt", type=float, default=21600.0, show_default=True, help="Seconds per step.")
@click.option("--radius", type=float, default=6.371e6, show_default=True)
def advection_cmd(
    source,
    adapter,
    options,
    starts,
    phi0,
    sigma,
    do_search,
    taus,
    controls,
    out,
    score,
    by_block,
    field,
    speed,
    wavenumbers,
    dt,
    radius,
) -> None:
    """A pattern carried by a uniform zonal flow: omega = k U."""
    from exege.laws.law import Envelope
    from exege.laws.waves import ZonalAdvection

    system = open_system(source, adapter, options)
    law = ZonalAdvection(
        field, speed, _wavenumbers(wavenumbers), Envelope(phi0, sigma), dt=dt, radius=radius
    )
    _report(
        system,
        system.states(parse_starts(starts)),
        law,
        do_search,
        taus,
        controls,
        out,
        score,
        by_block,
    )
