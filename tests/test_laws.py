"""LAWFUL on systems whose law is planted."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from exege.circuits import open_patchable  # noqa: E402
from exege.core.errors import RequestError  # noqa: E402
from exege.latents.grid import Grid  # noqa: E402
from exege.laws import (  # noqa: E402
    Envelope,
    Rossby,
    ZonalAdvection,
    components,
    consistency,
    deviations,
    measure,
    search,
)
from exege.laws.law import DAY, phase_rate, projection  # noqa: E402


@pytest.fixture(scope="module")
def toy():
    return open_patchable(None, "toy-transformer", n_lat=3, n_lon=24)


def _advection(**kw):
    # the toy's planted law: courant 0.5 moves a wave half a cell per step, at every s
    dlam = np.deg2rad(360 / 24)
    return ZonalAdvection(
        "tracer",
        speed=dlam / 2,
        wavenumbers=(1, 2, 3, 4, 5, 6),
        band=Envelope(0, 40),
        dt=1.0,
        radius=1.0,
        **kw,
    )


def test_consistency_is_lawfuls_eq_10():
    assert consistency([1, 2, 3], [1, 2, 3]) == 1.0
    assert consistency([2, 2, 2], [1, 2, 3]) == pytest.approx(0.0)  # the mean, no better
    assert consistency([3, 2, 1], [1, 2, 3]) == pytest.approx(-3.0)
    with pytest.raises(RequestError, match="same value"):
        consistency([1, 2], [5, 5])


def test_the_toy_obeys_its_planted_law_exactly(toy):
    r = measure(toy, toy.states([0, 1, 2]), _advection())
    assert pytest.approx(1.0, abs=1e-9) == r.M
    assert r.patched is None and len(r.rows()) == 6


def test_the_advecting_head_is_sufficient_and_necessary(toy):
    base = toy.states([0, 1, 2])
    d = deviations(toy, base, _advection(), ["a1.h0"], controls=[toy.states([5, 6, 7])])
    assert d.d_suf == pytest.approx(0.0, abs=1e-9)  # routing through a1.h0 alone recovers M0
    assert d.d_nec > 1.0  # routing through everything else does not
    assert d.consistent(0.9)
    assert d.random_suf[0] < d.M_suf  # a mismatched source does not recover it


def test_the_search_finds_the_planted_component_at_every_tau(toy):
    s = search(toy, toy.states([0, 1, 2]), _advection(), taus=(0.3, 0.6, 0.9))
    assert [c.components for c in s.circuits] == [("a1.h0",)] * 3
    assert s.of == len(components(toy)) and "embed" not in components(toy)


def test_patching_the_embedding_is_refused(toy):
    with pytest.raises(RequestError, match="embedding"):
        measure(toy, toy.states([0]), _advection(), ["embed"])


def test_a_model_the_law_does_not_hold_for_is_refused_a_search(toy):
    wrong = _advection()
    wrong = ZonalAdvection(
        "tracer",
        speed=-5 * wrong.speed,
        wavenumbers=wrong.wavenumbers,
        band=wrong.band,
        dt=1.0,
        radius=1.0,
    )
    with pytest.raises(RequestError, match="not consistent"):
        search(toy, toy.states([0, 1]), wrong)


# -- Rossby ------------------------------------------------------------------------------------


def _globe(n_lat=60, n_lon=120):
    lat = np.linspace(-89, 89, n_lat)
    lon = np.arange(n_lon) * 360.0 / n_lon
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    return Grid(lat=LA.ravel(), lon=LO.ravel(), shape=(n_lat, n_lon))


def _rossby(**kw):
    return Rossby(u=("U",), v=("V",), observe_field="V", ps="PS", band=Envelope(45, 10), **kw)


def test_the_planted_wave_is_balanced_and_sized():
    g, law = _globe(), _rossby()
    d = law.perturbation(g, {}, 6)
    assert np.abs(d["V"]).max() == pytest.approx(1.0, rel=0.05)
    # geostrophy: |p'| = rho f |v| / k at the band's center
    k = 6 / (law.radius * np.cos(np.deg2rad(45)))
    f = 2 * 7.292e-5 * np.sin(np.deg2rad(45))
    assert np.abs(d["PS"]).max() == pytest.approx(law.rho_s * f / k, rel=0.1)


def test_a_wave_moving_at_the_law_scores_one_and_the_doppler_shift_is_taken_out():
    g = _globe()
    U = 15.0
    fields = {"U": np.full((1, g.n_nodes), U), "V": np.zeros((1, g.n_nodes))}
    for intrinsic in (True, False):
        law = _rossby(intrinsic=intrinsic)
        model, expected = [], []
        for s in law.alphas:
            planted = law.perturbation(g, fields, s)["V"]
            # move the planted wave by the full (Doppler-shifted) dispersion relation
            omega = _rossby(intrinsic=False).expect(g, fields, s) / DAY
            moved = Rossby(u=("U",), v=("V",), observe_field="V", band=Envelope(45, 10))
            shift = omega * law.dt / s  # radians of longitude; v ~ sin(s lon - omega t)
            lon = np.deg2rad(g.lon)
            ratio = moved.perturbation(g, fields, s)["V"]
            assert np.allclose(ratio, planted)
            G = np.exp(-(((g.lat - 45) / 10) ** 2))
            A = law.amplitude * law.radius * np.cos(np.deg2rad(45)) / s
            after = (
                -(A * s / (law.radius * np.cos(np.deg2rad(g.lat)))) * G * np.sin(s * (lon - shift))
            )
            response = {"V": (after - planted)[None]}
            model.append(law.observe(g, fields, {"V": planted}, response, s))
            expected.append(law.expect(g, fields, s))
        assert consistency(model, expected) == pytest.approx(1.0, abs=1e-9)
    # the intrinsic phase speed is westward, and fastest for the longest waves
    law = _rossby()
    c = [law.expect(g, fields, s) / s for s in law.alphas]  # per unit wavenumber
    assert all(x < 0 for x in c) and all(abs(a) > abs(b) for a, b in zip(c, c[1:], strict=False))


def test_phase_rate_reads_a_rotation_and_a_growth():
    w, gr = phase_rate(1 + 0j, np.exp(-1j * 0.3 + 0.1), 2.0)
    assert w == pytest.approx(0.15) and gr == pytest.approx(0.05)
    g = _globe()
    band = Envelope(45, 10)
    v = band(g.lat) * np.cos(3 * np.deg2rad(g.lon))
    assert abs(projection(g, v, 3, band)) > 0 and abs(projection(g, v, 4, band)) < 1e-9


def test_the_advection_command_scores_and_searches_the_toy():
    from click.testing import CliRunner

    from exege._cli import cli

    args = [
        "laws",
        "advection",
        "--adapter",
        "toy-transformer",
        "--option",
        "n_lon=24",
        "--field",
        "tracer",
        "--speed",
        str(np.deg2rad(15) / 2),
        "--wavenumbers",
        "1-6",
        "--phi0",
        "0",
        "--sigma",
        "40",
        "--dt",
        "1",
        "--radius",
        "1",
        "--starts",
        "0-2",
        "--search",
        "--taus",
        "0.9",
    ]
    r = CliRunner().invoke(cli, args)
    assert r.exit_code == 0, r.output
    assert "M = 1.0000" in r.output and "a1.h0" in r.output
