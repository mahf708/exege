"""Associations against a rotation null: planted answers on a source held in memory."""

from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")

from click.testing import CliRunner  # noqa: E402

from exege._cli import cli  # noqa: E402
from exege.core.errors import RequestError  # noqa: E402
from exege.latents import (  # noqa: E402
    LatentInfo,
    LayerInfo,
    associate,
    benjamini_hochberg,
    feature_profile,
)
from exege.latents.calibrate import _empirical_p, _label, _leave_one_out_gaps  # noqa: E402
from exege.latents.grid import Grid  # noqa: E402

N_LAT, N_LON, N_TIMES, N_RANDOM = 12, 36, 8, 22
WAVE, ZONAL = 0, 1  # the two planted channels; the rest fire at random


def _smooth(rng, lat, lon):
    """A field smooth along each row and broad in wavenumber, with its own phases in
    every row: many degrees of freedom, so a turn of 30 degrees or more decorrelates it.
    (A field of two or three wavenumbers comes back nearly onto itself under some
    turns, and a rotation null then has little power: honestly so.)"""
    rows = np.unique(lat)
    phases = rng.uniform(0, 2 * np.pi, (rows.size, 12))
    row = np.searchsorted(rows, lat)
    out = np.zeros(lat.size)
    for k in range(1, 13):
        out += np.cos(k * np.radians(lon) + phases[row, k - 1]) / np.sqrt(k)
    return out


class FieldSource:
    """A ``LatentSource`` with ``ReferenceFields``, on a regular grid of 10-degree columns.

    ``wave`` moves from time to time, and channel 0 fires where it is high; ``zonal``
    depends on latitude only, and channel 1 fires where it is high (a real association
    that a rotation cannot see); ``noise`` is unrelated to anything; ``twin`` is ``wave``
    and a little noise. Channels 2 onward fire at random nodes.
    """

    def __init__(self, mask=None, holes=False, n_lon=N_LON):
        lat, lon = np.meshgrid(
            np.linspace(-82.5, 82.5, N_LAT), np.arange(n_lon) * (360.0 / n_lon), indexing="ij"
        )
        lat, lon = lat.ravel(), lon.ravel()
        self._grid = Grid(lat=lat, lon=lon, shape=(N_LAT, n_lon), mask=mask)
        rng = np.random.default_rng(1)
        n = lat.size
        self.reads = 0
        self._fields = {name: [] for name in ("wave", "zonal", "noise", "twin")}
        latents = []
        zonal = np.sin(np.radians(lat))
        for t in range(N_TIMES):
            wave = _smooth(rng, lat, lon)
            fields = {
                "wave": wave,
                "zonal": zonal + 0.01 * rng.normal(size=n),
                "noise": rng.normal(size=n),
                "twin": wave + 0.05 * rng.normal(size=n),
            }
            if holes and t == 0:
                fields["wave"] = fields["wave"].copy()
                fields["wave"][: N_LON // 2] = np.nan  # missing over part of a row
            for name, values in fields.items():
                self._fields[name].append(values)
            h = np.zeros((n, 2 + N_RANDOM), dtype=np.float32)
            h[:, WAVE] = np.maximum(wave - np.quantile(wave, 0.8), 0.0)
            h[:, ZONAL] = np.maximum(zonal - 0.6, 0.0)
            h[:, 2:] = (rng.uniform(size=(n, N_RANDOM)) > 0.85) * rng.uniform(
                0.5, 1.0, size=(n, N_RANDOM)
            )
            latents.append(h)
        self._latents = np.stack(latents)
        self._info = LatentInfo(
            source="fields",
            times=tuple(f"t{i}" for i in range(N_TIMES)),
            layers=(LayerInfo(0, "only", 2 + N_RANDOM),),
            n_nodes=n,
            model="mem",
            component="mem",
            checkpoint="mem.ckpt",
        )

    def info(self):
        return self._info

    def grid(self):
        return self._grid

    def load(self, time, layer, channels=None, nodes=None):
        self.reads += 1
        out = self._latents[self._info.time_index(time)]
        if nodes is not None:
            out = out[list(nodes)]
        if channels is not None:
            out = out[:, list(channels)]
        return out.copy()

    def field_names(self):
        return tuple(self._fields)

    def field(self, name, time, lead=0):
        return self._fields[name][self._info.time_index(time) + lead].copy()


def test_benjamini_hochberg_keeps_what_clears_its_rising_bar():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, np.nan])
    kept = benjamini_hochberg(p, q=0.05)
    assert kept.tolist() == [True, True, False, False, False, False, False, False, False]
    assert not benjamini_hochberg(np.array([np.nan, np.nan])).any()
    # step-up, not each rank on its own: 0.03 fails its own bar (0.025) but the larger
    # 0.04 clears its bar (0.05), and carries the smaller one with it
    assert benjamini_hochberg(np.array([0.03, 0.04]), q=0.05).tolist() == [True, True]


def test_a_planted_feature_is_labeled_with_its_field_and_chance_with_nothing():
    result = associate(FieldSource(), layer=0, fields=["wave", "zonal", "noise"])
    by_column = dict(zip(result.columns, result.label, strict=True))
    assert by_column[WAVE] == "wave"
    wave = result.fields.index("wave")
    assert result.z[WAVE, wave] > 5 and result.p[WAVE, wave] < 1e-6
    # an empirical p cannot go below one in n + 1
    assert result.p_empirical[WAVE, wave] == pytest.approx(1 / (len(result.shifts) + 1))
    labeled_at_random = [by_column[c] for c in range(2, 2 + N_RANDOM)]
    assert sum(name != "none" for name in labeled_at_random) <= 2
    assert result.calibration_label_rate <= 0.15
    assert result.summary()["columns"][0]["column"] == WAVE
    assert set(result.null_shape) == {"median_abs_skew", "median_excess_kurtosis"}


def test_a_purely_zonal_association_reads_null_although_it_is_real():
    """The documented limit: every rotation keeps a latitude-only alignment."""
    result = associate(FieldSource(), layer=0, columns=[ZONAL], fields=["zonal", "noise"])
    zonal = result.fields.index("zonal")
    assert result.statistic[0, zonal] > 1.0  # the raw association is large
    assert result.label[0] != "zonal"
    assert abs(result.z[0, zonal]) < 3 or not np.isfinite(result.z[0, zonal])


def test_two_fields_that_go_together_leave_the_feature_ambiguous():
    result = associate(FieldSource(), layer=0, columns=[WAVE], fields=["wave", "twin", "noise"])
    assert result.label == ("ambiguous",)
    assert result.gap[0] < result.gap_bar[0]


def test_masked_nodes_and_missing_values_are_left_out():
    mask = np.ones(N_LAT * N_LON, dtype=bool)
    mask[-N_LON:] = False  # the last row means nothing
    source = FieldSource(mask=mask, holes=True)
    source._latents[:, ~mask, :] = np.nan  # a masked node may hold anything
    result = associate(source, layer=0, fields=["wave", "noise"])
    assert result.label[WAVE] == "wave"
    assert np.isfinite(result.coverage).all()


def test_requests_are_refused_before_anything_is_read():
    source = FieldSource()
    for kwargs, message in [
        ({"fields": ["rain"]}, "no field"),
        ({"columns": [99]}, "no channel 99"),
        ({"columns": []}, "no channels"),
        ({"q": 1.5}, "false-discovery"),
        ({"n_rotations": 2}, "at least 4"),
        ({"min_shift_degrees": 179}, "rotation"),
    ]:
        with pytest.raises(RequestError, match=message):
            associate(source, layer=0, **kwargs)
    assert source.reads == 0
    mesh = FieldSource()
    mesh._grid = Grid(lat=mesh._grid.lat, lon=mesh._grid.lon)  # the same nodes, no shape
    with pytest.raises(RequestError, match="mesh"):
        associate(mesh, layer=0)
    assert mesh.reads == 0


def test_a_profile_can_measure_each_effect_against_its_rotations():
    source = FieldSource()
    plain = feature_profile(source, layer=0, column=WAVE)
    assert plain.null_z is None and plain.p_value is None
    rotated = feature_profile(source, layer=0, column=WAVE, rotations=16)
    assert rotated.fields == plain.fields
    np.testing.assert_allclose(rotated.effect, plain.effect)
    by_field = dict(zip(rotated.fields, rotated.p_value, strict=True))
    assert by_field["wave"] < 1e-4 and by_field["noise"] > 1e-3
    summary = rotated.summary()["fields"][0]
    assert {"null_z", "p_value"} <= set(summary)


def test_cli_associate(tmp_path):
    archive = tmp_path / "toy"
    runner = CliRunner()
    made = runner.invoke(cli, ["latents", "toy", str(archive)])
    assert made.exit_code == 0, made.output
    out = tmp_path / "assoc.json"
    result = runner.invoke(
        cli,
        ["latents", "associate", str(archive), "--layer", "3", "--channel", "5",
         "--channel", "0", "--threshold", "1", "--rotations", "12", "--out", str(out)],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "calibration" in result.output and "precipitation" in result.output
    data = json.loads(out.read_text())
    assert data["columns"][0]["column"] == 5
    assert data["columns"][0]["label"] == "precipitation"
    as_json = runner.invoke(
        cli,
        [
            "latents",
            "associate",
            str(archive),
            "--channel",
            "5",
            "--threshold",
            "1",
            "--rotations",
            "8",
            "--json",
        ],
    )
    assert json.loads(as_json.output)["settings"]["n_rotations"] == 8
    profiled = runner.invoke(
        cli, ["latents", "profile", str(archive), "--channel", "5", "--rotations", "8"]
    )
    assert profiled.exit_code == 0, profiled.output
    assert "NULL_Z" in profiled.output


# -- the numbers, by hand ------------------------------------------------------


class TinySource:
    """Two rows of six columns at +-45 degrees (equal areas), one time, one channel.
    ``field`` and ``active`` are per node; ``mask`` marks valid nodes."""

    def __init__(self, field, active, mask=None, n_lon=6):
        lat, lon = np.meshgrid([-45.0, 45.0], np.arange(n_lon) * (360.0 / n_lon), indexing="ij")
        self._grid = Grid(lat=lat.ravel(), lon=lon.ravel(), shape=(2, n_lon), mask=mask)
        self._field = np.asarray(field, dtype=np.float64)
        self._h = np.asarray(active, dtype=np.float32)[:, None]
        self._info = LatentInfo(
            source="tiny", times=("t0",), layers=(LayerInfo(0, "only", 1),),
            n_nodes=2 * n_lon, model="m", component="c", checkpoint="k",
        )  # fmt: skip

    def info(self):
        return self._info

    def grid(self):
        return self._grid

    def load(self, time, layer, channels=None, nodes=None):
        out = self._h if nodes is None else self._h[list(nodes)]
        return out.copy()

    def field_names(self):
        return ("f",)

    def field(self, name, time, lead=0):
        return self._field.copy()


def test_the_statistic_and_its_null_are_what_a_hand_computation_gives():
    field = np.array([1.0, 4.0, 2.0, 8.0, 5.0, 7.0, 3.0, 6.0, 0.0, 9.0, 2.0, 1.0])
    active = np.zeros(12)
    active[[1, 9]] = [1.0, 2.0]  # column 1 of each row; the weights are the area, not this
    result = associate(TinySource(field, active), layer=0, n_rotations=5, min_shift_degrees=0)
    assert result.shifts == (1, 2, 3, 4, 5)
    standard = (field - field.mean()) / field.std()  # equal areas: a plain mean and spread
    assert result.statistic[0, 0] == pytest.approx(standard[[1, 9]].mean())
    # shift k reads the node k columns east of each active node
    null = np.array([standard[[(1 + k) % 6, 6 + (3 + k) % 6]].mean() for k in range(1, 6)])
    assert result.null_mean[0, 0] == pytest.approx(null.mean())
    assert result.null_sd[0, 0] == pytest.approx(null.std(ddof=1))
    expected_z = (result.statistic[0, 0] - null.mean()) / null.std(ddof=1)
    assert result.z[0, 0] == pytest.approx(expected_z)
    assert result.n_null[0, 0] == 5


def test_the_gap_bar_leaves_each_draw_out_of_its_own_null():
    rng = np.random.default_rng(3)
    null = rng.normal(size=(3, 2, 7))
    gaps = _leave_one_out_gaps(null)

    def gap(z):
        size = np.sort(np.abs(z))[::-1]
        return size[0] - size[1]

    for c in range(3):
        for r in range(7):
            rest = np.delete(null[c], r, axis=-1)
            z = (null[c, :, r] - rest.mean(axis=-1)) / rest.std(axis=-1, ddof=1)
            assert gaps[c, r] == pytest.approx(gap(z))
            # standardized against a null that holds it, the same draw reads otherwise
            inside = (null[c, :, r] - null[c].mean(axis=-1)) / null[c].std(axis=-1, ddof=1)
            assert gaps[c, r] != pytest.approx(gap(inside))
    labels = _label(null[:, :, 0], null[:, :, 1:], 0.05)
    assert labels.gap_bar.shape == (3,)


def test_the_empirical_p_is_a_rank_over_the_data_and_its_finite_rotations():
    p, n = _empirical_p(np.array([3.0]), np.array([[-3.0, 0.0, 0.0, 0.0]]))
    assert p[0] == pytest.approx(2 / 5) and n[0] == 4
    p, n = _empirical_p(np.array([1.0]), np.array([[np.nan, 0.5, np.nan, 0.2]]))
    assert n[0] == 2 and p[0] == pytest.approx(1 / 3)  # missing rotations are not draws
    p, _ = _empirical_p(np.array([5.0]), np.array([[0.0, 0.0, 0.0]]))
    assert p[0] == pytest.approx(1 / 4)  # a constant null still ranks a spike
    p, _ = _empirical_p(np.array([0.3]), np.array([[0.3, 0.3 + 1e-15, 0.3]]))
    assert p[0] == pytest.approx(1.0)  # rounding is a tie, not a discovery


def test_the_calibration_runs_the_procedure_it_vouches_for():
    """A single wave and a narrow band of shifts around half way round: the rotations see
    much the same thing, so the null varies little and features are labeled. The
    calibration, run against rotations of its own rotation that stay as far from the data,
    must say so."""

    class Wave(TinySource):
        def __init__(self):
            n_lon, n_feat = 72, 36
            lat, lon = np.meshgrid(np.linspace(-75, 75, 6), np.arange(n_lon) * 5.0, indexing="ij")
            self._grid = Grid(lat=lat.ravel(), lon=lon.ravel(), shape=(6, n_lon))
            self._field = np.cos(np.radians(lon.ravel()))
            phases = np.arange(n_feat) * 2 * np.pi / n_feat
            wave = np.cos(np.radians(lon.ravel())[:, None] - phases[None, :])
            self._h = np.maximum(wave - 0.5, 0.0).astype(np.float32)
            self._info = LatentInfo(
                source="wave", times=("t0",), layers=(LayerInfo(0, "only", n_feat),),
                n_nodes=lat.size, model="m", component="c", checkpoint="k",
            )  # fmt: skip

    result = associate(Wave(), layer=0, min_shift_degrees=80)
    assert result.n_labeled > 9
    assert result.calibration_label_rate > 0.1  # the old held-out draw read 0 here
    assert result.calibration_shift == 36
    # so narrow a band that the calibration could only come back to the data: refused
    with pytest.raises(RequestError, match="calibration"):
        associate(Wave(), layer=0, min_shift_degrees=150)


def test_a_field_the_same_everywhere_carries_nothing():
    active = np.zeros(12)
    active[[1, 9]] = 1.0
    result = associate(TinySource(np.full(12, 0.1), active), layer=0, n_rotations=5,
                       min_shift_degrees=0)  # fmt: skip
    assert np.isnan(result.statistic[0, 0]) and result.label == ("none",)


def test_a_rotation_reads_nothing_at_a_masked_partner():
    """A longitudinal mask: whatever a masked node holds must not reach the null."""
    mask = np.ones(N_LAT * N_LON, dtype=bool)
    mask[np.arange(N_LAT) * N_LON + 7] = False  # one column, every row
    mask[3 * N_LON + 20 : 3 * N_LON + 26] = False  # and a stretch of one row
    found = {}
    for fill in (0.0, 1e6):
        source = FieldSource(mask=mask)
        for values in source._fields["wave"]:
            values[~mask] = fill
        profile = feature_profile(source, layer=0, column=WAVE, fields=["wave"], rotations=8)
        result = associate(source, layer=0, columns=[WAVE], fields=["wave"], n_rotations=8)
        found[fill] = (profile.null_z[0], result.z[0, 0], result.null_sd[0, 0])
    np.testing.assert_allclose(found[0.0], found[1e6])


def test_a_profile_of_a_field_of_latitude_alone_has_no_z_score():
    source = FieldSource()
    source._fields["zonal"] = [np.sin(np.radians(source._grid.lat)) for _ in range(N_TIMES)]
    profile = feature_profile(source, layer=0, column=ZONAL, fields=["zonal"], rotations=6)
    assert np.isnan(profile.null_z[0]) and np.isnan(profile.p_value[0])


@pytest.mark.parametrize("n_lon", [360, 180, 288, 128, 1440])
@pytest.mark.parametrize("min_degrees", [0.0, 30.0])
def test_the_calibration_null_never_comes_back_to_the_data(n_lon, min_degrees):
    from exege.latents.calibrate import calibration_shifts
    from exege.latents.grid import rotation_shifts
    from exege.latents.toy import toy_grid

    grid = toy_grid(n_lat=4, n_lon=n_lon)
    shifts = rotation_shifts(grid, 64, min_degrees)
    held, again = calibration_shifts(shifts, n_lon, min_degrees)
    spacing = 360.0 / n_lon
    assert abs(held * spacing - 180.0) <= spacing * max(1, n_lon // 64)
    assert len(again) >= 4 and 0 not in again
    assert all(min(a, n_lon - a) * spacing >= min_degrees - 1e-9 for a in again)


# -- the orbit null ------------------------------------------------------------------


class _Smooth(TinySource):
    """12 rows by 72 columns, one field and many channels, each a sum of random
    east-west waves up to wavenumber ``kmax``; nothing goes with anything."""

    def __init__(self, rng, kmax, n_channels=40, field=None, h=None, mask=None):
        lat, lon = np.meshgrid(np.linspace(-80, 80, 12), np.arange(72) * 5.0, indexing="ij")
        self._grid = Grid(lat=lat.ravel(), lon=lon.ravel(), shape=(12, 72), mask=mask)

        def smooth():
            out = np.zeros(lat.shape)
            for k in range(1, kmax + 1):
                phase = rng.uniform(0, 2 * np.pi, (12, 1))
                out += rng.normal(size=(12, 1)) * np.cos(np.radians(k * lon) + phase)
            return out.ravel()

        self._field = smooth() if field is None else field
        self._h = (np.stack([smooth() for _ in range(n_channels)], 1) if h is None else h).astype(
            np.float32
        )
        self._info = LatentInfo(
            source="smooth", times=("t0",), layers=(LayerInfo(0, "x", self._h.shape[1]),),
            n_nodes=lat.size, model="m", component="c", checkpoint="k",
        )  # fmt: skip


def test_the_orbit_reads_the_same_statistic_as_the_rotations():
    source = _Smooth(np.random.default_rng(0), 4)
    rotations, orbit = associate(source, layer=0), associate(source, layer=0, null="orbit")
    np.testing.assert_allclose(orbit.statistic, rotations.statistic, atol=1e-7)
    np.testing.assert_allclose(orbit.null_mean, rotations.null_mean, atol=1e-7)
    assert orbit.settings["null"] == "orbit" and orbit.calibration_rotations == 72
    assert orbit.feature_p is not None and rotations.feature_p is None


@pytest.mark.parametrize("kmax", [1, 4, 30])
def test_the_orbit_labels_independent_columns_at_about_q(kmax):
    """Wave-one, smooth and rough fields alike: the share of columns labeled when nothing
    goes with anything stays near q (here, pooled over 10 independent draws)."""
    rng = np.random.default_rng(kmax)
    labeled = [associate(_Smooth(rng, kmax), layer=0, null="orbit").n_labeled for _ in range(10)]
    assert np.mean(labeled) / 40 <= 0.09


def test_the_orbit_finds_a_planted_association():
    rng = np.random.default_rng(5)
    source = _Smooth(rng, 4, n_channels=8)
    h = source._h.copy()
    h[:, 0] = np.maximum(source._field, 0.0) + 0.1 * rng.normal(size=source._field.size)
    planted = _Smooth(rng, 4, field=source._field, h=h)
    result = associate(planted, layer=0, null="orbit")
    assert result.label[0] == "f" and result.feature_p[0] <= 0.05


def test_the_orbit_respects_the_mask():
    rng = np.random.default_rng(6)
    mask = np.ones(12 * 72, dtype=bool)
    mask[:200] = False
    source = _Smooth(rng, 4, mask=mask)
    plain = _Smooth(np.random.default_rng(6), 4, mask=mask)
    np.testing.assert_allclose(
        associate(source, layer=0, null="orbit").statistic,
        associate(plain, layer=0).statistic,
        atol=1e-7,
    )


def test_an_unknown_null_is_refused():
    with pytest.raises(RequestError, match="null must be"):
        associate(_Smooth(np.random.default_rng(0), 2), layer=0, null="gaussian")
