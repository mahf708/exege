"""How much of a column is fixed in place, and what latitude and the statics explain."""

from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")

from click.testing import CliRunner  # noqa: E402

from conftest import MemorySource, write_latent_archive  # noqa: E402
from exege._cli import cli  # noqa: E402
from exege.core.errors import RequestError  # noqa: E402
from exege.latents import feature_stationarity, save_basis  # noqa: E402
from exege.latents.basis import Dictionary  # noqa: E402

N_LAT, N_LON, N_TIMES = 12, 24, 20
N = N_LAT * N_LON


class WithFields(MemorySource):
    """A memory source that also keeps fields: ``land``, the same at every time, and
    ``tide``, which is not."""

    def __init__(self, data, fields, **kwargs):
        super().__init__(data, **kwargs)
        self._fields = fields

    def field_names(self):
        return tuple(self._fields)

    def field(self, name, time, lead=0):
        values = self._fields[name]
        return np.array(values[self._info.time_index(time)] if values.ndim == 2 else values)


@pytest.fixture
def planted():
    """Channel 0: one random map at every time, on a large offset. Channel 1: a fixed
    latitude profile. Channel 2: the static field ``land``. Channel 3: new noise at every
    time, so its time mean holds 1 / N_TIMES of its variance."""
    rng = np.random.default_rng(0)
    lat = np.repeat(np.linspace(-75.0, 75.0, N_LAT), N_LON)
    pattern, land = rng.normal(size=N), rng.normal(size=N)
    data = np.zeros((N_TIMES, N, 4))
    data[:, :, 0] = pattern + 1e4
    data[:, :, 1] = 3.0 * np.cos(np.radians(lat))
    data[:, :, 2] = land
    data[:, :, 3] = rng.normal(size=(N_TIMES, N))
    fields = {"land": land, "tide": rng.normal(size=(N_TIMES, N))}
    return WithFields({0: data.astype(np.float32)}, fields, n_lat=N_LAT, n_lon=N_LON), land


def test_the_planted_channels(planted):
    source, _ = planted
    result = feature_stationarity(source, layer=0)
    fixed, zonal, residual = result.fixed_share, result.zonal_share, result.residual_fixed_share
    assert fixed[:3] == pytest.approx(1.0, abs=1e-6)  # the offset costs no precision
    assert fixed[3] == pytest.approx(1 / N_TIMES, abs=0.03)
    assert zonal[1] == pytest.approx(1.0) and residual[1] == pytest.approx(0.0, abs=1e-9)
    # A random map is explained by twelve rows by chance only, and so is unexplained here.
    assert residual[0] > 0.85 and residual[2] > 0.85
    assert list(result.columns[result.ranked(top=2)]) in ([0, 2], [2, 0])
    assert result.mean[0] == pytest.approx(1e4, abs=0.2)
    assert result.statics == () and result.static_fixed_share.size == 0


def test_a_static_explains_what_it_is(planted):
    source, _ = planted
    result = feature_stationarity(source, layer=0, statics=["land"], chunk=3)
    assert result.static_share[2] > 0.9
    assert result.residual_fixed_share[2] == pytest.approx(0.0, abs=1e-9)
    assert result.residual_fixed_share[0] > 0.85  # a map land does not explain stays
    assert result.static_fixed_share == pytest.approx([1.0])
    assert result.columns[result.ranked(top=1)][0] == 0


def test_chunks_and_columns_do_not_change_the_answer(planted):
    source, _ = planted
    whole = feature_stationarity(source, layer=0, statics=["land"])
    for chunk in (1, 2):
        parts = feature_stationarity(source, layer=0, statics=["land"], chunk=chunk)
        assert parts.residual_fixed_share == pytest.approx(whole.residual_fixed_share)
    some = feature_stationarity(source, layer=0, columns=[3, 1])
    assert list(some.columns) == [3, 1]
    assert some.fixed_share[1] == pytest.approx(1.0)


def test_a_field_that_varies_says_so(planted):
    source, _ = planted
    result = feature_stationarity(source, layer=0, statics=["tide"])
    assert result.static_fixed_share[0] < 0.2


def test_a_mask_is_respected():
    rng = np.random.default_rng(1)
    mask = np.ones(N, dtype=bool)
    mask[:40] = False
    data = np.zeros((N_TIMES, N, 1))
    data[:, :, 0] = rng.normal(size=N)  # fixed where valid
    data[:, ~mask, 0] = rng.normal(size=(N_TIMES, 40)) * 100  # anything where not
    data[0, ~mask, 0] = np.nan
    source = MemorySource({0: data}, n_lat=N_LAT, n_lon=N_LON, mask=mask)
    result = feature_stationarity(source, layer=0)
    assert result.fixed_share[0] == pytest.approx(1.0)
    assert result.n_nodes_used == N - 40


def test_a_static_with_holes_leaves_those_nodes_out(planted):
    source, land = planted
    holed = land.copy()
    holed[:10] = np.nan
    source._fields["land"] = holed
    result = feature_stationarity(source, layer=0, statics=["land"])
    assert result.n_nodes_used == N - 10
    assert result.residual_fixed_share[2] == pytest.approx(0.0, abs=1e-9)


def test_a_mesh_has_no_rows():
    rng = np.random.default_rng(2)
    data = np.repeat(rng.normal(size=(1, 30, 2)), 5, axis=0)
    source = MemorySource({0: data}, n_lat=5, n_lon=6)
    source._grid = type(source._grid)(lat=source._grid.lat, lon=source._grid.lon)
    result = feature_stationarity(source, layer=0)
    assert np.isnan(result.zonal_share).all()
    assert result.residual_fixed_share == pytest.approx([1.0, 1.0])


def test_a_dictionary_is_read_feature_by_feature(planted):
    source, _ = planted
    frame = np.eye(4, dtype=np.float32)[[1, 3]]
    basis = Dictionary(
        encoder=frame, encoder_bias=np.zeros(2, np.float32), decoder=frame.copy(),
        decoder_bias=np.zeros(4, np.float32), input_mean=np.zeros(4, np.float32),
        activation="relu",
    )  # fmt: skip
    result = feature_stationarity(source, layer=0, basis=basis, allow_unverified_basis=True)
    assert result.settings["columns"] == "features"
    assert result.zonal_share[0] == pytest.approx(1.0)  # cos(lat) is positive: ReLU keeps it


def _by_hand(data, weights):
    """The fixed share of one column, written out: the variance of its time-mean map
    over the variance of everything, both weighted by ``weights``."""
    node_mean = data.mean(axis=0)
    overall = weights @ node_mean
    fixed = weights @ (node_mean - overall) ** 2
    temporal = weights @ data.var(axis=0)
    return fixed / (fixed + temporal)


def test_the_shares_are_weighted_by_area():
    """Fixed near the poles, varying at the equator: the share depends on how much area
    each holds, and a plain mean over nodes gets it wrong."""
    rng = np.random.default_rng(7)
    source = MemorySource({0: np.zeros((N_TIMES, N, 1))}, n_lat=N_LAT, n_lon=N_LON)
    grid = source.grid()
    polar = np.abs(grid.lat) > 50.0
    data = np.zeros((N_TIMES, N))
    data[:, polar] = rng.normal(size=polar.sum())
    data[:, ~polar] = rng.normal(size=(N_TIMES, (~polar).sum()))
    source._data[0] = data[:, :, None]
    result = feature_stationarity(source, layer=0)
    expected = _by_hand(data, grid.weights())
    assert result.fixed_share[0] == pytest.approx(expected, rel=1e-9)
    assert abs(_by_hand(data, np.full(N, 1.0 / N)) - expected) > 0.05  # weighting matters
    total = grid.weights() @ ((data - grid.weights() @ data.mean(axis=0)) ** 2).mean(axis=0)
    assert result.total_variance[0] == pytest.approx(total, rel=1e-9)


def test_coverage_is_the_share_of_area_and_time_above_the_threshold(planted):
    source, _ = planted
    weights = source.grid().weights()
    pattern = source._data[0][0, :, 0].astype(np.float64) - 1e4
    result = feature_stationarity(source, layer=0, threshold=1e4, columns=[0, 2])
    assert result.coverage[0] == pytest.approx(weights @ (pattern > 0.0), abs=1e-3)
    assert result.coverage[1] == pytest.approx(0.0)  # land is nowhere near 1e4
    assert feature_stationarity(source, layer=0, columns=[0]).coverage[0] == pytest.approx(1.0)


def test_statics_in_any_units_are_weighed_alike(planted):
    """A static near 1e-11 beside one near 5e4: each explains what it explains, and one
    that is a function of latitude adds nothing the rows did not."""
    source, land = planted
    lat = source.grid().lat
    rng = np.random.default_rng(8)
    source._fields["tiny"] = land * 1e-11
    source._fields["huge"] = 5e4 + 1e3 * rng.normal(size=N)
    source._fields["coslat"] = np.cos(np.radians(lat))
    result = feature_stationarity(source, layer=0, statics=["huge", "tiny", "coslat"])
    # land: what twelve rows explain of a random map by chance, the static explains the rest
    assert result.static_share[2] + result.zonal_share[2] == pytest.approx(1.0, abs=1e-6)
    assert result.static_share[2] > 0.9
    assert result.residual_fixed_share[2] == pytest.approx(0.0, abs=1e-6)
    assert result.zonal_share[1] == pytest.approx(1.0) and result.static_share[1] == pytest.approx(
        0.0, abs=1e-9
    )
    alone = feature_stationarity(source, layer=0, statics=["tiny"])
    assert alone.residual_fixed_share[2] == pytest.approx(0.0, abs=1e-6)
    assert alone.residual_fixed_share[0] > 0.85  # and what it does not explain stays


def test_a_column_that_varies_only_in_time_has_nothing_fixed(planted):
    rng = np.random.default_rng(9)
    data = np.repeat(rng.normal(size=(N_TIMES, 1, 1)), N, axis=1)  # the same everywhere
    source = MemorySource({0: data}, n_lat=N_LAT, n_lon=N_LON)
    result = feature_stationarity(source, layer=0)
    assert result.fixed_share[0] == 0.0 and result.residual_fixed_share[0] == 0.0
    assert np.isnan(result.zonal_share[0]) and np.isnan(result.static_share[0])
    assert result.columns[result.ranked()][0] == 0


def test_a_column_missing_at_some_times_at_a_node_is_not_read_at_all():
    rng = np.random.default_rng(10)
    data = rng.normal(size=(N_TIMES, N, 2))
    source = MemorySource({0: data}, n_lat=N_LAT, n_lon=N_LON)

    class Missing:
        kind, n_features, n_channels = "fake", 2, 2
        meta = {"sha256": "0" * 64}

        def transform(self, latents, features=None):
            out = np.array(latents, dtype=np.float64)
            if np.allclose(latents[5], data[3, 5]):
                out[5, 0] = np.nan
            return out if features is None else out[:, list(features)]

    result = feature_stationarity(source, layer=0, basis=Missing(), allow_unverified_basis=True)
    assert np.isnan(result.fixed_share[0]) and np.isnan(result.coverage[0])
    assert np.isfinite(result.fixed_share[1]) and np.isfinite(result.coverage[1])


def test_nodes_of_no_area_are_left_out():
    from exege.latents.grid import Grid

    rng = np.random.default_rng(11)
    data = np.repeat(rng.normal(size=(1, N, 1)), 3, axis=0)
    source = MemorySource({0: data}, n_lat=N_LAT, n_lon=N_LON)
    area = np.ones(N)
    area[:10] = 0.0
    grid = source._grid
    source._grid = Grid(lat=grid.lat, lon=grid.lon, shape=grid.shape, area=area)
    assert feature_stationarity(source, layer=0).n_nodes_used == N - 10


@pytest.mark.parametrize(
    ("kwargs", "words"),
    [
        ({"columns": [9]}, "no channel 9"),
        ({"columns": []}, "no channels"),
        ({"columns": [1, 1]}, "twice"),
        ({"statics": ["sea"]}, "no field"),
        ({"statics": ["land", "land"]}, "twice"),
        ({"chunk": 0}, "chunk"),
        ({"times": []}, "no times"),
        ({"times": [0]}, "at least two times"),
    ],
)
def test_requests_that_cannot_be_had(planted, kwargs, words):
    source, _ = planted
    with pytest.raises(RequestError, match=words):
        feature_stationarity(source, layer=0, **kwargs)


def test_statics_need_fields():
    source = MemorySource({0: np.zeros((2, 48, 1))})
    with pytest.raises(RequestError, match="keeps no physical fields"):
        feature_stationarity(source, layer=0, statics=["land"])


def test_the_command(tmp_path):
    path = write_latent_archive(tmp_path / "a")
    runner = CliRunner()
    out = runner.invoke(cli, ["latents", "stationary", str(path), "--layer", "1", "--top", "3"])
    assert out.exit_code == 0, out.output
    assert "channels of layer 1 over 2 time(s)" in out.output
    assert "UNEXPLAINED" in out.output
    as_json = runner.invoke(
        cli, ["latents", "stationary", str(path), "--channel", "4", "--channel", "0", "--json"]
    )
    summary = json.loads(as_json.output)
    by_column = {c["column"]: c for c in summary["columns"]}
    assert set(by_column) == {0, 4}
    assert by_column[4]["fixed_share"] > 0.99  # the bump sits still; noise does not
    assert by_column[0]["fixed_share"] < 0.9


def test_the_command_with_a_basis(tmp_path):
    path = write_latent_archive(tmp_path / "a")
    from exege.latents import accumulate_moments, open_source, pca_from_moments

    source = open_source(path)
    basis_path = save_basis(
        tmp_path / "pca.npz", pca_from_moments(accumulate_moments(source, layer=2), 2)
    )
    runner = CliRunner()
    out = runner.invoke(
        cli, ["latents", "stationary", str(path), "--basis", str(basis_path), "--feature", "0"]
    )
    assert out.exit_code == 0, out.output
    assert "features of layer 2" in out.output
    both = runner.invoke(cli, ["latents", "stationary", str(path), "--channel", "1",
                               "--feature", "0", "--basis", str(basis_path)])  # fmt: skip
    assert both.exit_code != 0 and "not both" in both.output
    alone = runner.invoke(cli, ["latents", "stationary", str(path), "--feature", "0"])
    assert alone.exit_code != 0 and "--feature needs --basis" in alone.output
