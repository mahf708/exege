"""A free run against a teacher-forced one, column by column."""

from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")

from click.testing import CliRunner  # noqa: E402

from conftest import MemorySource, write_latent_archive  # noqa: E402
from exege._cli import cli  # noqa: E402
from exege.core.errors import RequestError  # noqa: E402
from exege.latents import Region, feature_gap  # noqa: E402

N_TIMES, N = 4, 48


class Labeled(MemorySource):
    """A memory source whose times carry the labels given, and whose identity may differ."""

    def __init__(self, data, labels, checkpoint="mem.ckpt", **kwargs):
        super().__init__(data, **kwargs)
        from dataclasses import replace

        self._info = replace(self._info, times=tuple(labels), checkpoint=checkpoint)


def _reference(seed=0):
    rng = np.random.default_rng(seed)
    data = np.abs(rng.normal(size=(N_TIMES, N, 3))) + 0.5  # every value positive: active
    data[:, :, 2] = -1.0  # never active
    return data


LABELS = [f"t{i}" for i in range(N_TIMES)]


def test_a_damped_channel_reads_its_damping():
    data = _reference()
    damped = data.copy()
    damped[:, :, 0] *= 0.5
    result = feature_gap(Labeled({0: data}, LABELS), Labeled({0: damped}, LABELS), layer=0)
    assert result.ratio_mean[0] == pytest.approx(0.5)
    assert result.ratio_strength[0] == pytest.approx(0.5)
    assert result.ratio_mean[1] == pytest.approx(1.0) and result.deficit[1] == pytest.approx(0.0)
    assert result.deficit[0] == pytest.approx(0.5)  # half the size
    assert list(result.columns[result.ranked()])[0] == 0
    assert result.coverage[:, 0] == pytest.approx([1.0, 1.0])
    assert result.coverage[:, 2] == pytest.approx([0.0, 0.0])
    assert np.isnan(result.strength[:, 2]).all()  # never active: no strength
    assert result.run_mean.shape == (N_TIMES, 3)
    assert result.run_mean[:, 0] == pytest.approx(0.5 * result.reference_mean[:, 0])


def test_a_gap_that_grows_with_lead_shows_time_by_time():
    data = _reference()
    fading = data.copy()
    for t in range(N_TIMES):
        fading[t, :, 1] *= 1.0 - 0.2 * t
    result = feature_gap(Labeled({0: data}, LABELS), Labeled({0: fading}, LABELS), layer=0)
    ratios = result.run_mean[:, 1] / result.reference_mean[:, 1]
    assert ratios == pytest.approx([1.0, 0.8, 0.6, 0.4])


def test_the_deficit_reads_magnitude_whatever_the_sign():
    """A negative channel and one that swings about zero, damped by half, are weakened by
    half; an amplified negative channel is strengthened."""
    rng = np.random.default_rng(3)
    data = np.zeros((N_TIMES, N, 3))
    data[:, :, 0] = -2.0
    data[:, :, 1] = rng.choice([-1.0, 1.0], size=(N_TIMES, N))
    data[:, :, 2] = -2.0
    run = data * np.array([0.5, 0.5, 2.0])
    result = feature_gap(Labeled({0: data}, LABELS), Labeled({0: run}, LABELS), layer=0)
    assert result.deficit == pytest.approx([0.5, 0.5, -1.0])
    assert result.rms[0] == pytest.approx([2.0, 1.0, 2.0]) and result.rms[1] == pytest.approx(
        [1.0, 0.5, 4.0]
    )
    assert list(result.columns[result.ranked()])[-1] == 2


def test_everything_is_weighted_by_area():
    data = np.ones((N_TIMES, N, 1))
    reference = Labeled({0: data}, LABELS)
    grid = reference.grid()
    weights = grid.weights()
    polar = np.abs(grid.lat) > 50.0
    run = data.copy()
    run[:, polar, 0] = 0.0
    result = feature_gap(reference, Labeled({0: run}, LABELS), layer=0, threshold=0.5)
    expected = weights @ (~polar)
    assert result.ratio_mean[0] == pytest.approx(expected)
    assert result.coverage[1, 0] == pytest.approx(expected)
    assert abs(expected - (~polar).mean()) > 0.05  # a plain mean over nodes would differ
    assert result.strength[1, 0] == pytest.approx(1.0)  # where active, as strong as before


def test_coverage_and_strength_part_where_a_column_is_on_in_places():
    data = np.full((N_TIMES, N, 1), -1.0)
    reference = Labeled({0: data}, LABELS)
    grid = reference.grid()
    north = grid.lat > 0.0
    data[:, north, 0] = 2.0
    result = feature_gap(reference, reference, layer=0)
    share = grid.weights() @ north
    assert 0.0 < result.coverage[0, 0] < 1.0
    assert result.coverage[0, 0] == pytest.approx(share)
    assert result.strength[0, 0] == pytest.approx(2.0)
    assert result.mean[0, 0] == pytest.approx(2.0 * share - (1.0 - share))


def test_a_strength_of_zero_has_no_ratio():
    data = np.zeros((N_TIMES, N, 1))
    data[:, ::2, 0], data[:, 1::2, 0] = -0.5, 0.5
    run = np.full((N_TIMES, N, 1), 0.5)
    result = feature_gap(Labeled({0: data}, LABELS), Labeled({0: run}, LABELS), layer=0,
                         threshold=-1.0)  # fmt: skip
    assert np.isnan(result.ratio_strength[0]) and np.isnan(result.ratio_mean[0])


def test_a_basis_is_read_on_both_sides():
    from exege.latents import Dictionary

    data = _reference()
    damped = data.copy()
    damped[:, :, 1] *= 0.25
    frame = np.eye(3, dtype=np.float32)[[1, 2]]
    basis = Dictionary(
        encoder=frame, encoder_bias=np.zeros(2, np.float32), decoder=frame.copy(),
        decoder_bias=np.zeros(3, np.float32), input_mean=np.zeros(3, np.float32),
    )  # fmt: skip
    result = feature_gap(Labeled({0: data}, LABELS), Labeled({0: damped}, LABELS), layer=0,
                         basis=basis, allow_unverified_basis=True)  # fmt: skip
    assert result.settings["columns"] == "features"
    assert result.ratio_mean[0] == pytest.approx(0.25) and result.deficit[0] == pytest.approx(0.75)
    assert result.coverage[:, 1] == pytest.approx([0.0, 0.0])  # relu of -1: never on
    assert "basis" in result.provenance


def test_both_sides_are_read_over_the_nodes_both_have():
    """Where a basis leaves one side without a value, the other side is not read there
    either, so the two are never set against each other over different areas."""
    data = _reference()
    run = data.copy()
    run[1, :10, 0] = 1000.0  # this basis makes nothing of values this large

    class Gappy:
        kind, n_features, n_channels = "fake", 3, 3
        meta = {"sha256": "0" * 64}

        def transform(self, latents, features=None):
            out = np.where(np.abs(latents) > 100.0, np.nan, latents).astype(np.float64)
            return out if features is None else out[:, list(features)]

    result = feature_gap(Labeled({0: data}, LABELS), Labeled({0: run}, LABELS), layer=0,
                         basis=Gappy(), allow_unverified_basis=True)  # fmt: skip
    assert result.run_mean[1, 0] == pytest.approx(result.reference_mean[1, 0])
    assert result.ratio_mean[0] == pytest.approx(1.0)
    assert result.ratio_mean[1] == pytest.approx(1.0)


def test_times_are_paired_by_label():
    data = _reference()
    # The run keeps the last three times, in the same order, under the same labels.
    run = Labeled({0: data[1:]}, LABELS[1:])
    result = feature_gap(Labeled({0: data}, LABELS), run, layer=0)
    assert result.times == tuple(LABELS[1:])
    assert result.ratio_mean == pytest.approx([1.0, 1.0, 1.0])
    with pytest.raises(RequestError, match="no latents at t0"):
        feature_gap(Labeled({0: data}, LABELS), run, layer=0, times=["t0"])


def test_times_are_paired_by_label_whatever_order_each_keeps_them_in():
    data = _reference()
    for t in range(N_TIMES):
        data[t, :, 0] *= 1.0 + t  # each time its own size
    keep = [3, 0, 2]  # not a suffix, not in order
    run = Labeled({0: data[keep]}, [LABELS[t] for t in keep])
    result = feature_gap(Labeled({0: data}, LABELS), run, layer=0)
    assert set(result.times) == {"t0", "t2", "t3"}
    assert result.run_mean[:, 0] == pytest.approx(result.reference_mean[:, 0])
    assert result.ratio_mean == pytest.approx([1.0, 1.0, 1.0])


def test_different_networks_are_refused():
    data = _reference()
    with pytest.raises(RequestError, match="different networks"):
        feature_gap(Labeled({0: data}, LABELS), Labeled({0: data}, LABELS, "other.ckpt"), layer=0)
    result = feature_gap(
        Labeled({0: data}, LABELS), Labeled({0: data}, LABELS, "other.ckpt"), layer=0,
        across_models=True,
    )  # fmt: skip
    assert result.ratio_mean[0] == pytest.approx(1.0)


def test_only_nodes_valid_in_both_are_read():
    data = _reference()
    mask = np.ones(N, dtype=bool)
    mask[:6] = False
    run = data.copy()
    run[:, :6, :] = np.nan  # what a masked node holds is nobody's business
    result = feature_gap(Labeled({0: data}, LABELS), Labeled({0: run}, LABELS, mask=mask), layer=0)
    assert result.settings["n_nodes_compared"] == N - 6
    assert result.ratio_mean[0] == pytest.approx(1.0)


def test_a_region_narrows_the_nodes():
    data = _reference()
    run = data.copy()
    reference = Labeled({0: data}, LABELS)
    grid = reference.grid()
    inside = Region(lat=float(grid.lat[0]), lon=float(grid.lon[0]), radius_km=500.0)
    nodes = inside.nodes(grid)
    run[:, nodes, 0] *= 2.0
    result = feature_gap(reference, Labeled({0: run}, LABELS), layer=0, region=inside)
    assert result.ratio_mean[0] == pytest.approx(2.0)
    assert result.deficit[0] < 0.0
    with pytest.raises(RequestError, match="widen the region"):
        feature_gap(reference, reference, layer=0, region=Region(0.0, 1.0, 1.0))


def test_nodes_of_no_area_are_left_out_and_none_at_all_is_refused():
    from exege.latents.grid import Grid

    data = _reference()
    reference, run = Labeled({0: data}, LABELS), Labeled({0: data}, LABELS)
    grid = reference.grid()
    area = np.ones(N)
    area[:5] = 0.0
    for source in (reference, run):
        source._grid = Grid(lat=grid.lat, lon=grid.lon, shape=grid.shape, area=area)
    assert feature_gap(reference, run, layer=0).settings["n_nodes_compared"] == N - 5
    corner = Region(lat=float(grid.lat[0]), lon=float(grid.lon[0]), radius_km=10.0)
    with pytest.raises(RequestError, match="of any area"):
        feature_gap(reference, run, layer=0, region=corner)


def test_columns_are_chosen():
    data = _reference()
    result = feature_gap(Labeled({0: data}, LABELS), Labeled({0: data}, LABELS), layer=0,
                         columns=[2, 0])  # fmt: skip
    assert list(result.columns) == [2, 0]
    with pytest.raises(RequestError, match="no channel 5"):
        feature_gap(Labeled({0: data}, LABELS), Labeled({0: data}, LABELS), layer=0, columns=[5])


def test_the_command(tmp_path):
    reference = write_latent_archive(tmp_path / "control")
    run = write_latent_archive(tmp_path / "run", shift=(1, -1.0))  # the offset channel, at 50
    runner = CliRunner()
    out = runner.invoke(cli, ["latents", "gap", str(reference), str(run), "--layer", "2",
                              "--top", "3"])  # fmt: skip
    assert out.exit_code == 0, out.output
    assert "the run's largest deficit first" in out.output
    first_row = out.output.strip().splitlines()[3]
    assert first_row.split()[0] == "1"  # the channel the run carries less of
    as_json = runner.invoke(
        cli, ["latents", "gap", str(reference), str(run), "--channel", "1", "--json"]
    )
    summary = json.loads(as_json.output)
    column = summary["columns"][0]
    assert column["column"] == 1
    assert column["run"]["mean_by_time"][0] == pytest.approx(column["reference"]["mean_by_time"][0])
    assert column["difference"] == pytest.approx(-0.5, abs=1e-3)  # shifted at one time of two


def test_a_repeated_time_is_refused():
    data = _reference()
    with pytest.raises(RequestError, match="once"):
        feature_gap(Labeled({0: data}, LABELS), Labeled({0: data}, LABELS), layer=0,
                    times=["t0", "t0", "t1"])  # fmt: skip


def test_a_feature_too_rare_to_measure_is_ranked_last():
    """Feature 1 fires at one node of the reference and nowhere in the run: a deficit of
    one, from nothing. Below min_coverage it goes after the feature truly damped."""
    from exege.latents.basis import Dictionary

    data = _reference()
    data[:, :, 1] = 0.0
    data[:, 0, 1] = 5.0  # one node of 48
    run = data.copy()
    run[:, :, 0] *= 0.7
    run[:, 0, 1] = 0.0
    eye = np.eye(3, dtype=np.float32)
    basis = Dictionary(eye, np.zeros(3), eye, np.zeros(3), np.zeros(3), meta={})
    kw = {"layer": 0, "basis": basis, "allow_unverified_basis": True}
    rare_first = feature_gap(Labeled({0: data}, LABELS), Labeled({0: run}, LABELS),
                             min_coverage=0.0, **kw)  # fmt: skip
    assert rare_first.deficit[1] == pytest.approx(1.0)
    assert list(rare_first.columns[rare_first.ranked()])[:2] == [1, 0]
    floored = feature_gap(Labeled({0: data}, LABELS), Labeled({0: run}, LABELS), **kw)
    assert floored.coverage[0, 1] < 0.01
    assert list(floored.columns[floored.ranked()]) == [0, 1, 2]
    assert floored.deficit[1] == pytest.approx(1.0)  # still reported, only ranked last


def test_the_coverage_floor_is_for_sparse_features_and_silence_goes_last():
    """A PCA-like basis swings both ways: its components are ranked by deficit whatever
    their 'coverage'. A silent column has no deficit and goes after every one that has."""
    from exege.latents.basis import PCA

    data = _reference()
    data[:, :, 1] -= 10.0  # negative throughout: coverage 0 above a threshold of 0
    run = data.copy()
    run[:, :, 1] *= 0.2
    run[:, :, 0] *= 0.9
    pca = PCA(mean=np.zeros(3), components=np.eye(3)[:2], explained_variance_ratio=np.ones(2) / 2)
    result = feature_gap(Labeled({0: data}, LABELS), Labeled({0: run}, LABELS), layer=0,
                         basis=pca, allow_unverified_basis=True)  # fmt: skip
    assert result.settings["sparse"] is False
    assert list(result.columns[result.ranked()]) == [1, 0]
