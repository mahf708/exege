"""A token-normalized dictionary trained elsewhere, read into exege.

The layout is checked against a reference written here, step by step from its
description, not from the code under test: each node's channels are centered on their
own mean and divided by their own length, the pre-bias is taken off, the encoder's
ReLU is kept to its ``k`` largest, and the code is written back in raw units with the
pre-bias added.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from click.testing import CliRunner  # noqa: E402

from conftest import MemorySource, write_latent_archive  # noqa: E402
from exege._cli import cli  # noqa: E402
from exege.adapters.toy_dynamics import ToyDynamics  # noqa: E402
from exege.core.errors import RequestError  # noqa: E402
from exege.latents import (  # noqa: E402
    PCA,
    Dictionary,
    Steer,
    basis_hash,
    feature_direction,
    load_basis,
    open_source,
    run_steering,
    save_basis,
    token_normalized_dictionary,
)
from exege.latents.evaluate import measure  # noqa: E402
from exege.latents.source import check_basis_fits  # noqa: E402

N_FEATURES, N_CHANNELS, K = 12, 5, 3


def _published(seed: int = 0, n_features: int = N_FEATURES, n_channels: int = N_CHANNELS):
    """Arrays in the published layout: encoder (F, C), decoder (C, F) with unit columns."""
    rng = np.random.default_rng(seed)
    encoder = rng.normal(size=(n_features, n_channels)).astype(np.float32)
    decoder = rng.normal(size=(n_channels, n_features)).astype(np.float32)
    decoder /= np.linalg.norm(decoder, axis=0, keepdims=True)
    pre_bias = rng.normal(0.0, 0.1, n_channels).astype(np.float32)
    return encoder, decoder, pre_bias


def _reference(x, encoder, decoder, pre_bias, k):
    """The layout, written out plainly: (codes, reconstruction in raw units)."""
    x = np.asarray(x, dtype=np.float64)
    codes = np.zeros((x.shape[0], encoder.shape[0]))
    for i, row in enumerate(x):
        centered = row - row.mean()
        normalized = centered / max(np.linalg.norm(centered), 1e-6)
        pre = np.maximum((normalized - pre_bias) @ encoder.T, 0.0)
        keep = np.argsort(pre)[::-1][:k]
        codes[i, keep] = pre[keep]
    return codes, codes @ decoder.T + pre_bias


def test_it_encodes_and_decodes_as_the_layout_says():
    encoder, decoder, pre_bias = _published()
    basis = token_normalized_dictionary(encoder, decoder, pre_bias, K)
    x = np.random.default_rng(3).normal(5.0, 2.0, size=(40, N_CHANNELS)).astype(np.float32)
    codes, rebuilt = _reference(x, encoder, decoder, pre_bias, K)
    np.testing.assert_allclose(basis.transform(x), codes, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(basis.reconstruct(x), rebuilt, rtol=1e-4, atol=1e-4)
    assert (np.count_nonzero(basis.transform(x), axis=1) <= K).all()
    # a subset of features is the same columns: top-k competes over all of them
    np.testing.assert_allclose(basis.transform(x, [4, 1]), codes[:, [4, 1]], rtol=1e-4, atol=1e-5)


def test_a_node_rescaled_or_shifted_as_a_whole_has_the_same_code():
    encoder, decoder, pre_bias = _published()
    basis = token_normalized_dictionary(encoder, decoder, pre_bias, K)
    x = np.random.default_rng(4).normal(size=(10, N_CHANNELS)).astype(np.float32)
    np.testing.assert_allclose(basis.transform(3.0 * x + 7.0), basis.transform(x), atol=1e-4)


def test_the_decoder_may_be_either_way_round_unless_it_is_square():
    encoder, decoder, pre_bias = _published()
    a = token_normalized_dictionary(encoder, decoder, pre_bias, K)
    b = token_normalized_dictionary(encoder, decoder.T, pre_bias, K)
    np.testing.assert_array_equal(a.decoder, b.decoder)
    np.testing.assert_array_equal(a.decoder, decoder.T)
    square = _published(n_features=N_CHANNELS)
    with pytest.raises(RequestError, match="square"):
        token_normalized_dictionary(*square, K)
    told = token_normalized_dictionary(*square, K, decoder_layout="channels-first")
    np.testing.assert_array_equal(told.decoder, square[1].T)
    with pytest.raises(RequestError, match="other way round"):
        token_normalized_dictionary(encoder, decoder, pre_bias, K, decoder_layout="features-first")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"decoder": np.zeros((7, 3))}, "the decoder must be"),
        ({"pre_bias": np.zeros(N_CHANNELS + 1)}, "pre-bias must hold"),
        ({"k": 0}, "k must lie"),
        ({"k": N_FEATURES + 1}, "k must lie"),
        ({"pre_bias": np.full(N_CHANNELS, np.nan)}, "not finite"),
        ({"decoder_layout": "sideways"}, "decoder_layout must be"),
    ],
)
def test_a_layout_that_does_not_fit_is_refused(change, message):
    encoder, decoder, pre_bias = _published()
    given = {"encoder": encoder, "decoder": decoder, "pre_bias": pre_bias, "k": K, **change}
    with pytest.raises(RequestError, match=message):
        token_normalized_dictionary(**given)


def test_a_token_dictionary_cannot_also_carry_a_mean_or_a_scale():
    encoder, decoder, pre_bias = _published()
    good = token_normalized_dictionary(encoder, decoder, pre_bias, K)
    fields = {n: getattr(good, n) for n in ("encoder", "encoder_bias", "decoder", "decoder_bias")}
    with pytest.raises(ValueError, match="input_mean 0 and input_scale 1"):
        Dictionary(**fields, input_mean=np.ones(N_CHANNELS), input_norm="token")
    with pytest.raises(ValueError, match="input_mean 0 and input_scale 1"):
        Dictionary(**fields, input_mean=np.zeros(N_CHANNELS), input_scale=2.0, input_norm="token")
    with pytest.raises(ValueError, match="input_norm must be"):
        Dictionary(**fields, input_mean=np.zeros(N_CHANNELS), input_norm="layer")


def test_a_token_dictionary_cannot_be_a_transcoder():
    encoder, decoder, pre_bias = _published()
    good = token_normalized_dictionary(encoder, decoder, pre_bias, K)
    fields = {n: getattr(good, n) for n in ("encoder", "encoder_bias", "decoder", "decoder_bias")}
    common = {**fields, "input_mean": np.zeros(N_CHANNELS), "input_norm": "token"}
    with pytest.raises(ValueError, match="transcoder"):
        Dictionary(**common, output_mean=np.zeros(N_CHANNELS))
    with pytest.raises(ValueError, match="transcoder"):
        Dictionary(**common, output_scale=2.0)


def test_a_node_whose_channels_are_all_equal_meets_the_floor_not_a_division_by_zero():
    encoder, decoder, pre_bias = _published()
    basis = token_normalized_dictionary(encoder, decoder, pre_bias, K)
    flat = np.full((2, N_CHANNELS), 4.0, dtype=np.float32)
    codes, _ = _reference(flat, encoder, decoder, pre_bias, K)  # centered to zero: no length
    got = basis.transform(flat)
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, codes, rtol=1e-4, atol=1e-5)


def test_a_node_that_is_not_a_number_comes_out_as_none_and_alone():
    encoder, decoder, pre_bias = _published()
    basis = token_normalized_dictionary(encoder, decoder, pre_bias, K)
    x = np.random.default_rng(5).normal(size=(4, N_CHANNELS)).astype(np.float32)
    x[1, 2] = np.nan
    got = basis.transform(x)
    assert np.isnan(got[1]).all()  # features compete: one missing channel decides none
    codes, _ = _reference(np.delete(x, 1, axis=0), encoder, decoder, pre_bias, K)
    np.testing.assert_allclose(np.delete(got, 1, axis=0), codes, rtol=1e-4, atol=1e-5)


def test_its_directions_are_in_raw_units():
    encoder, decoder, pre_bias = _published()
    basis = token_normalized_dictionary(encoder, decoder, pre_bias, K)
    np.testing.assert_allclose(feature_direction(basis, 7), decoder[:, 7])


# -- the file -----------------------------------------------------------------------


def test_bases_that_leave_the_new_option_be_keep_their_hashes():
    """Pinned from the code before ``input_norm`` existed: adding it must not change
    the content hash of any basis a result already names."""
    weights = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    dictionary = Dictionary(
        encoder=weights,
        encoder_bias=np.array([0, -1, 0], dtype=np.float32),
        decoder=weights.copy(),
        decoder_bias=np.zeros(2, dtype=np.float32),
        input_mean=np.array([1, 0], dtype=np.float32),
        input_scale=2.0,
        activation="topk",
        k=2,
        meta={"fitted_on": {"layer": 3}},
    )
    pca = PCA(
        mean=np.array([1.0, 2.0]),
        components=np.array([[0.6, 0.8]]),
        explained_variance_ratio=np.array([0.9]),
    )
    assert basis_hash(dictionary) == (
        "8d91eea13a36a47337179de0390af087978ed4f3cba4b1af861ba7806896ee0c"
    )
    assert basis_hash(pca) == "e7c8e1eff4615f5b294bcc747c9cbd5d643c8e9e867b0f2c7850adaa9ee6c80d"


def test_a_token_dictionary_survives_its_file_and_its_hash_says_so(tmp_path):
    encoder, decoder, pre_bias = _published()
    basis = token_normalized_dictionary(encoder, decoder, pre_bias, K, meta={"note": "x"})
    save_basis(tmp_path / "b.npz", basis)
    loaded = load_basis(tmp_path / "b.npz", sha256=basis_hash(basis))
    assert loaded.input_norm == "token" and loaded.meta["hash_status"] == "verified"
    x = np.random.default_rng(5).normal(size=(8, N_CHANNELS)).astype(np.float32)
    np.testing.assert_array_equal(loaded.transform(x), basis.transform(x))
    # the same arrays read the standard way are another basis, with another hash
    standard = Dictionary(
        **{n: getattr(basis, n) for n in ("encoder", "encoder_bias", "decoder", "decoder_bias")},
        input_mean=basis.input_mean, activation="topk", k=K,
    )  # fmt: skip
    assert basis_hash(standard) != basis_hash(basis)


# -- reading and steering -----------------------------------------------------------


def test_its_explained_variance_is_about_the_mean_of_what_is_measured():
    """A dictionary that never fires rebuilds every node as its pre-bias. With the
    pre-bias at the measured times' area-weighted mean, it explains nothing: exactly 0."""
    rng = np.random.default_rng(6)
    data = rng.normal(3.0, 1.5, size=(3, 48, N_CHANNELS))
    source = MemorySource({0: data})
    weights = source.grid().weights()
    mean = np.mean([weights @ data[t] for t in range(3)], axis=0)
    silent = token_normalized_dictionary(
        np.zeros((N_FEATURES, N_CHANNELS)), np.eye(N_CHANNELS, N_FEATURES), mean, K
    )
    metrics = measure(silent, source, ["t0", "t1", "t2"], layer=0)
    assert metrics.explained_variance == pytest.approx(0.0, abs=1e-6)
    assert metrics.mean_active_features == 0.0


def test_a_token_dictionary_steers_in_raw_units():
    """Adding along a feature whose decoder row is channel 0 is the planted edit, so it
    gives exactly the planted dictionary's response."""
    system = ToyDynamics()
    planted = system.planted_dictionary()
    frame = np.eye(4, dtype=np.float32)[:3]
    token = token_normalized_dictionary(
        frame, frame, np.zeros(4), 2, meta={"fitted_on": dict(planted.meta["fitted_on"])}
    )
    steer = Steer(layer=1, feature=0, amount=1.5, times=(1,))
    common = {"steps": 5, "seeds": (0,), "n_random": 3}
    ours = run_steering(system, token, steer, **common).effects["temperature"]
    theirs = run_steering(system, planted, steer, **common).effects["temperature"]
    assert ours.feature_response == pytest.approx(theirs.feature_response)
    assert ours.feature_response == pytest.approx(1.5 * 2.0 * (1 + 0.5 + 0.25 + 0.125) / 4)


# -- the command --------------------------------------------------------------------


def _npz(path, **change):
    encoder, decoder, pre_bias = _published(n_channels=6)
    arrays = {"W_enc": encoder, "W_dec": decoder, "b_pre": pre_bias, **change}
    np.savez(path, **arrays)
    return path


def test_the_command_writes_a_basis_that_knows_where_it_belongs(tmp_path):
    archive = write_latent_archive(tmp_path / "latents")
    raw = _npz(tmp_path / "sae.npz")
    out = tmp_path / "basis.npz"
    run = CliRunner()
    done = run.invoke(
        cli, ["latents", "import-sae", str(raw), "--out", str(out), "--k", "2",
              "--like", str(archive), "--layer", "2"],
    )  # fmt: skip
    assert done.exit_code == 0, done.output
    basis = load_basis(out)
    assert f"--basis-sha256 {basis.meta['sha256']}" in done.output
    assert "records no network" not in done.output
    assert basis.meta["imported_from"]["path"] == str(raw)
    info = open_source(archive).info()
    check_basis_fits(basis, info, 2)  # accepted where it was said to belong
    with pytest.raises(RequestError, match="was not fitted here"):
        check_basis_fits(basis, info, 1)
    used = run.invoke(cli, ["latents", "census", str(archive), "--layer", "2", "--basis", str(out)])
    assert used.exit_code == 0, used.output


def test_without_an_archive_it_says_the_basis_is_unverified(tmp_path):
    archive = write_latent_archive(tmp_path / "latents")
    out = tmp_path / "basis.npz"
    run = CliRunner()
    done = run.invoke(
        cli, ["latents", "import-sae", str(_npz(tmp_path / "sae.npz")), "--out", str(out),
              "--k", "2"],
    )  # fmt: skip
    assert done.exit_code == 0, done.output
    assert "--allow-unverified-basis" in done.output
    refused = run.invoke(cli, ["latents", "census", str(archive), "--basis", str(out)])
    assert refused.exit_code != 0 and "unverified" in refused.output


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--decoder", "nothing"], "holds no array(s) nothing"),
        (["--layer", "2"], "together"),
    ],
)
def test_the_command_refuses_what_it_cannot_read(tmp_path, args, message):
    run = CliRunner()
    raw = _npz(tmp_path / "sae.npz")
    done = run.invoke(
        cli, ["latents", "import-sae", str(raw), "--out", str(tmp_path / "b.npz"), "--k", "2",
              *args],
    )  # fmt: skip
    assert done.exit_code != 0 and message in done.output


def test_the_command_refuses_a_dictionary_of_another_width(tmp_path):
    archive = write_latent_archive(tmp_path / "latents")
    encoder, decoder, pre_bias = _published(n_channels=4)
    raw = tmp_path / "narrow.npz"
    np.savez(raw, W_enc=encoder, W_dec=decoder, b_pre=pre_bias)
    done = CliRunner().invoke(
        cli, ["latents", "import-sae", str(raw), "--out", str(tmp_path / "b.npz"), "--k", "2",
              "--like", str(archive), "--layer", "2"],
    )  # fmt: skip
    assert done.exit_code != 0 and "reads 4 channel(s), and layer 2" in done.output
    mismatched = _npz(tmp_path / "bad.npz", W_dec=np.zeros((3, 3), dtype=np.float32))
    done = CliRunner().invoke(
        cli, ["latents", "import-sae", str(mismatched), "--out", str(tmp_path / "c.npz"),
              "--k", "2"],
    )  # fmt: skip
    assert done.exit_code != 0 and "the decoder must be" in done.output
