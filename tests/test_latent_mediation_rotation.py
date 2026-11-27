"""Mediation and rotation, against answers planted in the toy system, and their commands.

In ``ToyDynamics`` an add on channel 0 of layer 0 reaches moisture only through channel 1
of layer 1 (the mixer swaps them), so holding that channel takes the whole effect away and
holding channel 2 takes none. With ``positional`` set, channel 3 of layer 0 holds a pattern
that never moves; with ``static`` set, channel 2 holds a boundary field that moves only
when it is rotated with the state.
"""

from __future__ import annotations

import json
import shlex

import pytest

np = pytest.importorskip("numpy")

from click.testing import CliRunner  # noqa: E402

from exege._cli import cli  # noqa: E402
from exege.adapters.toy_dynamics import ToyDynamics  # noqa: E402
from exege.core.errors import RequestError  # noqa: E402
from exege.latents import (  # noqa: E402
    Dictionary,
    Directions,
    Mediator,
    Rotatable,
    Steer,
    rotation_test,
    run_mediation,
    save_basis,
)

STEPS = 5


def _linear(system, layer=1):
    """Every channel as a feature, with no ReLU: holding one holds the channel."""
    return Directions(
        mean=np.zeros(4), directions_=np.eye(4),
        meta={"fitted_on": {"provenance": system.info().identity(), "layer": layer}},
    )  # fmt: skip


@pytest.fixture
def system():
    return ToyDynamics(masked=3)


def _mediate(system, mediator, **kwargs):
    steer = Steer(layer=0, feature=0, amount=1.5, times=(1,))
    return run_mediation(system, system.planted_dictionary(0), steer, mediator, steps=STEPS,
                         **kwargs)  # fmt: skip


# -- mediation ----------------------------------------------------------------


def test_holding_the_channel_the_path_runs_through_takes_the_whole_effect(system):
    result = _mediate(system, Mediator(layer=1, feature=1, basis=_linear(system)), seeds=(0, 1))
    moisture = result.effects["moisture"]
    assert moisture.total == pytest.approx(0.703125, abs=1e-5)
    assert moisture.direct == pytest.approx(0.0, abs=1e-6)
    assert moisture.share == pytest.approx(1.0, abs=1e-5)
    assert result.mediator_change > 0.0
    assert result.exactness == 0.0


def test_holding_a_channel_off_the_path_takes_nothing(system):
    result = _mediate(system, Mediator(layer=1, feature=2, basis=_linear(system)))
    moisture = result.effects["moisture"]
    assert moisture.direct == pytest.approx(moisture.total)
    assert moisture.share == pytest.approx(0.0, abs=1e-9)
    assert result.mediator_change == 0.0
    assert result.effects["temperature"].share is None  # nothing to share out


def test_a_sparse_feature_holds_only_what_it_encodes(system):
    """The planted dictionary's feature is relu(channel): holding it cannot hold the part
    of the channel below zero, so it mediates some of the effect and not all."""
    result = _mediate(system, Mediator(layer=1, feature=1, basis=system.planted_dictionary(1)))
    share = result.effects["moisture"].share
    assert 0.0 < share < 1.0


def test_holding_the_mediator_alone_changes_nothing_at_all(system):
    result = _mediate(
        ToyDynamics(masked=3, feedback=0.3), Mediator(layer=1, feature=0, basis=_linear(system)),
        seeds=(0, 1, 2),
    )  # fmt: skip
    assert result.exactness == 0.0


def test_a_mediator_held_at_some_nodes_only(system):
    nodes = tuple(range(3, 12))
    part = _mediate(system, Mediator(layer=1, feature=1, basis=_linear(system), nodes=nodes))
    share = part.effects["moisture"].share
    grid = system.grid()
    weights = grid.weights()
    assert share == pytest.approx(weights[list(nodes)].sum(), abs=1e-5)
    with pytest.raises(RequestError, match="masked out"):
        _mediate(system, Mediator(layer=1, feature=1, basis=_linear(system), nodes=(0, 1)))


def test_a_mediator_basis_that_belongs_elsewhere_is_refused(system):
    with pytest.raises(RequestError, match="not fitted here"):
        _mediate(system, Mediator(layer=0, feature=1, basis=_linear(system, layer=1)))


# -- rotation -----------------------------------------------------------------


def _anchored(result):
    return {c: bool(a) for c, a in zip(result.columns, result.anchored, strict=True)}


@pytest.fixture
def rotating():
    return ToyDynamics(n_lat=8, n_lon=12, positional=5.0, static=5.0)


def test_the_toy_system_can_be_rotated(rotating):
    assert isinstance(rotating, Rotatable)
    assert rotating.rotation_parts() == ("state", "static")
    with pytest.raises(RequestError, match="no part"):
        rotating.rotate(rotating.initial_state(), 3, ["forcing"])


def test_turning_everything_leaves_only_what_the_model_holds_of_its_own(rotating):
    result = rotation_test(rotating, layer=0, starts=(0, 1))
    assert _anchored(result) == {0: False, 1: False, 2: False, 3: True}
    assert result.score[3] > 0.8 and (result.score[:3] < -0.8).all()


def test_turning_the_weather_alone_leaves_the_boundary_behind(rotating):
    result = rotation_test(rotating, layer=0, parts=["state"], starts=(0, 1))
    assert _anchored(result) == {0: False, 1: False, 2: True, 3: True}


def test_turning_the_boundary_alone_moves_only_what_reads_it(rotating):
    result = rotation_test(rotating, layer=0, parts=["static"])
    assert _anchored(result) == {0: True, 1: True, 2: False, 3: True}


def test_a_pattern_that_maps_onto_itself_has_no_power(rotating):
    """A turn by half the circle sends a pattern with two-fold symmetry onto itself."""
    result = rotation_test(rotating, layer=0, shifts=[180.0], columns=[0])
    assert result.n_powered[0] == 1  # random noise is not symmetric
    sym = ToyDynamics(n_lat=8, n_lon=12)
    state = sym.initial_state()
    half = state.reshape(8, 12, -1)[:, :6]
    sym_state = np.concatenate([half, half], axis=1).reshape(96, -1)
    sym.initial_state = lambda start=0: sym_state
    result = rotation_test(sym, layer=0, shifts=[180.0], columns=[0])
    assert result.n_powered[0] == 0 and np.isnan(result.score[0])


def test_a_rotation_that_cannot_be_done_is_refused(rotating, system):
    with pytest.raises(RequestError, match="whole number"):
        rotation_test(rotating, layer=0, shifts=[45.0])
    with pytest.raises(RequestError, match="onto itself"):
        rotation_test(rotating, layer=0, shifts=[360.0])
    with pytest.raises(RequestError, match="no part"):
        rotation_test(rotating, layer=0, parts=["forcing"])
    with pytest.raises(RequestError, match="columns must lie"):
        rotation_test(rotating, layer=0, columns=[7])

    class Fixed:  # can be run, cannot be turned: it has the four methods and no others
        def info(self):
            return system.info()

        def grid(self):
            return system.grid()

        def initial_state(self, start=0):
            return system.initial_state(start)

        def run(self, *args, **kwargs):
            return system.run(*args, **kwargs)

    with pytest.raises(RequestError, match="cannot be rotated"):
        rotation_test(Fixed(), layer=0)


def test_rotating_through_a_basis_reads_its_features(rotating):
    basis = _linear(rotating, layer=0)
    result = rotation_test(rotating, layer=0, basis=basis, columns=[3, 0])
    assert result.columns == (3, 0) and result.anchored.tolist() == [True, False]


# -- the commands -------------------------------------------------------------


def _invoke(args):
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    return result.output


@pytest.fixture
def planted(tmp_path):
    system = ToyDynamics(masked=3)
    path = save_basis(tmp_path / "planted.npz", system.planted_dictionary())
    along = save_basis(
        tmp_path / "along.npz",
        Directions(mean=np.zeros(4), directions_=np.eye(4)[:1],
                   meta={"fitted_on": {"provenance": system.info().identity(), "layer": 1}}),
    )  # fmt: skip
    return path, along


def test_rotate_and_mediate_commands(planted):
    out = _invoke(["latents", "rotate", "--adapter", "toy-dynamics", "--adapter-option",
                   "positional=5", "--adapter-option", "static=5", "--layer", "0",
                   "--part", "state", "--start", "0", "--start", "1"])  # fmt: skip
    rows = {line.split()[0]: line.split() for line in out.splitlines()[3:] if line.strip()}
    assert rows["3"][-1] == "yes" and rows["2"][-1] == "yes" and rows["0"][-1] != "yes"
    path, _ = planted
    out = _invoke(["latents", "mediate", "--adapter", "toy-dynamics", "--adapter-option",
                   "masked=3", "--basis", str(path), "--layer", "1", "--feature", "0",
                   "--amount", "1.5", "--time", "1", "--steps", "5", "--mediator-layer", "1",
                   "--mediator-feature", "2", "--json"])  # fmt: skip
    data = json.loads(out)
    assert data["exactness"] == 0.0
    assert data["effects"]["temperature"]["share"] == pytest.approx(0.0, abs=1e-9)


# -- holds and turns, more closely --------------------------------------------


def test_a_mediator_at_the_steers_place_is_held_after_the_steer(system):
    """Hold the steered channel itself, at the same layer and time: held after the steer,
    the edit is undone and nothing is left; held before, it would undo nothing."""
    steer = Steer(layer=1, feature=0, amount=1.5, times=(1,))
    mediator = Mediator(layer=1, feature=0, basis=_linear(system))
    result = run_mediation(system, system.planted_dictionary(), steer, mediator, steps=STEPS)
    temperature = result.effects["temperature"]
    assert temperature.total > 1.0
    assert temperature.direct == pytest.approx(0.0, abs=1e-6)


def test_how_far_the_steer_moved_the_mediator_is_measured_where_it_is_held(system):
    # +1.5 on channel 1 of layer 1 at step 1, nothing after: RMS over steps 1..4 is 0.75
    whole = _mediate(system, Mediator(layer=1, feature=1, basis=_linear(system)))
    assert whole.mediator_change == pytest.approx(0.75, abs=1e-6)
    part = _mediate(system, Mediator(layer=1, feature=1, basis=_linear(system),
                                     nodes=tuple(range(3, 12))))  # fmt: skip
    assert part.mediator_change == pytest.approx(0.75, abs=1e-6)


def test_an_untied_feature_is_held_where_it_was_not_where_its_decoder_would_put_it(system):
    """Encoder twice the decoder: a move of 1 along the direction moves the activation by 2.
    The hold still reaches its target, and mediates what the tied feature does."""
    tied = system.planted_dictionary(1)
    untied = Dictionary(
        encoder=2.0 * tied.encoder, encoder_bias=tied.encoder_bias, decoder=tied.decoder,
        decoder_bias=tied.decoder_bias, input_mean=tied.input_mean, meta=tied.meta,
    )  # fmt: skip
    a = _mediate(system, Mediator(layer=1, feature=1, basis=tied))
    b = _mediate(system, Mediator(layer=1, feature=1, basis=untied))
    assert b.hold_error < 1e-5
    assert b.effects["moisture"].share == pytest.approx(a.effects["moisture"].share, abs=1e-5)


def test_a_hold_that_is_not_a_no_op_shows_in_exactness(system):
    class Jittery:  # any run with hooks comes back a little off: not deterministic
        def __getattr__(self, name):
            return getattr(system, name)

        def run(self, state, steps, hooks=(), **kwargs):
            rollout = system.run(state, steps, hooks, **kwargs)
            if not hooks:
                return rollout
            fields = {k: v + 1e-3 for k, v in rollout.fields.items()}
            return type(rollout)(fields=fields, latents=rollout.latents)

    result = _mediate(Jittery(), Mediator(layer=1, feature=1, basis=_linear(system)))
    assert result.exactness == pytest.approx(1e-3, rel=1e-6)


def test_rotate_turns_east(rotating):
    state = rotating.initial_state()
    turned = rotating.rotate(state, 1, ["state"])["state"]
    before, after = state.reshape(8, 12, -1), turned.reshape(8, 12, -1)
    assert np.array_equal(after[:, 1:], before[:, :-1]) and np.array_equal(
        after[:, 0], before[:, -1]
    )
    kept = rotating.rotate(state, 1, ["static"])["state"]
    kept[:] = 0.0
    assert np.abs(state).sum() > 0.0  # the state given is left as it was
    with pytest.raises(RequestError, match="expected"):
        rotating.rotate(np.zeros((2, 6)), 1, ["state"])


def test_rotation_reads_only_valid_nodes():
    masked = ToyDynamics(n_lat=8, n_lon=12, positional=5.0, static=5.0, masked=13)
    result = rotation_test(masked, layer=0, parts=["state"], starts=(0, 1))
    assert _anchored(result) == {0: False, 1: False, 2: True, 3: True}


def test_a_turn_that_makes_a_pattern_its_own_opposite_has_no_power():
    """cos(2 lon) turned a quarter of the circle is -cos(2 lon): self-similarity -1. Any
    mixture of the map that stayed and the one that moved is then a multiple of one map,
    so the shift cannot say how much of each the column is."""
    sym = ToyDynamics(n_lat=8, n_lon=12)
    state = sym.initial_state().copy()
    lon = np.radians(sym.grid().lon)
    state[:, 3] = np.cos(2.0 * lon)  # u0, which channel 0 of layer 0 reads
    sym.initial_state = lambda start=0: state
    result = rotation_test(sym, layer=0, shifts=[90.0], columns=[0])
    assert result.self_similarity[0, 0, 0] == pytest.approx(-1.0, abs=1e-9)
    assert result.n_powered[0] == 0 and np.isnan(result.score[0])


@pytest.mark.parametrize("s", [-0.45, 0.0, 0.3, 0.45])
@pytest.mark.parametrize("alpha", [0.0, 0.3, 0.55, 1.0])
def test_the_score_is_the_anchored_share_whatever_the_self_similarity(alpha, s):
    """A turned map that is alpha of the original where it was and 1 - alpha of it turned
    along, the two maps correlated s: the score is 2 alpha - 1 for every s."""
    from exege.latents.rotation import _correlate, _score

    rng = np.random.default_rng(0)
    n = 4000
    original = rng.normal(size=n)
    moved = s * original + np.sqrt(1 - s * s) * rng.normal(size=n)
    turned = alpha * original + (1 - alpha) * moved
    w = np.ones(n)
    col = lambda v: v[:, None]  # noqa: E731
    stays = _correlate(col(turned), col(original), w)
    follows = _correlate(col(turned), col(moved), w)
    selfsim = _correlate(col(original), col(moved), w)
    assert _score(stays, follows, selfsim)[0] == pytest.approx(2 * alpha - 1, abs=0.05)


def test_a_column_that_vanishes_when_turned_is_not_anchored():
    class Vanishing(ToyDynamics):
        def rotate(self, state, columns, parts=("state", "static")):
            turned = super().rotate(state, columns, parts)
            turned["state"][:, 3:] = 0.0  # no forcing at all once turned
            return turned

    result = rotation_test(Vanishing(n_lat=8, n_lon=12), layer=0, columns=[0])
    assert result.n_powered[0] > 0 and not result.anchored[0] and np.isnan(result.score[0])
    assert result.summary()["columns"][0]["stays"] is None


def test_a_rotation_by_a_shift_that_is_not_a_number_is_refused(rotating):
    with pytest.raises(RequestError, match="finite"):
        rotation_test(rotating, layer=0, shifts=[float("nan")])


# -- edge cases -------------------------------------------------------------


def _feature_against_its_direction():
    """A ReLU feature on channel 1 whose decoder points the other way."""
    return Dictionary(
        encoder=np.array([[0, 1, 0, 0]], dtype=np.float32),
        encoder_bias=np.zeros(1, dtype=np.float32),
        decoder=np.array([[0, -1, 0, 0]], dtype=np.float32),
        decoder_bias=np.zeros(4, dtype=np.float32), input_mean=np.zeros(4, dtype=np.float32),
    )  # fmt: skip


def test_a_hold_finds_a_target_that_lies_against_the_direction():
    from exege.latents.mediation import _hold

    basis = _feature_against_its_direction()
    mediator = Mediator(layer=1, feature=0, basis=basis)
    latents = np.zeros((3, 4), dtype=np.float32)
    latents[:, 1] = -0.5  # off: activation 0
    edit = _hold(mediator, basis.decoder[0].astype(np.float64), np.ones(3), np.ones(3, bool))
    reached = basis.transform(edit(latents), [0])[:, 0]
    assert np.allclose(reached, 1.0, atol=1e-6)


def test_a_hold_with_nothing_to_move_leaves_every_bit_as_it_was():
    from exege.latents.mediation import _hold

    basis = _feature_against_its_direction()
    latents = np.array([[-0.0, 0.3, -0.0, 2.0]], dtype=np.float32)
    target = basis.transform(latents, [0])[:, 0]
    edit = _hold(Mediator(layer=1, feature=0, basis=basis), basis.decoder[0].astype(np.float64),
                 target, np.ones(1, bool))  # fmt: skip
    out = edit(latents)
    assert out.tobytes() == latents.tobytes()


def test_a_hold_only_run_that_is_not_a_number_is_infinitely_inexact(system):
    class Breaks:
        def __getattr__(self, name):
            return getattr(system, name)

        def run(self, state, steps, hooks=(), **kwargs):
            rollout = system.run(state, steps, hooks, **kwargs)
            if len(hooks) != steps - 1:  # only the hold-only run has a hook at every step after 0
                return rollout
            latents = {k: np.full_like(v, np.nan) for k, v in rollout.latents.items()}
            return type(rollout)(fields=rollout.fields, latents=latents)

    result = _mediate(Breaks(), Mediator(layer=1, feature=2, basis=_linear(system)))
    assert result.exactness == float("inf")


def test_constant_columns_correlate_with_nothing():
    from exege.latents.rotation import _correlate

    constant = np.full((3, 1), 3.1)
    varying = np.array([[1.0], [2.0], [4.0]])
    weights = np.array([0.1, 0.2, 0.3])
    assert np.isnan(_correlate(constant, constant, weights)).all()
    assert np.isnan(_correlate(constant, varying, weights)).all()
    assert _correlate(varying, varying, weights)[0] == pytest.approx(1.0)


def test_rotate_prints_columns_with_no_power():
    out = _invoke(["latents", "rotate", "--adapter", "toy-dynamics", "--layer", "0",
                   "--max-self-similarity", "-1"])  # fmt: skip
    assert "-" in out.splitlines()[-1]


def test_mediate_prints_a_command_that_reproduces_it_and_pins_its_mediator(planted, tmp_path):
    path, _ = planted
    system = ToyDynamics(masked=3)
    channels = save_basis(tmp_path / "channels.npz", _linear(system))
    args = ["latents", "mediate", "--adapter", "toy-dynamics", "--adapter-option", "masked=3",
            "--basis", str(path), "--layer", "1", "--feature", "0", "--amount", "1.5",
            "--time", "1", "--steps", "5", "--mediator-layer", "1", "--mediator-feature", "0",
            "--mediator-basis", str(channels)]  # fmt: skip
    out = _invoke(args)
    command = out.strip().splitlines()[-1].removeprefix("reproduce: ")
    assert "--mediator-basis-sha256" in command and "--basis-sha256" in command
    again = _invoke(shlex.split(command)[1:])
    assert again == out
    fitted = {"fitted_on": {"provenance": system.info().identity(), "layer": 1}}
    reversed_rows = Directions(mean=np.zeros(4), directions_=np.eye(4)[::-1].copy(), meta=fitted)
    save_basis(channels, reversed_rows)
    refused = CliRunner().invoke(cli, shlex.split(command)[1:])
    assert refused.exit_code != 0 and "sha256" in refused.output


def test_a_hold_that_misses_says_the_share_means_little():
    from exege.latents.mediation import MediationResult

    base = {"spec": {}, "provenance": {}, "effects": {}, "exactness": 0.0, "responses": {}}
    assert MediationResult(mediator_change=1.0, hold_rms=0.01, **base).held
    assert not MediationResult(mediator_change=1.0, hold_rms=0.5, **base).held
    assert MediationResult(mediator_change=1.0, hold_rms=0.5, **base).to_dict()["held"] is False
