"""The steering arms beyond the first four, against answers planted in the toy system.

References and the two modes that use them, open and closed loops, comparison and
matched arms, outcomes measured in a region, many cases, and dose curves with an amount
chosen leave-one-out. ``ToyDynamics`` is linear: adding ``a`` to channel 0 of layer 1 moves
temperature by ``gain * a`` on that step and ``decay`` times as much on each step after.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from conftest import MemorySource  # noqa: E402
from exege.adapters.toy_dynamics import ToyDynamics  # noqa: E402
from exege.core.errors import RequestError  # noqa: E402
from exege.latents import (  # noqa: E402
    Box,
    Case,
    Dictionary,
    Directions,
    Hook,
    Outcome,
    Region,
    Steer,
    basis_hash,
    dose_curve,
    feature_reference,
    leave_one_out,
    load_basis,
    mean_difference,
    run_cases,
    run_steering,
    save_basis,
)

GAIN, DECAY, STEPS, T0, AMOUNT = 2.0, 0.5, 5, 1, 1.5


@pytest.fixture
def system():
    return ToyDynamics(masked=3, gain=GAIN, decay=DECAY)


def _weights(system):
    grid = system.grid()
    return grid.weights()[grid.valid], grid.valid


def _control_h(system, seed=0, t=T0, layer=1, state=None):
    state = system.initial_state() if state is None else state
    rollout = system.run(state, STEPS, (), noise_seed=seed, record=[(layer, t)])
    return rollout.latents[(layer, t)].astype(np.float64)


def _step(result, arm="feature", **where):
    return result.pairing(arm, 0, **where).differences["temperature"][T0]


# -- references ---------------------------------------------------------------


def test_toward_one_sets_the_activation_to_the_reference(system):
    n = system.grid().n_nodes
    reference = np.linspace(0.0, 1.0, n)
    steer = Steer(layer=1, feature=0, mode="toward", amount=1.0, times=(T0,), reference=reference)
    result = run_steering(system, system.planted_dictionary(), steer, steps=STEPS, n_random=2)
    active = np.maximum(_control_h(system)[:, 0], 0.0)
    weights, valid = _weights(system)
    assert _step(result) == pytest.approx(GAIN * weights @ (reference - active)[valid], abs=1e-4)


@pytest.mark.parametrize("amount", [0.0, 2.0, 3.5])
def test_excess_scales_only_what_lies_above_the_reference(system, amount):
    reference = np.full(system.grid().n_nodes, 0.3)
    steer = Steer(layer=1, feature=0, mode="excess", amount=amount, times=(T0,),
                  reference=reference)  # fmt: skip
    result = run_steering(system, system.planted_dictionary(), steer, steps=STEPS, n_random=2)
    active = np.maximum(_control_h(system)[:, 0], 0.0)
    excess = np.maximum(active - 0.3, 0.0)
    weights, valid = _weights(system)
    expected = GAIN * (amount - 1.0) * (weights @ excess[valid])
    assert _step(result) == pytest.approx(expected, abs=1e-4)


def test_the_identity_amount_of_each_mode_changes_nothing(system):
    reference = np.zeros(system.grid().n_nodes)
    for mode in ("add", "scale", "toward", "excess"):
        kwargs = {"reference": reference} if mode in ("toward", "excess") else {}
        steer = Steer(layer=1, feature=0, mode=mode, times=(T0,), **kwargs)
        steer = Steer(layer=1, feature=0, mode=mode, amount=steer.identity_amount, times=(T0,),
                      **kwargs)  # fmt: skip
        result = run_steering(system, system.planted_dictionary(), steer, steps=STEPS, n_random=1)
        assert np.all(result.pairing("feature", 0).differences["temperature"] == 0.0)
    assert Steer(layer=1, feature=0, mode="clamp").identity_amount is None


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"mode": "toward"}, "needs a reference"),
        ({"mode": "add", "reference": np.zeros(3)}, "takes no reference"),
        ({"mode": "excess", "reference": np.zeros((2, 2))}, "one value per node"),
        ({"loop": "spiral"}, "loop must be"),
    ],
)
def test_a_reference_steer_is_checked_when_it_is_made(kwargs, match):
    with pytest.raises(RequestError, match=match):
        Steer(layer=1, feature=0, **kwargs)


def test_a_reference_that_does_not_fit_the_grid_or_is_missing_where_it_acts_is_refused(system):
    short = Steer(layer=1, feature=0, mode="toward", times=(T0,), reference=np.zeros(3))
    with pytest.raises(RequestError, match="value"):
        run_steering(system, system.planted_dictionary(), short, steps=STEPS, n_random=1)
    holed = np.zeros(system.grid().n_nodes)
    holed[5] = np.nan
    gap = Steer(layer=1, feature=0, mode="toward", times=(T0,), reference=holed)
    with pytest.raises(RequestError, match="not finite on a valid node"):
        run_steering(system, system.planted_dictionary(), gap, steps=STEPS, n_random=1)
    holed[5], holed[0] = 0.0, np.nan  # node 0 is masked: what it holds there does not matter
    fine = Steer(layer=1, feature=0, mode="toward", times=(T0,), reference=holed)
    run_steering(system, system.planted_dictionary(), fine, steps=STEPS, n_random=1)


def test_a_reference_is_named_in_the_spec_by_content_not_carried(system):
    reference = np.full(system.grid().n_nodes, 0.25)
    steer = Steer(layer=1, feature=0, mode="toward", times=(T0,), reference=reference,
                  reference_meta={"times": ["t0"]})  # fmt: skip
    spec = run_steering(system, system.planted_dictionary(), steer, steps=STEPS, n_random=1).spec
    named = spec["steer"]["reference"]
    assert named["times"] == ["t0"] and named["mean"] == pytest.approx(0.25)
    assert len(named["sha256"]) == 64


# -- open and closed loops ----------------------------------------------------


def _scale_twice(loop):
    return Steer(layer=1, feature=0, mode="scale", amount=2.0, times=(1, 2), loop=loop)


def test_without_feedback_the_two_loops_agree_exactly(system):
    basis = system.planted_dictionary()
    open_, closed = (
        run_steering(system, basis, _scale_twice(loop), steps=STEPS, n_random=2)
        for loop in ("open", "closed")
    )
    for arm in ("feature", "reconstruction"):
        a = open_.pairing(arm, 0).differences["temperature"]
        b = closed.pairing(arm, 0).differences["temperature"]
        assert np.array_equal(a, b)


def test_with_feedback_a_closed_loop_reads_what_its_earlier_edits_did():
    system = ToyDynamics(masked=3, gain=GAIN, decay=DECAY, feedback=0.4)
    basis = system.planted_dictionary()
    state = system.initial_state()
    control = system.run(state, STEPS, (), noise_seed=0, record=[(1, 1), (1, 2)])

    def scaled_own(h):  # what a closed loop does: double the activation this run has
        out = h.copy()
        out[:, 0] += np.maximum(h[:, 0], 0.0)
        return out

    def scaled_control(t):  # what an open loop does: the control's activation, doubled
        active = np.maximum(control.latents[(1, t)][:, 0], 0.0)

        def edit(h):
            out = h.copy()
            out[:, 0] += active.astype(h.dtype)
            return out

        return edit

    by_hand = {
        "closed": [Hook(1, 1, scaled_own), Hook(1, 2, scaled_own)],
        "open": [Hook(1, 1, scaled_control(1)), Hook(1, 2, scaled_control(2))],
    }
    found = {}
    for loop, hooks in by_hand.items():
        result = run_steering(system, basis, _scale_twice(loop), steps=STEPS, n_random=1)
        manual = system.run(state, STEPS, hooks, noise_seed=0).fields["temperature"]
        weights, valid = _weights(system)
        expected = manual[:, valid] @ weights - control.fields["temperature"][:, valid] @ weights
        found[loop] = result.pairing("feature", 0).differences["temperature"]
        assert np.allclose(found[loop], expected, atol=1e-5)
    assert not np.allclose(found["open"], found["closed"], atol=1e-3)


# -- comparison and matched arms ----------------------------------------------


def _directions(system, rows):
    rows = np.asarray(rows, dtype=np.float64)
    return Directions(
        mean=np.zeros(4), directions_=rows / np.linalg.norm(rows, axis=1, keepdims=True),
        meta={"fitted_on": {"provenance": system.info().identity(), "layer": 1}},
    )  # fmt: skip


def test_a_comparison_along_the_feature_does_what_the_feature_does(system):
    same = _directions(system, [[1, 0, 0, 0], [0, 0, 0, 1]])
    result = run_steering(
        system, system.planted_dictionary(), Steer(layer=1, feature=0, amount=AMOUNT, times=(T0,)),
        steps=STEPS, n_random=2, comparisons={"same": (same, 0), "pressure": (same, 1)},
    )  # fmt: skip
    effect = result.effects["temperature"]
    assert effect.comparison_responses["same"] == pytest.approx(effect.feature_response)
    assert effect.comparison_responses["pressure"] == pytest.approx(0.0, abs=1e-9)
    assert result.effects["pressure"].comparison_responses["pressure"] > 0.0
    assert set(result.spec["comparisons"]) == {"same", "pressure"}


def test_a_comparison_of_another_width_is_refused(system):
    wide = Directions(mean=np.zeros(5), directions_=np.eye(5)[:1])
    with pytest.raises(RequestError, match="channel"):
        run_steering(system, system.planted_dictionary(), Steer(layer=1, feature=0, times=(T0,)),
                     steps=STEPS, n_random=1, comparisons={"wide": (wide, 0)})  # fmt: skip


def test_matched_features_are_real_ones_that_reach_elsewhere(system):
    steer = Steer(layer=1, feature=0, mode="scale", amount=3.0, times=(T0,))
    result = run_steering(system, system.planted_dictionary(), steer, steps=STEPS, n_random=2,
                          n_matched=2)  # fmt: skip
    matched = result.spec["matched"]
    assert sorted(matched["features"]) == [1, 2]
    temperature = result.effects["temperature"]
    assert temperature.matched_rank == 1
    assert np.allclose(temperature.matched_responses, 0.0, atol=1e-9)
    moisture = result.effects["moisture"]
    assert moisture.matched_rank == 3  # feature 0 does nothing there: last, ties and all


def test_matched_features_are_picked_by_what_they_carry_closest_first():
    from exege.latents import matched_features

    basis = Dictionary(
        encoder=np.eye(4, dtype=np.float32), encoder_bias=np.zeros(4, dtype=np.float32),
        decoder=np.eye(4, dtype=np.float32), decoder_bias=np.zeros(4, dtype=np.float32),
        input_mean=np.zeros(4, dtype=np.float32),
    )  # fmt: skip
    carried = np.array([1.0, 0.0, 4.0, 1.5])
    assert matched_features(basis, 0, carried, n=3) == [3, 2]  # 1 carries nothing
    with pytest.raises(RequestError, match="carries nothing"):
        matched_features(basis, 1, carried, n=1)


def test_matched_features_under_a_reference_mode_need_their_own_references(system):
    steer = Steer(layer=1, feature=0, mode="excess", amount=2.0, times=(T0,),
                  reference=np.zeros(system.grid().n_nodes))  # fmt: skip
    with pytest.raises(RequestError, match="reference_for"):
        run_steering(system, system.planted_dictionary(), steer, steps=STEPS, n_matched=1)
    calls = []

    def reference_for(feature):
        calls.append(feature)
        return np.zeros(system.grid().n_nodes)

    run_steering(system, system.planted_dictionary(), steer, steps=STEPS, n_random=1,
                 n_matched=2, reference_for=reference_for)  # fmt: skip
    assert sorted(calls) == [1, 2]


# -- outcomes -----------------------------------------------------------------


def test_an_outcome_in_a_region_is_measured_there_only(system):
    grid = system.grid()
    region = Box(lat_min=0.0, lat_max=90.0, lon_min=0.0, lon_max=360.0)
    steer = Steer(layer=1, feature=0, amount=AMOUNT, times=(T0,))
    result = run_steering(system, system.planted_dictionary(), steer, steps=STEPS, n_random=1,
                          outcome=Outcome(region=region))  # fmt: skip
    # adding everywhere moves every valid node by the same amount, so any mean sees it whole
    assert _step(result) == pytest.approx(AMOUNT * GAIN, abs=1e-5)
    rollout = system.run(system.initial_state(), STEPS, (), noise_seed=0)
    nodes = region.nodes(grid)
    share = grid.weights()[nodes] / grid.weights()[nodes].sum()
    control = result.arm("control", 0).outcomes["temperature"]
    assert np.allclose(control, rollout.fields["temperature"][:, nodes] @ share)


def test_the_minimum_of_a_field_is_its_minimum(system):
    steer = Steer(layer=1, feature=0, amount=AMOUNT, times=(T0,))
    result = run_steering(system, system.planted_dictionary(), steer, steps=STEPS, n_random=1,
                          outcome=Outcome(statistic="min"))  # fmt: skip
    rollout = system.run(system.initial_state(), STEPS, (), noise_seed=0)
    valid = system.grid().valid
    expected = rollout.fields["temperature"][:, valid].min(axis=1)
    assert np.allclose(result.arm("control", 0).outcomes["temperature"], expected)
    assert result.spec["outcome"] == {"statistic": "min", "region": None}
    with pytest.raises(RequestError, match="outcome is one of"):
        Outcome(statistic="median")


def test_an_outcome_region_with_no_valid_node_is_refused_before_running(system):
    empty = Region(lat=0.0, lon=0.0, radius_km=1.0)
    with pytest.raises(RequestError, match="no valid nodes"):
        run_steering(system, system.planted_dictionary(), Steer(layer=1, feature=0),
                     steps=STEPS, outcome=Outcome(region=empty))  # fmt: skip


# -- cases --------------------------------------------------------------------


def test_each_case_starts_where_it_says_and_its_region_holds_the_edit(system):
    region = Box(lat_min=0.0, lat_max=90.0, lon_min=0.0, lon_max=360.0)
    cases = [Case(start=0, label="a"), Case(start=1, region=region, label="b")]
    study = run_cases(
        system, system.planted_dictionary(), Steer(layer=1, feature=0, amount=AMOUNT, times=(T0,)),
        cases, steps=STEPS, n_random=2,
    )  # fmt: skip
    a, b = study.results
    assert a.spec["steer"]["nodes"] is None
    assert b.spec["steer"]["nodes"] == [int(n) for n in region.nodes(system.grid())]
    assert b.spec["outcome"]["region"]["kind"] == "box"
    # inside its own region, an edit restricted to that region is felt whole
    assert _step(b) == pytest.approx(AMOUNT * GAIN, abs=1e-5)
    summary = study.summary()["temperature"]
    assert summary["cases"] == ["a", "b"] and summary["positive"] == 2


def test_cases_must_be_named_apart_and_not_given_a_state(system):
    steer = Steer(layer=1, feature=0, times=(T0,))
    with pytest.raises(RequestError, match="must differ"):
        run_cases(system, system.planted_dictionary(), steer, [Case(0), Case(0)], steps=STEPS)
    with pytest.raises(RequestError, match="initial_state"):
        run_cases(system, system.planted_dictionary(), steer, [Case(0)], steps=STEPS,
                  initial_state=system.initial_state())  # fmt: skip


# -- dose curves --------------------------------------------------------------


def test_leave_one_out_chooses_without_looking_at_the_case():
    scores = np.array(
        [
            [5.0, 1.0, 9.0],  # case 0 alone would pick column 1 ...
            [5.0, 9.0, 1.0],
            [5.0, 8.0, 2.0],
        ]
    )
    picks, values = leave_one_out(scores)
    assert picks.tolist() == [2, 1, 0]  # ... but the others point at column 2; a tie, the first
    assert values.tolist() == [9.0, 9.0, 5.0]
    picks, _ = leave_one_out(-scores, lower_is_better=False)
    assert picks.tolist() == [2, 1, 0]
    with pytest.raises(RequestError, match="two cases"):
        leave_one_out(scores[:1])


def test_a_dose_curve_grows_as_planted_and_finds_the_amount_that_meets_the_target(system):
    basis = system.planted_dictionary()
    steer = Steer(layer=1, feature=0, mode="add", times=(T0,))
    amounts = (0.0, 1.0, 2.0, 3.0)
    cases = [Case(0, label="a"), Case(1, label="b"), Case(2, label="c")]
    targets = {}
    for case in cases:  # what each case "should" have done: its control plus an add of 2
        rollout = system.run(system.initial_state(case.start), STEPS, (), noise_seed=0)
        weights, valid = _weights(system)
        series = rollout.fields["temperature"][:, valid] @ weights
        planted = np.array([0.0] * T0 + [2.0 * GAIN * DECAY**k for k in range(STEPS - T0)])
        targets[case.name] = {"temperature": series + planted}
    curve = dose_curve(system, basis, steer, amounts, steps=STEPS, cases=cases,
                       fields=["temperature"], targets=targets)  # fmt: skip
    response = curve.responses[:, :, 0]
    per_unit = GAIN * sum(DECAY**k for k in range(STEPS - T0)) / (STEPS - T0)
    assert np.allclose(response, np.array(amounts)[None, :] * per_unit, atol=1e-5)
    selection = curve.selections["temperature"]
    assert selection.loo_amounts == (2.0, 2.0, 2.0)
    assert selection.medians["loo"] == pytest.approx(0.0, abs=1e-5)
    assert selection.medians["baseline"] > 1.0
    assert curve.baseline_errors.shape == (3, 1)


def test_a_dose_curve_refuses_what_it_cannot_score(system):
    basis, steer = system.planted_dictionary(), Steer(layer=1, feature=0, times=(T0,))
    with pytest.raises(RequestError, match="at least one amount"):
        dose_curve(system, basis, steer, (), steps=STEPS)
    with pytest.raises(RequestError, match="must differ"):
        dose_curve(system, basis, steer, (1.0, 1.0), steps=STEPS)
    with pytest.raises(RequestError, match="no target for case"):
        dose_curve(system, basis, steer, (1.0,), steps=STEPS, targets={"x": {}})
    with pytest.raises(RequestError, match="one per step"):
        dose_curve(system, basis, steer, (1.0,), steps=STEPS, fields=["temperature"],
                   targets={"0": {"temperature": [0.0]}})  # fmt: skip


# -- references and directions read from an archive ---------------------------


def _archive():
    """Channel 0 is 1 at every node at the first two times and 3 at the last two; channel
    1 is a fixed pattern; the rest are zero."""
    data = np.zeros((4, 48, 3), dtype=np.float32)
    data[:2, :, 0], data[2:, :, 0] = 1.0, 3.0
    data[:, :, 1] = np.linspace(-1.0, 1.0, 48)
    return MemorySource({0: data})


def _identity_dictionary(source):
    return Dictionary(
        encoder=np.eye(3, dtype=np.float32), encoder_bias=np.zeros(3, dtype=np.float32),
        decoder=np.eye(3, dtype=np.float32), decoder_bias=np.zeros(3, dtype=np.float32),
        input_mean=np.zeros(3, dtype=np.float32),
        meta={"fitted_on": {"provenance": source.info().identity(), "layer": 0}},
    )  # fmt: skip


def test_a_reference_is_the_feature_averaged_over_the_times_named():
    source = _archive()
    basis = _identity_dictionary(source)
    reference, meta = feature_reference(source, basis, layer=0, feature=0, times=[0, 3])
    assert np.allclose(reference, 2.0)
    resample, _ = feature_reference(source, basis, layer=0, feature=1, times=["t1"])
    assert np.allclose(resample, np.maximum(np.linspace(-1.0, 1.0, 48), 0.0))
    assert meta["times"] == ["t0", "t3"] and meta["feature"] == 0
    with pytest.raises(RequestError, match="at least one time"):
        feature_reference(source, basis, layer=0, feature=0, times=[])


def test_a_mean_difference_points_from_quiet_to_event(tmp_path):
    source = _archive()
    direction = mean_difference(source, layer=0, event_times=[2, 3], quiet_times=[0, 1])
    assert np.allclose(direction.directions()[0], [1.0, 0.0, 0.0])
    assert direction.meta["length"] == pytest.approx(2.0)
    assert np.allclose(direction.transform(np.array([[3.0, 0.0, 0.0]])), 2.0)
    path = save_basis(tmp_path / "dom.npz", direction)
    again = load_basis(path)
    assert isinstance(again, Directions) and again.meta["hash_status"] == "verified"
    assert basis_hash(again) == basis_hash(direction)
    with pytest.raises(RequestError, match="both event and quiet"):
        mean_difference(source, layer=0, event_times=[1, 2], quiet_times=[1])


def test_directions_must_have_unit_length():
    with pytest.raises(ValueError, match="unit length"):
        Directions(mean=np.zeros(2), directions_=np.array([[2.0, 0.0]]))


# -- what the reviews asked of the arms ---------------------------------------


def test_a_comparison_fitted_elsewhere_is_refused(system):
    elsewhere = Directions(
        mean=np.zeros(4), directions_=np.eye(4)[:1],
        meta={"fitted_on": {"provenance": system.info().identity(), "layer": 0}},
    )  # fmt: skip
    with pytest.raises(RequestError, match="comparison 'x'.*not fitted here"):
        run_steering(system, system.planted_dictionary(), Steer(layer=1, feature=0, times=(T0,)),
                     steps=STEPS, n_random=1, comparisons={"x": (elsewhere, 0)})  # fmt: skip


def test_a_reference_read_on_other_nodes_is_refused(system):
    reference = np.zeros(system.grid().n_nodes)
    steer = Steer(layer=1, feature=0, mode="toward", times=(T0,), reference=reference,
                  reference_meta={"grid": "0" * 64})  # fmt: skip
    with pytest.raises(RequestError, match="laid out otherwise"):
        run_steering(system, system.planted_dictionary(), steer, steps=STEPS, n_random=1)
    from exege.latents.grid import node_digest

    fine = replace_meta(steer, {"grid": node_digest(system.grid())})
    run_steering(system, system.planted_dictionary(), fine, steps=STEPS, n_random=1)


def replace_meta(steer, meta):
    from dataclasses import replace

    return replace(steer, reference_meta=meta)


def test_in_a_closed_loop_other_directions_are_given_what_the_feature_did():
    """With feedback, a closed loop's second edit depends on the first. A comparison
    direction is given the feature arm's own changes, step by step, so it stays the same
    size: its run equals a run that adds those changes along it by hand."""
    system = ToyDynamics(masked=3, gain=GAIN, decay=DECAY, feedback=0.4)
    basis = system.planted_dictionary()
    mixed = _directions(system, [[1, 1, 0, 0]])
    steer = Steer(layer=1, feature=0, mode="scale", amount=2.0, times=(1, 2), loop="closed")
    result = run_steering(system, basis, steer, steps=STEPS, n_random=1,
                          comparisons={"mixed": (mixed, 0)})  # fmt: skip
    state = system.initial_state()
    applied = {}

    def doubled(t):
        def edit(h):
            applied[t] = np.maximum(h[:, 0].astype(np.float64), 0.0)
            out = h.copy()
            out[:, 0] += applied[t].astype(h.dtype)
            return out

        return edit

    system.run(state, STEPS, [Hook(1, 1, doubled(1)), Hook(1, 2, doubled(2))], noise_seed=0)
    unit = np.array([1.0, 1.0, 0.0, 0.0]) / np.sqrt(2.0)

    def pushed(t):
        return lambda h: h + (applied[t][:, None] * unit[None, :]).astype(h.dtype)

    manual = system.run(state, STEPS, [Hook(1, 1, pushed(1)), Hook(1, 2, pushed(2))],
                        noise_seed=0).fields["temperature"]  # fmt: skip
    control = system.run(state, STEPS, (), noise_seed=0).fields["temperature"]
    weights, valid = _weights(system)
    expected = manual[:, valid] @ weights - control[:, valid] @ weights
    found = result.pairing("comparison", 0, label="mixed").differences["temperature"]
    assert np.allclose(found, expected, atol=1e-5)


def test_a_zero_edit_does_not_rank_first(system):
    steer = Steer(layer=1, feature=0, amount=0.0, times=(T0,))
    effect = run_steering(system, system.planted_dictionary(), steer, steps=STEPS, n_random=4,
                          n_matched=2).effects["temperature"]  # fmt: skip
    assert effect.rank == 5 and effect.matched_rank == 3 and effect.p_value == 1.0


def test_directions_can_be_the_basis_steered(system):
    lin = _directions(system, np.eye(4))
    result = run_steering(system, lin, Steer(layer=1, feature=0, amount=AMOUNT, times=(T0,)),
                          steps=STEPS, n_random=2)  # fmt: skip
    assert _step(result) == pytest.approx(AMOUNT * GAIN, abs=1e-5)
    assert np.allclose(result.pairing("reconstruction", 0).differences["temperature"], 0.0,
                       atol=1e-5)  # a complete basis reconstructs exactly  # fmt: skip


def test_latents_recorded_as_not_numbers_are_refused(system):
    class Broken:
        def __getattr__(self, name):
            return getattr(system, name)

        def run(self, *args, **kwargs):
            rollout = system.run(*args, **kwargs)
            latents = {k: np.full_like(v, np.nan) for k, v in rollout.latents.items()}
            return type(rollout)(fields=rollout.fields, latents=latents)

    with pytest.raises(RequestError, match="not finite on valid nodes at layer"):
        run_steering(Broken(), system.planted_dictionary(), Steer(layer=1, feature=0),
                     steps=STEPS, n_random=1)  # fmt: skip


def test_cases_that_start_apart_respond_apart(system):
    """Scaling depends on the activation there is, so cases that start from different
    states respond differently, each as its own control says."""
    steer = Steer(layer=1, feature=0, mode="scale", amount=3.0, times=(T0,))
    study = run_cases(system, system.planted_dictionary(), steer,
                      [Case(0), Case(1), Case(2)], steps=STEPS, n_random=1)  # fmt: skip
    weights, valid = _weights(system)
    for case, result in zip(study.cases, study.results, strict=True):
        h = _control_h(system, state=system.initial_state(case.start))
        expected = GAIN * 2.0 * (weights @ np.maximum(h[:, 0], 0.0)[valid])
        assert _step(result) == pytest.approx(expected, abs=1e-4)
    responses = study.responses("temperature")
    assert len(set(np.round(responses, 6))) == 3


def test_matched_features_take_their_own_references(system):
    """Under excess, a matched feature's reference decides how much of it is scaled: a
    reference above every activation scales nothing."""
    n = system.grid().n_nodes
    steer = Steer(layer=1, feature=0, mode="excess", amount=3.0, times=(T0,),
                  reference=np.zeros(n))  # fmt: skip
    high = run_steering(system, system.planted_dictionary(), steer, steps=STEPS, n_random=1,
                        n_matched=2, reference_for=lambda j: np.full(n, 1e6))  # fmt: skip
    assert np.allclose(high.effects["moisture"].matched_responses, 0.0)
    low = run_steering(system, system.planted_dictionary(), steer, steps=STEPS, n_random=1,
                       n_matched=2, reference_for=lambda j: np.zeros(n))  # fmt: skip
    assert np.abs(low.effects["moisture"].matched_responses).max() > 1e-3
    assert set(low.spec["matched"]["references"]) == {"1", "2"}


def test_leave_one_out_parts_from_hindsight_when_the_best_amount_swings():
    errors = np.array([[3.0, 1.0, 4.0], [3.0, 4.0, 1.0], [3.0, 4.0, 1.5], [3.0, 1.2, 4.0]])
    picks, loo = leave_one_out(errors)
    oracle = errors.min(axis=1)
    assert np.median(loo) > np.median(oracle)
    assert picks.tolist() == [2, 1, 1, 2]


def test_a_mean_difference_weighs_nodes_by_their_area():
    """A difference only near a pole, where nodes are small, is a small direction next to
    the same difference at the equator, where they are large."""
    source = MemorySource({0: np.zeros((2, 48, 2), dtype=np.float32)})
    grid = source.grid()
    data = np.zeros((2, 48, 2), dtype=np.float32)
    polar = np.abs(grid.lat) > 70.0
    data[1, polar, 0] = 1.0
    data[1, ~polar, 1] = 1.0
    source = MemorySource({0: data})
    direction = mean_difference(source, layer=0, event_times=[1], quiet_times=[0])
    w = grid.weights()
    expected = np.array([w[polar].sum(), w[~polar].sum()])
    assert np.allclose(direction.directions()[0], expected / np.linalg.norm(expected))
