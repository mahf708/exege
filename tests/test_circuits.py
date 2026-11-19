"""Circuits on a system whose circuits are planted (adapters/toy_transformer.py)."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from exege.adapters.toy_transformer import PLANTED  # noqa: E402
from exege.circuits import (  # noqa: E402
    Graph,
    LinearMetric,
    attribute,
    core,
    edge_frequency,
    faithfulness_curve,
    field_mean,
    jaccard,
    open_patchable,
    prune,
    residual_graph,
    top_k,
)
from exege.core.errors import AdapterError, RequestError  # noqa: E402


@pytest.fixture(scope="module")
def toy():
    return open_patchable(None, "toy-transformer")


@pytest.fixture(scope="module")
def runs(toy):
    return toy.states([0, 1, 2]), toy.states([3, 4, 5])


def _tracer(toy):
    g = toy.grid()
    return field_mean(g, "tracer", where=g.within(0, 90, 4000))


# -- the graph ---------------------------------------------------------------------------------


def test_the_reduced_graph_has_heads_as_senders_and_one_attention_input_per_block():
    g = residual_graph(depth=3, heads=4)
    assert len(g.senders) == 1 + 3 * 5 and len(g.receivers) == 3 * 2 + 1
    assert g.upstream("a0") == [0]
    assert [g.senders[i] for i in g.upstream("m0")] == ["embed"] + [f"a0.h{h}" for h in range(4)]
    assert len(g.upstream("out")) == len(g.senders)
    # each receiver hears everything computed before it, and nothing after
    assert g.n_edges == sum(len(g.upstream(r)) for r in g.receivers)


def test_a_graph_round_trips_and_names_its_edges():
    g = residual_graph(2, 2)
    assert Graph.from_dict(g.to_dict()) == g
    m = g.mask_of(["embed->out", "a1.h0->out"])
    assert sorted(g.names(m)) == ["a1.h0->out", "embed->out"]
    with pytest.raises(RequestError, match="not an edge"):
        g.mask_of(["m1->a0"])  # downstream to upstream
    with pytest.raises(RequestError, match="no sender"):
        g.mask_of(["nobody->out"])
    with pytest.raises(RequestError):
        g.check_mask(np.ones((2, 2)))
    assert not g.check_mask(np.ones(g.shape))[~g.edges].any()  # off-graph entries are dropped


# -- metrics -----------------------------------------------------------------------------------


def test_a_field_mean_is_area_weighted_over_its_region(toy):
    g = toy.grid()
    m = field_mean(g, "tracer")
    ones = {"tracer": np.ones((2, g.n_nodes)), "ripple": np.zeros((2, g.n_nodes))}
    assert m(ones) == pytest.approx(1.0)
    polar = {"tracer": np.where(np.abs(g.lat) > 30, 1.0, 0.0)[None]}
    assert field_mean(g, "tracer", where=np.abs(g.lat) > 30)(polar) == pytest.approx(1.0)
    with pytest.raises(RequestError):
        field_mean(g, "tracer", where=np.zeros(g.n_nodes, dtype=bool))
    with pytest.raises(RequestError, match="did not write"):
        m({"ripple": ones["ripple"]})


# -- finding the planted circuit ---------------------------------------------------------------


def test_pruning_finds_exactly_the_planted_circuit(toy, runs):
    circuit = prune(toy, *runs, _tracer(toy), tau=1e-6)
    assert sorted(circuit.edges()) == sorted(PLANTED)
    assert circuit.recovery == pytest.approx(1.0)
    assert circuit.components() == ["embed", "a1.h0"]
    assert circuit.to_dict()["n_edges"] == 3


def test_a_loose_threshold_prunes_more_and_recovers_less(toy, runs):
    tight = prune(toy, *runs, field_mean(toy.grid(), "ripple"), tau=1e-6)
    loose = prune(toy, *runs, field_mean(toy.grid(), "ripple"), tau=0.3)
    assert loose.n_edges < tight.n_edges
    assert abs(1 - loose.recovery) <= 0.3 * loose.n_edges + 1e-9  # each kept step moved < tau
    assert tight.recovery == pytest.approx(1.0)


def test_eap_agrees_with_patching_where_the_path_is_linear(toy, runs):
    m = _tracer(toy)
    exact = attribute(toy, *runs, m, method="patch")
    eap = attribute(toy, *runs, m, method="eap")
    np.testing.assert_allclose(eap.scores, exact.scores, atol=1e-7)
    assert {name for name, s in exact.ranked(3)} == set(PLANTED)
    assert attribute(toy, *runs, m).method == "eap"  # auto takes the gradient when it can


def test_eap_needs_a_linear_metric(toy, runs):
    def nonlinear(out):
        return float(np.abs(out["tracer"]).mean())

    assert attribute(toy, *runs, nonlinear).method == "patch"
    with pytest.raises(RequestError, match="LinearMetric"):
        attribute(toy, *runs, nonlinear, method="eap")


def test_the_faithfulness_curve_is_flat_once_the_circuit_is_in(toy, runs):
    m = _tracer(toy)
    a = attribute(toy, *runs, m, method="patch")
    curve = faithfulness_curve(toy, *runs, m, a, sizes=[0, 1, 3, 10])
    assert curve.recoveries[0] == pytest.approx(0.0)
    assert curve.recoveries[2] == pytest.approx(1.0) and curve.recoveries[3] == pytest.approx(1.0)


def test_nothing_to_explain_is_refused(toy):
    same = toy.states([0])
    with pytest.raises(RequestError, match="same metric"):
        prune(toy, same, same, _tracer(toy))


# -- comparing circuits ------------------------------------------------------------------------


def test_overlap_frequency_and_core():
    g = residual_graph(1, 2)
    a, b = g.mask_of(["embed->out", "a0.h0->out"]), g.mask_of(["embed->out", "a0.h1->out"])
    assert jaccard(a, a) == 1.0 and jaccard(a, b) == pytest.approx(1 / 3)
    freq = edge_frequency([a, b])
    assert freq[g.sender("embed"), g.receiver("out")] == 1.0
    assert g.names(core([a, b], at=0.9)) == ["embed->out"]
    assert top_k(np.where(g.edges, 1.0, 0.0), g, 2).sum() == 2


def test_circuits_from_different_bases_share_the_planted_core(toy):
    m = _tracer(toy)
    found = [prune(toy, toy.states([i]), toy.states([i + 8]), m, tau=1e-6) for i in range(4)]
    assert sorted(toy.graph().names(core(found))) == sorted(PLANTED)


def test_an_adapter_that_cannot_be_patched_is_refused():
    with pytest.raises(AdapterError, match="Patchable"):
        open_patchable(None, "toy-dynamics")


def test_a_linear_metric_refuses_nan_where_it_reads():
    m = LinearMetric({"x": np.array([0.0, 1.0])})
    assert m({"x": np.array([[np.nan, 2.0]])}) == 2.0  # a zero weight ignores the node
    with pytest.raises(RequestError):
        m({"x": np.array([[1.0, np.nan]])})


# -- the command line --------------------------------------------------------------------------


def test_the_prune_command_prints_the_planted_circuit_and_writes_json(tmp_path):
    import json

    from click.testing import CliRunner

    from exege._cli import cli

    out = tmp_path / "c.json"
    args = [
        "circuits",
        "prune",
        "--adapter",
        "toy-transformer",
        "--field",
        "tracer",
        "--lat",
        "0",
        "--lon",
        "90",
        "--radius-km",
        "4000",
        "--base",
        "0-2",
        "--source-starts",
        "3,4,5",
        "--tau",
        "1e-6",
        "--out",
        str(out),
    ]
    r = CliRunner().invoke(cli, args)
    assert r.exit_code == 0, r.output
    assert "3 of 21" in r.output and "a1.h0->out" in r.output
    assert sorted(json.loads(out.read_text())["edges"]) == sorted(PLANTED)
    bad = CliRunner().invoke(
        cli, ["circuits", "graph", "--adapter", "toy-transformer", "--option", "nonsense"]
    )
    assert bad.exit_code != 0 and "key=value" in bad.output
