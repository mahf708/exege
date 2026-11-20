"""The samudra adapter replays fme's Samudra.forward exactly and its edges mean what they say."""

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("fme") is None or importlib.util.find_spec("torch") is None,
    reason="needs fme and torch",
)


@pytest.fixture(scope="module")
def toy():
    from exege.adapters.samudra import Samudra

    return Samudra(n_lat=16, n_lon=32, seed=0)


def test_an_unpatched_step_is_the_module_forward(toy):
    import torch

    s = toy.states([0, 1, 2])
    with torch.no_grad():
        y = toy.net(s.x.clone())
    out = toy.step(s)
    for i, n in enumerate(toy.out_names):
        ref = (y[:, i] * toy.stds[n] + toy.means[n]).reshape(3, -1).cpu().numpy()
        np.testing.assert_array_equal(out[n], ref)


def test_the_two_level0_senders_add_up_to_layer0(toy):
    import torch

    s = toy.states([0])
    L0 = toy.net.layers[0]
    with torch.no_grad():
        _, vals = toy._run(s.x, lambda r, v: v)
        ref = L0(s.x.clone())
    torch.testing.assert_close(
        vals["l0.skip_module"] + vals["l0.convblock"], ref, rtol=0, atol=1e-12
    )


def test_skip0_and_deep_path_split_a_linear_response(toy):
    """With base = perturbed and source = clean, opening only the skip.0 edges plus the deep path's
    first edge (enc1) recovers the full response, up to the network's nonlinearity."""
    from exege.circuits.contract import EdgePatch

    clean = toy.states([0])
    pert = toy.perturb(clean, {"FLUS": np.full(toy.grid().n_nodes, 1e-3)})
    g = toy.graph()
    full = toy.step(pert)["sst"] - toy.step(clean)["sst"]

    def opened(edges):
        m = g.mask(0.0)
        for s, r in [("input", "l0.skip_module"), ("input", "l0.convblock")] + edges:
            m[g.sender(s), g.receiver(r)] = 1.0
        return toy.step(pert, EdgePatch(clean, m))["sst"] - toy.step(clean)["sst"]

    skip0 = opened([("l0.skip_module", "out"), ("l0.convblock", "out")])
    m = g.mask(1.0)
    m[g.sender("l0.skip_module"), g.receiver("enc1")] = 0.0
    m[g.sender("l0.convblock"), g.receiver("enc1")] = 0.0
    deep = full - (toy.step(pert, EdgePatch(clean, m))["sst"] - toy.step(clean)["sst"])
    np.testing.assert_allclose(skip0 + deep, full, rtol=1e-3, atol=1e-9)
