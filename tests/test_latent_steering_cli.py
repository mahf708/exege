"""The steering commands: the new arms, many starts, references read from an archive and
pinned, dose curves, and difference-of-means directions, each run and reproduced from the
command it prints."""

from __future__ import annotations

import json
import shlex

import pytest

np = pytest.importorskip("numpy")

from click.testing import CliRunner  # noqa: E402

from exege._cli import cli  # noqa: E402
from exege.adapters.toy_dynamics import ToyDynamics  # noqa: E402
from exege.latents import Directions, save_basis  # noqa: E402


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


_BASE = ["latents", "steer", "--adapter", "toy-dynamics", "--adapter-option", "masked=3",
         "--layer", "1", "--feature", "0", "--time", "1", "--steps", "5"]  # fmt: skip


def test_steer_runs_the_new_arms_and_reproduces_itself(planted):
    path, along = planted
    args = [*_BASE, "--basis", str(path), "--mode", "scale", "--amount", "3", "--matched", "2",
            "--compare", f"same={along}:0", "--loop", "closed", "--random-draws", "4",
            "--outcome", "max", "--field", "temperature"]  # fmt: skip
    out = _invoke(args)
    assert "MATCHED_RANK" in out and "SAME" in out and "closed loop" in out
    command = out.strip().splitlines()[-1].removeprefix("reproduce: ")
    again = _invoke(shlex.split(command)[1:])
    assert again.splitlines()[:6] == out.splitlines()[:6]


def test_steer_from_several_starts_reports_each_and_the_spread(planted):
    path, _ = planted
    args = [*_BASE, "--basis", str(path), "--amount", "1.5", "--start", "0", "--start", "1",
            "--lat", "40", "--lon", "60", "--radius-km", "4000", "--random-draws", "3"]  # fmt: skip
    out = _invoke(args)
    assert "case 0" in out and "case 1" in out and "across cases" in out
    data = json.loads(_invoke([*args, "--json"]))
    assert [c["label"] for c in data["cases"]] == ["0", "1"]


def test_steer_refuses_a_reference_mode_without_a_reference(planted):
    path, _ = planted
    result = CliRunner().invoke(cli, [*_BASE, "--basis", str(path), "--mode", "excess",
                                      "--amount", "2"])  # fmt: skip
    assert result.exit_code != 0 and "--reference-from" in result.output


def _archive_of(system, path, starts=(0, 1, 2)):
    """An archive of ``system``'s own network: its grid, its identity, and layer 1's
    latents at the first step of each start, as an exporter would keep them."""
    from exege.adapters.latent_archive import write_archive

    info = system.info()
    latents = [
        system.run(system.initial_state(s), 1, (), noise_seed=0, record=[(1, 0)]).latents[(1, 0)]
        for s in starts
    ]
    layer0 = np.zeros_like(np.stack(latents))
    return write_archive(
        path, grid=system.grid(), times=[f"0001-01-0{i + 1}T00:00:00" for i in range(len(starts))],
        layers=[("encoder", layer0), ("mixed", np.stack(latents))], model=info.model,
        component=info.component, checkpoint=info.checkpoint, calendar="noleap",
        timestep_seconds=21600, dtype="float32",
    )  # fmt: skip


def test_steer_with_a_reference_read_from_an_archive_and_pinned(planted, tmp_path):
    path, along = planted
    archive = _archive_of(ToyDynamics(masked=3), tmp_path / "archive")
    args = [*_BASE, "--basis", str(path), "--mode", "toward", "--amount", "1",
            "--reference-from", str(archive), "--reference-time", "0", "--reference-time", "1",
            "--compare", f"same={along}:0", "--random-draws", "3",
            "--field", "temperature"]  # fmt: skip
    out = _invoke(args)
    command = out.strip().splitlines()[-1].removeprefix("reproduce: ")
    assert "--reference-sha256" in command and "--compare-sha256" in command
    again = _invoke(shlex.split(command)[1:])
    assert again.splitlines()[:4] == out.splitlines()[:4]
    # a reference read from other times is not the one pinned
    words = shlex.split(command)[1:]
    moved = [w if w != "1" or words[i - 1] != "--reference-time" else "2"
             for i, w in enumerate(words)]  # fmt: skip
    refused = CliRunner().invoke(cli, moved)
    assert refused.exit_code != 0 and "pinned" in refused.output
    # nor is a comparison file whose content changed
    save_basis(along, Directions(mean=np.zeros(4), directions_=np.eye(4)[1:2],
               meta={"fitted_on": {"provenance": ToyDynamics().info().identity(),
                                   "layer": 1}}))  # fmt: skip
    refused = CliRunner().invoke(cli, shlex.split(command)[1:])
    assert refused.exit_code != 0 and "sha256" in refused.output


def test_dose_prints_the_curve_and_the_amount_chosen_leave_one_out(planted, tmp_path):
    path, _ = planted
    system = ToyDynamics(masked=3)
    grid = system.grid()
    weights = grid.weights()[grid.valid]
    targets = {}
    for start in (0, 1, 2):
        rollout = system.run(system.initial_state(start), 5, (), noise_seed=0)
        series = rollout.fields["temperature"][:, grid.valid] @ weights
        targets[str(start)] = {
            "temperature": (series + np.array([0.0, 4.0, 2.0, 1.0, 0.5])).tolist()
        }
    goal = tmp_path / "targets.json"
    goal.write_text(json.dumps(targets))
    args = ["latents", "dose", "--adapter", "toy-dynamics", "--adapter-option", "masked=3",
            "--basis", str(path), "--layer", "1", "--feature", "0", "--time", "1", "--steps", "5",
            "--amounts", "0,1,2,3", "--start", "0", "--start", "1", "--start", "2",
            "--field", "temperature", "--targets", str(goal)]  # fmt: skip
    out = _invoke(args)
    assert "amount chosen leave-one-out" in out
    data = json.loads(_invoke([*args, "--json"]))
    assert data["selections"]["temperature"]["loo_amounts"] == [2.0, 2.0, 2.0]
    command = out.strip().splitlines()[-1].removeprefix("reproduce: ")
    assert "--targets" in command and "--amounts 0.0,1.0,2.0,3.0" in command


def test_direction_writes_a_basis_steer_can_compare_with(tmp_path):
    from exege.latents import load_basis
    from exege.latents.toy import write_toy

    archive = write_toy(tmp_path / "archive")
    out = _invoke(["latents", "direction", str(archive), "--layer", "3", "--event-time", "0",
                   "--quiet-time", "6", "--lat", "10", "--lon", "-140", "--radius-km", "2500",
                   "--out", str(tmp_path / "dom.npz")])  # fmt: skip
    assert "--basis-sha256" in out
    basis = load_basis(tmp_path / "dom.npz")
    assert basis.kind == "directions" and basis.meta["fitted_on"]["event_times"]


def test_matched_references_are_pinned_in_the_command(planted, tmp_path):
    path, _ = planted
    archive = _archive_of(ToyDynamics(masked=3), tmp_path / "archive")
    args = [*_BASE, "--basis", str(path), "--mode", "excess", "--amount", "2",
            "--reference-from", str(archive), "--reference-time", "0", "--reference-time", "1",
            "--matched", "2", "--random-draws", "2", "--field", "moisture"]  # fmt: skip
    out = _invoke(args)
    command = out.strip().splitlines()[-1].removeprefix("reproduce: ")
    assert "--matched-references-sha256" in command
    _invoke(shlex.split(command)[1:])
    words = shlex.split(command)[1:]
    at = words.index("--matched-references-sha256") + 1
    words[at] = "0" * 64
    refused = CliRunner().invoke(cli, words)
    assert refused.exit_code != 0 and "matched features' references" in refused.output


def test_a_reference_is_read_at_the_archives_layer_in_the_same_network_place(planted, tmp_path):
    """An exporter that keeps only some layers numbers them its own way (here network layer
    1 is the archive's layer 0): the reference is read where the network place matches."""
    from exege.adapters.latent_archive import write_archive

    path, _ = planted
    system = ToyDynamics(masked=3)
    info = system.info()
    latents = [
        system.run(system.initial_state(s), 1, (), noise_seed=0, record=[(1, 0)]).latents[(1, 0)]
        for s in (0, 1)
    ]
    kept = write_archive(
        tmp_path / "kept", grid=system.grid(), times=["0001-01-01T00:00:00", "0001-01-02T00:00:00"],
        layers=[("mixed", np.stack(latents))], network_layers=[1], model=info.model,
        component=info.component, checkpoint=info.checkpoint, calendar="noleap",
        timestep_seconds=21600, dtype="float32",
    )  # fmt: skip
    args = [*_BASE, "--basis", str(path), "--mode", "toward", "--amount", "1",
            "--reference-from", str(kept), "--reference-time", "0", "--reference-time", "1",
            "--random-draws", "2", "--field", "temperature"]  # fmt: skip
    full = _archive_of(system, tmp_path / "full", starts=(0, 1))
    same = [*args[: args.index("--reference-from") + 1], str(full),
            *args[args.index("--reference-from") + 2 :]]  # fmt: skip
    assert _invoke(args).splitlines()[:4] == _invoke(same).splitlines()[:4]
