"""`exege` is exege-core with its extras, released together: the two must agree."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from exege import __version__

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "packages" / "exege" / "pyproject.toml"

# An sdist of exege-core holds the tests but not packages/: nothing to compare there.
pytestmark = pytest.mark.skipif(not FULL.is_file(), reason="not a checkout")


def _project(path: Path) -> dict:
    return tomllib.loads(path.read_text())["project"]


def test_the_full_install_is_this_version_and_pins_it():
    full = _project(FULL)
    assert full["version"] == __version__
    (pin,) = full["dependencies"]
    assert pin.startswith("exege-core[") and pin.endswith(f"]=={__version__}")
    assert full["optional-dependencies"] == {"nn": [f"exege-core[nn]=={__version__}"]}


def test_every_extra_but_nn_is_reached_from_the_full_install():
    core = _project(ROOT / "pyproject.toml")["optional-dependencies"]
    (pin,) = _project(FULL)["dependencies"]
    named = pin[len("exege-core[") : pin.index("]")].split(",")

    reached, todo = set(), list(named)
    while todo:
        extra = todo.pop()
        if extra in reached:
            continue
        reached.add(extra)
        for requirement in core[extra]:
            if requirement.startswith("exege-core["):
                todo.extend(requirement[len("exege-core[") : requirement.index("]")].split(","))
    assert reached == set(core) - {"nn"}
