"""laws -- whether a model obeys a physical law, and which of its parts do.

    from exege.circuits import open_patchable
    from exege.laws import Rossby, measure, search

    system = open_patchable(ckpt, "vit-interp")
    base = system.states([0, 40, 80, 120])
    law = Rossby(u=("U_3",), v=("V_3",), observe_field="V_3")
    measure(system, base, law).M        # LAWFUL's consistency over s = 2..14
    search(system, base, law).circuits  # the heads and MLPs that carry it

After LAWFUL (Chen, Parker & Arora 2026), on any ``circuits.Patchable`` system.

- ``law``      the ``Law`` contract, LAWFUL's consistency score, and wave tools on a sphere
- ``waves``    ``ZonalAdvection`` (omega = k U) and ``Rossby`` (barotropic dispersion)
- ``lawful``   ``measure``, ``deviations`` (sufficiency, necessity, controls), ``search``

Needs the ``laws`` extra (numpy). Names load on first use, as in ``exege.latents``.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from exege.laws.law import Envelope, Law, consistency, phase_rate, projection
    from exege.laws.lawful import (
        Deviations,
        LawCircuit,
        LawResult,
        Search,
        components,
        deviations,
        measure,
        search,
    )
    from exege.laws.waves import Rossby, ZonalAdvection

_LAZY = {
    "Envelope": "law",
    "Law": "law",
    "consistency": "law",
    "phase_rate": "law",
    "projection": "law",
    "Deviations": "lawful",
    "LawCircuit": "lawful",
    "LawResult": "lawful",
    "Search": "lawful",
    "components": "lawful",
    "deviations": "lawful",
    "measure": "lawful",
    "search": "lawful",
    "Rossby": "waves",
    "ZonalAdvection": "waves",
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{_LAZY[name]}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_LAZY])


__all__ = [
    "Deviations",
    "Envelope",
    "Law",
    "LawCircuit",
    "LawResult",
    "Rossby",
    "Search",
    "ZonalAdvection",
    "components",
    "consistency",
    "deviations",
    "measure",
    "phase_rate",
    "projection",
    "search",
]
