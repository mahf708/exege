"""A physical law, paired with a model: LAWFUL's bridges, family and consistency score.

LAWFUL (Chen, Parker & Arora 2026) asks whether a model that predicts a system well has
learned the law that governs it, and uses it. A ``Law`` here holds what that pairing needs:

- the **perturbation family** ``alphas`` and, for each alpha, the **input bridge** run
  backwards: ``perturbation(grid, fields, alpha)``, the change to the model's input fields
  that realises the physical counterfactual (a planted wave of wavenumber alpha, say);
- the **output bridge** on the model's side, ``observe(...)``: the observable the law speaks
  about, read off how the model's step responded;
- the **physics model** with its own bridge, ``expect(grid, fields, alpha)``: the same
  observable as the law predicts it, at the same base state.

Consistency over the family is LAWFUL's eq. 10, ``consistency(m_f, m_g)``: one minus the
squared misfit over the variance of the law's own values across the family -- so the law
must vary across the family for the score to mean anything.

The model's response is its *tendency* response, ``[step(x') - x'] - [step(x) - x]``: what
the network adds on top of carrying the state forward. A time-stepper's skip connection
carries any perturbation forward unchanged and is not physics any component did, and a
patched run (whose input is the base state) has the same form, ``step_C(x) - step(x)``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from exege.core.errors import RequestError
from exege.core.extras import missing_extra
from exege.latents.grid import Grid

try:
    import numpy as np
except ImportError as exc:
    raise missing_extra("numpy", "laws") from exc

EARTH_RADIUS_M = 6.371e6
OMEGA = 7.292e-5  # rad/s
DAY = 86400.0


@runtime_checkable
class Law(Protocol):
    """A law over a perturbation family. ``fields`` are the base state's, ``(batch,
    n_nodes)`` per name; ``delta`` what ``perturbation`` returned; ``response`` the model's
    tendency response, ``(batch, n_nodes)`` per field the step writes."""

    name: str

    @property
    def alphas(self) -> tuple[float, ...]: ...

    def perturbation(
        self, grid: Grid, fields: Mapping[str, np.ndarray], alpha: float
    ) -> dict[str, np.ndarray]: ...

    def observe(
        self,
        grid: Grid,
        fields: Mapping[str, np.ndarray],
        delta: Mapping[str, np.ndarray],
        response: Mapping[str, np.ndarray],
        alpha: float,
    ) -> float: ...

    def expect(self, grid: Grid, fields: Mapping[str, np.ndarray], alpha: float) -> float: ...

    def to_dict(self) -> dict[str, Any]: ...


SCORES = ("eq10", "correlation")


def consistency(m_f: Sequence[float], m_g: Sequence[float], score: str = "eq10") -> float:
    """How well the model's values follow the law's across the family.

    ``"eq10"`` is LAWFUL's eq. 10, ``1 - sum (m_f - m_g)^2 / sum (m_g - mean m_g)^2``: at most
    1, and 1 exactly when the model matches the law at every alpha; an offset or a wrong
    scale costs as much as a wrong shape. ``"correlation"`` is the Pearson correlation of the
    two across the family (LAWFUL's "similarity-based" choice of Phi): 1 when the model
    follows the law's shape, whatever its offset and scale."""
    if score not in SCORES:
        raise RequestError(f"score must be one of {', '.join(SCORES)}, not {score!r}")
    f, g = np.asarray(m_f, dtype=np.float64), np.asarray(m_g, dtype=np.float64)
    if f.shape != g.shape or f.ndim != 1 or f.size < 2:
        raise RequestError("consistency needs the same two or more alphas on both sides")
    spread = ((g - g.mean()) ** 2).sum()
    if spread == 0:
        raise RequestError("the law predicts the same value at every alpha: nothing to score")
    if not (np.isfinite(f).all() and np.isfinite(g).all()):
        raise RequestError("a model or law value is not finite")
    if score == "correlation":
        df, dg = f - f.mean(), g - g.mean()
        norm = np.sqrt((df**2).sum() * (dg**2).sum())
        return 0.0 if norm == 0 else float((df * dg).sum() / norm)
    return float(1 - ((f - g) ** 2).sum() / spread)


# -- waves on a sphere: what the wave laws share -----------------------------------------------


@dataclass(frozen=True)
class Envelope:
    """A Gaussian band in latitude: ``exp(-((lat - phi0) / sigma)^2)``, degrees."""

    phi0: float = 45.0
    sigma: float = 10.0

    def __call__(self, lat: np.ndarray) -> np.ndarray:
        return np.exp(-(((np.asarray(lat, dtype=np.float64) - self.phi0) / self.sigma) ** 2))

    def d_dphi(self, lat: np.ndarray) -> np.ndarray:
        """Derivative in latitude, per radian."""
        lat = np.asarray(lat, dtype=np.float64)
        return self(lat) * (-2 * (lat - self.phi0) / self.sigma**2) * (180 / np.pi)

    def l2(self, radius: float) -> float:
        """Mean squared meridional wavenumber of the band, ``2 / (sigma a)^2``, 1/m^2."""
        return 2.0 / (np.deg2rad(self.sigma) * radius) ** 2


def projection(grid: Grid, values: np.ndarray, s: int, band: Envelope) -> complex:
    """Complex amplitude of ``values`` (``(n_nodes,)`` or ``(batch, n_nodes)``, averaged over
    the batch) on ``band(lat) exp(i s lon)``, area weighted: the phase of a zonal wave."""
    v = np.asarray(values, dtype=np.float64).reshape(-1, grid.n_nodes)
    w = grid.weights() * band(grid.lat)
    e = np.exp(-1j * s * np.deg2rad(grid.lon))
    keep = w != 0
    if not np.isfinite(v[:, keep]).all():
        raise RequestError("a field is not finite where the wave is measured")
    return complex((v[:, keep] @ (w[keep] * e[keep])).mean())


def band_mean(grid: Grid, values: np.ndarray, band: Envelope) -> float:
    """Mean of ``values`` over the band, weighted by area and the envelope squared (where the
    wave's energy is), averaged over the batch."""
    v = np.asarray(values, dtype=np.float64).reshape(-1, grid.n_nodes)
    w = grid.weights() * band(grid.lat) ** 2
    keep = w != 0
    return float((v[:, keep] @ w[keep]).mean() / w[keep].sum())


def phase_rate(a0: complex, a1: complex, dt: float) -> tuple[float, float]:
    """Angular frequency (rad/s, eastward propagation positive for ``exp(i(s lon - w t))``)
    and growth rate (1/s) from a wave's complex amplitude before and after ``dt``."""
    if a0 == 0 or a1 == 0:
        raise RequestError("the wave has no amplitude to follow")
    r = a1 / a0
    return float(-np.angle(r) / dt), float(np.log(abs(r)) / dt)


def check_alphas(alphas: Sequence[float]) -> tuple[float, ...]:
    out = tuple(alphas)
    if len(out) < 2:
        raise RequestError("a perturbation family needs at least two alphas")
    return out
