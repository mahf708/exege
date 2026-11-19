"""Laws of zonal waves on a sphere, over a family of zonal wavenumbers.

Both laws plant a small wave of wavenumber ``s`` in a latitude band on the model's own
input states, take one step, and read the wave's angular frequency off the phase advance
of the model's response projected on ``band(lat) exp(i s lon)``:

    a0 = proj(planted),   a1 = a0 + proj(tendency response),   omega = -arg(a1 / a0) / dt

- ``ZonalAdvection``: a passive pattern carried by a uniform zonal speed, ``omega = k U``
  with ``k = s / (a cos phi0)``. The toy transformer's planted law, and a sanity check for
  any emulator's tracer.
- ``Rossby``: barotropic Rossby waves on the base state's own zonal flow,
  ``omega = k U_bar - beta k / (k^2 + l^2)``, ``beta = 2 Omega cos(phi0) / a``. The planted
  wave is non-divergent (winds from a streamfunction ``psi = A band(lat) cos(s lon)``) and,
  optionally, geostrophically balanced (surface pressure ``rho_s f psi``). By default the
  score is on the *intrinsic* frequency, ``omega - k U_bar = -beta k / K^2``: the part the
  beta effect alone is responsible for, which Doppler shifting by the jet would otherwise
  swamp.

Frequencies are reported in rad/day.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from exege.core.errors import RequestError
from exege.core.extras import missing_extra
from exege.latents.grid import Grid
from exege.laws.law import (
    DAY,
    EARTH_RADIUS_M,
    OMEGA,
    Envelope,
    band_mean,
    check_alphas,
    phase_rate,
    projection,
)

try:
    import numpy as np
except ImportError as exc:
    raise missing_extra("numpy", "laws") from exc


def _k(s: float, phi0: float, radius: float) -> float:
    return s / (radius * np.cos(np.deg2rad(phi0)))


def _need(fields: Mapping[str, np.ndarray], *names: str) -> None:
    missing = [n for n in names if n not in fields]
    if missing:
        raise RequestError(f"the state has no field {', '.join(missing)}")


@dataclass(frozen=True)
class ZonalAdvection:
    """``omega = k * speed``: a pattern in ``field`` carried east at ``speed`` (m/s at phi0)."""

    field: str
    speed: float
    wavenumbers: Sequence[int] = (1, 2, 3, 4, 5)
    band: Envelope = field(default_factory=lambda: Envelope(0.0, 30.0))
    amplitude: float = 1.0
    dt: float = 21600.0
    radius: float = EARTH_RADIUS_M
    name: str = "zonal advection"

    @property
    def alphas(self) -> tuple[float, ...]:
        return check_alphas(self.wavenumbers)

    def perturbation(self, grid: Grid, fields, alpha) -> dict[str, np.ndarray]:
        lam = np.deg2rad(grid.lon)
        return {self.field: self.amplitude * self.band(grid.lat) * np.cos(alpha * lam)}

    def observe(self, grid, fields, delta, response, alpha) -> float:
        a0 = projection(grid, delta[self.field], int(alpha), self.band)
        a1 = a0 + projection(grid, response[self.field], int(alpha), self.band)
        return phase_rate(a0, a1, self.dt)[0] * DAY

    def expect(self, grid, fields, alpha) -> float:
        return _k(alpha, self.band.phi0, self.radius) * self.speed * DAY

    def to_dict(self) -> dict[str, Any]:
        return {"law": self.name, **asdict(self)}


@dataclass(frozen=True)
class Rossby:
    """Barotropic Rossby waves of zonal wavenumber ``s`` in a band, on the base flow.

    ``u`` and ``v`` name the wind fields the wave is planted in (one per level: the wave is
    barotropic); ``observe`` the meridional wind it is read from; ``u_bar`` the zonal wind
    whose band mean is the flow the law Doppler-shifts by (default: the ``u`` at the
    observed level). ``ps`` (optional) receives the geostrophic surface pressure.
    """

    u: tuple[str, ...]
    v: tuple[str, ...]
    observe_field: str
    u_bar: str | None = None
    ps: str | None = None
    wavenumbers: Sequence[int] = tuple(range(2, 15))
    band: Envelope = field(default_factory=Envelope)
    amplitude: float = 1.0  # peak meridional wind, m/s
    intrinsic: bool = True
    l2: float | None = None  # meridional wavenumber squared, 1/m^2; default from the band
    beta_from_flow: bool = False  # beta - U_yy, from the band's zonal-mean flow
    dt: float = 21600.0
    radius: float = EARTH_RADIUS_M
    rho_s: float = 1.2
    name: str = "barotropic Rossby"

    def __post_init__(self) -> None:
        if len(self.u) != len(self.v) or not self.u:
            raise RequestError("name one u and one v field per level")
        if self.observe_field not in self.v:
            raise RequestError(f"observe_field must be one of {self.v}")

    @property
    def alphas(self) -> tuple[float, ...]:
        return check_alphas(self.wavenumbers)

    @property
    def _u_bar(self) -> str:
        return self.u_bar or self.u[self.v.index(self.observe_field)]

    def perturbation(self, grid: Grid, fields, alpha) -> dict[str, np.ndarray]:
        s = float(alpha)
        phi = np.deg2rad(grid.lat)
        lam = np.deg2rad(grid.lon)
        A = self.amplitude * self.radius * np.cos(np.deg2rad(self.band.phi0)) / s
        G = self.band(grid.lat)
        psi = A * G * np.cos(s * lam)
        u = -(A / self.radius) * self.band.d_dphi(grid.lat) * np.cos(s * lam)
        v = -(A * s / (self.radius * np.cos(phi))) * G * np.sin(s * lam)
        out = {}
        for un, vn in zip(self.u, self.v, strict=True):
            out[un], out[vn] = u, v
        if self.ps is not None:
            out[self.ps] = self.rho_s * 2 * OMEGA * np.sin(phi) * psi
        return out

    def flow(self, grid: Grid, fields) -> float:
        """The band's zonal-mean zonal wind, m/s."""
        _need(fields, self._u_bar)
        return band_mean(grid, fields[self._u_bar], self.band)

    def beta(self, grid: Grid, fields) -> float:
        b = 2 * OMEGA * np.cos(np.deg2rad(self.band.phi0)) / self.radius
        if not self.beta_from_flow:
            return b
        # U_yy of the zonal-mean flow at phi0, from three bands a sigma apart
        h = self.band.sigma
        U = [
            self.flow(grid, fields)
            if d == 0
            else band_mean(grid, fields[self._u_bar], Envelope(self.band.phi0 + d, self.band.sigma))
            for d in (-h, 0, h)
        ]
        dy = np.deg2rad(h) * self.radius
        return b - (U[0] - 2 * U[1] + U[2]) / dy**2

    def observe(self, grid, fields, delta, response, alpha) -> float:
        s = int(alpha)
        _need(response, self.observe_field)
        a0 = projection(grid, delta[self.observe_field], s, self.band)
        a1 = a0 + projection(grid, response[self.observe_field], s, self.band)
        omega = phase_rate(a0, a1, self.dt)[0]
        if self.intrinsic:
            omega -= _k(s, self.band.phi0, self.radius) * self.flow(grid, fields)
        return omega * DAY

    def expect(self, grid, fields, alpha) -> float:
        k = _k(alpha, self.band.phi0, self.radius)
        l2 = self.band.l2(self.radius) if self.l2 is None else self.l2
        omega = -self.beta(grid, fields) * k / (k**2 + l2)
        if not self.intrinsic:
            omega += k * self.flow(grid, fields)
        return omega * DAY

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["wavenumbers"] = list(self.wavenumbers)
        return {"law": self.name, **d}
