"""A toy system that can be steered, with a planted answer. For tests and docs.

Not a model of anything: a small linear dynamical system over a lat-lon grid, written so
that what an edit at a latent layer does to the physical fields can be worked out by hand.

    state x (n_nodes, 3): temperature, moisture, pressure
    forcing u (n_nodes, 3): drifts on its own, with noise: u' = 0.9 u + noise * eps_u
                            (+ feedback * x, temperature and moisture swapped)
    layer 0   h0 = u @ E          (4 channels; + static * S on channel 2, + positional * P
                                   on channel 3, when those are set)
    layer 1   h1 = h0 @ M         (M swaps channels 0 and 1)
    next      x' = decay * x + h1 @ D + noise * eps_x

``D`` sends channel 0 to temperature with weight ``gain``, channel 1 to moisture, channel
2 to pressure, channel 3 to pressure at half weight. So adding ``a`` to channel 0 of layer 1
(the planted feature) moves temperature by ``gain * a`` on that step and by ``decay`` times
as much on each step after, and nothing else; the same edit at layer 0 reaches moisture
instead, by way of ``M``; and a random direction mostly does neither. The latents do not
read ``x`` back, so no edit feeds into the next one's input: responses are exactly those.
With ``feedback`` set they do: ``u`` takes ``feedback * x`` each step with temperature and
moisture swapped, so that each of the first two features of layer 1 reads back, through
the mixer, what it wrote the step before. That is what makes
a steer worked out from the arm's own latents (closed loop) differ from one worked out
from the control's (open loop).

Two fixed per-node patterns stand in for what a real model holds that is not weather.
``P`` is the model's own (a learned positional embedding): it never moves. ``S`` is a
boundary field it is given (orography, land): it moves only when the state it belongs to is
rotated. ``rotate`` turns the state east by whole columns, its ``"state"`` part (``x`` and
``u``), its ``"static"`` part (``S``) or both, so the rotation experiment has a planted
answer: channels 0 and 1 follow the weather, channel 3 stays with ``P`` when ``positional``
dominates it, and channel 2 follows ``S`` when ``static`` dominates it -- if, and only if,
``S`` was rotated too.

Noise is a function of the seed alone (drawn up front, not from the state), so two runs
with one seed differ only by what the hooks did. The initial state is ``(n_nodes, 6)``:
``x`` then ``u``; a rotated one is a ``{"state": ..., "static_columns": k}`` mapping.

``planted_dictionary`` is a dictionary that reads three of the four channels, so it has a
reconstruction error with a known cause: the fourth channel, and what its ReLU drops.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from exege.core.errors import RequestError
from exege.core.extras import missing_extra
from exege.latents.basis import Dictionary
from exege.latents.grid import Grid, longitude_partner
from exege.latents.source import LatentInfo, LayerInfo
from exege.latents.steering import Hook, Rollout

try:
    import numpy as np
except ImportError as exc:
    raise missing_extra("numpy", "latents") from exc

FIELDS = ("temperature", "moisture", "pressure")
PARTS = ("state", "static")
WIDTH = 4


class ToyDynamics:
    """``Intervenable`` and ``Rotatable``: it reads no source, so its options are
    keyword-only. ``masked`` nodes, the first ones, have fields that are NaN, like land."""

    def __init__(
        self,
        *,
        n_lat: int = 4,
        n_lon: int = 6,
        gain: float = 2.0,
        decay: float = 0.5,
        noise: float = 0.05,
        masked: int = 0,
        seed: int = 0,
        feedback: float = 0.0,
        positional: float = 0.0,
        static: float = 0.0,
    ) -> None:
        if not 0 <= masked < n_lat * n_lon:
            raise RequestError(f"masked must leave a node valid: 0 <= masked < {n_lat * n_lon}")
        self._options = {
            "n_lat": n_lat, "n_lon": n_lon, "gain": gain, "decay": decay, "noise": noise,
            "masked": masked, "seed": seed, "feedback": feedback,
            "positional": positional, "static": static,
        }  # fmt: skip
        self.decay, self.noise, self.seed, self.feedback = decay, noise, seed, feedback
        self.positional, self.static = positional, static
        lat, lon = np.meshgrid(
            np.linspace(-75.0, 75.0, n_lat), np.arange(n_lon) * (360.0 / n_lon), indexing="ij"
        )
        mask = np.arange(n_lat * n_lon) >= masked
        self._grid = Grid(lat=lat.ravel(), lon=lon.ravel(), shape=(n_lat, n_lon), mask=mask)
        self.position = np.random.default_rng([seed, 7]).normal(size=n_lat * n_lon)
        self.boundary = np.random.default_rng([seed, 11]).normal(size=n_lat * n_lon)
        self.encoder = np.array([[1, 0, 0, 0.5], [0, 1, 0, 0.5], [0, 0, 1, 0]], dtype=np.float64)
        self.mixer = np.eye(WIDTH)[[1, 0, 2, 3]]
        self.decoder = np.array([[gain, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0.5]], dtype=np.float64)
        self._info = LatentInfo(
            source="toy-dynamics",
            times=("start",),
            layers=(LayerInfo(0, "encoder", WIDTH), LayerInfo(1, "mixed", WIDTH)),
            n_nodes=self._grid.n_nodes,
            model="toy-dynamics",
            component="linear",
            checkpoint=f"planted-{seed}",
            options=self._options,
        )

    def info(self) -> LatentInfo:
        return self._info

    def grid(self) -> Grid:
        return self._grid

    def initial_state(self, start: str | int = 0) -> np.ndarray:
        if isinstance(start, str):
            raise RequestError("this system starts from a number, the seed offset of its state")
        rng = np.random.default_rng([self.seed, int(start)])
        return rng.normal(size=(self._grid.n_nodes, 2 * len(FIELDS)))

    def run(
        self,
        initial_state: Any,
        steps: int,
        hooks: Sequence[Hook] = (),
        *,
        noise_seed: int | None = None,
        record: Sequence[tuple[int, int]] = (),
    ) -> Rollout:
        n = self._grid.n_nodes
        static_columns = 0
        if isinstance(initial_state, Mapping):
            static_columns = int(initial_state.get("static_columns", 0))
            initial_state = initial_state["state"]
        boundary = self._east(self.boundary, static_columns)
        state = np.asarray(initial_state, dtype=np.float64)
        if state.shape != (n, 2 * len(FIELDS)):
            raise RequestError(f"the state is {state.shape}, expected {(n, 2 * len(FIELDS))}")
        x, u = state[:, : len(FIELDS)], state[:, len(FIELDS) :]
        if steps < 1:
            raise RequestError("steps must be at least 1")
        edits = {}
        for hook in hooks:
            if hook.layer not in (0, 1) or not 0 <= hook.time < steps:
                raise RequestError(f"no place at layer {hook.layer}, time {hook.time}")
            edits.setdefault((hook.layer, hook.time), []).append(hook.edit)
        for place in record:
            if place[0] not in (0, 1) or not 0 <= place[1] < steps:
                raise RequestError(f"cannot record layer {place[0]} at time {place[1]}")
        eps = (
            None
            if noise_seed is None
            else np.random.default_rng(noise_seed).normal(size=(steps, n, 2 * len(FIELDS)))
        )
        written = np.empty((steps, n, len(FIELDS)))
        kept: dict[tuple[int, int], np.ndarray] = {}

        def hooked(h: np.ndarray, layer: int, t: int) -> np.ndarray:
            h = h.astype(np.float32)  # latents are float32, as a recorded model's are
            for edit in edits.get((layer, t), ()):
                h = np.asarray(edit(h.copy()), dtype=np.float32)
            if (layer, t) in record:
                kept[(layer, t)] = h.copy()
            return h

        for t in range(steps):
            h0 = u @ self.encoder
            if self.static:
                h0[:, 2] += self.static * boundary
            if self.positional:
                h0[:, 3] += self.positional * self.position
            h0 = hooked(h0, 0, t)
            h1 = hooked(h0.astype(np.float64) @ self.mixer, 1, t)
            x = self.decay * x + h1.astype(np.float64) @ self.decoder
            u = 0.9 * u
            if self.feedback:
                u = u + self.feedback * x[:, [1, 0, 2]]
            if eps is not None:
                x = x + self.noise * eps[t, :, : len(FIELDS)]
                u = u + self.noise * eps[t, :, len(FIELDS) :]
            written[t] = x
        fields = {name: written[:, :, i].copy() for i, name in enumerate(FIELDS)}
        for values in fields.values():
            values[:, ~self._grid.valid] = np.nan
        return Rollout(fields=fields, latents=kept)

    def _east(self, values: np.ndarray, columns: int) -> np.ndarray:
        """``values`` per node, moved ``columns`` columns east."""
        return values if not columns else values[longitude_partner(self._grid, -columns)]

    def rotation_parts(self) -> tuple[str, ...]:
        return PARTS

    def rotate(self, state: Any, columns: int, parts: Sequence[str] = PARTS) -> dict[str, Any]:
        """``state`` turned ``columns`` columns east: the parts named, and nothing else."""
        if unknown := [p for p in parts if p not in PARTS]:
            raise RequestError(f"no part(s) {unknown} to rotate; there are {', '.join(PARTS)}")
        if isinstance(state, Mapping):
            static_columns, state = int(state.get("static_columns", 0)), state["state"]
        else:
            static_columns = 0
        array = np.array(state, dtype=np.float64)  # a copy: the state given is left as it was
        expected = (self._grid.n_nodes, 2 * len(FIELDS))
        if array.shape != expected:
            raise RequestError(f"the state is {array.shape}, expected {expected}")
        if "state" in parts:
            array = array[longitude_partner(self._grid, -int(columns))]
        if "static" in parts:
            static_columns += int(columns)
        return {"state": array, "static_columns": static_columns}

    def planted_dictionary(self, layer: int = 1) -> Dictionary:
        """A dictionary for ``layer`` with three features: channels 0, 1 and 2. Feature 0
        of layer 1 is the planted one. It carries the identity a basis fitted here would."""
        if layer not in (0, 1):
            raise RequestError(f"no layer {layer}")
        frame = np.eye(WIDTH, dtype=np.float32)[:3]
        return Dictionary(
            encoder=frame,
            encoder_bias=np.zeros(3, dtype=np.float32),
            decoder=frame.copy(),
            decoder_bias=np.zeros(WIDTH, dtype=np.float32),
            input_mean=np.zeros(WIDTH, dtype=np.float32),
            meta={"fitted_on": {"provenance": self._info.identity(), "layer": layer}},
        )
