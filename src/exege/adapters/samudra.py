"""Samudra (the U-Net ocean of SamudrACE) as a ``circuits.Patchable`` with ``EdgeGradients``.

The graph follows the network's additive structure. Layer 0, the only reader of the input, is two
senders: ``l0.skip_module`` (a 1x1 conv) and ``l0.convblock`` (conv -> norm -> act -> conv -> norm
-> act -> 1x1); their sum is level 0's encoder output, read by the deep path (``enc1``) and, through
the full-resolution skip, by ``out``. Below that the U-Net's encoder levels, bottleneck and decoder
levels are senders, and every decoder receiver reads the upsampled level below plus the encoder skip
of its own level::

    input -> l0.skip_module, l0.convblock
    l0.*  -> enc1, out                       (deep path and skip.0)
    enc_k -> enc_k+1 ... enc_N-1 -> bottleneck
    bottleneck -> dec_N-1;  enc_k -> dec_k   (skip.k)
    dec_k+1 -> dec_k;  dec1 -> out

A receiver with several senders reads their sum; an upsample is applied to one sender's (mixed)
value, so edge masks mix exactly as ``EdgePatch`` defines. ``out`` is the last ConvNeXt and the
output conv.

Two ways to open it: - ``source`` = a capture directory (``x_###.npy`` normalized network inputs,
``meta.json`` with ``in_names``, ``out_names``, ``means``, ``stds``; optional ``meta.npz`` with
``lat``, ``lon``, ``out_mask``) and ``checkpoint=`` an fme checkpoint (coupled or ocean-only) whose
ocean module is used. Runs in float32. - no source: a tiny randomly initialised Samudra on a small
grid with states drawn from ``seed``, for tests and docs (float64, so exactness checks hold to
1e-6). Needs ``fme`` (and torch) installed.

Fields are physical: ``fields(state)`` returns every input channel (``x * std + mean``), ``perturb``
adds to them, and ``step`` writes the output channels in physical units. States are never
modified.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from exege.circuits.contract import ComponentPatch, EdgePatch, Patch
from exege.circuits.graph import Graph
from exege.core.errors import AdapterError, RequestError
from exege.core.extras import missing_extra
from exege.latents.grid import Grid

try:
    import numpy as np
except ImportError as exc:
    raise missing_extra("numpy", "circuits") from exc


@dataclass(frozen=True)
class SamudraState:
    """Normalized network inputs ``x[b, channel, lat, lon]``. Opaque to exege."""

    x: Any  # torch.Tensor


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise AdapterError("the samudra adapter needs torch") from exc
    return torch


class Samudra:
    """``Patchable`` and ``EdgeGradients`` over one Samudra forward call."""

    def __init__(
        self,
        source: str | None = None,
        *,
        checkpoint: str | None = None,
        device: str = "cpu",
        n_lat: int = 16,
        n_lon: int = 32,
        seed: int = 0,
    ) -> None:
        torch = _torch()
        self.torch = torch
        self.dev = torch.device(device)
        if source is None:
            self._toy(n_lat, n_lon, seed)
        else:
            self._open(source, checkpoint)
        self.dtype = next(self.net.parameters()).dtype
        L = self.net.layers
        self.N = (len(L) - 2) // 4  # encoder levels
        if len(L) != 4 * self.N + 2 or self.N < 2:
            raise AdapterError(f"unexpected Samudra layout ({len(L)} layers)")
        enc = [f"enc{k}" for k in range(1, self.N)]
        dec = [f"dec{k}" for k in range(self.N - 1, 0, -1)]
        senders = ["input", "l0.skip_module", "l0.convblock", *enc, "bottleneck", *dec]
        receivers = ["l0.skip_module", "l0.convblock", *enc, "bottleneck", *dec, "out"]
        E = np.zeros((len(senders), len(receivers)), bool)

        def edge(s, r):
            E[senders.index(s), receivers.index(r)] = True

        edge("input", "l0.skip_module")
        edge("input", "l0.convblock")
        for s in ("l0.skip_module", "l0.convblock"):
            edge(s, "enc1")
            edge(s, "out")
        for k in range(1, self.N - 1):
            edge(f"enc{k}", f"enc{k + 1}")
        edge(f"enc{self.N - 1}", "bottleneck")
        edge("bottleneck", f"dec{self.N - 1}")
        for k in range(1, self.N):
            edge(f"enc{k}", f"dec{k}")
        for k in range(self.N - 1, 1, -1):
            edge(f"dec{k}", f"dec{k - 1}")
        edge("dec1", "out")
        self._graph = Graph(tuple(senders), tuple(receivers), E)

    # -- opening ----------------------------------------------------------------------------
    def _toy(self, n_lat: int, n_lon: int, seed: int) -> None:
        torch = self.torch
        try:
            from fme.ace.models.ocean.m2lines.samudra import Samudra as FmeSamudra
        except ImportError as exc:
            raise AdapterError("the samudra adapter's toy needs fme installed") from exc
        torch.manual_seed(seed)
        self.in_names = ["LANDFRAC", "FLUS", "FLDS", "sst", "temperatureCoarsened_0"]
        self.out_names = ["sst", "temperatureCoarsened_0"]
        self.net = FmeSamudra(
            input_channels=len(self.in_names),
            output_channels=len(self.out_names),
            ch_width=[6, 8],
            dilation=[1, 2],
            n_layers=[1, 1],
            upscale_factor=2,
        )
        self.net = self.net.double().to(self.dev).eval()
        names = sorted(set(self.in_names) | set(self.out_names))
        self.means = {n: 0.0 for n in names} | {"sst": 290.0, "FLUS": 400.0, "FLDS": 350.0}
        self.stds = {n: 1.0 for n in names} | {"sst": 5.0, "FLUS": 30.0, "FLDS": 30.0}
        g = np.random.default_rng(seed)
        # states = one shared field + small per-state differences:
        # patching contrasts nearby states, as it does in use
        base = g.standard_normal((1, len(self.in_names), n_lat, n_lon))
        self._x = base + 0.02 * g.standard_normal((8, len(self.in_names), n_lat, n_lon))
        lat = np.linspace(-80, 80, n_lat)
        lon = np.arange(n_lon) * 360.0 / n_lon
        self._labels = [str(i) for i in range(len(self._x))]
        self._mask = np.ones((n_lat, n_lon), bool)
        self._lat, self._lon = lat, lon

    def _open(self, source: str, checkpoint: str | None) -> None:
        torch = self.torch
        meta_p = os.path.join(source, "meta.json")
        if not os.path.isdir(source) or not os.path.exists(meta_p):
            raise AdapterError(f"{source!r} is not a Samudra capture directory (no meta.json)")
        if checkpoint is None:
            raise AdapterError("opening a capture directory needs checkpoint=<fme checkpoint>")
        with open(meta_p) as fh:
            meta = json.load(fh)
        self.in_names, self.out_names = list(meta["in_names"]), list(meta["out_names"])
        self.means, self.stds = meta["means"], meta["stds"]
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        coupled = "ocean" in ck["stepper"]["config"]
        del ck
        if coupled:
            from fme.coupled.inference.inference import load_stepper

            st = load_stepper(checkpoint).ocean
        else:
            from fme.ace.stepper.single_module import load_stepper

            st = load_stepper(checkpoint)
        self.net = st.modules[0].module.float().to(self.dev).eval()
        self.net.checkpoint_strategy = None
        for layer in self.net.layers:
            if hasattr(layer, "checkpoint_strategy"):
                layer.checkpoint_strategy = None
        files = sorted(f for f in os.listdir(source) if f.startswith("x_") and f.endswith(".npy"))
        if not files:
            raise AdapterError(f"no x_###.npy inputs in {source!r}")
        self._files = [os.path.join(source, f) for f in files]
        self._labels = [f[2:-4] for f in files]
        self._x = None
        npz = os.path.join(source, "meta.npz")
        H, W = np.load(self._files[0], mmap_mode="r").shape[-2:]
        if os.path.exists(npz):
            z = np.load(npz)
            self._lat, self._lon, self._mask = z["lat"], z["lon"], z["out_mask"][0]
        else:
            self._lat, self._lon = np.linspace(-89.5, 89.5, H), np.arange(W) + 0.5
            self._mask = np.ones((H, W), bool)

    # -- protocol -----------------------------------------------------------------------------
    def graph(self) -> Graph:
        return self._graph

    def grid(self) -> Grid:
        LA, LO = np.meshgrid(self._lat, self._lon, indexing="ij")
        return Grid(lat=LA.ravel(), lon=LO.ravel(), shape=LA.shape, mask=self._mask.ravel())

    def field_names(self) -> tuple[str, ...]:
        return tuple(self.out_names)

    def states(self, starts: Sequence[str | int]) -> SamudraState:
        if len(starts) == 0:
            raise RequestError("no starts")
        idx = []
        for s in starts:
            i = self._labels.index(s) if isinstance(s, str) and s in self._labels else s
            if not isinstance(i, (int, np.integer)) or not 0 <= int(i) < len(self._labels):
                raise RequestError(f"no start {s!r}: {len(self._labels)} states")
            idx.append(int(i))
        if self._x is not None:
            x = np.stack([self._x[i] for i in idx])
        else:
            x = np.stack([np.load(self._files[i]) for i in idx])
        t = self.torch.as_tensor(np.nan_to_num(x), dtype=self._dtype(), device=self.dev)
        return SamudraState(t)

    def fields(self, state: SamudraState) -> dict[str, np.ndarray]:
        x = state.x.detach().cpu().numpy()
        B = x.shape[0]
        return {
            n: (x[:, i] * self.stds[n] + self.means[n]).reshape(B, -1)
            for i, n in enumerate(self.in_names)
        }

    def perturb(self, state: SamudraState, delta: Mapping[str, np.ndarray]) -> SamudraState:
        x = state.x.clone()
        B, _, H, W = x.shape
        for name, d in delta.items():
            if name not in self.in_names:
                raise RequestError(f"no input field {name!r}")
            d = np.asarray(d, dtype=float)
            if d.shape not in ((H * W,), (B, H * W)):
                raise RequestError(f"a delta is (n_nodes,) or (batch, n_nodes), got {d.shape}")
            dt = self.torch.as_tensor(
                np.broadcast_to(d, (B, H * W)).reshape(B, H, W) / self.stds[name],
                dtype=x.dtype,
                device=x.device,
            )
            x[:, self.in_names.index(name)] += dt
        return SamudraState(x)

    def step(self, state: SamudraState, patch: Patch | None = None) -> dict[str, np.ndarray]:
        with self.torch.no_grad():
            y = self._forward(state, patch)
        return self._physical(y)

    def edge_gradients(self, state: SamudraState, source: SamudraState, weights) -> np.ndarray:
        torch = self.torch
        g = self._graph
        mask = torch.ones(g.shape, dtype=self._dtype(), device=self.dev, requires_grad=True)
        y = self._forward(state, EdgePatch(source, mask))
        B = y.shape[0]
        metric = 0.0
        for i, n in enumerate(self.out_names):
            if n in weights:
                w = torch.as_tensor(np.asarray(weights[n]), dtype=y.dtype, device=y.device)
                metric = metric + (y[:, i].reshape(B, -1) * self.stds[n] * w).sum() / B
        (grad,) = torch.autograd.grad(metric, mask)
        out = grad.detach().cpu().numpy().astype(float)
        out[~g.edges] = 0.0
        return out

    # -- internals ----------------------------------------------------------------------------
    def _dtype(self):
        return next(self.net.parameters()).dtype

    def _physical(self, y) -> dict[str, np.ndarray]:
        a = y.detach().cpu().numpy()
        B = a.shape[0]
        return {
            n: (a[:, i] * self.stds[n] + self.means[n]).reshape(B, -1)
            for i, n in enumerate(self.out_names)
        }

    def _pad_to(self, u, like):
        F = self.torch.nn.functional
        ph = like.shape[-2] - u.shape[-2]
        pw = like.shape[-1] - u.shape[-1]
        u = F.pad(u, (pw // 2, pw - pw // 2, 0, 0), mode=self.net.pad)
        return F.pad(u, (0, 0, ph // 2, ph - ph // 2), mode="constant")

    def _convblock(self, l0, x):
        F = self.torch.nn.functional
        for layer in l0.convblock:
            if isinstance(layer, self.torch.nn.Conv2d) and layer.kernel_size[0] != 1:
                x = F.pad(x, (l0.N_pad, l0.N_pad, 0, 0), mode=l0.pad)
                x = F.pad(x, (0, 0, l0.N_pad, l0.N_pad), mode="constant")
            x = layer(x)
        return x

    def _run(self, x, read):
        """One forward. ``read(receiver, {sender: value})`` returns the per-sender values the
        receiver uses."""
        L, N, F = self.net.layers, self.N, self.torch.nn.functional
        out = {"input": x}
        r = read("l0.skip_module", {"input": out["input"]})
        out["l0.skip_module"] = L[0].skip_module(r["input"])
        r = read("l0.convblock", {"input": out["input"]})
        out["l0.convblock"] = self._convblock(L[0], r["input"])
        r = read("enc1", {s: out[s] for s in ("l0.skip_module", "l0.convblock")})
        out["enc1"] = L[2](L[1](r["l0.skip_module"] + r["l0.convblock"]))
        for k in range(2, N):
            r = read(f"enc{k}", {f"enc{k - 1}": out[f"enc{k - 1}"]})
            out[f"enc{k}"] = L[2 * k](L[2 * k - 1](r[f"enc{k - 1}"]))
        r = read("bottleneck", {f"enc{N - 1}": out[f"enc{N - 1}"]})
        out["bottleneck"] = L[2 * N](L[2 * N - 1](r[f"enc{N - 1}"]))
        below = "bottleneck"
        for j in range(N - 1, 0, -1):
            up, conv = L[2 * N + 1 + 2 * (N - 1 - j)], L[2 * N + 2 + 2 * (N - 1 - j)]
            r = read(f"dec{j}", {below: out[below], f"enc{j}": out[f"enc{j}"]})
            out[f"dec{j}"] = conv(self._pad_to(up(r[below]), r[f"enc{j}"]) + r[f"enc{j}"])
            below = f"dec{j}"
        r = read(
            "out",
            {
                "dec1": out["dec1"],
                "l0.skip_module": out["l0.skip_module"],
                "l0.convblock": out["l0.convblock"],
            },
        )
        e0 = r["l0.skip_module"] + r["l0.convblock"]
        s0 = self._pad_to(L[2 * N + 1 + 2 * (N - 1)](r["dec1"]), e0) + e0
        c0 = L[4 * N](s0)
        npad = self.net.N_pad
        c0 = F.pad(c0, (npad, npad, 0, 0), mode=self.net.pad)
        c0 = F.pad(c0, (0, 0, npad, npad), mode="constant")
        return L[4 * N + 1](c0), out

    def _forward(self, state: SamudraState, patch: Patch | None):
        g = self._graph
        if patch is None:
            return self._run(state.x, lambda r, v: v)[0]
        if not isinstance(patch.source, SamudraState) or patch.source.x.shape != state.x.shape:
            raise RequestError("a patch's source must be a state of the same batch size")
        _, src = self._run(patch.source.x, lambda r, v: v)
        if isinstance(patch, ComponentPatch):
            bad = [c for c in patch.components if c not in g.senders]
            if bad:
                raise RequestError(f"no component {bad[0]!r}; senders are {', '.join(g.senders)}")
            comps = set(patch.components)
            return self._run(
                state.x, lambda r, v: {s: (src[s] if s in comps else val) for s, val in v.items()}
            )[0]
        if isinstance(patch, EdgePatch):
            m = patch.mask
            if tuple(m.shape) != g.shape:
                raise RequestError(f"an edge mask is {g.shape}, got {tuple(m.shape)}")
            if not self.torch.is_tensor(m):
                m = self.torch.as_tensor(
                    np.asarray(m, dtype=float), dtype=self._dtype(), device=self.dev
                )

            def read(r, vals):
                j = g.receivers.index(r)
                return {
                    s: src[s] + m[g.senders.index(s), j] * (v - src[s]) for s, v in vals.items()
                }

            return self._run(state.x, read)[0]
        raise RequestError(f"unknown patch {type(patch).__name__}")
