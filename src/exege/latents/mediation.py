"""Mediation: does a steer act *through* another feature?

A steer on feature ``i`` changes a field. Did it do so by way of feature ``j`` -- at a later
layer, or at later steps -- or around it? Run the steer twice from one state under one
noise: once as it is (the *total* effect), and once while ``j`` is held, at every node it
is held at and every step from the first edit on, at the value it took in the control (the
*direct* effect, the part that does not pass through ``j``). What the hold takes away is
the *indirect* effect, through ``j``::

    total    = steer i                       - control
    direct   = steer i, j held at control    - control
    indirect = total - direct

A fourth run holds ``j`` with no steer at all. In a deterministic system its every field
and latent must equal the control's exactly -- holding ``j`` at the value it already has
changes nothing -- and ``exactness`` reports the largest difference found, so a hold that
is not a no-op (a lossy encode, a nondeterministic forward pass) shows before any share is
read. This is the step from a feature that matters to the path it matters along: from
concepts to circuits. Needs the ``latents`` extra.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from exege.core.errors import RequestError
from exege.core.extras import missing_extra
from exege.latents.basis import Decomposition, basis_provenance, result_provenance
from exege.latents.source import check_basis_fits
from exege.latents.steering import (
    Hook,
    Intervenable,
    Outcome,
    Rollout,
    Steer,
    _guarded,
    _plan,
    _reduce,
    feature_direction,
)

try:
    import numpy as np
except ImportError as exc:
    raise missing_extra("numpy", "latents") from exc

TOTAL, DIRECT, HOLD = "total", "direct", "hold"


@dataclass(frozen=True)
class Mediator:
    """The feature to hold: ``feature`` of ``basis``, read at ``layer``, held at ``nodes``
    (None: every valid node) at every step from the steer's first edit on."""

    layer: int
    feature: int
    basis: Decomposition
    nodes: tuple[int, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer, "feature": self.feature,
            "nodes": None if self.nodes is None else list(self.nodes),
            "basis": basis_provenance(self.basis),
        }  # fmt: skip


@dataclass(frozen=True)
class MediationEffect:
    """On one field, seed-means of the paired responses (``*_se``: standard errors across
    seeds, None for one): ``total``, ``direct`` and ``indirect`` (their difference), and
    ``share`` the indirect over the total (None where the total is zero)."""

    field: str
    total: float
    direct: float
    indirect: float
    share: float | None
    total_se: float | None
    direct_se: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field, "total": self.total, "direct": self.direct,
            "indirect": self.indirect, "share": self.share, "total_se": self.total_se,
            "direct_se": self.direct_se,
        }  # fmt: skip


@dataclass(frozen=True)
class MediationResult:
    """Per field, a ``MediationEffect``; ``mediator_change`` the area-weighted RMS, over
    the nodes it is held at and the steps from the first edit on, of how far the steer
    alone moved the mediator's activation (zero: the steer never reached it there, and
    there is nothing to mediate); ``exactness`` the largest difference between the
    hold-only run and the control, over every field measured and recorded latent at valid nodes
    (zero in a deterministic system; infinite where the hold-only run is not a number
    where it should be); ``hold_error`` the largest gap, in the direct run, between the
    mediator's activation where it is held and the value it was held at (what the hold
    could not reach: a ReLU feature asked to go below zero, say), and ``hold_rms`` that
    gap's RMS, weighted as ``mediator_change`` is. Where ``hold_rms`` is not small beside
    ``mediator_change`` the direct run did not hold the mediator, and ``share`` says
    little: a top-k feature, whose activation jumps as the layer moves along it, often
    cannot be held at all. ``held`` is False when ``hold_rms`` is a tenth of
    ``mediator_change`` or more."""

    spec: dict[str, Any]
    provenance: dict[str, Any]
    effects: dict[str, MediationEffect]
    mediator_change: float
    exactness: float
    responses: dict[str, dict[str, list[float]]]
    hold_error: float = 0.0
    hold_rms: float = 0.0

    @property
    def held(self) -> bool:
        return self.hold_rms < 0.1 * self.mediator_change

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec, "provenance": self.provenance,
            "effects": {n: e.to_dict() for n, e in self.effects.items()},
            "mediator_change": self.mediator_change, "exactness": self.exactness,
            "hold_error": self.hold_error, "hold_rms": self.hold_rms, "held": self.held,
            "responses": self.responses,
        }  # fmt: skip


def _se(values: Sequence[float]) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    return float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else None


_SECANT_STEPS = 2
_BISECTIONS = 50


def _hold(
    mediator: Mediator, direction: np.ndarray, target: np.ndarray, held: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """An edit that sets the mediator's activation, where it is held, to ``target``.

    Moving the layer by ``delta`` along the feature's direction moves its activation by
    ``delta`` only when its encoder and decoder agree (a tied dictionary, a unit
    direction); an untied one, or a token-normalized one, moves it by some other amount,
    and a ReLU stops moving it at all past zero -- where many moves reach the same target,
    and the one wanted is the smallest. So the move is found per node: a couple of secant
    steps from ``target - now``, exact at once where encoder and decoder agree and wherever
    nothing needs moving (the hold is then a no-op, to the bit); then, for the nodes still
    off their target or whose path to it bends (the activation halfway is not halfway
    between), a bisection for a crossing of the target, read on those nodes
    alone. A target that no move along the direction reaches is left as near as it got,
    and ``hold_error`` says by how much."""
    feature, basis = mediator.feature, mediator.basis

    def read(latents: np.ndarray) -> np.ndarray:
        return basis.transform(latents, [feature])[:, 0]

    def moved(latents: np.ndarray, delta: np.ndarray) -> np.ndarray:
        # rows that do not move are left as they are, to the bit (-0.0 stays -0.0)
        out = latents.copy()
        rows = delta != 0.0
        out[rows] += (delta[rows, None] * direction[None, :]).astype(latents.dtype)
        return out

    def off(reached: np.ndarray, goal: np.ndarray) -> np.ndarray:
        return np.abs(goal - reached) > 1e-9 * (1.0 + np.abs(goal))

    def edit(latents: np.ndarray) -> np.ndarray:
        now = read(latents)
        delta = np.where(held, target - now, 0.0)
        for _ in range(_SECANT_STEPS):
            reached = read(moved(latents, delta))
            if not (held & off(reached, target)).any():
                break
            with np.errstate(invalid="ignore", divide="ignore"):
                slope = (reached - now) / delta
            usable = held & (delta != 0.0) & np.isfinite(slope) & (np.abs(slope) > 1e-9)
            step = np.where(usable, (target - reached) / np.where(usable, slope, 1.0), 0.0)
            delta = delta + step
        reached = read(moved(latents, delta))
        halfway = read(moved(latents, 0.5 * delta))
        bent = off(halfway, 0.5 * (now + reached))
        rows = np.flatnonzero(held & (target != now) & (off(reached, target) | bent))
        if rows.size:
            delta[rows] = _bisect(read, moved, latents[rows], now[rows], target[rows],
                                  delta[rows], reached[rows])  # fmt: skip
        return moved(latents, delta)

    return edit


def _bisect(read, moved, latents, now, goal, guess, reached):
    """Per row, a move along the direction whose activation is ``goal``: a bracket is
    sought between no move and a move grown from the guess (or ``goal - now``), first in
    that direction and then in the other -- a feature whose encoder and decoder point
    apart moves against its direction -- and bisected for a crossing. Where the activation
    along the direction is monotone that crossing is the nearest; where it is not (top-k,
    token norm) it is a crossing within the bracket, not always the nearest.
    Rows with no bracket either way keep ``guess``, and are not bisected."""
    sign = np.sign(goal - now)
    start = np.where(guess != 0.0, guess, np.where(goal != now, goal - now, 1.0))

    def bracket(first):
        hi, crossed = first.copy(), np.zeros(now.shape, dtype=bool)
        for _ in range(12):
            side = np.sign(read(moved(latents, hi)) - goal)
            crossed |= side != -sign
            hi = np.where(crossed, hi, 2.0 * hi)
            if crossed.all():
                break
        return hi, crossed

    hi, crossed = bracket(start)
    if not crossed.all():
        back, back_crossed = bracket(-start)
        hi = np.where(crossed, hi, back)
        crossed = crossed | back_crossed
    rows = np.flatnonzero(crossed)
    out = guess.copy()
    if not rows.size:
        return out
    part, g, s0 = latents[rows], goal[rows], sign[rows]
    lo, high = np.zeros(rows.size), hi[rows]
    for _ in range(_BISECTIONS):
        mid = 0.5 * (lo + high)
        below = np.sign(read(moved(part, mid)) - g) == -s0
        lo, high = np.where(below, mid, lo), np.where(below, high, mid)
    at_hi, at_lo = read(moved(part, high)), read(moved(part, lo))
    best = np.where(np.abs(at_hi - g) <= np.abs(at_lo - g), high, lo)
    # the nearest crossing is taken when it does no worse than the guess: where both reach
    # the goal (a ReLU's flat side), the smaller move is the one meant
    tolerance = 1e-9 * (1.0 + np.abs(g))
    better = np.abs(read(moved(part, best)) - g) <= np.abs(reached[rows] - g) + tolerance
    out[rows] = np.where(better, best, guess[rows])
    return out


def _compose(hooks: Sequence[Hook]) -> list[Hook]:
    """Hooks at one place, made one hook that applies them in the order given: a system
    is not promised to apply two hooks at one place in any order."""
    by_place: dict[tuple[int, int], list[Hook]] = {}
    for hook in hooks:
        by_place.setdefault((hook.layer, hook.time), []).append(hook)
    out = []
    for (layer, time), group in by_place.items():
        if len(group) == 1:
            out.append(group[0])
            continue

        def edit(latents: np.ndarray, group=group) -> np.ndarray:
            for hook in group:
                latents = hook.edit(latents)
            return latents

        out.append(Hook(layer, time, edit))
    return out


def run_mediation(
    system: Intervenable,
    basis: Decomposition,
    steer: Steer,
    mediator: Mediator,
    *,
    steps: int,
    seeds: Sequence[int] = (0,),
    fields: Sequence[str] | None = None,
    outcome: Outcome | None = None,
    initial_state: Any = None,
    allow_unverified_basis: bool = False,
) -> MediationResult:
    """Total, direct and indirect effects of ``steer`` through ``mediator``, per field.

    The steer is applied as ``run_steering`` applies it (``steer.loop`` included). The
    mediator is held at the control's value from the steer's first time to the last step,
    at its own layer: at the steer's layer and time it is held *after* the steer acts, so
    a mediator that is the steered feature itself takes the whole effect away. Each seed
    has its own control, and its own values to hold at.
    """
    info = system.info()
    check_basis_fits(mediator.basis, info, mediator.layer, allow_unverified=allow_unverified_basis)
    m_direction = feature_direction(mediator.basis, mediator.feature)
    plan = _plan(
        system, basis, steer, steps=steps, seeds=seeds, fields=fields,
        record_layers=[mediator.layer], outcome=outcome,
        allow_unverified_basis=allow_unverified_basis,
    )  # fmt: skip
    grid = plan.grid
    held = grid.valid.copy()
    if mediator.nodes is not None:
        if any(not 0 <= n < grid.n_nodes for n in mediator.nodes):
            raise RequestError(f"mediator nodes must lie in 0..{grid.n_nodes - 1}")
        chosen = np.zeros(grid.n_nodes, dtype=bool)
        chosen[list(mediator.nodes)] = True
        held &= chosen
        if not held.any():
            raise RequestError("every mediator node asked for is masked out")
    hold_times = range(plan.first, plan.steps)
    state = system.initial_state() if initial_state is None else initial_state
    weights = grid.weights()
    share = weights[held] / weights[held].sum()

    def activation(rollout: Rollout, t: int) -> np.ndarray:
        latents = rollout.latents[(mediator.layer, t)]
        return mediator.basis.transform(latents, [mediator.feature])[:, 0]

    names: list[str] = []
    responses: dict[str, dict[str, list[float]]] = {}
    change_sq, change_n, exactness, hold_error = 0.0, 0, 0.0, 0.0
    miss_sq, miss_n = 0.0, 0
    valid = grid.valid

    def gap(a: np.ndarray, b: np.ndarray) -> float:
        """The largest difference at valid nodes. A field may be missing where the grid is
        valid (an ocean level below the sea floor): missing in both runs at the same
        places is the same; missing in one only, or another kind of non-number, is
        infinitely inexact."""
        a = np.asarray(a, dtype=np.float64)[..., valid]
        b = np.asarray(b, dtype=np.float64)[..., valid]
        ok = np.isfinite(a)
        if (ok != np.isfinite(b)).any():
            return float("inf")
        if not ok.all() and not np.array_equal(a[~ok], b[~ok], equal_nan=True):
            return float("inf")
        return float(np.abs(a[ok] - b[ok]).max()) if ok.any() else 0.0

    for seed in plan.seeds:
        control = plan.run(state, (), seed)
        if not names:
            names = plan.names(control)
            responses = {n: {TOTAL: [], DIRECT: []} for n in names}
        series = _reduce(control.fields, names, grid, plan.steps, plan.outcome)
        targets = {t: activation(control, t) for t in hold_times}
        holds = [
            _guarded(Hook(mediator.layer, t, _hold(mediator, m_direction, targets[t], held)),
                     grid.valid)
            for t in hold_times
        ]  # fmt: skip
        steering = plan.hooks(steer, plan.direction, control)
        runs = {
            TOTAL: steering,
            DIRECT: _compose([*steering, *holds]),
            HOLD: holds,
        }
        for arm, hooks in runs.items():
            rollout = plan.run(state, hooks, seed, check_finite=arm != HOLD)
            if arm == HOLD:
                for name in names:  # the fields measured, held to the steering rule
                    exactness = max(exactness, gap(rollout.fields[name], control.fields[name]))
                for place in plan.places:
                    exactness = max(
                        exactness, gap(rollout.latents[place].T, control.latents[place].T)
                    )
                continue
            outcomes = _reduce(rollout.fields, names, grid, plan.steps, plan.outcome)
            for n in names:
                difference = outcomes[n] - series[n]
                responses[n][arm].append(float(difference[plan.first :].mean()))
            for t in hold_times:
                moved = activation(rollout, t) - targets[t]
                if arm == TOTAL:
                    change_sq += float(share @ (moved[held] ** 2))
                    change_n += 1
                else:
                    hold_error = max(hold_error, float(np.abs(moved[held]).max()))
                    miss_sq += float(share @ (moved[held] ** 2))
                    miss_n += 1
    effects = {}
    for n in names:
        total = float(np.mean(responses[n][TOTAL]))
        direct = float(np.mean(responses[n][DIRECT]))
        effects[n] = MediationEffect(
            field=n, total=total, direct=direct, indirect=total - direct,
            share=None if total == 0.0 else (total - direct) / total,
            total_se=_se(responses[n][TOTAL]), direct_se=_se(responses[n][DIRECT]),
        )  # fmt: skip
    spec = {
        "steer": steer.to_dict(), "mediator": mediator.to_dict(), "steps": plan.steps,
        "seeds": list(plan.seeds), "fields": names, "outcome": plan.outcome.to_dict(),
        "hold_times": list(hold_times),
    }  # fmt: skip
    return MediationResult(
        spec=spec,
        provenance=result_provenance(info, basis=basis),
        effects=effects,
        mediator_change=float(np.sqrt(change_sq / max(change_n, 1))),
        exactness=exactness,
        responses=responses,
        hold_error=hold_error,
        hold_rms=float(np.sqrt(miss_sq / max(miss_n, 1))),
    )
