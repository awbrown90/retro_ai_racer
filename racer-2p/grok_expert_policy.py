#!/usr/bin/env python3
"""Grok expert client for play-racer-2p.py — drive red or blue and win.

The server is lockstep: it writes ``state.json`` / ``obs.npy`` into one
player folder and waits for a matching ``action.json``.  This process is
given either colour and then lives entirely inside that folder.  It never
opens the other player's files, the server, or the shared track code.

Usage:

    python grok_expert_policy.py red
    python grok_expert_policy.py blue
    python grok_expert_policy.py --player blue-player

How it wins
-----------
The stage, centrifugal push, grass cap and traffic hit-boxes are public
constants from racer_env.py.  Combined with the telemetry in ``state.json``
that is enough to simulate a turn *exactly* the way the server will hold
the action.  Vision on ``obs.npy`` finds the painted traffic cars so the
planner can pick a committed passing lane instead of sitting on a bumper.

Each decision is a short beam search over the nine legal button combos,
rolled out for the server's frame-skip.  Staying on the tarmac and reaching
the next checkpoint before the clock dies outrank everything else: a car
that runs out of time is disqualified on the spot, however far ahead it was.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Public physics, mirrored from racer_env.py.  The client never imports the
# environment — it only uses the numbers the README already documents.

FPS = 50
DT = 1.0 / FPS
SEG_LEN = 200.0
ROAD_W = 2400.0
FOV = 100.0
CAM_HEIGHT = 1300.0
WIDTH, HEIGHT = 320, 200

MAX_SPEED = SEG_LEN / DT
ACCEL = MAX_SPEED / 3.6
BRAKE = -MAX_SPEED / 1.6
DECEL = -MAX_SPEED / 7.0
OFF_DECEL = -MAX_SPEED / 1.4
OFF_LIMIT = MAX_SPEED / 3.6
CENTRIFUGAL = 0.32
TOP_MPH = 186.0

TRAFFIC_W = 1050.0
PLAYER_CAR_PX = 76.0
HIT_AHEAD = 280.0
HIT_STANDOFF = 240.0
CLOSING_HIT = 1.15
CHECKPOINT_EVERY = 550
CHECKPOINT_BONUS = 22.0

CAM_DEPTH = 1.0 / math.tan((FOV / 2.0) * math.pi / 180.0)
PLAYER_Z = CAM_HEIGHT * CAM_DEPTH
_SCALE_P = CAM_DEPTH / PLAYER_Z
PLAYER_HALF_W = (PLAYER_CAR_PX / 2.0) / (_SCALE_P * ROAD_W * WIDTH / 2.0)
TRAFFIC_HALF_W = (TRAFFIC_W / 2.0) / ROAD_W
TRAFFIC_REACH = PLAYER_HALF_W + TRAFFIC_HALF_W
RIVAL_REACH = PLAYER_HALF_W * 2.0

STEER_PER_TICK = DT * 2.4
ACCEL_PER_TICK = ACCEL * DT / MAX_SPEED
BRAKE_PER_TICK = -BRAKE * DT / MAX_SPEED
COAST_PER_TICK = -DECEL * DT / MAX_SPEED
OFFROAD_PER_TICK = -OFF_DECEL * DT / MAX_SPEED
OFFROAD_LIMIT = OFF_LIMIT / MAX_SPEED

ROAD_SOFT = 0.86
ROAD_HARD = 1.0
PASS_EDGE = 0.78

ACCEL_A = (1, 0, 0, 0)
ACCEL_L = (1, 0, 1, 0)
ACCEL_R = (1, 0, 0, 1)
COAST = (0, 0, 0, 0)
COAST_L = (0, 0, 1, 0)
COAST_R = (0, 0, 0, 1)
BRAKE_A = (0, 1, 0, 0)
BRAKE_L = (0, 1, 1, 0)
BRAKE_R = (0, 1, 0, 1)

ACTIONS: tuple[tuple[int, int, int, int], ...] = (
    ACCEL_A, ACCEL_L, ACCEL_R,
    COAST, COAST_L, COAST_R,
    BRAKE_A, BRAKE_L, BRAKE_R,
)

ACTION_NAME = {
    ACCEL_A: "accel",
    ACCEL_L: "accel-left",
    ACCEL_R: "accel-right",
    COAST: "coast",
    COAST_L: "coast-left",
    COAST_R: "coast-right",
    BRAKE_A: "brake",
    BRAKE_L: "brake-left",
    BRAKE_R: "brake-right",
}

# Hand-authored stage from racer_env._build_track: (enter, hold, leave, curve)
ROAD_SPECS = (
    (60, 120, 60, 0.0),
    (50, 90, 50, 2.6),
    (40, 70, 40, 0.0),
    (50, 100, 50, -3.2),
    (30, 40, 30, 4.4),
    (40, 80, 40, 0.0),
    (25, 30, 25, -5.0),
    (25, 30, 25, 5.0),
    (25, 30, 25, -5.0),
    (60, 140, 60, 0.0),
    (50, 110, 50, 3.0),
    (40, 60, 40, -2.2),
    (30, 40, 30, 6.0),
    (60, 160, 60, 0.0),
    (45, 80, 45, -4.0),
    (35, 50, 35, 3.4),
    (70, 180, 70, 0.0),
)

TRAFFIC_LIVERIES = (
    ("yellow", (228, 192, 56)),
    ("white", (216, 216, 220)),
    ("purple", (150, 66, 172)),
    ("orange", (232, 132, 40)),
    ("cyan", (86, 198, 196)),
)


def ease_in(a: float, b: float, p: float) -> float:
    return a + (b - a) * p * p


def ease_in_out(a: float, b: float, p: float) -> float:
    return a + (b - a) * (-math.cos(p * math.pi) / 2.0 + 0.5)


def normalize_player(value: str) -> str:
    name = value.strip().lower().replace("_", "-")
    if name.endswith("-player"):
        name = name[: -len("-player")]
    if name not in ("red", "blue"):
        raise argparse.ArgumentTypeError(
            "player must be red, blue, red-player or blue-player"
        )
    return name


# ---------------------------------------------------------------------------
# Fixed course


class Track:
    """Curvature and racing line for the public, unchanging stage."""

    def __init__(self) -> None:
        curves: list[float] = []
        for enter, hold, leave, curve in ROAD_SPECS:
            curves.extend(ease_in(0.0, curve, i / enter) for i in range(enter))
            curves.extend([curve] * hold)
            curves.extend(ease_in_out(curve, 0.0, i / leave)
                          for i in range(leave))
        self.curves = np.asarray(curves, dtype=np.float64)
        self.n_segs = int(self.curves.size)
        self.track_len = self.n_segs * SEG_LEN

    def curve_at(self, distance: float) -> float:
        index = int(np.clip(distance / SEG_LEN, 0, self.n_segs - 1))
        return float(self.curves[index])

    def lookahead(self, distance: float, speed_pct: float) -> float:
        reach = 4200.0 + 9000.0 * max(0.22, speed_pct)
        samples = (0.0, 0.16, 0.36, 0.62, 1.0)
        weights = (1.00, 0.90, 0.70, 0.46, 0.22)
        values = [self.curve_at(distance + reach * frac) for frac in samples]
        blend = sum(v * w for v, w in zip(values, weights)) / sum(weights)
        current = values[0]
        if abs(current) > 0.32:
            blend = 0.70 * current + 0.30 * blend
        return blend

    def racing_lane(self, distance: float, speed_pct: float) -> float:
        curve = self.lookahead(distance, speed_pct)
        if abs(curve) < 0.14:
            return 0.0
        # Positive curve is a right-hander; the inside is +x.
        mag = 0.68 * math.tanh(abs(curve) / 2.55)
        mag *= 0.82 + 0.18 * min(1.0, speed_pct / 0.85)
        return math.copysign(mag, curve)

    def next_checkpoint(self, distance: float) -> float:
        seg = distance / SEG_LEN
        nxt = math.ceil((seg + 1e-6) / CHECKPOINT_EVERY) * CHECKPOINT_EVERY
        nxt = min(float(self.n_segs), nxt)
        return max(0.0, nxt * SEG_LEN - distance)


# ---------------------------------------------------------------------------
# Vision


@dataclass
class Blob:
    label: str
    x0: int
    x1: int
    y0: int
    y1: int
    area: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0 + 1

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0


@dataclass
class CarSighting:
    label: str
    lane: float
    gap: float
    width_px: float
    speed_pct: float = 0.36
    confidence: float = 1.0


@dataclass
class Obstacle:
    label: str
    lane: float
    gap: float
    speed_pct: float
    radius: float
    confidence: float = 1.0


def _components(mask: np.ndarray, label: str) -> list[Blob]:
    height, width = mask.shape
    runs: list[tuple[int, int, int]] = []
    parent: list[int] = []
    previous: list[int] = []

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[rb] = ra

    for y in range(height):
        padded = np.empty(width + 2, dtype=np.int8)
        padded[0] = padded[-1] = 0
        padded[1:-1] = mask[y]
        edges = np.diff(padded)
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1) - 1
        current: list[int] = []
        for x0, x1 in zip(starts.tolist(), ends.tolist()):
            if x1 - x0 + 1 < 2:
                continue
            idx = len(runs)
            runs.append((y, x0, x1))
            parent.append(idx)
            current.append(idx)
            for old in previous:
                _, ox0, ox1 = runs[old]
                if x0 <= ox1 + 1 and x1 + 1 >= ox0:
                    union(idx, old)
        previous = current

    groups: dict[int, list[int]] = {}
    for i in range(len(runs)):
        groups.setdefault(root(i), []).append(i)

    blobs: list[Blob] = []
    for members in groups.values():
        picked = [runs[i] for i in members]
        blobs.append(Blob(
            label,
            min(r[1] for r in picked),
            max(r[2] for r in picked),
            min(r[0] for r in picked),
            max(r[0] for r in picked),
            sum(r[2] - r[1] + 1 for r in picked),
        ))
    return blobs


def road_band(obs: np.ndarray, row: int) -> tuple[float, float] | None:
    h, w, _ = obs.shape
    row = int(np.clip(row, 96, h - 23))
    y0, y1 = max(0, row - 2), min(h, row + 3)
    band = obs[y0:y1].astype(np.int16)
    spread = band.max(axis=2) - band.min(axis=2)
    level = band.mean(axis=2)
    tarmac = (spread <= 15) & (level >= 52) & (level <= 122)
    counts = tarmac.sum(axis=0)
    occupied = counts > 0
    if occupied.sum() < 10:
        return None
    dense = occupied.copy()
    dense[1:] |= occupied[:-1]
    dense[:-1] |= occupied[1:]
    xs = np.flatnonzero(dense & occupied)
    if xs.size < 10:
        return None
    left, right = float(xs.min()), float(xs.max())
    if right - left < 18:
        return None
    return (left + right) / 2.0, (right - left) / 2.0


class TrafficVision:
    """Find the five traffic liveries the server actually paints."""

    def _modeled_road(
        self,
        track: Track,
        distance: float,
        player_lane: float,
        car_width: float,
        screen_w: int,
    ) -> tuple[float, float] | None:
        if car_width < 2.0:
            return None
        proj = TRAFFIC_W * CAM_DEPTH * float(screen_w) / 2.0
        dz = proj / car_width
        base = int(distance / SEG_LEN)
        target = int(round((distance + dz) / SEG_LEN))
        steps = target - base
        if steps < 0 or steps > 240:
            return None
        x = 0.0
        dx = -track.curve_at(distance) * ((distance % SEG_LEN) / SEG_LEN)
        for offset in range(steps):
            idx = base + offset
            if idx >= track.n_segs:
                break
            x += dx
            dx += float(track.curves[idx])
        half = car_width * ROAD_W / TRAFFIC_W
        centre = screen_w / 2.0 + half * (x / ROAD_W - player_lane)
        return centre, half

    def detect(
        self,
        obs: np.ndarray,
        track: Track,
        distance: float,
        player_lane: float,
    ) -> list[CarSighting]:
        if not isinstance(obs, np.ndarray) or obs.ndim != 3:
            return []
        h, w, c = obs.shape
        if c != 3 or h < 120 or w < 200:
            return []

        y0, y1 = 88, min(h - 21, 179)
        crop = obs[y0:y1].astype(np.int16)
        blobs: list[Blob] = []
        for label, rgb in TRAFFIC_LIVERIES:
            target = np.asarray(rgb, dtype=np.int16)
            mask = np.max(np.abs(crop - target), axis=2) <= 3
            for blob in _components(mask, label):
                blob.y0 += y0
                blob.y1 += y0
                blobs.append(blob)

        proj = TRAFFIC_W * CAM_DEPTH * float(w) / 2.0
        found: list[CarSighting] = []
        for blob in blobs:
            if blob.width < 3 or blob.area < 4:
                continue
            geometry = self._modeled_road(
                track, distance, player_lane, float(blob.width), w
            )
            if geometry is None:
                road_row = int(round(blob.y1 + 0.10 * blob.width))
                geometry = road_band(obs, road_row)
            if geometry is None:
                continue
            centre, half = geometry
            lane = (blob.cx - centre) / max(half, 1.0)
            if abs(lane) > 1.28:
                continue
            gap = proj / max(float(blob.width), 1.0) - PLAYER_Z
            if not 0.0 < gap < 55_000.0:
                continue
            found.append(CarSighting(
                label=blob.label,
                lane=float(np.clip(lane, -1.4, 1.4)),
                gap=float(gap),
                width_px=float(blob.width),
                confidence=min(1.0, blob.width / 10.0),
            ))

        found.sort(key=lambda car: car.gap)
        unique: list[CarSighting] = []
        for car in found:
            if any(
                car.label == old.label
                and abs(car.lane - old.lane) < 0.12
                and abs(car.gap - old.gap) < 1800.0
                for old in unique
            ):
                continue
            unique.append(car)
        return unique


# ---------------------------------------------------------------------------
# Predictive controller


@dataclass(frozen=True)
class SimState:
    distance: float
    speed_pct: float
    lane: float


@dataclass
class SimResult:
    state: SimState
    trajectory: list[tuple[float, float, float]]
    gain: float
    off_fraction: float
    soft_cost: float
    max_abs_lane: float


@dataclass
class BeamNode:
    state: SimState
    score: float
    first: tuple[int, int, int, int] | None


class GrokExpertPolicy:
    """Plan a winning line from one player's private observation."""

    def __init__(
        self,
        player: str,
        initial_frame_skip: int = 5,
        verbose: bool = False,
    ) -> None:
        self.player = player
        self.track = Track()
        self.vision = TrafficVision()
        self.frame_skip = max(1, int(initial_frame_skip))
        self.verbose = verbose
        self.prev_tick: int | None = None
        self.prev_distance: float | None = None
        self.prev_cars: list[CarSighting] = []
        self.pass_target: float | None = None
        self.pass_hold = 0
        self.last_action = ACCEL_A

    def simulate(
        self,
        start: SimState,
        action: tuple[int, int, int, int],
        steps: int,
    ) -> SimResult:
        acc, brk, left, right = action
        distance, speed, lane = start.distance, start.speed_pct, start.lane
        path: list[tuple[float, float, float]] = []
        off = 0
        soft = 0.0
        widest = abs(lane)
        for _ in range(steps):
            curve = self.track.curve_at(distance)
            steer = STEER_PER_TICK * speed
            lane += steer * (right - left)
            lane -= steer * speed * curve * CENTRIFUGAL
            if acc:
                speed += ACCEL_PER_TICK
            elif brk:
                speed -= BRAKE_PER_TICK
            else:
                speed -= COAST_PER_TICK
            if abs(lane) > ROAD_HARD and speed > OFFROAD_LIMIT:
                speed -= OFFROAD_PER_TICK
                off += 1
            speed = float(np.clip(speed, 0.0, 1.0))
            lane = float(np.clip(lane, -2.4, 2.4))
            distance += speed * MAX_SPEED * DT
            path.append((distance, lane, speed))
            extra = max(0.0, abs(lane) - ROAD_SOFT)
            soft += extra * extra
            widest = max(widest, abs(lane))
        return SimResult(
            SimState(distance, speed, lane),
            path,
            distance - start.distance,
            off / max(1, steps),
            soft / max(1, steps),
            widest,
        )

    def _learn_skip(self, state: dict) -> None:
        tick = int(state.get("tick", 0))
        turn = int(state.get("turn", 0))
        if self.prev_tick is not None and tick > self.prev_tick:
            self.frame_skip = max(1, tick - self.prev_tick)
        elif turn > 0 and tick > 0:
            guess = int(round(tick / turn))
            if guess > 0:
                self.frame_skip = guess

    def _track_speeds(self, cars: list[CarSighting], state: dict) -> None:
        tick = int(state.get("tick", 0))
        distance = float(state.get("distance", 0.0))
        if (
            self.prev_tick is None
            or self.prev_distance is None
            or tick <= self.prev_tick
        ):
            return
        elapsed = (tick - self.prev_tick) * DT
        own = distance - self.prev_distance
        unused = set(range(len(self.prev_cars)))
        for car in cars:
            best: tuple[float, int] | None = None
            for idx in unused:
                old = self.prev_cars[idx]
                if old.label != car.label:
                    continue
                expected = old.gap - own + 0.36 * MAX_SPEED * elapsed
                cost = abs(car.lane - old.lane) * 5000.0 + abs(car.gap - expected)
                if best is None or cost < best[0]:
                    best = (cost, idx)
            if best is None or best[0] > 6500.0:
                continue
            old = self.prev_cars[best[1]]
            estimate = (car.gap - old.gap + own) / max(elapsed * MAX_SPEED, 1.0)
            if 0.15 <= estimate <= 0.62:
                car.speed_pct = 0.68 * estimate + 0.32 * old.speed_pct
            unused.discard(best[1])

    def _obstacles(
        self, cars: list[CarSighting], state: dict
    ) -> list[Obstacle]:
        out = [
            Obstacle(
                car.label,
                car.lane,
                car.gap,
                car.speed_pct,
                TRAFFIC_REACH + 0.07,
                car.confidence,
            )
            for car in cars
            if car.gap < 26_000.0
        ]
        opp = state.get("opponent") or {}
        lead = float(opp.get("lead", 1e9))
        if 0.0 < lead < 22_000.0:
            out.append(Obstacle(
                "rival",
                float(opp.get("lane_offset", 0.0)),
                lead,
                float(opp.get("speed_mph", 0.0)) / TOP_MPH,
                RIVAL_REACH + 0.06,
                1.0,
            ))
        return out

    def _clock_pressure(self, state: dict) -> float:
        """0 = plenty of time, 1 = about to be disqualified."""
        remaining = float(state.get("time_left", 32.0))
        distance = float(state.get("distance", 0.0))
        speed = max(0.18, float(state.get("speed_mph", 0.0)) / TOP_MPH)
        to_cp = self.track.next_checkpoint(distance)
        eta = to_cp / max(speed * MAX_SPEED, 1.0)
        if remaining <= 0.4:
            return 1.0
        slack = remaining - eta
        if slack > 10.0:
            return 0.0
        if slack < 1.2:
            return 1.0
        return float(np.clip((8.0 - slack) / 8.0, 0.0, 1.0))

    def _choose_lane(
        self,
        state: dict,
        obstacles: list[Obstacle],
        base: float,
        pressure: float,
    ) -> tuple[float, bool]:
        current = float(state.get("lane_offset", 0.0))
        speed = float(state.get("speed_mph", 0.0)) / TOP_MPH
        behind = float((state.get("opponent") or {}).get("lead", 0.0)) > 80.0

        def must_pass(car: Obstacle) -> bool:
            closing = speed - car.speed_pct
            if car.gap < 2800.0:
                return True
            if closing <= 0.02:
                return False
            return (car.gap / (closing * MAX_SPEED)) < (4.6 if behind else 4.0)

        relevant = [
            car for car in obstacles
            if 0.0 < car.gap < 24_000.0 and must_pass(car)
        ]
        threatened = [
            car for car in relevant
            if abs(current - car.lane) < car.radius + 0.22
            or (
                self.pass_target is not None
                and abs(self.pass_target - car.lane) < car.radius + 0.14
            )
        ]

        if self.pass_target is not None:
            blocked = any(
                abs(self.pass_target - car.lane) < car.radius + 0.06
                for car in relevant
            )
            if not blocked:
                if relevant:
                    self.pass_hold = max(
                        self.pass_hold,
                        max(2, int(math.ceil(0.55 * FPS / self.frame_skip))),
                    )
                elif self.pass_hold > 0:
                    self.pass_hold -= 1
                if relevant or self.pass_hold > 0:
                    return self.pass_target, True
                self.pass_target = None

        if threatened:
            candidates = (
                -PASS_EDGE, -0.64, -0.40, -0.18, 0.0, 0.18, 0.40, 0.64, PASS_EDGE
            )
            best_lane, best = base, -1e9
            for lane in candidates:
                score = -0.48 * abs(lane - current)
                score -= 0.18 * abs(lane - base)
                score -= 0.16 * abs(lane)
                if pressure > 0.55:
                    score -= 0.55 * max(0.0, abs(lane) - 0.72)
                for car in relevant:
                    clear = abs(lane - car.lane) - car.radius
                    urgency = float(np.clip(
                        (20_000.0 - car.gap) / 15_000.0, 0.15, 1.0
                    ))
                    if behind:
                        urgency = min(1.0, urgency * 1.15)
                    if clear < 0.06:
                        score -= (8.4 + 26.0 * (0.06 - clear)) * urgency
                    else:
                        score += min(clear, 0.92) * 0.58 * urgency
                if score > best:
                    best_lane, best = lane, score
            self.pass_target = best_lane
            self.pass_hold = max(3, int(math.ceil(0.9 * FPS / self.frame_skip)))
            return best_lane, True

        if self.pass_target is not None and self.pass_hold > 0:
            self.pass_hold -= 1
            return self.pass_target, True

        self.pass_target = None
        self.pass_hold = 0

        opp = state.get("opponent") or {}
        lead = float(opp.get("lead", 1e9))
        # Legal defence: contact only delays the car behind.
        if (
            -4600.0 < lead < -180.0
            and not relevant
            and abs(base) < 0.50
            and pressure < 0.7
        ):
            block = float(np.clip(float(opp.get("lane_offset", 0.0)), -0.66, 0.66))
            return block, False
        return base, False

    def _collision(
        self,
        path: Sequence[tuple[float, float, float]],
        obstacles: Sequence[Obstacle],
        start_distance: float,
        elapsed: int,
        pressure: float,
    ) -> float:
        penalty = 0.0
        for car in obstacles:
            hit = False
            near = 0.0
            for local, (distance, lane, _) in enumerate(path, start=1):
                tick = elapsed + local
                rel = (
                    car.gap
                    + car.speed_pct * MAX_SPEED * DT * tick
                    - (distance - start_distance)
                )
                sep = abs(lane - car.lane)
                if -100.0 < rel < 520.0:
                    if sep < car.radius:
                        hit = True
                    elif sep < car.radius + 0.08:
                        near = max(near, 2.6)
                elif 0.0 < rel < 3600.0:
                    short = car.radius + 0.10 - sep
                    if short > 0.0:
                        near = max(near, 18.0 * ((3600.0 - rel) / 3600.0) * short)
            if hit:
                # Rival contact only holds the trailer; traffic is a real shunt.
                if car.label == "rival":
                    penalty += 11.0 if pressure < 0.75 else 6.0
                else:
                    penalty += 88.0
            penalty += near * car.confidence
        return penalty

    def _score(
        self,
        before: SimState,
        result: SimResult,
        target: float,
        obstacles: Sequence[Obstacle],
        start_distance: float,
        elapsed: int,
        steps: int,
        action: tuple[int, int, int, int],
        pressure: float,
    ) -> float:
        ceiling = MAX_SPEED * DT * steps
        score = 10.2 * result.gain / max(ceiling, 1.0)
        score += (0.62 + 0.35 * pressure) * result.state.speed_pct
        score -= 78.0 * result.off_fraction
        score -= 155.0 * result.soft_cost
        if abs(before.lane) <= ROAD_HARD < abs(result.state.lane):
            score -= 46.0
        if result.max_abs_lane > ROAD_HARD:
            score -= 125.0 * (result.max_abs_lane - ROAD_HARD) ** 2
        if abs(result.state.lane) > 0.94:
            score -= 14.0 * (abs(result.state.lane) - 0.94)
        score -= (1.00 + 0.35 * pressure) * abs(result.state.lane - target)
        if abs(before.lane) > ROAD_HARD:
            score += 6.0 * (abs(before.lane) - abs(result.state.lane))
        score -= self._collision(
            result.trajectory, obstacles, start_distance, elapsed, pressure
        )
        # High frame-skip: a steer is a full-road swipe. Prefer still wheels
        # unless the line actually needs them.
        if action[2] or action[3]:
            score -= 0.04 if self.frame_skip >= 20 else 0.02
        if pressure > 0.65 and action[1]:
            score -= 0.8 * pressure
        return score

    def _search(
        self,
        start: SimState,
        desired: float,
        passing: bool,
        obstacles: list[Obstacle],
        steps: int,
        pressure: float,
    ) -> tuple[int, int, int, int]:
        turn_s = steps * DT
        horizon = int(math.ceil(1.85 / max(turn_s, DT)))
        horizon = int(np.clip(horizon, 3, 18))
        width = 56 if steps <= 10 else 48
        beam = [BeamNode(start, 0.0, None)]
        for depth in range(horizon):
            kids: list[BeamNode] = []
            elapsed = depth * steps
            for node in beam:
                for action in ACTIONS:
                    result = self.simulate(node.state, action, steps)
                    target = desired if passing else self.track.racing_lane(
                        result.state.distance, result.state.speed_pct
                    )
                    score = node.score + self._score(
                        node.state,
                        result,
                        target,
                        obstacles,
                        start.distance,
                        elapsed,
                        steps,
                        action,
                        pressure,
                    )
                    kids.append(BeamNode(
                        result.state, score, node.first or action
                    ))
            kids.sort(
                key=lambda n: (
                    n.score
                    + 0.62 * n.state.speed_pct
                    - 0.16 * abs(n.state.lane - desired)
                ),
                reverse=True,
            )
            beam = kids[:width]
        return beam[0].first or ACCEL_A

    def decide(
        self, obs: np.ndarray, state: dict
    ) -> tuple[int, int, int, int]:
        self._learn_skip(state)
        distance = float(state.get("distance", 0.0))
        lane = float(state.get("lane_offset", 0.0))
        cars = self.vision.detect(obs, self.track, distance, lane)
        self._track_speeds(cars, state)
        obstacles = self._obstacles(cars, state)
        speed = float(state.get("speed_mph", 0.0)) / TOP_MPH
        pressure = self._clock_pressure(state)
        start = SimState(distance, speed, lane)
        base = self.track.racing_lane(distance, speed)
        desired, passing = self._choose_lane(state, obstacles, base, pressure)
        action = self._search(
            start, desired, passing, obstacles, self.frame_skip, pressure
        )

        if self.verbose:
            near = min(obstacles, key=lambda c: c.gap, default=None)
            traffic = "clear" if near is None else (
                f"{near.label}@{near.gap:5.0f}/x{near.lane:+.2f}"
            )
            print(
                f"{self.player} turn {int(state.get('turn', -1)):4d} "
                f"fs={self.frame_skip:2d} "
                f"{float(state.get('speed_mph', 0.0)):5.1f}mph "
                f"x={lane:+.3f} tgt={desired:+.2f} "
                f"clk={pressure:.2f} {traffic:>22s} -> {ACTION_NAME[action]}",
                flush=True,
            )

        self.prev_tick = int(state.get("tick", 0))
        self.prev_distance = distance
        self.prev_cars = cars
        self.last_action = action
        return action


# ---------------------------------------------------------------------------
# Folder protocol — this process only touches one player directory.


def read_state(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def read_obs(path: Path) -> np.ndarray | None:
    try:
        obs = np.load(path, allow_pickle=False)
    except (OSError, ValueError, EOFError):
        return None
    if obs.shape != (HEIGHT, WIDTH, 3) or obs.dtype != np.uint8:
        return None
    return obs


def write_action(path: Path, turn: int, action: Sequence[int]) -> None:
    payload = {
        "turn": int(turn),
        "accelerate": int(action[0]),
        "brake": int(action[1]),
        "left": int(action[2]),
        "right": int(action[3]),
    }
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def run_client(args: argparse.Namespace) -> int:
    game_dir = Path(args.game_dir).expanduser().resolve()
    player_dir = game_dir / f"{args.player}-player"
    if not player_dir.is_dir():
        print(f"error: player folder does not exist: {player_dir}",
              file=sys.stderr)
        return 2

    state_path = player_dir / "state.json"
    obs_path = player_dir / "obs.npy"
    action_path = player_dir / "action.json"
    policy = GrokExpertPolicy(
        args.player,
        initial_frame_skip=args.initial_frame_skip,
        verbose=args.verbose,
    )

    answered: int | None = None
    last_activity = time.monotonic()
    started = time.time()
    saw_race = False
    announced = False

    while True:
        state = read_state(state_path)
        if state is None:
            if (
                args.idle_timeout > 0
                and time.monotonic() - last_activity > args.idle_timeout
            ):
                print(
                    f"{args.player}: no server state for "
                    f"{args.idle_timeout:g}s",
                    file=sys.stderr,
                )
                return 1
            time.sleep(args.poll_interval)
            continue

        status = state.get("status")
        turn = int(state.get("turn", -1))
        if status == "race_over":
            if not saw_race:
                try:
                    mtime = state_path.stat().st_mtime
                except OSError:
                    mtime = 0.0
                if mtime < started:
                    if (
                        args.idle_timeout > 0
                        and time.monotonic() - last_activity > args.idle_timeout
                    ):
                        print(
                            f"{args.player}: no new server race for "
                            f"{args.idle_timeout:g}s",
                            file=sys.stderr,
                        )
                        return 1
                    time.sleep(args.poll_interval)
                    continue
            result = state.get("result") or {}
            print(
                f"{args.player}: race over - winner "
                f"{result.get('winner', '?')} ({result.get('reason', '?')}), "
                f"place {state.get('you_placed', state.get('place', '?'))}",
                flush=True,
            )
            return 0

        if status != "awaiting_action" or turn < 0 or turn == answered:
            time.sleep(args.poll_interval)
            continue
        saw_race = True

        obs = read_obs(obs_path)
        if obs is None:
            time.sleep(args.poll_interval)
            continue

        if not announced:
            print(
                f"{args.player}: grok expert connected to {player_dir}",
                flush=True,
            )
            announced = True

        action = policy.decide(obs, state)
        write_action(action_path, turn, action)
        answered = turn
        last_activity = time.monotonic()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "player",
        nargs="?",
        type=normalize_player,
        help="red, blue, red-player or blue-player",
    )
    parser.add_argument(
        "--player",
        dest="player_flag",
        type=normalize_player,
        help="same as the positional colour argument",
    )
    parser.add_argument(
        "--game-dir",
        default=str(Path(__file__).resolve().parent),
        help="racer-2p directory (default: this script's folder)",
    )
    parser.add_argument(
        "--initial-frame-skip",
        type=int,
        default=5,
        help="turn-length guess until state.tick reveals the real skip",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.004,
    )
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=300.0,
        help="quit after this many seconds with no server; 0 waits forever",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="print every planned action",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    colour = args.player_flag or args.player
    if not colour:
        build_parser().error("specify red or blue (or red-player / blue-player)")
    args.player = colour
    try:
        return run_client(args)
    except KeyboardInterrupt:
        print(f"\n{args.player}: stopped", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
