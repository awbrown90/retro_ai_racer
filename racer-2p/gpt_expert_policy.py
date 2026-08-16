#!/usr/bin/env python3
"""Expert file-protocol client for the two-player Retro AI Racer race.

The server publishes a private ``state.json`` and ``obs.npy`` in each
player's folder.  This client reads only the selected player's folder and
writes a turn-matched ``action.json`` there.

Usage:

    python gpt_expert_policy.py red
    python gpt_expert_policy.py blue

The policy combines three sources of information:

* exact, predictive control for the fixed stage and its centrifugal forces;
* color/component detection in ``obs.npy`` for traffic and passing lanes;
* the rival telemetry in ``state.json`` for safe overtakes and legal blocking.

It evaluates complete actions over the server's frame-skip, then performs a
short beam search over future turns.  This matters at large frame-skips where
one full second of steering can otherwise put a car straight onto the grass.
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
# Game constants.  They mirror the public physics constants in racer_env.py;
# the client does not instantiate an environment or inspect hidden race state.

FPS = 50
DT = 1.0 / FPS
SEG_LEN = 200.0
MAX_SPEED = 10_000.0
TOP_MPH = 186.0
ACCEL_PER_TICK = DT / 3.6
BRAKE_PER_TICK = DT / 1.6
COAST_PER_TICK = DT / 7.0
OFFROAD_PER_TICK = DT / 1.4
OFFROAD_LIMIT = 1.0 / 3.6
STEER_PER_TICK = DT * 2.4
CENTRIFUGAL = 0.32

CAM_DEPTH = 1.0 / math.tan(math.radians(50.0))
PLAYER_Z = 1300.0 * CAM_DEPTH
TRAFFIC_WIDTH = 1050.0
PLAYER_HALF_WIDTH = 0.129
TRAFFIC_HALF_WIDTH = (TRAFFIC_WIDTH / 2.0) / 2400.0
TRAFFIC_REACH = PLAYER_HALF_WIDTH + TRAFFIC_HALF_WIDTH
RIVAL_REACH = PLAYER_HALF_WIDTH * 2.0

ROAD_SOFT_EDGE = 0.84
ROAD_HARD_EDGE = 1.0
PASS_EDGE = 0.80

# accelerate, brake, left, right
ACCEL = (1, 0, 0, 0)
ACCEL_LEFT = (1, 0, 1, 0)
ACCEL_RIGHT = (1, 0, 0, 1)
COAST = (0, 0, 0, 0)
COAST_LEFT = (0, 0, 1, 0)
COAST_RIGHT = (0, 0, 0, 1)
BRAKE = (0, 1, 0, 0)
BRAKE_LEFT = (0, 1, 1, 0)
BRAKE_RIGHT = (0, 1, 0, 1)

ACTIONS: tuple[tuple[int, int, int, int], ...] = (
    ACCEL,
    ACCEL_LEFT,
    ACCEL_RIGHT,
    COAST,
    COAST_LEFT,
    COAST_RIGHT,
    BRAKE,
    BRAKE_LEFT,
    BRAKE_RIGHT,
)

ACTION_NAMES = {
    ACCEL: "accelerate",
    ACCEL_LEFT: "accelerate-left",
    ACCEL_RIGHT: "accelerate-right",
    COAST: "coast",
    COAST_LEFT: "coast-left",
    COAST_RIGHT: "coast-right",
    BRAKE: "brake",
    BRAKE_LEFT: "brake-left",
    BRAKE_RIGHT: "brake-right",
}


# ---------------------------------------------------------------------------
# Fixed course model

ROAD_SPECS = (
    # enter, hold, leave, curve
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


def ease_in(a: float, b: float, p: float) -> float:
    return a + (b - a) * p * p


def ease_in_out(a: float, b: float, p: float) -> float:
    return a + (b - a) * (-math.cos(p * math.pi) / 2.0 + 0.5)


class TrackModel:
    """Curvature and racing-line model for the fixed stage."""

    def __init__(self) -> None:
        curves: list[float] = []
        for enter, hold, leave, curve in ROAD_SPECS:
            curves.extend(ease_in(0.0, curve, i / enter)
                          for i in range(enter))
            curves.extend([curve] * hold)
            curves.extend(ease_in_out(curve, 0.0, i / leave)
                          for i in range(leave))
        self.curves = np.asarray(curves, dtype=np.float64)
        self.track_len = float(len(curves)) * SEG_LEN

    def curve_at(self, distance: float) -> float:
        index = min(len(self.curves) - 1,
                    max(0, int(distance / SEG_LEN)))
        return float(self.curves[index])

    def lookahead_curve(self, distance: float, speed_pct: float) -> float:
        """Blend the current bend with the road reached in roughly 1.2 s."""
        reach = 4500.0 + 8000.0 * max(0.25, speed_pct)
        fractions = (0.0, 0.18, 0.40, 0.67, 1.0)
        weights = (1.00, 0.92, 0.72, 0.48, 0.25)
        values = [self.curve_at(distance + reach * f) for f in fractions]

        # Nearby road wins in an S-bend.  On a straight before turn-in, the
        # first non-trivial future curve establishes the side to prepare.
        blend = sum(v * w for v, w in zip(values, weights)) / sum(weights)
        current = values[0]
        if abs(current) > 0.35:
            blend = 0.68 * current + 0.32 * blend
        return blend

    def racing_lane(self, distance: float, speed_pct: float) -> float:
        curve = self.lookahead_curve(distance, speed_pct)
        if abs(curve) < 0.16:
            return 0.0
        # Positive curvature is a right turn, whose inside is positive x.
        magnitude = 0.66 * math.tanh(abs(curve) / 2.7)
        return math.copysign(magnitude, curve)


# ---------------------------------------------------------------------------
# Vision

TRAFFIC_COLORS: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("yellow", (228, 192, 56)),
    ("white", (216, 216, 220)),
    ("purple", (150, 66, 172)),
    ("orange", (232, 132, 40)),
    ("cyan", (86, 198, 196)),
)


@dataclass
class PixelComponent:
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
    def height(self) -> int:
        return self.y1 - self.y0 + 1

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0


@dataclass
class CarDetection:
    label: str
    lane: float
    gap: float
    width_px: float
    bottom_y: float
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


def _horizontal_components(mask: np.ndarray, label: str) -> list[PixelComponent]:
    """Group same-color horizontal runs without requiring scipy/OpenCV."""
    height, width = mask.shape
    runs: list[tuple[int, int, int]] = []
    parents: list[int] = []
    previous: list[int] = []

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(a: int, b: int) -> None:
        ra, rb = root(a), root(b)
        if ra != rb:
            parents[rb] = ra

    for y in range(height):
        row = mask[y]
        padded = np.empty(width + 2, dtype=np.int8)
        padded[0] = padded[-1] = 0
        padded[1:-1] = row
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1) - 1
        current: list[int] = []
        for x0, x1 in zip(starts.tolist(), ends.tolist()):
            if x1 - x0 + 1 < 2:
                continue
            index = len(runs)
            runs.append((y, x0, x1))
            parents.append(index)
            current.append(index)
            for old in previous:
                _, old_x0, old_x1 = runs[old]
                if x0 <= old_x1 + 1 and x1 + 1 >= old_x0:
                    union(index, old)
        previous = current

    grouped: dict[int, list[int]] = {}
    for i in range(len(runs)):
        grouped.setdefault(root(i), []).append(i)

    result: list[PixelComponent] = []
    for indices in grouped.values():
        selected = [runs[i] for i in indices]
        y0 = min(run[0] for run in selected)
        y1 = max(run[0] for run in selected)
        x0 = min(run[1] for run in selected)
        x1 = max(run[2] for run in selected)
        area = sum(run[2] - run[1] + 1 for run in selected)
        result.append(PixelComponent(label, x0, x1, y0, y1, area))
    return result


def road_bounds(obs: np.ndarray, row: int) -> tuple[float, float] | None:
    """Return the visible tarmac centre and half-width at a screen row."""
    height, width, _ = obs.shape
    row = int(np.clip(row, 96, height - 23))
    y0, y1 = max(0, row - 2), min(height, row + 3)
    band = obs[y0:y1].astype(np.int16)
    spread = band.max(axis=2) - band.min(axis=2)
    level = band.mean(axis=2)
    tarmac = (spread <= 15) & (level >= 52) & (level <= 122)
    counts = tarmac.sum(axis=0)
    xs = np.flatnonzero(counts > 0)
    if xs.size < 10:
        return None

    # Isolated grey scenery should not define an edge.  Keep columns close to
    # another road-colored column in the small vertical band.
    occupied = counts > 0
    dense = occupied.copy()
    dense[1:] |= occupied[:-1]
    dense[:-1] |= occupied[1:]
    xs = np.flatnonzero(dense & (counts > 0))
    if xs.size < 10:
        return None
    left, right = float(xs.min()), float(xs.max())
    if right - left < 18:
        return None
    return (left + right) / 2.0, (right - left) / 2.0


class TrafficVision:
    """Extract traffic cars from their exact, server-defined liveries."""

    @staticmethod
    def _modeled_road_geometry(
        track: TrackModel,
        distance: float,
        player_lane: float,
        car_width: float,
        screen_width: int,
    ) -> tuple[float, float] | None:
        """Road centre/width at a car, including bends and camera offset.

        Near a bend the tarmac often extends beyond both screen edges, so its
        visible min/max pixels are not its real projected boundaries.  A
        traffic body's width gives its projection scale directly; walking the
        public fixed-course curvature then recovers the unclipped road centre.
        """
        if car_width < 2.0:
            return None
        projection_numerator = (
            TRAFFIC_WIDTH * CAM_DEPTH * float(screen_width) / 2.0
        )
        projected_dz = projection_numerator / car_width
        base_index = int(distance / SEG_LEN)
        target_index = int(round((distance + projected_dz) / SEG_LEN))
        steps = target_index - base_index
        if steps < 0 or steps > 240:
            return None

        base_pct = (distance % SEG_LEN) / SEG_LEN
        x = 0.0
        dx = -track.curve_at(distance) * base_pct
        for offset in range(steps):
            index = base_index + offset
            if index >= len(track.curves):
                break
            x += dx
            dx += float(track.curves[index])

        road_half = car_width * 2400.0 / TRAFFIC_WIDTH
        road_center = (
            screen_width / 2.0
            + road_half * (x / 2400.0 - player_lane)
        )
        return road_center, road_half

    def detect(self, obs: np.ndarray, *, track: TrackModel | None = None,
               distance: float = 0.0,
               player_lane: float = 0.0) -> list[CarDetection]:
        if not isinstance(obs, np.ndarray) or obs.ndim != 3:
            return []
        height, width, channels = obs.shape
        if channels != 3 or height < 120 or width < 200:
            return []

        crop_y0, crop_y1 = 88, min(height - 21, 179)
        crop = obs[crop_y0:crop_y1].astype(np.int16)
        components: list[PixelComponent] = []

        for label, rgb in TRAFFIC_COLORS:
            target = np.asarray(rgb, dtype=np.int16)
            # Pygame's primitives are exact-color; a tiny tolerance also
            # survives RGB conversion or a future encoder change.
            mask = np.max(np.abs(crop - target), axis=2) <= 3
            for component in _horizontal_components(mask, label):
                component.y0 += crop_y0
                component.y1 += crop_y0
                components.append(component)

        detections: list[CarDetection] = []
        projection_numerator = (
            TRAFFIC_WIDTH * CAM_DEPTH * float(width) / 2.0
        )
        for component in components:
            if component.width < 3 or component.area < 4:
                continue
            # A traffic body occupies the upper 78% of its projected height;
            # its road-contact row is just below the exact-color rectangle.
            road_row = int(round(component.y1 + 0.10 * component.width))
            geometry = None
            if track is not None:
                geometry = self._modeled_road_geometry(
                    track,
                    distance,
                    player_lane,
                    float(component.width),
                    width,
                )
            if geometry is None:
                geometry = road_bounds(obs, road_row)
            if geometry is None:
                continue
            road_center, road_half = geometry
            lane = (component.cx - road_center) / max(road_half, 1.0)

            # Checkpoint signs share yellow with traffic but sit well outside
            # the tarmac.  The road-normalized lane removes them cleanly.
            if abs(lane) > 1.28:
                continue
            gap = projection_numerator / max(float(component.width), 1.0)
            gap -= PLAYER_Z
            if not 0.0 < gap < 55_000.0:
                continue

            confidence = min(1.0, component.width / 10.0)
            detections.append(CarDetection(
                label=component.label,
                lane=float(np.clip(lane, -1.4, 1.4)),
                gap=float(gap),
                width_px=float(component.width),
                bottom_y=float(component.y1),
                confidence=confidence,
            ))

        # Duplicate fragments of the same body are rare, but when one occurs
        # keep the larger/nearer interpretation.
        detections.sort(key=lambda car: car.gap)
        filtered: list[CarDetection] = []
        for car in detections:
            duplicate = any(
                car.label == old.label
                and abs(car.lane - old.lane) < 0.12
                and abs(car.gap - old.gap) < 1800.0
                for old in filtered
            )
            if not duplicate:
                filtered.append(car)
        return filtered


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
    soft_edge_cost: float
    max_abs_lane: float


@dataclass
class BeamNode:
    state: SimState
    score: float
    first_action: tuple[int, int, int, int] | None


class ExpertPolicy:
    def __init__(self, player: str, initial_frame_skip: int = 5,
                 verbose: bool = False) -> None:
        self.player = player
        self.track = TrackModel()
        self.vision = TrafficVision()
        self.frame_skip = max(1, int(initial_frame_skip))
        self.verbose = verbose

        self.previous_tick: int | None = None
        self.previous_turn: int | None = None
        self.previous_distance: float | None = None
        self.previous_detections: list[CarDetection] = []
        self.pass_target: float | None = None
        self.pass_memory = 0
        self.last_action = ACCEL

    # --------------------------------------------------------------- dynamics

    def simulate_turn(self, start: SimState,
                      action: tuple[int, int, int, int],
                      steps: int) -> SimResult:
        acc, brake, left, right = action
        distance = start.distance
        speed = start.speed_pct
        lane = start.lane
        trajectory: list[tuple[float, float, float]] = []
        off_ticks = 0
        soft_cost = 0.0
        max_abs = abs(lane)

        for _ in range(steps):
            curve = self.track.curve_at(distance)
            steer = STEER_PER_TICK * speed
            lane += steer * (right - left)
            lane -= steer * speed * curve * CENTRIFUGAL

            if acc:
                speed += ACCEL_PER_TICK
            elif brake:
                speed -= BRAKE_PER_TICK
            else:
                speed -= COAST_PER_TICK

            if abs(lane) > ROAD_HARD_EDGE and speed > OFFROAD_LIMIT:
                speed -= OFFROAD_PER_TICK
                off_ticks += 1

            speed = float(np.clip(speed, 0.0, 1.0))
            lane = float(np.clip(lane, -2.4, 2.4))
            distance += speed * MAX_SPEED * DT
            trajectory.append((distance, lane, speed))

            excess = max(0.0, abs(lane) - ROAD_SOFT_EDGE)
            soft_cost += excess * excess
            max_abs = max(max_abs, abs(lane))

        return SimResult(
            state=SimState(distance, speed, lane),
            trajectory=trajectory,
            gain=distance - start.distance,
            off_fraction=off_ticks / max(1, steps),
            soft_edge_cost=soft_cost / max(1, steps),
            max_abs_lane=max_abs,
        )

    # --------------------------------------------------------------- tracking

    def _learn_frame_skip(self, state: dict) -> None:
        tick = int(state.get("tick", 0))
        turn = int(state.get("turn", 0))
        if self.previous_tick is not None and tick > self.previous_tick:
            self.frame_skip = max(1, tick - self.previous_tick)
        elif turn > 0 and tick > 0:
            estimate = int(round(tick / turn))
            if estimate > 0:
                self.frame_skip = estimate

    def _estimate_traffic_speeds(self, detections: list[CarDetection],
                                 state: dict) -> None:
        tick = int(state.get("tick", 0))
        distance = float(state.get("distance", 0.0))
        if (self.previous_tick is None or self.previous_distance is None
                or tick <= self.previous_tick):
            return
        elapsed = (tick - self.previous_tick) * DT
        own_gain = distance - self.previous_distance

        unused = set(range(len(self.previous_detections)))
        for car in detections:
            best: tuple[float, int] | None = None
            for index in unused:
                old = self.previous_detections[index]
                if old.label != car.label:
                    continue
                expected_gap = old.gap - own_gain + 0.36 * MAX_SPEED * elapsed
                cost = (abs(car.lane - old.lane) * 5000.0
                        + abs(car.gap - expected_gap))
                if best is None or cost < best[0]:
                    best = (cost, index)
            if best is None or best[0] > 6500.0:
                continue
            old = self.previous_detections[best[1]]
            estimate = (
                car.gap - old.gap + own_gain
            ) / max(elapsed * MAX_SPEED, 1.0)
            if 0.15 <= estimate <= 0.62:
                car.speed_pct = 0.65 * estimate + 0.35 * old.speed_pct
            unused.discard(best[1])

    def _obstacles(self, detections: list[CarDetection],
                   state: dict) -> list[Obstacle]:
        obstacles = [
            Obstacle(
                label=car.label,
                lane=car.lane,
                gap=car.gap,
                speed_pct=car.speed_pct,
                radius=TRAFFIC_REACH + 0.075,
                confidence=car.confidence,
            )
            for car in detections
            if car.gap < 26_000.0
        ]

        opponent = state.get("opponent") or {}
        lead = float(opponent.get("lead", 1e9))
        if 0.0 < lead < 22_000.0:
            obstacles.append(Obstacle(
                label="rival",
                lane=float(opponent.get("lane_offset", 0.0)),
                gap=lead,
                speed_pct=float(opponent.get("speed_mph", 0.0)) / TOP_MPH,
                radius=RIVAL_REACH + 0.07,
                confidence=1.0,
            ))
        return obstacles

    # -------------------------------------------------------------- strategy

    def _choose_pass_lane(self, state: dict, obstacles: list[Obstacle],
                          base_lane: float) -> tuple[float, bool]:
        current = float(state.get("lane_offset", 0.0))
        speed_pct = float(state.get("speed_mph", 0.0)) / TOP_MPH

        def needs_pass(car: Obstacle) -> bool:
            closing = speed_pct - car.speed_pct
            if car.gap < 3000.0:
                return True
            if closing <= 0.025:
                return False
            time_to_car = car.gap / (closing * MAX_SPEED)
            return time_to_car < 4.2

        relevant = [
            car for car in obstacles
            if 0.0 < car.gap < 24_000.0 and needs_pass(car)
        ]
        threatened = [
            car for car in relevant
            if abs(current - car.lane) < car.radius + 0.22
            or (self.pass_target is not None
                and abs(self.pass_target - car.lane) < car.radius + 0.14)
        ]

        # Passing works only if the car commits to one side.  Keep a chosen
        # lane unless a newly visible car actually blocks it; recomputing from
        # the current lane every turn causes a costly left/right slalom in a
        # dense pack.
        if self.pass_target is not None:
            target_blocked = any(
                abs(self.pass_target - car.lane) < car.radius + 0.07
                for car in relevant
            )
            if not target_blocked:
                if relevant:
                    self.pass_memory = max(
                        self.pass_memory,
                        max(2, int(math.ceil(0.55 * FPS
                                             / self.frame_skip))),
                    )
                elif self.pass_memory > 0:
                    self.pass_memory -= 1
                if relevant or self.pass_memory > 0:
                    return self.pass_target, True
                self.pass_target = None

        if threatened:
            candidates = (-PASS_EDGE, -0.62, -0.38, 0.0,
                          0.38, 0.62, PASS_EDGE)
            best_lane, best_score = base_lane, -1e9
            for lane in candidates:
                score = -0.52 * abs(lane - current)
                score -= 0.22 * abs(lane - base_lane)
                score -= 0.20 * abs(lane)
                for car in relevant:
                    clearance = abs(lane - car.lane) - car.radius
                    urgency = float(np.clip(
                        (20_000.0 - car.gap) / 15_000.0, 0.15, 1.0
                    ))
                    if clearance < 0.06:
                        score -= (8.0 + 24.0 * (0.06 - clearance)) * urgency
                    else:
                        score += min(clearance, 0.9) * 0.55 * urgency
                if score > best_score:
                    best_lane, best_score = lane, score
            self.pass_target = best_lane
            self.pass_memory = max(
                3, int(math.ceil(0.9 * FPS / self.frame_skip))
            )
            return best_lane, True

        if self.pass_target is not None and self.pass_memory > 0:
            self.pass_memory -= 1
            return self.pass_target, True

        self.pass_target = None
        self.pass_memory = 0

        # When leading, matching a very close trailing rival's lane is a safe,
        # legal defense: race contact only delays the car behind.  Traffic and
        # track preparation always take priority over blocking.
        opponent = state.get("opponent") or {}
        lead = float(opponent.get("lead", 1e9))
        if (-4800.0 < lead < -150.0 and not relevant
                and abs(base_lane) < 0.48):
            block_lane = float(opponent.get("lane_offset", 0.0))
            return float(np.clip(block_lane, -0.68, 0.68)), False

        return base_lane, False

    def _collision_penalty(
        self,
        trajectory: Sequence[tuple[float, float, float]],
        obstacles: Sequence[Obstacle],
        initial_distance: float,
        elapsed_ticks: int,
    ) -> float:
        penalty = 0.0
        for car in obstacles:
            collision = False
            near_risk = 0.0
            for local_tick, (distance, lane, _) in enumerate(
                    trajectory, start=1):
                tick = elapsed_ticks + local_tick
                own_gain = distance - initial_distance
                relative_gap = (
                    car.gap
                    + car.speed_pct * MAX_SPEED * DT * tick
                    - own_gain
                )
                separation = abs(lane - car.lane)
                if -100.0 < relative_gap < 520.0:
                    if separation < car.radius:
                        collision = True
                    elif separation < car.radius + 0.08:
                        near_risk = max(near_risk, 2.5)
                elif 0.0 < relative_gap < 3600.0:
                    lateral_shortfall = car.radius + 0.10 - separation
                    if lateral_shortfall > 0.0:
                        proximity = (3600.0 - relative_gap) / 3600.0
                        near_risk = max(
                            near_risk,
                            18.0 * proximity * lateral_shortfall,
                        )
            if collision:
                # Rear-ending traffic is a real crash.  Rival contact is
                # different: the server merely holds the trailing car to the
                # leader's pace, so it should prompt a pass without making the
                # planner dive onto the grass to avoid one harmless tick.
                penalty += (13.0 if car.label == "rival" else 85.0)
            penalty += near_risk * car.confidence
        return penalty

    def _transition_score(
        self,
        before: SimState,
        result: SimResult,
        desired_lane: float,
        obstacles: Sequence[Obstacle],
        initial_distance: float,
        elapsed_ticks: int,
        steps: int,
        action: tuple[int, int, int, int],
    ) -> float:
        max_gain = MAX_SPEED * DT * steps
        score = 10.0 * result.gain / max(max_gain, 1.0)
        score += 0.55 * result.state.speed_pct

        # A brief brush of the rumble strip is tolerable; grass is not.
        score -= 72.0 * result.off_fraction
        score -= 150.0 * result.soft_edge_cost
        if (abs(before.lane) <= ROAD_HARD_EDGE
                and abs(result.state.lane) > ROAD_HARD_EDGE):
            score -= 42.0
        if result.max_abs_lane > ROAD_HARD_EDGE:
            score -= 115.0 * (result.max_abs_lane - ROAD_HARD_EDGE) ** 2
        if abs(result.state.lane) > 0.94:
            score -= 13.0 * (abs(result.state.lane) - 0.94)

        score -= 1.05 * abs(result.state.lane - desired_lane)
        if abs(before.lane) > ROAD_HARD_EDGE:
            score += 5.0 * (abs(before.lane) - abs(result.state.lane))

        score -= self._collision_penalty(
            result.trajectory,
            obstacles,
            initial_distance,
            elapsed_ticks,
        )

        # Tiny regularizer: on an otherwise straight/equal choice, keep the
        # wheel still and retain the option to steer next turn.
        if action[2] or action[3]:
            score -= 0.025
        return score

    def _search(self, start: SimState, desired_lane: float,
                pass_active: bool, obstacles: list[Obstacle],
                steps: int) -> tuple[int, int, int, int]:
        turn_seconds = steps * DT
        horizon = int(math.ceil(1.8 / max(turn_seconds, DT)))
        horizon = int(np.clip(horizon, 3, 18))
        beam_width = 54
        beam = [BeamNode(start, 0.0, None)]

        for depth in range(horizon):
            expanded: list[BeamNode] = []
            elapsed_ticks = depth * steps
            for node in beam:
                for action in ACTIONS:
                    result = self.simulate_turn(node.state, action, steps)
                    if pass_active:
                        target = desired_lane
                    else:
                        target = self.track.racing_lane(
                            result.state.distance,
                            result.state.speed_pct,
                        )
                    score = node.score + self._transition_score(
                        node.state,
                        result,
                        target,
                        obstacles,
                        start.distance,
                        elapsed_ticks,
                        steps,
                        action,
                    )
                    first = node.first_action or action
                    expanded.append(BeamNode(result.state, score, first))

            # The terminal heuristic helps the beam preserve speed and arrive
            # with lateral room for the next replan.
            expanded.sort(
                key=lambda node: (
                    node.score
                    + 0.6 * node.state.speed_pct
                    - 0.18 * abs(node.state.lane - desired_lane)
                ),
                reverse=True,
            )
            beam = expanded[:beam_width]

        return beam[0].first_action or ACCEL

    # ---------------------------------------------------------------- decide

    def decide(self, obs: np.ndarray, state: dict) -> tuple[int, int, int, int]:
        self._learn_frame_skip(state)
        distance = float(state.get("distance", 0.0))
        lane = float(state.get("lane_offset", 0.0))
        detections = self.vision.detect(
            obs,
            track=self.track,
            distance=distance,
            player_lane=lane,
        )
        self._estimate_traffic_speeds(detections, state)
        obstacles = self._obstacles(detections, state)

        speed_pct = float(state.get("speed_mph", 0.0)) / TOP_MPH
        start = SimState(distance, speed_pct, lane)

        base_lane = self.track.racing_lane(distance, speed_pct)
        desired_lane, pass_active = self._choose_pass_lane(
            state, obstacles, base_lane
        )
        action = self._search(
            start,
            desired_lane,
            pass_active,
            obstacles,
            self.frame_skip,
        )

        if self.verbose:
            nearest = min(obstacles, key=lambda car: car.gap,
                          default=None)
            traffic_text = "clear"
            if nearest is not None:
                traffic_text = (
                    f"{nearest.label}@{nearest.gap:5.0f}"
                    f"/x{nearest.lane:+.2f}"
                )
            print(
                f"{self.player} turn {int(state.get('turn', -1)):4d} "
                f"fs={self.frame_skip:2d} "
                f"{float(state.get('speed_mph', 0.0)):5.1f}mph "
                f"x={lane:+.3f} target={desired_lane:+.2f} "
                f"{traffic_text:>22s} -> {ACTION_NAMES[action]}",
                flush=True,
            )

        self.previous_tick = int(state.get("tick", 0))
        self.previous_turn = int(state.get("turn", 0))
        self.previous_distance = distance
        self.previous_detections = detections
        self.last_action = action
        return action


# ---------------------------------------------------------------------------
# File-protocol client


def read_state(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def read_observation(path: Path) -> np.ndarray | None:
    try:
        obs = np.load(path, allow_pickle=False)
    except (OSError, ValueError, EOFError):
        return None
    if obs.shape != (200, 320, 3) or obs.dtype != np.uint8:
        return None
    return obs


def write_action_atomic(path: Path, turn: int,
                        action: Sequence[int]) -> None:
    document = {
        "turn": int(turn),
        "accelerate": int(action[0]),
        "brake": int(action[1]),
        "left": int(action[2]),
        "right": int(action[3]),
    }
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(document, handle, separators=(",", ":"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


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
    policy = ExpertPolicy(
        args.player,
        initial_frame_skip=args.initial_frame_skip,
        verbose=args.verbose,
    )

    answered: int | None = None
    last_server_activity = time.monotonic()
    client_started_wall = time.time()
    saw_live_race = False
    announced = False

    while True:
        state = read_state(state_path)
        if state is None:
            if (args.idle_timeout > 0
                    and time.monotonic() - last_server_activity
                    > args.idle_timeout):
                print(f"{args.player}: no server state for "
                      f"{args.idle_timeout:g}s", file=sys.stderr)
                return 1
            time.sleep(args.poll_interval)
            continue

        status = state.get("status")
        turn = int(state.get("turn", -1))
        if status == "race_over":
            # Player folders intentionally retain the final state.  If this
            # process was launched before the next server, do not mistake an
            # old result for a race we actually joined.
            if not saw_live_race:
                try:
                    state_mtime = state_path.stat().st_mtime
                except OSError:
                    state_mtime = 0.0
                if state_mtime < client_started_wall:
                    if (args.idle_timeout > 0
                            and time.monotonic() - last_server_activity
                            > args.idle_timeout):
                        print(f"{args.player}: no new server race for "
                              f"{args.idle_timeout:g}s", file=sys.stderr)
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
        saw_live_race = True

        obs = read_observation(obs_path)
        if obs is None:
            time.sleep(args.poll_interval)
            continue

        if not announced:
            print(f"{args.player}: expert policy connected to {player_dir}",
                  flush=True)
            announced = True

        action = policy.decide(obs, state)
        write_action_atomic(action_path, turn, action)
        answered = turn
        last_server_activity = time.monotonic()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Expert client for play-racer-2p.py",
    )
    parser.add_argument(
        "player",
        choices=("red", "blue"),
        help="which private player folder to use",
    )
    parser.add_argument(
        "--game-dir",
        default=str(Path(__file__).resolve().parent),
        help="racer-2p directory (default: directory containing this script)",
    )
    parser.add_argument(
        "--initial-frame-skip",
        type=int,
        default=5,
        help="turn length guess until it can be inferred from state.tick",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.004,
        help="seconds between protocol polls (default: 0.004)",
    )
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=300.0,
        help="quit after this many seconds without a server; 0 waits forever",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="print every tactical decision",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return run_client(args)
    except KeyboardInterrupt:
        print(f"\n{args.player}: stopped", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
