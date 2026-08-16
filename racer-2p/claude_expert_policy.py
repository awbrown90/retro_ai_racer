#!/usr/bin/env python3
"""Expert client for play-racer-2p.py - drives red or blue to the finish.

    python claude_expert_policy.py red
    python claude_expert_policy.py --player blue

It speaks the folder protocol: waits for ``<tag>-player/state.json`` to show a
new turn, reads ``obs.npy``, and writes ``action.json`` back atomically.

How it drives
-------------
The car's lateral physics are simple but unforgiving:

    steer_per_tick = DT * 2.4 * (speed / MAX_SPEED)
    drift_per_tick = -steer_per_tick * (speed / MAX_SPEED) * curve * 0.32

Steering authority is *proportional to speed*. A turn here is a batch of
physics ticks, so with the 50-tick turns a slow agent gets, one steering input
at 186mph moves the car ~2.4 lane units - wider than the whole road, which is
+-1.0. At top speed there is no such thing as a small correction: you either
hold the wheel straight or you cross the entire road.

The flip side is that the curve's outward push scales with speed *squared*, so
push/steer is 0.32 * speed_pct * curve. Lifting genuinely buys back control:
at half speed a steering input is worth twice as much against the curve.

Rather than hand-tuned rules, this policy:

  1. reconstructs the road's curvature profile ahead from the frame, by
     inverting the renderer's own segment walk (fit_curve),
  2. re-measures the curve it just drove from telemetry alone (measure_curve),
  3. keeps other cars in world coordinates, including through the blind zone
     nearer than dz ~2100, where the road is too wide to measure and our own
     bodywork covers them,
  4. notices from physics when something it cannot see is holding it up - the
     engine failing to deliver the acceleration it owes us,
  5. runs a diverse beam search over action sequences through an exact copy of
     the server's physics, and plays the first action of the best branch.

Because the rollout *is* the real physics, "stay off the grass" and "do not
rear-end anyone" are not rules - they fall out of the speed those things cost.

Measured on the offline bench (seeds 3/7/11 at frame-skip 5 and 50) it
finishes the stage in all six runs. Head to head on the server it beats the
bundled agent.py comfortably, and loses to gpt_expert_policy.py, which covers
the stage in ~616 turns against this policy's ~870.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- game truth
# Mirrored from games/racer_env.py. These are the physics the server runs.
FPS = 50
DT = 1.0 / FPS
SEG_LEN = 200
ROAD_W = 2400
FOV = 100
CAM_HEIGHT = 1300
WIDTH, HEIGHT = 320, 200

MAX_SPEED = SEG_LEN / DT                 # 10000 world units / second
ACCEL = MAX_SPEED / 3.6
BRAKE = -MAX_SPEED / 1.6
DECEL = -MAX_SPEED / 7.0                 # coasting
OFF_DECEL = -MAX_SPEED / 1.4             # extra scrub on the grass
OFF_LIMIT = MAX_SPEED / 3.6              # grass speed cap (~51.7 mph)
CENTRIFUGAL = 0.32
TOP_MPH = 186

TRAFFIC_W = 1050
PLAYER_CAR_PX = 76.0
HIT_AHEAD = 280
HIT_BEHIND = 80
HIT_STANDOFF = 240
CLOSING_HIT = 1.15

CHECKPOINT_BONUS = 22.0
CHECKPOINT_EVERY = 550                   # segments

CAM_DEPTH = 1.0 / math.tan((FOV / 2.0) * math.pi / 180.0)
PLAYER_Z = CAM_HEIGHT * CAM_DEPTH        # the nose sits this far ahead of cam
_SCALE_P = CAM_DEPTH / PLAYER_Z
PLAYER_HALF_W = (PLAYER_CAR_PX / 2) / (_SCALE_P * ROAD_W * WIDTH / 2)
TRAFFIC_HALF_W = (TRAFFIC_W / 2) / ROAD_W
HIT_REACH = PLAYER_HALF_W + TRAFFIC_HALF_W        # ~0.347 lane units
RIVAL_REACH = PLAYER_HALF_W * 2                   # racer-to-racer contact

# depth of a scanline, from how wide the road is on it: dz = KZ / half_px
KZ = CAM_DEPTH * ROAD_W * WIDTH / 2

# ------------------------------------------------------------------ tuning
ROWS = tuple(range(100, 180, 2))         # scanlines sampled for road geometry
MIN_HALF = 22.0                          # px; narrower rows are too noisy
CURVE_ERR = 0.35          # how wrong the curve fit can plausibly be
CENTRE_PULL = 7000.0     # mild preference for the middle of the road
BEAM = 27
PER_FIRST = 3            # keep this many branches per opening move
HORIZON_S = 1.5          # seconds of road to think over, whatever the
                         # server's frame-skip makes a turn worth
HORIZON_MIN, HORIZON_MAX = 3, 24
NEAR_MISS = 0.18         # lane units of clearance we would like when passing
SPEED_VALUE = 2.2                        # seconds of future progress a unit
                                         # of speed is worth at the horizon


# ============================================================== perception

def road_rows(obs):
    """Road centre and half-width on each sampled scanline.

    The tarmac is the only near-grey thing in the palette. A car sitting on
    the road splits the grey run in two; taking the widest run then reports
    half a road and drags the apparent centre sideways, which reads as a bend
    that is not there. So runs are merged back across anything car-sized.
    """
    a = obs.astype(np.int16)
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    grey = (np.abs(r - g) < 16) & (np.abs(g - b) < 20) & (r > 42) & (r < 140)

    out = {}
    for row in ROWS:
        xs = np.flatnonzero(grey[row])
        if xs.size < 6:
            continue
        runs = [q for q in np.split(xs, np.flatnonzero(np.diff(xs) > 7) + 1)
                if q.size >= 3]
        if not runs:
            continue
        main = max(runs, key=len)
        lo, hi = int(main[0]), int(main[-1])
        for q in runs:                                   # merge across cars
            a0, b0 = int(q[0]), int(q[-1])
            if a0 < lo and lo - b0 < 110:
                lo = a0
            if b0 > hi and a0 - hi < 110:
                hi = b0
        half = (hi - lo) / 2.0
        if half < 4:
            continue
        out[row] = (lo, hi, (lo + hi) / 2.0, half)
    return out


def usable_rows(rows):
    """Rows whose geometry can be trusted.

    Perspective forces the road to widen as it comes toward the car, so a row
    narrower than one nearer to us has had an edge eaten by traffic parked on
    the kerb - its centre is fiction. Clipped rows lie about the centre too.
    """
    # a lower row index is further away, so half-width has to grow as we walk
    # down the frame; a row narrower than one behind it has lost an edge
    good, prev_half = {}, 0.0
    for row in sorted(rows):                             # far -> near
        lo, hi, ctr, half = rows[row]
        if half < prev_half - 4:                         # impossibly narrow
            continue
        prev_half = max(prev_half, half)
        if half >= MIN_HALF and lo > 1 and hi < WIDTH - 2:
            good[row] = (ctr, half)
    return good


def fit_curve(good, lane):
    """Recover the curvature profile ahead from the shape of the road.

    The renderer walks segments accumulating ``x += dx; dx += curve``, so the
    road's lateral offset k segments ahead is the double integral of curve:

        X(k) / ROAD_W  =  norm(k) + player_x        (call this e)

    For curve(s) = c0 + c1*s that is  e = c0*k^2/2 + c1*k^3/6, linear in the
    unknowns, so a least-squares fit over the visible rows recovers both the
    curve we are entering and how it is developing.

    Returns (c0, c1) in the same units as the segment curves the server uses.
    """
    ks, es = [], []
    for row, (ctr, half) in good.items():
        k = (KZ / half) / SEG_LEN                        # segments ahead
        if not (3.0 <= k <= 90.0):
            continue
        ks.append(k)
        es.append((ctr - WIDTH / 2) / half + lane)
    if len(ks) < 4:
        return 0.0, 0.0
    k = np.asarray(ks)
    e = np.asarray(es) * ROAD_W                          # into world units
    A = np.stack([k ** 2 / 2.0, k ** 3 / 6.0], axis=1)
    try:
        sol, *_ = np.linalg.lstsq(A, e, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, 0.0
    c0 = float(np.clip(sol[0], -12.0, 12.0))
    c1 = float(np.clip(sol[1], -1.2, 1.2))
    return c0, c1


def detect_cars(obs, rows, good):
    """Other cars on the tarmac, as (lane_lo, lane_hi, dz).

    Traffic is the only strongly saturated thing on the road; the kerbs are
    excluded by trimming the road span, and our own bodywork by ignoring the
    bottom-centre block where it is drawn.
    """
    a = obs.astype(np.int16)
    mx, mn = a.max(axis=2), a.min(axis=2)
    green = (a[:, :, 1] > a[:, :, 0] + 25) & (a[:, :, 1] > a[:, :, 2] + 25)
    lit = ((mx - mn) > 55) & ~green                      # coloured cars
    pale = (mn > 150)                                    # white/silver cars
    cand = lit | pale

    mask = np.zeros(cand.shape, bool)
    for row, (lo, hi, ctr, half) in rows.items():
        if row < 104:
            continue
        pad = max(4, int(half * 0.13))                   # keep off the kerbs
        mask[row, lo + pad:hi - pad + 1] = True
    mask = _smear(mask, 7)
    hit = cand & mask
    hit[142:, 104:216] = False                           # our own car

    out = []
    for y0, y1, x0, x1, n in _blobs(hit):
        if n < 22 or (y1 - y0) < 2:
            continue
        near = min(good, key=lambda r: abs(r - y1)) if good else None
        if near is None:
            continue
        ctr, half = good[near]
        if half < 8:
            continue
        lo, hi = (x0 - ctr) / half, (x1 - ctr) / half
        # A car is a fixed size in the world, so however far away it is it
        # covers about the same slice of road: traffic 0.44 lane units, the
        # rival 0.26. Anything much thinner is a lane dash, and anything
        # sitting off the tarmac is kerb paint, not a car.
        span = hi - lo
        if not (0.17 <= span <= 1.00):
            continue
        if min(abs(lo), abs(hi)) > 1.02:
            continue
        out.append((lo, hi, KZ / half))

    out.sort(key=lambda c: c[2])
    keep = []                                            # drop duplicates
    for lo, hi, dz in out:
        if any(abs(dz - d2) < 900 and lo < h2 and l2 < hi
               for l2, h2, d2 in keep):
            continue
        keep.append((lo, hi, dz))
    return keep


def _smear(mask, k):
    """Cheap vertical dilation so sparse scanlines cover the gaps between."""
    out = mask.copy()
    for d in range(1, k + 1):
        out[d:] |= mask[:-d]
        out[:-d] |= mask[d:]
    return out


def _blobs(mask):
    """Connected components, labelled with a small union-find over runs."""
    h, w = mask.shape
    parent = {}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    runs, prev = [], []
    for y in range(h):
        xs = np.flatnonzero(mask[y])
        cur = []
        if xs.size:
            for q in np.split(xs, np.flatnonzero(np.diff(xs) > 1) + 1):
                idx = len(runs)
                parent[idx] = idx
                runs.append([y, y, int(q[0]), int(q[-1]), int(q.size)])
                for pidx in prev:                        # touching above?
                    if runs[pidx][2] <= q[-1] and q[0] <= runs[pidx][3]:
                        union(idx, pidx)
                cur.append(idx)
        prev = cur

    merged = {}
    for i, (y0, y1, x0, x1, n) in enumerate(runs):
        root = find(i)
        if root in merged:
            m = merged[root]
            m[0], m[1] = min(m[0], y0), max(m[1], y1)
            m[2], m[3] = min(m[2], x0), max(m[3], x1)
            m[4] += n
        else:
            merged[root] = [y0, y1, x0, x1, n]
    return [tuple(v) for v in merged.values()]


# =============================================================== car tracking

class Traffic:
    """Keeps other cars in world coordinates so contact can be predicted.

    A car close enough to actually hit is large, low in the frame and partly
    behind our own bodywork - exactly when vision is worst. Tracking it in
    world space means it stays predictable through the moment it matters.
    """

    def __init__(self):
        self.tracks = []        # dicts: z, x_lo, x_hi, speed, z0, t0, seen
        self.clock = 0.0

    def update(self, cars, position, dt):
        self.clock += dt
        for t in self.tracks:
            t['z'] += t['speed'] * dt
        for lo, hi, dz in cars:
            z = position + dz
            best, best_d = None, 1e9
            for t in self.tracks:
                d = abs(t['z'] - z)
                lat = abs((t['x_lo'] + t['x_hi']) / 2 - (lo + hi) / 2)
                if d < 2600 and lat < 0.55 and d < best_d:
                    best, best_d = t, d
            if best is None:
                self.tracks.append({'z': z, 'x_lo': lo, 'x_hi': hi,
                                    'speed': 0.38 * MAX_SPEED,
                                    'z0': z, 't0': self.clock,
                                    'seen': self.clock})
            else:
                # Depth comes from how wide the road is on the blob's row, so
                # it is quantised hard - one pixel is hundreds of units out at
                # range. Differencing that turn to turn gives nonsense speeds,
                # so measure over the longest baseline the track has.
                base = self.clock - best['t0']
                if base > 0.35:
                    v = (z - best['z0']) / base
                    best['speed'] = float(np.clip(v, 0.20 * MAX_SPEED,
                                                  0.60 * MAX_SPEED))
                    if base > 1.5:                   # keep the window moving
                        best['z0'], best['t0'] = z, self.clock
                best.update(z=z, x_lo=lo, x_hi=hi, seen=self.clock)
        # A car we have lost for a moment is usually a car in the blind zone:
        # closer than dz ~2100, where the road is too wide to measure and our
        # own bodywork covers it. Hold onto those until they are demonstrably
        # behind us. Anything further out gets dropped quickly instead, so a
        # noisy speed estimate cannot invent an obstacle out in the distance.
        kept = []
        for t in self.tracks:
            lost = self.clock - t['seen']
            blind = position - 900 < t['z'] < position + 3400
            if lost < (1.1 if blind else 0.4) and t['z'] > position - 1500:
                kept.append(t)
        self.tracks = kept
        self.tracks.sort(key=lambda t: t['z'])
        del self.tracks[12:]

    def snapshot(self):
        return [(t['z'], (t['x_lo'] + t['x_hi']) / 2.0, t['speed'])
                for t in self.tracks]


# ================================================================== rollout

ACTIONS = [(a, b, l, r)
           for (a, b) in ((1, 0), (0, 0), (0, 1))       # power / coast / brake
           for (l, r) in ((0, 0), (1, 0), (0, 1))]      # straight / left / right


def curve_at(seg_ahead, c0, c1, horizon_segs):
    """Curvature this far ahead, decayed to straight past what we can see."""
    if seg_ahead > horizon_segs:
        fade = max(0.0, 1.0 - (seg_ahead - horizon_segs) / 60.0)
        return (c0 + c1 * horizon_segs) * fade
    return c0 + c1 * max(0.0, seg_ahead)


def roll(state, action, ticks, c0, c1, horizon_segs, ref_seg, cars, rival,
         next_cp_seg):
    """Advance one turn through the server's own physics.

    Returns the new state plus what it cost us, so the planner can price a
    trip onto the grass or into someone's boot in the only currency that
    matters: how far up the road we end up.

    ``ref_seg`` is the segment the camera was at when the frame was taken -
    the curve profile is measured from there, so every branch of the search
    has to index into it from the same origin.
    """
    pos, spd, x, tleft, touching = state
    acc, brk, left, right = action
    crashed = 0
    off_ticks = 0
    graze = 0
    cars = [[z, cx, cs] for z, cx, cs in cars]
    rival = list(rival) if rival is not None else None

    for _ in range(ticks):
        sp = spd / MAX_SPEED
        steer = DT * 2.4 * sp
        if left:
            x -= steer
        if right:
            x += steer
        curve = curve_at(pos / SEG_LEN - ref_seg, c0, c1, horizon_segs)
        x -= steer * sp * curve * CENTRIFUGAL

        if acc:
            spd += ACCEL * DT
        elif brk:
            spd += BRAKE * DT
        else:
            spd += DECEL * DT

        if abs(x) > 1.0 and spd > OFF_LIMIT:
            spd += OFF_DECEL * DT
            off_ticks += 1
        spd = min(max(spd, 0.0), MAX_SPEED)
        x = min(max(x, -2.4), 2.4)

        nose = pos + PLAYER_Z
        now_touching = False
        for car in cars:
            gap = car[0] - nose
            if -HIT_BEHIND < gap < HIT_AHEAD + 260:
                slack = abs(x - car[1]) - HIT_REACH
                if 0.0 <= slack < NEAR_MISS:
                    graze += 1
            if -HIT_BEHIND < gap < HIT_AHEAD and abs(x - car[1]) < HIT_REACH:
                now_touching = True
                pos = min(pos, max(0.0, car[0] - HIT_STANDOFF - PLAYER_Z))
                if not touching and spd > car[2] * CLOSING_HIT:
                    spd = car[2] * 0.85          # ran into the back of them
                    crashed += 1
                else:
                    spd = min(spd, car[2])       # just held to their pace
            car[0] += car[2] * DT
        touching = now_touching

        if rival is not None:
            gap = rival[0] - pos
            if 0 <= gap < HIT_AHEAD and abs(x - rival[1]) < RIVAL_REACH:
                pos = min(pos, max(0.0, rival[0] - HIT_STANDOFF))
                spd = min(spd, rival[2])
            rival[0] += rival[2] * DT

        pos += spd * DT
        tleft -= DT

    # checkpoints top the clock back up; missing one is a disqualification
    while next_cp_seg is not None and pos / SEG_LEN >= next_cp_seg:
        tleft += CHECKPOINT_BONUS
        next_cp_seg += CHECKPOINT_EVERY

    return ((pos, spd, x, tleft, touching),
            [tuple(c) for c in cars],
            tuple(rival) if rival is not None else None,
            next_cp_seg, crashed, off_ticks, graze)


def safe_limit(ticks, speed):
    """How close to the edge we can sit and still survive a mis-read bend.

    A curve we underestimated by ``CURVE_ERR`` drags us sideways by
    ``steer * speed_pct * curve * 0.32`` every tick. Reserve two turns of
    that, because that is how long it takes to notice and answer it. The
    faster we are going the more room we have to leave - which is exactly
    the discipline that is hard to hold by hand.
    """
    sp = speed / MAX_SPEED
    slip = 2.0 * ticks * (DT * 2.4 * sp) * sp * CURVE_ERR * CENTRIFUGAL
    return float(min(0.92, max(0.30, 1.0 - slip)))


def plan(state, ticks, c0, c1, horizon_segs, ref_seg, cars, rival, track_len,
         next_cp_seg, depth):
    """Beam search over action sequences; return the best first action.

    Scoring is in world units so everything is priced in the same currency:
    how far up the road a branch leaves us, plus the speed it carries out of
    the horizon, minus what the grass and other people's bodywork cost.
    """
    beam = [(0.0, state, cars, rival, next_cp_seg, None, 0, 0)]
    for _ in range(depth):
        nxt = []
        for _, st, cs, rv, cp, first, ncrash, noff in beam:
            if st[0] >= track_len:                       # already home
                nxt.append((1e12, st, cs, rv, cp, first, ncrash, noff))
                continue
            for act in ACTIONS:
                st2, cs2, rv2, cp2, crashed, off, graze = roll(
                    st, act, ticks, c0, c1, horizon_segs, ref_seg, cs, rv, cp)
                pos, spd, x, tleft, _ = st2
                tc, to = ncrash + crashed, noff + off
                if tleft <= 0.0 and pos < track_len:
                    val = -1e9                           # out of time: dead
                else:
                    val = pos + spd * SPEED_VALUE
                    lim = safe_limit(ticks, spd)
                    val -= 90000.0 * max(0.0, abs(x) - lim) ** 2
                    val -= CENTRE_PULL * max(0.0, abs(x) - 0.30) ** 2
                    val -= 9000.0 * tc
                    val -= 45.0 * to
                    val -= 12.0 * graze
                    if pos >= track_len:
                        val += 5e5                       # crossing the line
                nxt.append((val, st2, cs2, rv2, cp2,
                            first if first is not None else act, tc, to))
        # Diverse beam. While a slower car pins us the server clamps us to its
        # bumper, so every branch shows the *same* position and the ones that
        # steer aside look strictly worse - they pay to leave the middle and
        # have nothing to show for it yet. A plain beam prunes them long
        # before the overtake pays off, and the car queues behind traffic for
        # the rest of the race. Reserving room for every opening move keeps
        # the escape alive until it can prove itself.
        nxt.sort(key=lambda t: -t[0])
        picked, per = [], {}
        for item in nxt:
            key = item[5]
            if per.get(key, 0) < PER_FIRST:
                per[key] = per.get(key, 0) + 1
                picked.append(item)
            if len(picked) >= BEAM:
                break
        if len(picked) < BEAM:
            seen_ids = {id(i) for i in picked}
            picked += [i for i in nxt if id(i) not in seen_ids][:BEAM - len(picked)]
        beam = picked
        if not beam:
            break
    return beam[0][5] if beam and beam[0][5] is not None else (1, 0, 0, 0)


# =================================================================== driver

class Driver:
    def __init__(self, folder, verbose=True):
        self.folder = Path(folder)
        self.verbose = verbose
        self.traffic = Traffic()
        self.prev = None            # (tick, position, lane, speed, steer_cmd)
        self.ticks_per_turn = 50
        self.track_len = None
        self.measured_curve = 0.0
        self.blind_x = None          # last seen lane of a car in the blind zone
        self.last_act = (1, 0, 0, 0)

    # ---------------------------------------------------------- measurement

    def measure_curve(self, st):
        """Invert the lateral physics to read the curve we just drove.

        No vision involved: given how far we slid sideways, what we asked the
        wheel for, and how fast we were going, only one curvature explains it.
        """
        if self.prev is None:
            return None
        ptick, ppos, plane, pspd, psteer = self.prev
        n = st['tick'] - ptick
        if n <= 0 or pspd <= 1.0:
            return None
        sp = pspd / MAX_SPEED
        dx = st['lane_offset'] - plane
        steer_tick = DT * 2.4 * sp
        expect = n * steer_tick * psteer
        denom = n * steer_tick * sp * CENTRIFUGAL
        if abs(denom) < 1e-9:
            return None
        return float(np.clip((expect - dx) / denom, -12.0, 12.0))

    def free_speed(self, prev_speed, act, ticks, off_road):
        """What the engine alone would have done to our speed."""
        spd = prev_speed
        for _ in range(ticks):
            if act[0]:
                spd += ACCEL * DT
            elif act[1]:
                spd += BRAKE * DT
            else:
                spd += DECEL * DT
            if off_road and spd > OFF_LIMIT:
                spd += OFF_DECEL * DT
            spd = min(max(spd, 0.0), MAX_SPEED)
        return spd

    # ---------------------------------------------------------------- think

    def decide(self, st, obs):
        lane = st['lane_offset']
        speed = st['speed_mph'] / TOP_MPH * MAX_SPEED
        pos = st['distance']
        if self.track_len is None and st['progress'] > 1e-6:
            self.track_len = pos / st['progress']
        track_len = self.track_len or 1e9

        rows = road_rows(obs)
        good = usable_rows(rows)
        c0, c1 = fit_curve(good, lane)

        # the frame shows the road ahead; telemetry shows the bit just driven.
        # Trust the measurement for where we are, vision for what is coming.
        meas = self.measure_curve(st)
        if meas is not None:
            self.measured_curve = meas
            c0 = 0.55 * meas + 0.45 * c0

        horizon_segs = 0.0
        if good:
            horizon_segs = max((KZ / h) / SEG_LEN for _, h in good.values())

        # the rival first: telemetry gives its position and speed exactly
        opp = st.get('opponent') or {}
        rival = None
        if 'lead' in opp:
            rz = pos + float(opp['lead'])
            rspd = float(opp.get('speed_mph', 0.0)) / TOP_MPH * MAX_SPEED
            if rz > pos - 400:                    # only matters when ahead
                rival = (rz, float(opp.get('lane_offset', 0.0)), rspd)

        # The rival is painted on the road like anything else, so vision finds
        # it too - and then guesses its speed badly, turning a car that is
        # pulling away into an imaginary slow obstacle to be dodged. We know
        # exactly where it is from telemetry, so drop the duplicate sighting.
        seen = detect_cars(obs, rows, good)
        if rival is not None:
            seen = [c for c in seen
                    if abs((pos + c[2]) - rival[0]) > 1600
                    or abs((c[0] + c[1]) / 2 - rival[1]) > 0.45]

        dt_turn = self.ticks_per_turn * DT
        self.traffic.update(seen, pos, dt_turn)
        cars = self.traffic.snapshot()

        # A car close enough to actually block us sits at dz ~1300, which is
        # nearer than the lowest scanline the road is still measurable on - so
        # the one car that matters most is the one we cannot see. Physics
        # gives it away instead: full throttle that does not produce the
        # acceleration the engine owes us means somebody is in the way.
        for t in self.traffic.tracks:
            if t['z'] - pos < 3000:
                self.blind_x = (t['x_lo'] + t['x_hi']) / 2.0
        if cars and min(c[0] for c in cars) - pos > 6000:
            self.blind_x = None
        blocked = False
        if self.prev is not None:
            n = st['tick'] - self.prev[0]
            want = self.free_speed(self.prev[3], self.last_act, max(n, 1),
                                   abs(self.prev[2]) > 1.0)
            if speed < want - 0.02 * MAX_SPEED and abs(lane) <= 1.0:
                blocked = True
        if blocked and rival is not None:
            gap = rival[0] - pos
            if 0 <= gap < HIT_AHEAD + 400 and abs(lane - rival[1]) < 0.45:
                blocked = False          # the rival explains it, exactly
        if blocked:
            bx = self.blind_x if self.blind_x is not None else lane
            cars = [c for c in cars if c[0] - pos > 2600]
            cars.append((pos + PLAYER_Z + HIT_STANDOFF, bx, speed))

        seg_now = pos / SEG_LEN
        next_cp = (int(seg_now // CHECKPOINT_EVERY) + 1) * CHECKPOINT_EVERY

        # only cars we could actually reach inside the horizon are worth
        # carrying through every branch of the search
        # a lane change takes 0.7 / (ticks*DT*2.4*speed_pct) turns, and if the
        # horizon is shorter than that the planner cannot see the far side of
        # the manoeuvre and will never start it
        sp = max(speed / MAX_SPEED, 0.05)
        lane_turns = 0.75 / max(self.ticks_per_turn * DT * 2.4 * sp, 1e-4)
        depth = max(HORIZON_S / max(dt_turn, 1e-3), lane_turns + 3)
        depth = int(max(HORIZON_MIN, min(HORIZON_MAX, round(depth))))
        span = speed * HORIZON_S + 4 * PLAYER_Z
        cars = [c for c in cars if pos - 1500 < c[0] < pos + span]

        act = plan((pos, speed, lane, st['time_left'], False),
                   self.ticks_per_turn, c0, c1, horizon_segs, seg_now,
                   cars, rival, track_len, next_cp, depth)

        if self.verbose:
            names = {(0, 0): '   ', (1, 0): '  <', (0, 1): '>  '}
            pedal = 'ACC' if act[0] else ('BRK' if act[1] else 'cst')
            print(f"t{st['turn']:4d} {st['speed_mph']:5.1f}mph x={lane:+.2f} "
                  f"prog={st['progress']*100:5.1f}% c0={c0:+.2f} c1={c1:+.3f} "
                  f"cars={len(cars)}{'B' if blocked else ' '} "
                  f"-> {pedal}{names[(act[2], act[3])]}",
                  flush=True)

        self.prev = (st['tick'], pos, lane, speed, act[3] - act[2])
        self.last_act = act
        return act

    # ----------------------------------------------------------- the loop

    def write_action(self, turn, act):
        doc = {'turn': turn, 'accelerate': int(act[0]), 'brake': int(act[1]),
               'left': int(act[2]), 'right': int(act[3])}
        tmp = self.folder / 'action.json.tmp'
        tmp.write_text(json.dumps(doc))
        os.replace(tmp, self.folder / 'action.json')

    def read_state(self):
        try:
            return json.loads((self.folder / 'state.json').read_text())
        except (OSError, ValueError):
            return None

    def run(self, idle_timeout=600.0):
        answered = -1
        last_seen = time.monotonic()
        while True:
            st = self.read_state()
            if st is None:
                if time.monotonic() - last_seen > idle_timeout:
                    print('no server, giving up', file=sys.stderr)
                    return 1
                time.sleep(0.005)
                continue
            last_seen = time.monotonic()

            if st.get('status') == 'race_over':
                res = st.get('result', {})
                print(f"race over: winner={res.get('winner')} "
                      f"({res.get('reason')}), I placed {st.get('you_placed')} "
                      f"at {st.get('progress', 0) * 100:.1f}%")
                return 0

            turn = st.get('turn', -1)
            if turn == answered or st.get('status') != 'awaiting_action':
                time.sleep(0.004)
                continue

            try:
                with open(self.folder / 'obs.npy', 'rb') as fh:
                    obs = np.load(fh)
            except (OSError, ValueError):
                time.sleep(0.004)
                continue

            if self.prev is not None:
                n = st['tick'] - self.prev[0]
                if n > 0:
                    self.ticks_per_turn = n

            try:
                act = self.decide(st, obs)
            except Exception as exc:                     # never miss a turn
                print(f'  ! decide failed on turn {turn}: {exc!r}',
                      file=sys.stderr)
                act = (1, 0, 0, 0)
                self.prev = (st['tick'], st['distance'], st['lane_offset'],
                             st['speed_mph'] / TOP_MPH * MAX_SPEED, 0)

            self.write_action(turn, act)
            answered = turn


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('player', nargs='?', choices=('red', 'blue'),
                    help='which car to drive')
    ap.add_argument('--player', dest='player_flag', choices=('red', 'blue'),
                    help='which car to drive')
    ap.add_argument('--dir', default=None,
                    help='player folder (defaults to <player>-player next to '
                         'this script)')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    tag = args.player_flag or args.player
    if tag is None and args.dir is None:
        ap.error('say which car to drive: red or blue')

    folder = Path(args.dir) if args.dir else \
        Path(__file__).resolve().parent / f'{tag}-player'
    if not folder.is_dir():
        ap.error(f'no such folder: {folder}')

    print(f'claude expert policy driving {tag or folder.name} from {folder}')
    return Driver(folder, verbose=not args.quiet).run()


if __name__ == '__main__':
    sys.exit(main())
