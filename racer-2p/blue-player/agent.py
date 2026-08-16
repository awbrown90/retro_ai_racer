#!/usr/bin/env python3
"""Blue agent - measured: looks further ahead and lifts for traffic.

Self-contained on purpose: everything it needs lives in this folder, and it
only ever reads state.json / obs.npy here and writes action.json here.

Protocol:
    wait for state.json to show a new turn with status "awaiting_action"
    read obs.npy                (200, 320, 3) uint8, the frame it can see
    write action.json           {"turn": N, "accelerate": .., "brake": ..,
                                 "left": .., "right": ..}
    stop when status is "race_over"
"""
import json
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
IDLE_TIMEOUT = 180.0

# personality
LOOK_AHEAD = 0.58    # looks further up the road, so turns in earlier
DEADZONE = 4.0       # px of steering slack before correcting
EDGE = 0.62          # lane offset past which we haul it back regardless
BLOB = 110           # reacts to traffic sooner than red
CLEARANCE = 42.0     # gives a car more room than red does
LIFT = True          # blue backs off rather than barging through
LIFT_ABOVE = 95.0    # only ever lift above this speed


def road_pixels(obs, row_frac):
    """Column indices of tarmac on one scanline.

    Tarmac is the only near-grey thing in the palette: verges are green,
    rumble strips red/white, lane dashes near-white, sky blue.
    """
    h, w, _ = obs.shape
    row = obs[int(h * row_frac)].astype(int)
    r, g, b = row[:, 0], row[:, 1], row[:, 2]
    road = (abs(r - g) < 12) & (abs(g - b) < 14) & (r > 55) & (r < 115)
    return np.flatnonzero(road)


def car_ahead(obs):
    """Size and screen column of the nearest car sitting in our path.

    Cars are the only strongly saturated things on the road: tarmac and lane
    dashes are grey, and scenery is filtered out by the green test and by
    ignoring the outer quarter of the frame.
    """
    h, w, _ = obs.shape
    band = obs[int(h * 0.55):int(h * 0.74)].astype(int)
    hi, lo = band.max(axis=2), band.min(axis=2)
    green = ((band[:, :, 1] > band[:, :, 0] + 30)
             & (band[:, :, 1] > band[:, :, 2] + 30))
    car = ((hi - lo) > 60) & ~green
    left, right = int(w * 0.25), int(w * 0.75)
    cols = car[:, left:right].sum(axis=0)
    n = int(cols.sum())
    if n == 0:
        return 0, w / 2.0
    idx = np.flatnonzero(cols) + left
    return n, float(idx.mean())


def decide(obs, state):
    h, w, _ = obs.shape
    lane = state['lane_offset']
    accelerate, brake = 1, 0

    xs = road_pixels(obs, LOOK_AHEAD)
    if xs.size > 8:
        target = float(xs.mean())
        lo, hi = float(xs.min()), float(xs.max())
    else:
        # tarmac out of sight - almost certainly off it, aim back by telemetry
        target, lo, hi = w / 2 - lane * 90.0, 0.0, float(w)

    blob, blob_x = car_ahead(obs)
    if blob > BLOB:
        # aim past the car itself, down whichever side has more tarmac.
        # Aiming at "the roomier half of the road" instead steers straight
        # into it whenever the car happens to sit on that side.
        if (blob_x - lo) > (hi - blob_x):
            target = blob_x - CLEARANCE
        else:
            target = blob_x + CLEARANCE
        # only lift if we are actually carrying speed into it - lifting
        # unconditionally means a car ahead at the lights pins us at 0 mph
        # for the whole race, because we never get going in the first place
        if LIFT and state.get('speed_mph', 0.0) > LIFT_ABOVE:
            accelerate = 0

    err = target - w / 2
    left = 1 if err < -DEADZONE else 0
    right = 1 if err > DEADZONE else 0

    if lane > EDGE:                     # running wide, straighten it up
        left, right = 1, 0
    elif lane < -EDGE:
        left, right = 0, 1

    return [accelerate, brake, left, right]


def write_action(turn, action):
    doc = {'turn': turn, 'accelerate': action[0], 'brake': action[1],
           'left': action[2], 'right': action[3]}
    tmp = HERE / 'action.json.tmp'
    tmp.write_text(json.dumps(doc))
    os.replace(tmp, HERE / 'action.json')   # atomic, so no half-read


def read_state():
    try:
        return json.loads((HERE / 'state.json').read_text())
    except (OSError, ValueError):
        return None


def main():
    answered = -1
    last_seen = time.monotonic()
    while True:
        state = read_state()
        if state is None:
            if time.monotonic() - last_seen > IDLE_TIMEOUT:
                print('blue: no server, giving up')
                return
            time.sleep(0.005)
            continue

        if state['status'] == 'race_over':
            res = state.get('result', {})
            print(f'blue: race over - winner {res.get("winner")} '
                  f'({res.get("reason")}), I came '
                  f'{state.get("you_placed")}')
            return

        if state['turn'] == answered:
            time.sleep(0.004)
            continue

        try:
            obs = np.load(HERE / 'obs.npy')
        except (OSError, ValueError):
            time.sleep(0.004)
            continue

        write_action(state['turn'], decide(obs, state))
        answered = state['turn']
        last_seen = time.monotonic()


if __name__ == '__main__':
    main()
