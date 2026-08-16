#!/usr/bin/env python3
"""Lockstep race server: two agents race red vs blue over the play-racer stage.

Each agent lives entirely inside its own folder and never touches the other's.
The server drives a strict turn loop:

    1. writes the current frame and status into both player folders
    2. waits until *both* agents have written an action for that turn
    3. applies both actions to the same physics ticks, so neither agent gets
       to react to the other's move within a turn
    4. writes the resulting frame back, and goes again
    5. when the race is over it says so, and renders the split-screen mp4

Files, per player folder:

    state.json   server -> agent   turn number, status, place, telemetry
    obs.png      server -> agent   the frame that agent can see
    obs.npy      server -> agent   the same frame as a (200, 320, 3) array
    action.json  agent -> server   {"turn": N, "accelerate": 1, "brake": 0,
                                    "left": 0, "right": 1}

An action is only accepted if its ``turn`` matches the turn the server is
waiting on, so a stale file can never be replayed. Write it atomically
(temp file then rename) - the server retries on a partial read, but a rename
avoids the race entirely.

Run it:

    python racer-2p/play-racer-2p.py              # bundled bots
    python racer-2p/play-racer-2p.py --no-spawn   # your own agents
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')  # renders offscreen

import numpy as np  # noqa: E402
import pygame  # noqa: E402

from race_env import L, Race  # noqa: E402  (L is racer_env, via race_env)

HERE = Path(__file__).resolve().parent
FOLDERS = {'red': HERE / 'red-player', 'blue': HERE / 'blue-player'}
COAST = np.zeros(4, dtype=np.int8)


# ------------------------------------------------------------------ protocol

def write_atomic(path, data, binary=False):
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'wb' if binary else 'w') as fh:
        fh.write(data)
    os.replace(tmp, path)


def write_obs(folder, arr):
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    surf = pygame.image.frombuffer(arr.tobytes(), (w, h), 'RGB')
    tmp = folder / 'obs.png.tmp'
    pygame.image.save(surf, str(tmp))
    os.replace(tmp, folder / 'obs.png')

    tmp = folder / 'obs.npy.tmp'
    with open(tmp, 'wb') as fh:
        np.save(fh, arr)
    os.replace(tmp, folder / 'obs.npy')


def publish(race, turn, status, deadline_s=None, result=None):
    """Push the current frame and status into both player folders."""
    for tag, folder in FOLDERS.items():
        if status == 'awaiting_action':
            # drop any previous answer so it cannot be mistaken for this one
            (folder / 'action.json').unlink(missing_ok=True)
        write_obs(folder, race.obs[tag])
        state = {'turn': turn, 'tick': race.tick, 'status': status}
        state.update(race.status(tag))
        state['obs'] = {'png': 'obs.png', 'npy': 'obs.npy',
                        'shape': list(race.obs[tag].shape)}
        state['action_file'] = 'action.json'
        state['action_format'] = {'turn': turn, 'accelerate': '0|1',
                                  'brake': '0|1', 'left': '0|1',
                                  'right': '0|1'}
        if deadline_s is not None:
            state['deadline_s'] = deadline_s
        if result is not None:
            state['result'] = result
            state['you_placed'] = race.status(tag)['place']
        write_atomic(folder / 'state.json', json.dumps(state, indent=2))


def parse_action(doc):
    if 'action' in doc:
        raw = list(doc['action'])[:4]
    else:
        raw = [doc.get('accelerate', 0), doc.get('brake', 0),
               doc.get('left', 0), doc.get('right', 0)]
    raw += [0] * (4 - len(raw))
    return np.array([1 if int(v) else 0 for v in raw], dtype=np.int8)


def read_action(folder, turn):
    """Return the agent's action for this turn, or None if it isn't ready."""
    path = folder / 'action.json'
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return None  # missing, or caught mid-write - just try again
    try:
        if int(doc.get('turn', -1)) != turn:
            return None  # stale answer from an earlier turn
        return parse_action(doc)
    except (TypeError, ValueError):
        return None


def wait_for_actions(turn, timeout, previous):
    """Block until both agents answer, or the deadline passes."""
    deadline = time.monotonic() + timeout
    got, late = {}, []
    while len(got) < 2:
        for tag, folder in FOLDERS.items():
            if tag not in got:
                action = read_action(folder, turn)
                if action is not None:
                    got[tag] = action
        if len(got) == 2:
            break
        if time.monotonic() > deadline:
            for tag in FOLDERS:
                if tag not in got:
                    got[tag] = previous[tag]
                    late.append(tag)
            break
        time.sleep(0.004)
    return got, late


# --------------------------------------------------------------------- video

class Encoder:
    """Streams frames straight into ffmpeg so nothing piles up in memory."""

    def __init__(self, path, w, h, fps, scale):
        self.path = path
        cmd = ['ffmpeg', '-y', '-loglevel', 'error',
               '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', f'{w}x{h}', '-r', str(fps), '-i', '-', '-an',
               '-vf', f'scale={w * scale}:{h * scale}:flags=neighbor',
               '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20',
               str(path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.frames = 0

    def write(self, frame):
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        self.frames += 1

    def close(self):
        if self.proc.stdin and not self.proc.stdin.closed:
            self.proc.stdin.close()
        return self.proc.wait()


# ---------------------------------------------------------------------- main

def spawn_agents():
    procs = {}
    for tag, folder in FOLDERS.items():
        script = folder / 'agent.py'
        if not script.exists():
            print(f'  ! no agent.py in {folder.name}, nothing to spawn')
            continue
        procs[tag] = subprocess.Popen([sys.executable, str(script)],
                                      cwd=str(folder))
        print(f'  spawned {tag} agent (pid {procs[tag].pid})')
    return procs


def tidy(folder):
    for name in ('action.json', 'state.json', 'obs.png', 'obs.npy'):
        (folder / name).unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--seed', type=int, default=7,
                    help='traffic layout; the stage itself is always the same')
    ap.add_argument('--frame-skip', type=int, default=5,
                    help='physics ticks each action is held for (default 5, '
                         'ie 10 decisions a second). Raise it for slow agents')
    ap.add_argument('--turn-timeout', type=float, default=60.0,
                    help='seconds to wait for an agent before holding its '
                         'previous action')
    ap.add_argument('--max-turns', type=int, default=None,
                    help='safety cap on turns. Left alone it is derived from '
                         'the checkpoint clock, so the race always ends by '
                         'someone finishing or being disqualified rather '
                         'than being cut short with time still on the cars')
    ap.add_argument('--out', default=str(HERE / 'race.mp4'))
    ap.add_argument('--fps', type=int, default=50)
    ap.add_argument('--scale', type=int, default=2)
    ap.add_argument('--no-spawn', dest='spawn', action='store_false',
                    help='do not launch the bundled agent.py bots; wait for '
                         'your own agents to answer instead')
    args = ap.parse_args()

    for folder in FOLDERS.values():
        folder.mkdir(parents=True, exist_ok=True)
        tidy(folder)

    race = Race(seed=args.seed,
                max_ticks=None if args.max_turns is None
                else args.max_turns * args.frame_skip)
    max_turns = args.max_turns
    if max_turns is None:
        max_turns = race.max_ticks // args.frame_skip + 2
    w, h = race.split_size
    enc = Encoder(args.out, w, h, args.fps, args.scale)
    print(f'stage: {race.track_len:.0f} units, seed {args.seed}, '
          f'frame-skip {args.frame_skip}')
    print(f'safety cap: {max_turns} turns / {race.max_ticks} ticks '
          f'({race.max_ticks * L.DT:.0f}s of race clock)'
          + ('' if args.max_turns is None else '  [set by --max-turns]'))
    print(f'recording {w}x{h} -> {w * args.scale}x{h * args.scale} '
          f'into {args.out}')

    procs = spawn_agents() if args.spawn else {}
    if not procs:
        print('waiting for agents to write action.json in their folders...')

    previous = {'red': COAST.copy(), 'blue': COAST.copy()}
    turn = 0
    started = time.monotonic()
    try:
        while not race.over and turn < max_turns:
            publish(race, turn, 'awaiting_action', deadline_s=args.turn_timeout)
            actions, late = wait_for_actions(turn, args.turn_timeout, previous)
            for tag in late:
                print(f'  ! {tag} missed turn {turn}, held its last action')

            for _ in range(args.frame_skip):
                race.step(actions['red'], actions['blue'])
                enc.write(race.split_frame())
                if race.over:
                    break

            previous = actions
            turn += 1
            if turn % 25 == 0:
                r, b = race.status('red'), race.status('blue')
                print(f'  turn {turn:4d}  red {r["progress"] * 100:5.1f}% '
                      f'{r["speed_mph"]:5.1f}mph   '
                      f'blue {b["progress"] * 100:5.1f}% '
                      f'{b["speed_mph"]:5.1f}mph')

        result = race.result()
        publish(race, turn, 'race_over', result=result)
        for _ in range(int(args.fps * 1.5)):   # hold the last frame
            enc.write(race.split_frame())
    finally:
        enc.close()
        for tag, proc in procs.items():
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.terminate()

    took = time.monotonic() - started
    print()
    print(f'race over after {turn} turns / {race.tick} ticks '
          f'({took:.1f}s wall)')
    print(f'  reason : {result["reason"]}')
    print(f'  winner : {result["winner"]}')
    for tag in ('red', 'blue'):
        d = result[tag]
        print(f'  {tag:5s}: {ordinal(d["place"])}  {d["progress"] * 100:5.1f}% '
              f'crashes={d["crashes"]} finished={d["finished"]}')
    print(f'  video  : {args.out} ({enc.frames} frames)')


def ordinal(place):
    return {1: '1st', 2: '2nd'}.get(place, '-')


if __name__ == '__main__':
    main()
