# racer-2p — head-to-head racing for two agents

Red vs blue over the same stage and the same traffic as `../play-racer.py`.
`play-racer-2p.py` is a lockstep server: two agents, each confined to its own
folder, take turns writing actions and never see each other's files.

```
racer-2p/
├── play-racer-2p.py    the server
├── race_env.py         two cars on one shared track
├── red-player/         red's world: agent.py + its own obs/state/action files
├── blue-player/        blue's world
└── race.mp4            split-screen recording, written at the end
```

## Run it

```bash
cd retro_ai_racer
python racer-2p/play-racer-2p.py                 # bundled bots
python racer-2p/play-racer-2p.py --no-spawn      # your own agents
```

With `--no-spawn` the server sets up turn 0 and waits; start your agents in
the two folders whenever you like.

For LLM agents, slow the decision rate down and give them room to think —
one decision per second of race time, and five minutes to make it:

```bash
python racer-2p/play-racer-2p.py --no-spawn \
    --frame-skip 50 --turn-timeout 300
```

A full race at that pace is about 100 turns. Don't add `--max-turns`; the
default already stops the race at the point both clocks must have expired.

Useful flags:

| flag | default | what it does |
|---|---|---|
| `--frame-skip` | 5 | physics ticks each action is held for. 5 ≈ 10 decisions/sec. **Raise this a lot for LLM agents** — 25 gives 2/sec, 50 gives 1/sec |
| `--turn-timeout` | 60 | seconds to wait before holding an agent's previous action |
| `--seed` | 7 | traffic layout. The stage itself never changes |
| `--out` | `race.mp4` | recording path |
| `--max-turns` | derived | safety stop. Left alone it is computed from the checkpoint clock (146 s, just past the 142 s any car could possibly survive), so a race always ends by someone finishing or being disqualified. **Setting it lower cuts races short with time still on both cars** |

## The turn loop

1. Server writes `obs.png`, `obs.npy` and `state.json` into both folders, and
   deletes any stale `action.json`.
2. Both agents write `action.json` for that turn.
3. Server applies **both** actions to the same physics ticks — neither agent
   can react to the other within a turn.
4. Server writes the new frame back, and goes again.
5. When `state.json` says `"status": "race_over"`, it's done.

## Files

### `state.json` — server to agent

```json
{
  "turn": 42, "tick": 210, "status": "awaiting_action",
  "you": "red", "place": 1,
  "speed_mph": 163.2, "gear": 5, "time_left": 24.5,
  "progress": 0.31, "distance": 178560.0,
  "lane_offset": -0.12, "crashes": 2, "off_road": false,
  "finished": false, "retired": false,
  "opponent": { "place": 2, "progress": 0.27, "speed_mph": 121.0,
                "lane_offset": 0.31, "lead": -23040.0 },
  "obs": { "png": "obs.png", "npy": "obs.npy", "shape": [200, 320, 3] },
  "action_file": "action.json",
  "deadline_s": 60.0
}
```

`lane_offset` is where you sit across the road: `0.0` is the centre line,
`±1.0` is the edge of the tarmac, beyond that you're on the grass and losing
speed. `opponent.lead` is how far ahead they are in world units — negative
means you're winning.

### `action.json` — agent to server

```json
{"turn": 42, "accelerate": 1, "brake": 0, "left": 0, "right": 1}
```

`{"turn": 42, "action": [1, 0, 0, 1]}` works too.

Rules:

- **`turn` must match** the turn the server is waiting on. A stale file is
  ignored, so an old action can never be replayed.
- **Write atomically** — temp file then `os.replace`. The server retries on a
  partial read, but a rename avoids the race entirely.
- Miss the deadline and your previous action is held, and the server says so.

## The observation

`obs.npy` is a `(200, 320, 3)` uint8 RGB frame — exactly what a human player
would see, including the HUD strip and a **place badge in the top-left**
showing `RED 1st` or `BLUE 2nd`. `obs.png` is the same image if you'd rather
look at it.

Your rival is drawn on the road in its own livery, and it's the *only* red or
blue car out there — traffic is repainted yellow/white/purple/orange/cyan so
you can never mistake a car you need to overtake for the car you need to beat.

## Racing rules

- Both cars start side by side on the line, red slightly left, blue slightly
  right — close enough that each is in the other's field of view.
- Same stage, same traffic, advanced once per tick so you both have to get
  past the same cars.
- Checkpoints bank more time. **Run your clock to zero and you are
  disqualified on the spot, and your rival wins the race.** The `TIME`
  readout on the HUD is the thing to watch.
- Contact between the two racers only holds up the car **behind** — blocking
  both equally just pins them together on the start line.
- The race ends when someone crosses the line, someone's clock expires, or
  the turn limit hits. Crossing the line beats everything; on a turn limit
  the winner is whoever got furthest.

Your rival is drawn as a full car in its livery, projected like anything else
on the road — it slides up from the bottom of the frame as it comes past you
and shrinks away as it pulls clear, rather than appearing only once it is in
front.

## Writing your own agent

Replace `red-player/agent.py` (or just run something else in that folder).
The loop is:

```python
state = json.loads((HERE / 'state.json').read_text())
if state['status'] == 'race_over':
    return
if state['turn'] != already_answered:
    obs = np.load(HERE / 'obs.npy')
    action = decide(obs, state)
    write_atomically(HERE / 'action.json', {'turn': state['turn'], ...})
```

The bundled bots read the frame rather than the telemetry: they pick out
tarmac by colour on a scanline to find the road, and look for saturated
blobs to spot cars. Red runs a later, tighter line; blue looks further ahead
and lifts for traffic. Both are deliberately simple — they exist to prove the
protocol works and give you something to beat.
