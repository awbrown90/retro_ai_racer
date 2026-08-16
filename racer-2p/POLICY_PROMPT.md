# Policy prompt

The brief you hand to a **coding agent** (Claude Code, Codex, Grok Build, …) to
have it author a driving policy that races head-to-head.

This is the *offline* mode: the agent writes a program once, and that program
drives every turn at full speed. For the *online* mode — an LLM reading the
frame and deciding each turn itself — use [`AGENT_PROMPT.md`](AGENT_PROMPT.md)
instead.

Replace `{{NAME}}` with a short handle for the model (`grok`, `claude`, `gpt`)
and `{{REPO}}` with the path to your clone. Give one copy to each agent, in a
separate session, and **do not let them see each other's code** — the point is
that two independently written policies meet for the first time on the track.

---

You are writing a driving policy for a two-car racing game, and it will be
raced head-to-head against a policy written by a different model.

## What you are producing

A single standalone Python file, `{{REPO}}/racer-2p/{{NAME}}_policy.py`, run as:

```bash
python {{NAME}}_policy.py red      # or: blue
```

It must depend only on the standard library and `numpy`. It must not import
the game — a policy that imports `racer_env` is reading the answer key rather
than driving. Mirroring the published physics constants into your own file is
fine and expected; that is how you can plan ahead.

## The protocol you speak

Your process owns exactly one folder: `racer-2p/red-player/` or
`racer-2p/blue-player/`. Read and write only in there. Never touch the other
player's folder, the server (`play-racer-2p.py`), or the track code
(`race_env.py`, `../racer_env.py`).

Loop until the race ends:

1. Read `state.json` in your folder.
2. `status == "race_over"` → print `result.winner` and your place, and exit.
3. `turn` equals the turn you already answered → sleep briefly, read again.
4. Read `obs.npy` — a `(200, 320, 3)` uint8 RGB frame, the view from your car.
5. Decide, then write `action.json` **atomically** (temp file, then
   `os.replace`):

   ```json
   {"turn": 42, "accelerate": 1, "brake": 0, "left": 0, "right": 1}
   ```

The `turn` in your action must match the `turn` in `state.json` or the server
discards it as stale. Miss the deadline and your previous action is held.

The full field-by-field spec is in [`PROTOCOL.md`](PROTOCOL.md) — read it.

## What you are told each turn

`state.json` carries your telemetry (`speed_mph`, `gear`, `time_left`,
`progress`, `lane_offset`, `crashes`, `off_road`) and your opponent's
(`progress`, `speed_mph`, `lane_offset`, `lead` — negative means you lead).

`obs.npy` is the rendered frame: grey tarmac, white lane dashes, red/white
kerbs, green verges, your car at the bottom centre, HUD strip along the
bottom, place badge top-left. Traffic to overtake is yellow, white, purple,
orange or cyan. Your rival is the only other red-or-blue car on the road.

## The physics you are driving against

Published in `../racer_env.py` — read it, and mirror what you need:

- Top speed is 186 mph. **On the grass you are capped near 56 mph.** Staying
  on the tarmac matters more than anything else you can do.
- Steering authority is proportional to speed, and the curve's outward push
  scales with speed *squared*. At top speed a single input can cross the whole
  road; lifting genuinely buys back control.
- `lane_offset` is `0.0` at the centre line, `±1.0` at the edge of the tarmac.
- Rear-ending traffic costs a lot of speed. Pick a side early and commit.
- Contact between the two racers only holds up the car **behind**.
- One turn is `--frame-skip` physics ticks, so one decision is held for all of
  them. Read `state.tick` to infer the real skip rather than assuming it.

## How the race is won

- `progress` 1.0 is the finish. Crossing the line wins outright.
- **If your clock reaches zero you are disqualified on the spot and your rival
  wins**, however far ahead you were. Checkpoints are the only thing that puts
  time back on the clock, so keep reaching them.
- If the turn limit is hit first, whoever got furthest wins. Never stop
  pushing.

## Suggestions, not requirements

Nothing above dictates *how* you drive — that is the part you are being asked
to invent. Approaches that have worked: reconstructing the road's curvature
ahead by inverting the renderer's segment walk; tracking other cars in world
coordinates through the blind zone your own bodywork covers; running a beam
search over action sequences through a re-implementation of the server's
physics and playing the first action of the best branch.

Build yourself an offline bench so you can iterate without a live opponent —
drive the single-car env (`play-racer.py` / `Racer-v0`) headless across a few
seeds and both frame-skips, and only then take it to the server.
