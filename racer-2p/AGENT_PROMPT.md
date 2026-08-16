# Agent prompt

Replace `{{COLOR}}` with `red` or `blue` and give one copy to each LLM.
Everything else is identical between the two.

---

You are the **{{COLOR}}** driver in a two-car race against another agent.

Your entire world is the folder `<repo>/racer-2p/{{COLOR}}-player/`.
Work only in there. Do not read or write the other player's folder, and do not
modify the server (`play-racer-2p.py`) or the shared track code (`race_env.py`,
`../racer_env.py`). You win by driving well, not by changing the game.

A server is already running and is waiting for you. Your goal is to beat the
other car.

## The loop

Repeat until the race is over:

1. Read `state.json` in your folder.
2. If `status` is `"race_over"` — stop. Report `result.winner` and your `place`.
3. If `turn` is the number you already answered, wait a moment and read again.
4. Read `obs.npy` — a `(200, 320, 3)` uint8 RGB array, the view from your car.
   (`obs.png` is the same picture if you would rather look at it.)
5. Decide, then write `action.json` in your folder:

   ```json
   {"turn": 42, "accelerate": 1, "brake": 0, "left": 0, "right": 1}
   ```

   Write it atomically — write a temp file, then rename it over `action.json`.
6. Go back to step 1.

Rules:

- The `turn` in your action **must match** the `turn` in `state.json`, or the
  server ignores it as stale.
- Each control is `0` or `1`, and they combine — accelerate + left is normal.
- Take too long and the server holds your previous action and moves on, so
  answer promptly rather than deliberating forever.

## What you are told — `state.json`

| field | meaning |
|---|---|
| `place` | 1 or 2, right now |
| `speed_mph`, `gear`, `time_left` | your telemetry |
| `progress` | 0.0 → 1.0 along the stage |
| `lane_offset` | where you sit across the road: `0.0` centre, `±1.0` the edge of the tarmac, beyond that you are on grass |
| `crashes`, `off_road` | how it is going |
| `opponent` | their `place`, `progress`, `speed_mph` and `lead` — world units they are ahead, negative means you lead |

## What you see — `obs.npy`

Behind-the-car view: grey tarmac with white lane dashes and red/white kerbs,
green verges, hills on the horizon, your car at the bottom centre, and a place
badge (`{{COLOR}} 1st` / `{{COLOR}} 2nd`) top-left. Traffic to overtake is
blocky and painted yellow, white, purple, orange or cyan. Your rival is the
**only other red-or-blue car** on the road, and it is the only one drawn in
the same sleek shape as your own car — you can see it alongside you, not just
when it is ahead.

## How to win

- The stage is a fixed course; `progress` 1.0 is the finish. Crossing the line
  wins outright. If the turn limit is reached first, **whoever got furthest
  wins** — so never stop pushing.
- **Watch `time_left`. If your clock reaches zero you are disqualified on the
  spot and your rival wins the race**, however far ahead you were. Checkpoints
  are the only thing that puts time back on it, so keep reaching them.
- The clock in the game is game time, not real time. Thinking for a while does
  not cost you race time. But don't stall past the server's deadline.
- **Staying on the tarmac matters more than anything.** On the grass you are
  capped near 56 mph against a 186 mph top speed.
- Curves push you toward the outside. Turn in early rather than correcting late.
- Checkpoints bank extra time. Run the clock to zero and you are out.
- Rear-ending traffic costs a lot of speed. Pick a side early and commit,
  rather than sitting on a slower car's bumper at their pace.
- Contact with your rival only holds up whichever car is **behind**.

Decide every turn yourself from the frame and the telemetry — do not write an
autonomous bot to play the race for you.

Stop when `status` is `"race_over"`.
