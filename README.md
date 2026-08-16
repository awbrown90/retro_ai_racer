# retro_ai_racer

A from-scratch pseudo-3D arcade racer — original code and art, no ROM, no
ripped assets — wrapped as a Gymnasium environment, plus a lockstep server
that lets **two AI agents race each other over the same stage and the same
traffic**.

Play it yourself with the arrow keys, or hand the brief to two coding agents,
have each write a driving policy, and watch them meet for the first time on
the track.

---

## The races

Two policies, written independently by three different models, split-screen —
red on the left, blue on the right. Each pair drove the same stage with the
same traffic seed, one physics tick at a time, neither able to react to the
other within a turn.

### GPT‑5.6 (red) vs Grok 4.6 (blue)

![GPT-5.6 vs Grok 4.6](media/race_grok_vs_gpt.gif)

A straight fight to the flag. GPT‑5.6 crosses the line at 186 mph with Grok
4.6 still on the road at 98% — a couple of seconds in it. Full-quality
recording: [`media/race_grok_vs_gpt.mp4`](media/race_grok_vs_gpt.mp4)

### Opus 5 (red) vs Grok 4.6 (blue)

![Opus 5 vs Grok 4.6](media/race_grok_vs_claude.gif)

Not close. Grok 4.6 finishes with 80 seconds still banked on its clock and 2
crashes; Opus 5 gets tangled in traffic — 15 crashes, dropped to 2nd gear, and
stranded at 51% of the stage when the race ends. Full-quality recording:
[`media/race_grok_vs_claude.mp4`](media/race_grok_vs_claude.mp4)

*(The GIFs are played at 2× speed and downsampled; the mp4s are the real
50 fps recordings the server wrote.)*

### Who wrote what

Each policy in `racer-2p/` was written by a coding agent given nothing but the
protocol spec and the physics — no example policy, and no sight of the other
models' code.

| policy | model | reasoning | harness |
|---|---|---|---|
| `grok_expert_policy.py` | Grok 4.6 | xhigh | Grok Build |
| `claude_expert_policy.py` | Opus 5 | max | Claude Code |
| `gpt_expert_policy.py` | ChatGPT 5.6 Sol | max | Codex |

They are committed here as-written, and they are what you race against if you
want to try beating them.

---

## Install

Python 3.10+, and `ffmpeg` on your `PATH` if you want the server to record.

```bash
git clone https://github.com/awbrown90/retro_ai_racer.git
cd retro_ai_racer
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # gymnasium, numpy, pygame
```

Everything below is run from the repo root with that venv active.

---

## 1. Play it yourself

```bash
python play-racer.py
```

| key | |
|---|---|
| <kbd>↑</kbd> / <kbd>Z</kbd> | accelerate |
| <kbd>↓</kbd> / <kbd>X</kbd> | brake |
| <kbd>←</kbd> <kbd>→</kbd> / <kbd>A</kbd> <kbd>D</kbd> | steer |
| <kbd>R</kbd> | restart the stage |
| <kbd>Esc</kbd> | quit |

Reach each checkpoint before the clock runs out — every one banks more time.
Leaving the tarmac scrubs speed hard (you are capped near 56 mph on the grass,
against a 186 mph top speed), and rear-ending traffic costs more still, so the
racing line through the bends is what wins you the stage.

### As a Gymnasium environment

```python
import gymnasium, numpy as np
import racer_env                       # registers Racer-v0

env = gymnasium.make('Racer-v0')
obs, info = env.reset(seed=7)          # obs: (200, 320, 3) uint8 RGB
obs, reward, terminated, truncated, info = env.step(
    np.array([1, 0, 0, 0], dtype=np.int8))   # accel, brake, left, right
```

The stage is hand-authored and fixed — the seed only shuffles traffic, so the
road is the same road every run. `info` carries `speed_mph`, `gear`,
`time_left`, `progress`.

---

## 2. Watch the bundled bots race

```bash
python racer-2p/play-racer-2p.py
```

That spawns the two simple bots in `racer-2p/red-player/agent.py` and
`racer-2p/blue-player/agent.py`, races them, prints the result, and writes a
split-screen `racer-2p/race.mp4`. They exist to prove the protocol works and
to give you something to beat.

---

## 3. Race the shipped model policies against each other

Start the server with `--no-spawn` so it waits for your own agents, then start
one policy in each colour — three terminals, or background the server:

```bash
# terminal 1 — the server
python racer-2p/play-racer-2p.py --no-spawn --out racer-2p/race.mp4

# terminal 2 — red
cd racer-2p && python gpt_expert_policy.py red

# terminal 3 — blue
cd racer-2p && python grok_expert_policy.py blue
```

Any pairing works — swap in `claude_expert_policy.py`, or run the same policy
in both colours as a control. The server prints a running scoreboard and, at
the end, the winner and the path to the recording.

Useful server flags:

| flag | default | what it does |
|---|---|---|
| `--seed` | 7 | traffic layout; the stage itself never changes |
| `--frame-skip` | 5 | physics ticks each action is held for. 5 ≈ 10 decisions/sec |
| `--turn-timeout` | 60 | seconds to wait before holding an agent's previous action |
| `--out` | `race.mp4` | recording path |
| `--no-spawn` | off | don't launch the bundled bots; wait for your own agents |
| `--max-turns` | derived | safety stop. Left alone it is computed from the checkpoint clock, so a race always ends by someone finishing or being disqualified |

---

## 4. Have AI agents write their own policies and race them

This is the part that produced the videos above. Two coding agents, working
independently, each write a policy; the two policies then race.

**Step 1 — brief each agent separately.** Open a session per model (Claude
Code, Codex, Grok Build, …), each in its own clone or worktree so they cannot
see each other's work in progress. Hand each one
[`racer-2p/POLICY_PROMPT.md`](racer-2p/POLICY_PROMPT.md), substituting:

- `{{NAME}}` → a short handle for that model (`grok`, `claude`, `gpt`)
- `{{REPO}}` → the path to that agent's clone

The prompt tells it what to build (`racer-2p/{{NAME}}_policy.py`), the file
protocol it must speak, the physics it is driving against, and the rules it
wins by. It deliberately does **not** tell it how to drive — that is the part
being tested. It points the agent at
[`racer-2p/PROTOCOL.md`](racer-2p/PROTOCOL.md) for the field-by-field spec and
at `racer_env.py` for the physics.

**Step 2 — let each agent iterate offline.** A policy is a normal Python
program, so an agent can bench it against the single-car env across several
seeds and both frame-skips long before it meets an opponent. Encourage this —
every shipped policy here was tuned that way first.

**Step 3 — collect the policies into one clone.** Copy each agent's
`{{NAME}}_policy.py` into `racer-2p/`. They are standalone files; nothing else
needs to move.

**Step 4 — race them.**

```bash
python racer-2p/play-racer-2p.py --no-spawn --out racer-2p/race_a_vs_b.mp4 &
sleep 2
(cd racer-2p && python a_policy.py red)  &
(cd racer-2p && python b_policy.py blue) &
wait
```

**Step 5 — read the result.** The server prints the winner and the reason
(`finished`, `disqualified`, `tick_limit`), and both players' final
`state.json` carry the same `result` block. The split-screen mp4 is written to
`--out`.

### The other mode: an LLM driving live, turn by turn

Instead of writing a program, an LLM can read the frame and decide each turn
itself. Give it [`racer-2p/AGENT_PROMPT.md`](racer-2p/AGENT_PROMPT.md) and
slow the race down so it has room to think — one decision per second of race
time, five minutes to make it:

```bash
python racer-2p/play-racer-2p.py --no-spawn --frame-skip 50 --turn-timeout 300
```

A full race at that pace is about 100 turns. Don't add `--max-turns`; the
default already stops the race at the point both clocks must have expired.

---

## How the race is kept fair

The server is a strict lockstep turn loop:

1. It writes `obs.png`, `obs.npy` and `state.json` into **both** player
   folders, and deletes any stale `action.json`.
2. Both agents write `action.json` for that turn.
3. It applies **both** actions to the same physics ticks — neither agent can
   react to the other within a turn.
4. It writes the new frame back, and goes again.

An action is only accepted if its `turn` matches the turn the server is
waiting on, so a stale file can never be replayed. Each agent is confined to
its own folder and never sees the other's files. Both cars run the same stage
and the same traffic, advanced once per tick — what red overtakes is the same
car blue has to overtake.

Each racer is drawn in its own livery, and traffic is repainted
yellow/white/purple/orange/cyan, so the only red or blue car on the road is
the one you are racing.

Full spec: [`racer-2p/PROTOCOL.md`](racer-2p/PROTOCOL.md).

---

## Layout

```
retro_ai_racer/
├── racer_env.py                  the game: physics, renderer, Racer-v0
├── play-racer.py                 play it yourself with the keyboard
├── requirements.txt
├── racer-2p/
│   ├── play-racer-2p.py          the lockstep race server
│   ├── race_env.py               two cars on one shared track
│   ├── PROTOCOL.md               field-by-field spec of the file protocol
│   ├── POLICY_PROMPT.md          brief for a coding agent writing a policy
│   ├── AGENT_PROMPT.md           brief for an LLM driving live, turn by turn
│   ├── grok_expert_policy.py     Grok 4.6 xhigh, via Grok Build
│   ├── claude_expert_policy.py   Opus 5 max, via Claude Code
│   ├── gpt_expert_policy.py      ChatGPT 5.6 Sol max, via Codex
│   ├── red-player/agent.py       red's world — simple bundled bot
│   └── blue-player/agent.py      blue's world — simple bundled bot
└── media/                        the recorded races
```

`racer_env.py` is the single-car game. `racer-2p/race_env.py` is the two-car
wrapper around it — similar names, different jobs.
