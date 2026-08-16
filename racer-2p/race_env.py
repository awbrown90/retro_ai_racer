"""Two-car race on the play-racer stage: red vs blue, head to head.

Both cars run the same hand-authored stage and the same dynamic traffic from
``games/racer_env.py`` - one shared set of cars, advanced exactly once per
tick, so what red overtakes is the same car blue has to overtake.

Each racer sees the other as a car on the road, in their livery, and can be
blocked or shunted by them under the same rules as ordinary traffic.
"""
import pathlib
import sys

import numpy as np
import pygame

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import racer_env as L  # noqa: E402

RED = (214, 52, 44)
BLUE = (56, 96, 220)

# Traffic is repainted off the red/blue end of the spectrum so the only red
# car on the road is red's rival and the only blue car is blue's. The
# single-player palette has a red and a blue in it, which reads as "that's
# my opponent" when it is really just a car to overtake.
RACE_TRAFFIC = [(228, 192, 56), (216, 216, 220), (150, 66, 172),
                (232, 132, 40), (86, 198, 196)]
START_X = 0.20        # how far either side of the centre line they line up.
                      # Wider than ~0.21 and each car is outside the other's
                      # field of view on the line, so they start blind to
                      # each other; this keeps both fully in frame.
RIVAL_MIN_DZ = 320    # nearer than this the rival is past you and off-screen
DIVIDER = 8           # px between the two halves of the split screen
ORDINAL = {1: '1st', 2: '2nd'}


class Racer(L.RacerEnv):
    """One car in the race.

    ``_advance_traffic`` is a no-op here: the Race owns the shared traffic and
    rolls it forward once per tick, otherwise two racers would advance the
    same cars twice and the field would run at double speed.
    """

    tag = '?'
    color = (128, 128, 128)
    place = 1
    rival = None      # {'z', 'x', 'color'} - the other racer, set each tick

    def __init__(self, tag, color):
        super().__init__()
        self.tag = tag
        self.color = color
        self.body_color = color   # you see your own car in your own livery
        self.rival = None
        self.place = 1
        self.done = False
        self.done_tick = None

    def _advance_traffic(self):
        pass

    def _project_rival(self):
        """Where the rival sits on screen, at any distance.

        Placing it in ``traffic`` only worked once it was far enough ahead to
        clear the near edge of the drawn road, so it popped into existence the
        moment it overtook you. Projecting it directly means it slides up from
        the bottom of the screen the way anything else on the road does.

        Mirrors the curve walk in ``RacerEnv._frame`` so it sits on the road
        rather than beside it through a bend.
        """
        if not self.rival:
            return None
        rz = self.rival['z']
        dz = rz - self.position
        if dz < RIVAL_MIN_DZ:
            return None                      # fully past you, off the bottom

        n = self.n_segs
        base_i = int(self.position / L.SEG_LEN) % n
        base_pct = (self.position % L.SEG_LEN) / L.SEG_LEN
        target = int(rz / L.SEG_LEN) % n
        steps = (target - base_i) % n
        if steps > L.DRAW_DIST:
            return None                      # beyond the draw distance

        x = 0.0
        dx = -(self.segments[base_i]['curve'] * base_pct)
        for k in range(steps):
            x += dx
            dx += self.segments[(base_i + k) % n]['curve']

        frac = (rz % L.SEG_LEN) / L.SEG_LEN
        seg = self.segments[target]
        road_y = seg['y1'] + (seg['y2'] - seg['y1']) * frac

        p_i = int((self.position + self.player_z) / L.SEG_LEN) % n
        p_seg = self.segments[p_i]
        p_pct = ((self.position + self.player_z) % L.SEG_LEN) / L.SEG_LEN
        player_y = p_seg['y1'] + (p_seg['y2'] - p_seg['y1']) * p_pct
        cam_y = player_y + L.CAM_HEIGHT

        sx, sy, sw, _ = self._project(x + dx * frac
                                      - self.player_x * L.ROAD_W,
                                      road_y, rz, 0.0, cam_y, self.position)
        # the same physical car as yours, so alongside it matches your sprite
        w = L.PLAYER_CAR_PX * self.player_z / dz
        if w < 1.2 or sy > L.HEIGHT + 160:
            return None
        return sx + sw * self.rival['x'], sy - 26.0 * w / L.PLAYER_CAR_PX, w

    def _draw_player(self, s):
        # rival first, so your own car always paints over it when you touch
        spot = self._project_rival()
        if spot is not None:
            cx, y, w = spot
            self._draw_car(s, cx, y, w, self.rival['color'])
        super()._draw_player(s)

    def _draw_hud(self, s):
        super()._draw_hud(s)
        label = f'{self.tag.upper()} {ORDINAL.get(self.place, "-")}'
        t = self.big_font.render(label, True, (255, 255, 255))
        box = pygame.Rect(3, 3, t.get_width() + 8, t.get_height() + 3)
        pygame.draw.rect(s, (0, 0, 0), box.inflate(2, 2))
        pygame.draw.rect(s, self.color, box)
        s.blit(t, (box.x + 4, box.y + 1))


class Race:
    """Lockstep two-car race. Both actions are applied to the same tick."""

    def __init__(self, seed=0, max_ticks=None):
        self.red = Racer('red', RED)
        self.blue = Racer('blue', BLUE)
        self.seed = seed
        self.track_len = self.red.track_len
        # A racer can never hold more than the start clock plus every
        # checkpoint bonus, so once that has elapsed both clocks are certainly
        # zero. Defaulting the cap to that means a race is always decided on
        # the road - somebody finishes, or somebody is disqualified - and
        # never cut short with time still showing on both cars.
        if max_ticks is None:
            longest = (L.START_TIME
                       + len(self.red.checkpoints) * L.CHECKPOINT_BONUS)
            max_ticks = int(longest / L.DT) + 200
        self.max_ticks = max_ticks
        self.reset()

    # ------------------------------------------------------------------ setup

    def reset(self):
        self.red.reset(seed=self.seed)
        self.blue.reset(seed=self.seed)
        # identical seeds give identical fields; share one list so they stay
        # in lockstep for the rest of the race
        self.traffic = self.red.traffic
        self.blue.traffic = self.traffic
        for i, car in enumerate(self.traffic):
            car['color'] = RACE_TRAFFIC[i % len(RACE_TRAFFIC)]

        # line up side by side on the start line
        self.red.player_x = -START_X
        self.blue.player_x = START_X
        self.red.done = self.blue.done = False
        self.red.done_tick = self.blue.done_tick = None

        self.tick = 0
        self.over = False
        self.winner = None
        self.reason = None
        self._update_places()
        self._publish_views()
        self.obs = {'red': self.red.render(), 'blue': self.blue.render()}
        return self.obs

    def _opponent_car(self, other):
        """Where the other racer's car actually is, for drawing it."""
        return {'z': other.position + other.player_z, 'x': other.player_x,
                'speed': other.speed, 'color': other.color}

    def _resolve_contact(self):
        """Wheel-to-wheel contact between the two racers.

        Only the trailing car is held up. Blocking both symmetrically makes
        them shove each other backwards in equal measure, which pins them to
        the start line the moment they touch - they never get going at all.
        """
        if self.red.position >= self.blue.position:
            lead, trail = self.red, self.blue
        else:
            lead, trail = self.blue, self.red
        if lead.done or trail.done:
            return
        gap = lead.position - trail.position
        reach = trail.player_half_w * 2
        side_by_side = abs(lead.player_x - trail.player_x) < reach
        if 0 <= gap < L.HIT_AHEAD and side_by_side:
            trail.position = min(trail.position,
                                 max(0.0, lead.position - L.HIT_STANDOFF))
            trail.speed = min(trail.speed, lead.speed)

    def _publish_views(self):
        """Shared field for collisions, rival position for drawing.

        The rival is kept out of ``traffic`` entirely: it is drawn by
        ``_project_rival`` as a proper car, and racer-to-racer contact is
        settled by ``_resolve_contact``, which knows who is in front.
        """
        self.red.traffic = self.traffic
        self.blue.traffic = self.traffic
        self.red.rival = self._opponent_car(self.blue)
        self.blue.rival = self._opponent_car(self.red)

    def _update_places(self):
        if self.red.position >= self.blue.position:
            self.red.place, self.blue.place = 1, 2
        else:
            self.red.place, self.blue.place = 2, 1

    # ------------------------------------------------------------------- step

    def step(self, red_action, blue_action):
        """Advance one physics tick with both actions applied together."""
        if self.over:
            return self.obs

        # one shared field, rolled forward exactly once
        for car in self.traffic:
            car['z'] = (car['z'] + car['speed'] * L.DT) % self.track_len

        # both rivals are sampled before either moves, so neither gets to
        # react to the other's move within the same tick
        self._publish_views()

        for racer, action in ((self.red, red_action),
                              (self.blue, blue_action)):
            if racer.done:
                self.obs[racer.tag] = racer.render()
                continue
            obs, _, terminated, _, _ = racer.step(action)
            self.obs[racer.tag] = obs
            if terminated:
                racer.done = True
                racer.done_tick = self.tick

        self._resolve_contact()
        self.tick += 1
        self._update_places()
        self._check_over()
        return self.obs

    def _check_over(self):
        if self.red.finished or self.blue.finished:
            self.over, self.reason = True, 'finished'
        elif self.red.done or self.blue.done:
            # a checkpoint clock reaching zero is a disqualification, and it
            # ends the race then and there - the surviving car takes it
            self.over, self.reason = True, 'disqualified'
        elif self.tick >= self.max_ticks:
            self.over, self.reason = True, 'tick_limit'
        if self.over and self.winner is None:
            self.winner = self._decide_winner()

    def _dq(self, racer):
        """Out of time, rather than across the line."""
        return racer.done and not racer.finished

    def _decide_winner(self):
        if self.red.finished and not self.blue.finished:
            return 'red'
        if self.blue.finished and not self.red.finished:
            return 'blue'
        if self._dq(self.red) and not self._dq(self.blue):
            return 'blue'
        if self._dq(self.blue) and not self._dq(self.red):
            return 'red'
        if self.red.position > self.blue.position:
            return 'red'
        if self.blue.position > self.red.position:
            return 'blue'
        return 'draw'

    # ------------------------------------------------------------------ views

    def status(self, tag):
        """Everything a racer is told about itself and its rival."""
        me = self.red if tag == 'red' else self.blue
        them = self.blue if tag == 'red' else self.red
        info = me._info()
        return {
            'you': tag,
            'place': me.place,
            'speed_mph': round(info['speed_mph'], 1),
            'gear': info['gear'],
            'time_left': round(info['time_left'], 2),
            'progress': round(info['progress'], 5),
            'distance': round(info['distance'], 1),
            'lane_offset': round(me.player_x, 3),
            'crashes': info['crashes'],
            'off_road': abs(me.player_x) > 1.0,
            'finished': me.finished,
            'disqualified': self._dq(me),
            'opponent': {
                'place': them.place,
                'progress': round(them._info()['progress'], 5),
                'speed_mph': round(them._info()['speed_mph'], 1),
                'lane_offset': round(them.player_x, 3),
                'lead': round(them.position - me.position, 1),
            },
        }

    def split_frame(self):
        """Red on the left, blue on the right, for the recording."""
        r, b = self.obs['red'], self.obs['blue']
        h, w = r.shape[:2]
        out = np.empty((h, w * 2 + DIVIDER, 3), np.uint8)
        out[:, :w] = r
        out[:, w:w + DIVIDER] = (28, 30, 36)
        out[:, w + DIVIDER:] = b
        return out

    @property
    def split_size(self):
        h, w = self.obs['red'].shape[:2]
        return w * 2 + DIVIDER, h

    def result(self):
        return {
            'winner': self.winner,
            'reason': self.reason,
            'ticks': self.tick,
            'red': {'progress': round(self.red._info()['progress'], 5),
                    'finished': self.red.finished,
                    'disqualified': self._dq(self.red),
                    'time_left': round(self.red.time_left, 2),
                    'crashes': self.red.crashes,
                    'place': self.red.place},
            'blue': {'progress': round(self.blue._info()['progress'], 5),
                     'finished': self.blue.finished,
                     'disqualified': self._dq(self.blue),
                     'time_left': round(self.blue.time_left, 2),
                     'crashes': self.blue.crashes,
                     'place': self.blue.place},
        }
