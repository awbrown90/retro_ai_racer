"""A from-scratch pseudo-3D racer in the style of the 16-bit arcade classics.

Original code and art - no ROM, no ripped assets. What it reproduces is the
*feel* of the era's road racers: a behind-the-car road-strip renderer,
centrifugal push through the curves, scenery streaming past the verge, traffic
to overtake, and a checkpoint clock that keeps the run alive.

Exposed as a Gymnasium environment, ``Racer-v0``:

    action       MultiBinary(4) - accelerate, brake, steer left, steer right
    observation  Box(0, 255, (200, 320, 3), uint8) - the rendered frame
    reward       distance covered, plus checkpoint bonuses, minus crashes
    terminated   clock hits zero, or the stage is finished

The stage is hand-authored and fixed, the way a 1991 stage would be - the
seed only shuffles traffic, so the road is the same road every run.
"""
import math

import gymnasium
import numpy as np
import pygame
from gymnasium import spaces

WIDTH, HEIGHT = 320, 200
FPS = 50
DT = 1.0 / FPS

SEG_LEN = 200            # world units per road segment
ROAD_W = 2400            # half-width of the tarmac
RUMBLE_FRAC = 0.14       # rumble strip, as a fraction of road half-width
LANE_FRAC = 0.02
DRAW_DIST = 240          # segments drawn ahead of the camera
CAM_HEIGHT = 1300
FOV = 100
RUMBLE_SEGS = 4          # segments per light/dark stripe

MAX_SPEED = SEG_LEN / DT              # never skip a segment in one tick
ACCEL = MAX_SPEED / 3.6
BRAKE = -MAX_SPEED / 1.6
DECEL = -MAX_SPEED / 7.0
OFF_DECEL = -MAX_SPEED / 1.4
OFF_LIMIT = MAX_SPEED / 3.6
CENTRIFUGAL = 0.32
TOP_MPH = 186                         # what MAX_SPEED reads as on the HUD

# Car dimensions, shared by the renderer and the collision test so the hit
# box can never drift away from what you actually see on screen.
TRAFFIC_W = 1050                      # world width of a traffic car
TRAFFIC_H = 700                       # world height of a traffic car
PLAYER_CAR_PX = 76.0                  # drawn width of the player's car
HIT_AHEAD = 280                       # reach in front of the player's nose
HIT_BEHIND = 80                       # tolerance for overshooting in one tick
HIT_STANDOFF = 240                    # where a blocked car sits; must stay
                                      # inside HIT_AHEAD or contact flickers
                                      # off and the shunt re-fires every tick
CLOSING_HIT = 1.15                    # speed ratio that counts as a shunt
                                      # rather than just catching them up

START_TIME = 32.0
CHECKPOINT_BONUS = 22.0
CHECKPOINT_EVERY = 550                # segments

# Stage 1 palette: bright, hazy, green-verged - the clear-weather opener
SKY_TOP = (26, 62, 158)
SKY_BOT = (146, 194, 240)
HILL_FAR = (74, 122, 96)
HILL_NEAR = (52, 100, 72)
GRASS = ((22, 104, 40), (30, 124, 48))
ROAD = ((70, 70, 78), (80, 80, 88))
RUMBLE = ((188, 40, 40), (222, 222, 228))
LANE = (226, 226, 232)
FOG = (150, 190, 226)

CAR_BODY = (16, 92, 48)      # racing green
CAR_TRIM = (232, 206, 64)
CAR_GLASS = (30, 44, 40)
CAR_LIGHT = (226, 58, 44)
TRAFFIC = [(196, 48, 44), (44, 66, 188), (224, 190, 56),
           (216, 216, 220), (150, 66, 172)]


def _shade(color, f):
    """A darker (or lighter) cast of a livery, for wings and shadows."""
    return tuple(max(0, min(255, int(c * f))) for c in color)


def _ease_in(a, b, p):
    return a + (b - a) * p * p


def _ease_out(a, b, p):
    return a + (b - a) * (1.0 - (1.0 - p) ** 2)


def _ease_in_out(a, b, p):
    return a + (b - a) * (-math.cos(p * math.pi) / 2.0 + 0.5)


class RacerEnv(gymnasium.Env):
    """Behind-the-car arcade racer, 16-bit style."""

    metadata = {'render_modes': ['rgb_array'], 'render_fps': FPS}

    body_color = CAR_BODY   # your own car's livery; subclasses can repaint it

    def __init__(self, render_mode='rgb_array'):
        self.render_mode = render_mode
        self.action_space = spaces.MultiBinary(4)
        self.observation_space = spaces.Box(
            0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)

        self.cam_depth = 1.0 / math.tan((FOV / 2.0) * math.pi / 180.0)
        self.player_z = CAM_HEIGHT * self.cam_depth

        # Half-widths in player_x units (1.0 == road half-width), derived from
        # the drawn sizes: traffic is sized in world units, but the player's
        # car is drawn at a fixed pixel width, so convert it back through the
        # projection scale at the depth the car sits at.
        scale_p = self.cam_depth / self.player_z
        self.player_half_w = (PLAYER_CAR_PX / 2) / (
            scale_p * ROAD_W * WIDTH / 2)
        self.traffic_half_w = (TRAFFIC_W / 2) / ROAD_W

        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()
        self.surf = pygame.Surface((WIDTH, HEIGHT))
        self.font = pygame.font.Font(None, 15)
        self.big_font = pygame.font.Font(None, 24)

        self._build_track()
        self.np_random = np.random.default_rng(0)
        self._reset_state()

    # ---------------------------------------------------------------- track

    def _seg(self, curve, y):
        i = len(self.segments)
        self.segments.append({
            'index': i, 'curve': curve,
            'y1': self.segments[-1]['y2'] if self.segments else 0.0, 'y2': y,
            'sprites': [], 'cars': [],
        })

    def _road(self, enter, hold, leave, curve=0.0, height=0.0):
        start_y = self.segments[-1]['y2'] if self.segments else 0.0
        end_y = start_y + height
        total = enter + hold + leave
        for i in range(enter):
            self._seg(_ease_in(0.0, curve, i / enter),
                      _ease_in_out(start_y, end_y, i / total))
        for i in range(hold):
            self._seg(curve,
                      _ease_in_out(start_y, end_y, (enter + i) / total))
        for i in range(leave):
            self._seg(_ease_in_out(curve, 0.0, i / leave),
                      _ease_in_out(start_y, end_y, (enter + hold + i) / total))

    def _build_track(self):
        """Hand-authored stage: fast opener, sweepers, crests, tight esses."""
        self.segments = []
        self._road(60, 120, 60)                              # start straight
        self._road(50, 90, 50, curve=2.6)                    # long right
        self._road(40, 70, 40, height=1600)                  # climb
        self._road(50, 100, 50, curve=-3.2)                  # long left
        self._road(30, 40, 30, curve=4.4, height=-1200)      # dropping right
        self._road(40, 80, 40)
        self._road(25, 30, 25, curve=-5.0)                   # esses
        self._road(25, 30, 25, curve=5.0)
        self._road(25, 30, 25, curve=-5.0)
        self._road(60, 140, 60, height=2400)                 # big crest
        self._road(50, 110, 50, curve=3.0, height=-2400)
        self._road(40, 60, 40, curve=-2.2)
        self._road(30, 40, 30, curve=6.0)                    # hairpin-ish
        self._road(60, 160, 60)                              # back straight
        self._road(45, 80, 45, curve=-4.0, height=1400)
        self._road(35, 50, 35, curve=3.4, height=-1400)
        self._road(70, 180, 70)                              # run to the flag

        self.n_segs = len(self.segments)
        self.track_len = self.n_segs * SEG_LEN

        # roadside scenery: trees and marker poles, denser on the verges
        rng = np.random.default_rng(1991)
        for i in range(20, self.n_segs - 20):
            seg = self.segments[i]
            if i % 12 == 0:
                seg['sprites'].append((-1.35, 'pole'))
            if i % 12 == 6:
                seg['sprites'].append((1.35, 'pole'))
            if rng.random() < 0.22:
                side = -1.0 if rng.random() < 0.5 else 1.0
                off = side * (1.7 + rng.random() * 2.6)
                seg['sprites'].append((off, 'tree'))
            if i % CHECKPOINT_EVERY == 0:
                seg['sprites'].append((-1.6, 'sign'))
                seg['sprites'].append((1.6, 'sign'))

        self.checkpoints = [i for i in range(CHECKPOINT_EVERY, self.n_segs,
                                             CHECKPOINT_EVERY)]

    # ---------------------------------------------------------------- state

    def _reset_state(self):
        self.position = 0.0
        self.player_x = 0.0
        self.speed = 0.0
        self.time_left = START_TIME
        self.elapsed = 0.0
        self.next_cp = 0
        self.crashes = 0
        self.touching = False
        self.finished = False
        self.last_event = ''
        self.event_timer = 0.0
        self._spawn_traffic()

    def _spawn_traffic(self):
        self.traffic = []
        for _ in range(48):
            seg = int(self.np_random.integers(40, self.n_segs - 40))
            self.traffic.append({
                'z': seg * SEG_LEN,
                'x': float(self.np_random.uniform(-0.72, 0.72)),
                'speed': MAX_SPEED * float(self.np_random.uniform(0.24, 0.52)),
                'color': TRAFFIC[int(self.np_random.integers(len(TRAFFIC)))],
            })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        self._reset_state()
        return self._frame(), self._info()

    # ----------------------------------------------------------------- step

    def step(self, action):
        acc, brk, left, right = (int(a) for a in action)
        seg = self.segments[int(self.position / SEG_LEN) % self.n_segs]
        speed_pct = self.speed / MAX_SPEED
        reward = 0.0

        # steering: quicker at speed, and the curve pushes you outward
        steer = DT * 2.4 * speed_pct
        if left:
            self.player_x -= steer
        if right:
            self.player_x += steer
        self.player_x -= steer * speed_pct * seg['curve'] * CENTRIFUGAL

        if acc:
            self.speed += ACCEL * DT
        elif brk:
            self.speed += BRAKE * DT
        else:
            self.speed += DECEL * DT

        # off the tarmac: scrub speed hard and rattle the car
        off_road = abs(self.player_x) > 1.0
        if off_road and self.speed > OFF_LIMIT:
            self.speed += OFF_DECEL * DT
            reward -= 0.05
        self.speed = float(np.clip(self.speed, 0.0, MAX_SPEED))
        self.player_x = float(np.clip(self.player_x, -2.4, 2.4))

        hit = self._update_traffic()
        if hit:
            reward -= 4.0
            self.crashes += 1
            self._flash('CRASH!')

        before = self.position
        self.position += self.speed * DT
        reward += (self.position - before) / SEG_LEN * 0.1

        self.elapsed += DT
        self.time_left -= DT
        if self.event_timer > 0:
            self.event_timer -= DT

        terminated = False
        seg_now = int(self.position / SEG_LEN)
        while (self.next_cp < len(self.checkpoints)
               and seg_now >= self.checkpoints[self.next_cp]):
            self.time_left += CHECKPOINT_BONUS
            self.next_cp += 1
            reward += 25.0
            self._flash('CHECKPOINT +%d' % int(CHECKPOINT_BONUS))

        if self.position >= self.track_len:
            self.finished = True
            terminated = True
            reward += 200.0
            self._flash('FINISH!')
        elif self.time_left <= 0.0:
            self.time_left = 0.0
            terminated = True
            reward -= 20.0
            self._flash('OUT OF TIME')

        return self._frame(), reward, terminated, False, self._info()

    def _flash(self, text):
        self.last_event = text
        self.event_timer = 1.6

    def _update_traffic(self):
        self._advance_traffic()
        return self._collide_traffic()

    def _advance_traffic(self):
        """Roll traffic forward one tick.

        Split out from the collision test so a two-car race can advance one
        shared set of cars exactly once per tick instead of once per racer.
        """
        for car in self.traffic:
            car['z'] = (car['z'] + car['speed'] * DT) % self.track_len

    def _collide_traffic(self):
        """Report whether the player just hit a car.

        The player's car is drawn ``player_z`` in front of the camera, so the
        hit box has to sit there too - testing against ``position`` alone put
        it roughly five segments behind the car you can see, which fired on
        traffic that was still well up the road.

        Lateral reach is the sum of the two half-widths rather than a guessed
        constant, so a hit means the sprites really do overlap.

        Staying behind a slower car is a block, not a crash: only the first
        contact at real closing speed counts as a shunt, after which you are
        simply held to their pace until you steer around them.
        """
        hit = False
        touching = False
        nose = self.position + self.player_z
        reach = self.player_half_w + self.traffic_half_w
        for car in self.traffic:
            gap = car['z'] - nose
            if not -HIT_BEHIND < gap < HIT_AHEAD:
                continue
            if abs(self.player_x - car['x']) >= reach:
                continue

            touching = True
            # you cannot drive through them: hold station on their bumper
            self.position = min(
                self.position,
                max(0.0, car['z'] - HIT_STANDOFF - self.player_z))
            if not self.touching and self.speed > car['speed'] * CLOSING_HIT:
                self.speed = car['speed'] * 0.85   # ran into the back of them
                hit = True
            else:
                self.speed = min(self.speed, car['speed'])  # just held up
        self.touching = touching
        return hit

    def _info(self):
        return {
            'speed_mph': self.speed / MAX_SPEED * TOP_MPH,
            'time_left': self.time_left,
            'distance': self.position,
            'progress': self.position / self.track_len,
            'gear': self._gear(),
            'crashes': self.crashes,
            'finished': self.finished,
        }

    def _gear(self):
        return min(5, 1 + int(self.speed / MAX_SPEED * 5.0))

    # ---------------------------------------------------------------- render

    def _project(self, world_x, world_y, world_z, cam_x, cam_y, cam_z):
        dz = max(world_z - cam_z, 1.0)
        scale = self.cam_depth / dz
        sx = WIDTH / 2 + scale * (world_x - cam_x) * WIDTH / 2
        sy = HEIGHT / 2 - scale * (world_y - cam_y) * HEIGHT / 2
        sw = scale * ROAD_W * WIDTH / 2
        return sx, sy, sw, scale

    def _frame(self):
        s = self.surf
        base_i = int(self.position / SEG_LEN) % self.n_segs
        base = self.segments[base_i]
        base_pct = (self.position % SEG_LEN) / SEG_LEN
        p_seg = self.segments[
            int((self.position + self.player_z) / SEG_LEN) % self.n_segs]
        p_pct = ((self.position + self.player_z) % SEG_LEN) / SEG_LEN
        player_y = p_seg['y1'] + (p_seg['y2'] - p_seg['y1']) * p_pct

        self._draw_sky(s, base['curve'])

        cam_y = player_y + CAM_HEIGHT
        x = 0.0
        dx = -(base['curve'] * base_pct)
        maxy = float(HEIGHT)
        drawn = []

        for n in range(DRAW_DIST):
            idx = (base_i + n) % self.n_segs
            seg = self.segments[idx]
            looped = idx < base_i
            cam_z = self.position - (self.track_len if looped else 0)

            z1 = idx * SEG_LEN + (self.track_len if looped else 0)
            # The player offset is a *camera* x, so it enters negated: the
            # road slides the opposite way to the car. Getting this backwards
            # renders traffic at (car_x + player_x) instead of the true
            # (car_x - player_x), which puts cars nowhere near where they are.
            x1, y1, w1, _ = self._project(
                x - self.player_x * ROAD_W, seg['y1'], z1, 0.0, cam_y, cam_z)
            x2, y2, w2, sc2 = self._project(
                x + dx - self.player_x * ROAD_W, seg['y2'], z1 + SEG_LEN,
                0.0, cam_y, cam_z)

            x += dx
            dx += seg['curve']

            if y2 >= y1 or y2 >= maxy:
                continue
            self._draw_segment(s, idx, x1, y1, w1, x2, y2, w2,
                               n / DRAW_DIST)
            drawn.append((idx, x1, y1, w1, maxy))
            maxy = y2

        # scenery and traffic, painted back to front over the road
        for idx, sx, sy, sw, clip in reversed(drawn):
            seg = self.segments[idx]
            scale = sw / (ROAD_W * WIDTH / 2)
            for off, kind in seg['sprites']:
                self._draw_sprite(s, kind, sx + sw * off, sy, scale, clip)
            for car in self.traffic:
                if int(car['z'] / SEG_LEN) % self.n_segs == idx:
                    self._draw_traffic(s, car, sx + sw * car['x'], sy,
                                       scale, clip)

        self._draw_player(s)
        self._draw_hud(s)
        return np.transpose(pygame.surfarray.array3d(s), (1, 0, 2))

    def _draw_sky(self, s, curve):
        horizon = HEIGHT // 2
        for i in range(horizon):
            p = i / horizon
            col = tuple(int(SKY_TOP[c] + (SKY_BOT[c] - SKY_TOP[c]) * p)
                        for c in range(3))
            pygame.draw.line(s, col, (0, i), (WIDTH, i))
        s.fill(GRASS[0], (0, horizon, WIDTH, HEIGHT - horizon))
        # distant hills drift opposite the curve, the old parallax trick
        shift = -curve * 26.0
        for band, col in ((26, HILL_FAR), (14, HILL_NEAR)):
            pts = []
            for i in range(0, WIDTH + 24, 24):
                h = band * (0.6 + 0.4 * math.sin((i + shift) * 0.045))
                pts.append((i, horizon - h))
            pts += [(WIDTH + 24, horizon), (0, horizon)]
            pygame.draw.polygon(s, col, pts)

    def _draw_segment(self, s, idx, x1, y1, w1, x2, y2, w2, depth):
        dark = (idx // RUMBLE_SEGS) % 2
        grass = GRASS[dark]
        road = ROAD[dark]
        rumble = RUMBLE[dark]

        s.fill(grass, (0, int(y2), WIDTH, int(y1 - y2) + 1))

        r1, r2 = w1 * RUMBLE_FRAC, w2 * RUMBLE_FRAC
        self._quad(s, rumble, x1 - w1 - r1, y1, x1 - w1, y1,
                   x2 - w2, y2, x2 - w2 - r2, y2)
        self._quad(s, rumble, x1 + w1 + r1, y1, x1 + w1, y1,
                   x2 + w2, y2, x2 + w2 + r2, y2)
        self._quad(s, road, x1 - w1, y1, x1 + w1, y1,
                   x2 + w2, y2, x2 - w2, y2)

        if not dark:
            l1, l2 = w1 * LANE_FRAC, w2 * LANE_FRAC
            for lane in (-0.34, 0.34):
                self._quad(s, LANE,
                           x1 + w1 * lane - l1, y1, x1 + w1 * lane + l1, y1,
                           x2 + w2 * lane + l2, y2, x2 + w2 * lane - l2, y2)

        if depth > 0.62:  # haze out the far end of the draw distance
            a = int(min(1.0, (depth - 0.62) / 0.38) * 200)
            if a > 4:
                haze = pygame.Surface((WIDTH, max(1, int(y1 - y2) + 1)))
                haze.set_alpha(a)
                haze.fill(FOG)
                s.blit(haze, (0, int(y2)))

    @staticmethod
    def _quad(s, col, ax, ay, bx, by, cx, cy, dx_, dy):
        pygame.draw.polygon(s, col, ((ax, ay), (bx, by), (cx, cy), (dx_, dy)))

    def _draw_sprite(self, s, kind, sx, sy, scale, clip):
        """Sprites are sized in world units, then projected like the road.

        Vertical world->screen is ``scale * HEIGHT/2`` and horizontal is
        ``scale * WIDTH/2`` - the same factors _project uses.
        """
        if sy > clip or sy < 0 or scale <= 0:
            return
        vy = scale * HEIGHT / 2
        vx = scale * WIDTH / 2
        if kind == 'tree':
            h, w = 3000 * vy, 1500 * vx
            if h < 1.2:
                return
            pygame.draw.rect(s, (78, 54, 34),
                             (sx - w * 0.08, sy - h * 0.34,
                              max(1, w * 0.16), h * 0.34 + 1))
            pygame.draw.polygon(s, (24, 86, 36), (
                (sx, sy - h), (sx - w / 2, sy - h * 0.26),
                (sx + w / 2, sy - h * 0.26)))
            pygame.draw.polygon(s, (34, 110, 44), (
                (sx, sy - h * 0.92), (sx - w * 0.34, sy - h * 0.4),
                (sx + w * 0.34, sy - h * 0.4)))
        elif kind == 'pole':
            h, w = 1000 * vy, 90 * vx
            if h < 1.0:
                return
            pygame.draw.rect(s, (232, 232, 236),
                             (sx, sy - h, max(1, w), h))
            pygame.draw.rect(s, (208, 48, 44),
                             (sx, sy - h, max(1, w), max(1, h * 0.24)))
        elif kind == 'sign':
            h, w = 1700 * vy, 1400 * vx
            if h < 1.4:
                return
            pygame.draw.rect(s, (150, 150, 156),
                             (sx - w * 0.03, sy - h, max(1, w * 0.06), h))
            pygame.draw.rect(s, (232, 208, 56),
                             (sx - w / 2, sy - h, w, h * 0.42))
            pygame.draw.rect(s, (40, 40, 46),
                             (sx - w * 0.36, sy - h * 0.86,
                              max(1, w * 0.72), max(1, h * 0.1)))

    def _draw_traffic(self, s, car, sx, sy, scale, clip):
        if sy > clip or scale <= 0:
            return
        w = TRAFFIC_W * scale * WIDTH / 2
        h = TRAFFIC_H * scale * HEIGHT / 2
        if w < 1.5:
            return
        body = pygame.Rect(sx - w / 2, sy - h, w, h * 0.78)
        pygame.draw.rect(s, car['color'], body)
        pygame.draw.rect(s, (26, 30, 34),
                         (sx - w / 2, sy - h * 0.22, w, max(1, h * 0.22)))
        pygame.draw.rect(s, (40, 52, 58),
                         (sx - w * 0.32, sy - h * 0.92,
                          max(1, w * 0.64), max(1, h * 0.3)))
        if w > 6:
            for lx in (-0.36, 0.28):
                pygame.draw.rect(s, (232, 96, 72),
                                 (sx + w * lx, sy - h * 0.42,
                                  max(1, w * 0.08), max(1, h * 0.14)))

    def _draw_car(self, s, cx, y, w, body):
        """The Esprit silhouette at any size - your car, or a rival's.

        ``y`` is where the wheels meet the road, ``w`` the body width; every
        detail scales off ``w`` so the same sprite works from a full-size car
        alongside you down to a speck up the road.
        """
        k = w / PLAYER_CAR_PX
        h = 26.0 * k

        # taillights have to read against the bodywork: the stock red pair
        # disappears entirely on a red car, so go amber when the body is
        # already red
        lights = CAR_LIGHT
        if body[0] > body[1] + 60 and body[0] > body[2] + 60:
            lights = (255, 176, 64)

        pygame.draw.ellipse(s, _shade(body, 0.55),
                            (cx - w * 0.55, y - 3 * k, w * 1.1,
                             max(1, 9 * k)))
        for side in (-1, 1):
            pygame.draw.rect(s, (24, 24, 28),
                             (cx + side * w * 0.42 - 7 * k, y - h * 0.52,
                              max(1, 14 * k), max(1, 20 * k)))
        pygame.draw.polygon(s, body, (
            (cx - w / 2, y), (cx - w * 0.44, y - h * 0.62),
            (cx - w * 0.3, y - h * 0.92), (cx + w * 0.3, y - h * 0.92),
            (cx + w * 0.44, y - h * 0.62), (cx + w / 2, y)))
        pygame.draw.polygon(s, CAR_GLASS, (
            (cx - w * 0.26, y - h * 0.66), (cx - w * 0.21, y - h * 0.9),
            (cx + w * 0.21, y - h * 0.9), (cx + w * 0.26, y - h * 0.66)))
        pygame.draw.rect(s, CAR_TRIM,
                         (cx - w * 0.36, y - h * 0.55, w * 0.72,
                          max(1, 3 * k)))
        for side in (-1, 1):
            pygame.draw.rect(s, lights,
                             (cx + side * w * 0.34 - 6 * k, y - h * 0.44,
                              max(1, 12 * k), max(1, 6 * k)))
        # rear wing
        pygame.draw.rect(s, _shade(body, 0.78),
                         (cx - w * 0.34, y - h * 1.02, w * 0.68,
                          max(1, 4 * k)))

    def _draw_player(self, s):
        """Rear view of the Esprit, leaning as it slides through a bend."""
        cx = WIDTH / 2
        base = HEIGHT - 26
        lean = float(np.clip(self.player_x * 0.5, -1.0, 1.0))
        bob = math.sin(self.elapsed * 26.0) * (self.speed / MAX_SPEED) * 1.1
        if abs(self.player_x) > 1.0:
            bob += math.sin(self.elapsed * 61.0) * 2.2  # off-road rattle
        cx += lean * 9.0
        self._draw_car(s, cx, base + bob, PLAYER_CAR_PX, self.body_color)

    def _draw_hud(self, s):
        bar = HEIGHT - 20
        s.fill((10, 12, 16), (0, bar, WIDTH, 20))
        pygame.draw.line(s, (60, 70, 80), (0, bar), (WIDTH, bar))

        mph = int(self.speed / MAX_SPEED * TOP_MPH)
        s.blit(self.font.render(f'{mph:3d} MPH', True, (250, 230, 90)),
               (4, bar + 5))
        s.blit(self.font.render(f'GEAR {self._gear()}', True, (170, 220, 255)),
               (66, bar + 5))
        pct = int(self.position / self.track_len * 100)
        s.blit(self.font.render(f'{pct:3d}%', True, (200, 200, 210)),
               (128, bar + 5))
        col = (250, 90, 80) if self.time_left < 6 else (140, 250, 140)
        s.blit(self.font.render(f'TIME {self.time_left:5.1f}', True, col),
               (166, bar + 5))
        s.blit(self.font.render(f'CP {self.next_cp}/{len(self.checkpoints)}',
                                True, (200, 200, 210)), (250, bar + 5))

        # rev bar, sweeping and resetting with each gear
        rev = (self.speed / MAX_SPEED * 5.0) % 1.0
        pygame.draw.rect(s, (40, 44, 52), (4, bar - 6, 90, 4))
        pygame.draw.rect(s, (250, 120, 60), (4, bar - 6, int(90 * rev), 4))

        if self.event_timer > 0 and self.last_event:
            t = self.big_font.render(self.last_event, True, (255, 240, 120))
            s.blit(t, (WIDTH // 2 - t.get_width() // 2, 30))

    def render(self):
        return self._frame()

    def close(self):
        pass


gymnasium.register(id='Racer-v0', entry_point=RacerEnv,
                   max_episode_steps=None)
