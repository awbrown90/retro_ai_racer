"""Play Retro AI Racer - a pseudo-3D arcade racer, built from scratch.

The stage, car, scenery and physics are original code in racer_env.py,
exposed as the Gymnasium environment Racer-v0. No ROM involved.

Controls:
    UP or Z               accelerate
    DOWN or X             brake
    LEFT / RIGHT or A / D steer
    R                     restart the stage
    ESC                   quit

Reach each checkpoint before the clock runs out - every one banks more time.
Leaving the tarmac scrubs speed hard, and rear-ending traffic costs more
still, so the racing line through the bends is what wins you the stage.
"""
import gymnasium
import numpy as np
import pygame

import racer_env  # noqa: F401  registers Racer-v0

SCALE = 3
FPS = 50  # matches the env's render_fps, so one step per frame is real time
TITLE = "Retro AI Racer - Stage 1"
END_HOLD = int(FPS * 2.2)  # freeze on FINISH / OUT OF TIME so you can read it


def action_from_keys(held):
    """Map the set of held pygame keys to the env's MultiBinary action."""
    return np.array([
        1 if held.intersection((pygame.K_UP, pygame.K_z)) else 0,
        1 if held.intersection((pygame.K_DOWN, pygame.K_x)) else 0,
        1 if held.intersection((pygame.K_LEFT, pygame.K_a)) else 0,
        1 if held.intersection((pygame.K_RIGHT, pygame.K_d)) else 0,
    ], dtype=np.int8)


def main():
    print(__doc__)
    env = gymnasium.make('Racer-v0')
    obs, info = env.reset()

    pygame.init()
    h, w = obs.shape[:2]
    screen = pygame.display.set_mode((w * SCALE, h * SCALE))
    pygame.display.set_caption(TITLE)
    clock = pygame.time.Clock()

    held = set()
    hold = 0
    frame = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    hold = 0
                held.add(event.key)
            elif event.type == pygame.KEYUP:
                held.discard(event.key)
            elif event.type == pygame.WINDOWFOCUSLOST:
                held.clear()

        if hold:
            # stage over: keep showing the final frame and its banner
            obs = env.render()
            hold -= 1
            if hold == 0:
                obs, info = env.reset()
        else:
            obs, reward, terminated, truncated, info = env.step(
                action_from_keys(held))
            if terminated or truncated:
                hold = END_HOLD

        surf = pygame.image.frombuffer(
            np.ascontiguousarray(obs).tobytes(), (w, h), 'RGB').convert()
        pygame.transform.scale(surf, screen.get_size(), screen)
        pygame.display.flip()

        frame += 1
        if frame % 15 == 0:
            pygame.display.set_caption(
                f"{TITLE} - {info['speed_mph']:3.0f}mph "
                f"gear={info['gear']} time={info['time_left']:4.1f} "
                f"{info['progress'] * 100:3.0f}%")
        clock.tick(FPS)

    env.close()
    pygame.quit()


if __name__ == '__main__':
    main()
