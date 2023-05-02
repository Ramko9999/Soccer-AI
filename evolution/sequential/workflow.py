from evolution.sequential.hold_ball import HoldBall
from evolution.sequential.seek_ball import SeekBall
from visualization.visualizer import BluelockEnvironmentVisualizer


def evolve_sequential():
    seek_ball = SeekBall()
    seek_ball.evolve(100)
    hold_ball = HoldBall()
    hold_ball.evolve(200)


def watch_seek_ball():
    seek_ball = SeekBall()
    best_seeker = seek_ball.get_best_model()
    offender_id = 5
    dt, alotted = 8, 3000
    episodes = 20
    for factory in [seek_ball.get_env_factory() for _ in range(episodes)]:
        env = seek_ball.with_controls(factory(offender_id), offender_id, best_seeker)
        vis = BluelockEnvironmentVisualizer(env)
        for _ in range(0, alotted, dt):
            env.update(dt)
            vis.draw()


def watch_hold_ball():
    hold_ball = HoldBall()
    best_holder = hold_ball.get_best_model()
    offender_id, defender_id = 5, 1
    dt, alotted = 5, 50000
    episodes = 20
    for episode, factory in enumerate(
        [hold_ball.get_env_factory() for _ in range(episodes)]
    ):
        env = hold_ball.with_controls(
            factory(offender_id, defender_id), offender_id, defender_id, best_holder
        )
        vis = BluelockEnvironmentVisualizer(env)
        for _ in range(0, alotted, dt):
            if env.get_player(defender_id).has_possession():
                print(f"Defense got possession on {episode}")
                break
            env.update(dt)
            vis.draw()


def watch_sequential():
    watch_hold_ball()
