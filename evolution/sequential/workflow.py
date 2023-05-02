from evolution.sequential.hold_ball import HoldBall
from evolution.sequential.seek_ball import SeekBall, apply_seek
from evolution.sequential.pass_evaluate import PassEvaluate, apply_pass_evaluate
from evolution.util import with_offense_controls
from util import get_beeline_orientation
from visualization.visualizer import BluelockEnvironmentVisualizer
import time


def evolve_sequential():
    seek_ball = SeekBall()
    seek_ball.evolve(100)
    hold_ball = HoldBall()
    hold_ball.evolve(100)
    pass_eval = PassEvaluate(seek_ball.get_best_model())
    pass_eval.evolve(50)


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


def watch_pass_evaluate():
    seeker_model = SeekBall().get_best_model()
    pass_evaluate = PassEvaluate(seeker_model)
    pass_eval_model = pass_evaluate.get_best_model()
    episodes = 10
    passer_id, target_id, defender_id = 1, 2, 3
    for i, factory in enumerate(
        [pass_evaluate.get_env_factory() for _ in range(episodes)]
    ):
        env = factory(passer_id, target_id, defender_id)
        controls = {}
        controls[target_id] = lambda env, offender: apply_seek(
            env, offender.id, seeker_model
        )
        env = with_offense_controls(env, controls)
        vis = BluelockEnvironmentVisualizer(env)
        should_pass = apply_pass_evaluate(env, target_id, defender_id, pass_eval_model)
        passer, target, defender = (
            env.get_player(passer_id),
            env.get_player(target_id),
            env.get_player(defender_id),
        )
        # todo(): use passing model here
        print(f"{i} should pass: {should_pass}")
        passer.set_rotation(get_beeline_orientation(target.position - passer.position))
        passer.shoot()
        elapsed = 0
        dt, alotted = 5, 3000
        while elapsed < alotted and not (
            target.has_possession() or defender.has_possession()
        ):
            env.update(dt)
            vis.draw()
            elapsed += dt
        time.sleep(5)


def watch_sequential():
    watch_pass_evaluate()
