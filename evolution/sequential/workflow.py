from evolution.sequential.hold_ball import HoldBall, apply_hold_ball
from evolution.sequential.seek_ball import SeekBall, apply_seek
from evolution.sequential.pass_evaluate import PassEvaluate, apply_pass_evaluate
from evolution.sequential.find_space import FindSpace, apply_find_space
from evolution.util import with_offense_controls
from util import get_beeline_orientation
from visualization.visualizer import BluelockEnvironmentVisualizer


def evolve_sequential():
    seek_ball = SeekBall()
    seek_ball.evolve(100)
    hold_ball = HoldBall()
    hold_ball.evolve(100)
    pass_eval = PassEvaluate(seek_ball.get_best_model())
    pass_eval.evolve(50)
    find_space = FindSpace(
        seek_ball.get_best_model(),
        hold_ball.get_best_model(),
        pass_eval.get_best_model(),
    )
    find_space.evolve(50)


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
    episodes = 100
    passer_id, target_id, defender_id = 1, 2, 3
    confusion_matrix = [[0, 0], [0, 0]]
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
            # vis.draw()
            elapsed += dt
        if should_pass:
            if target.has_possession():
                confusion_matrix[0][0] += 1
            elif defender.has_possession():
                confusion_matrix[0][1] += 1
        else:
            if defender.has_possession():
                confusion_matrix[1][1] += 1
            elif target.has_possession():
                confusion_matrix[1][0] += 1
    print(confusion_matrix)


def watch_find_space():
    find_space = FindSpace(
        SeekBall().get_best_model(),
        HoldBall().get_best_model(),
        PassEvaluate(SeekBall().get_best_model()).get_best_model(),
    )
    episodes = 5
    passer_id, find_spacer_id, defender_id = 1, 2, 3
    factories = [find_space.get_env_factory() for _ in range(episodes)]
    net = find_space.get_best_model()
    for factory in factories:
        env = factory(passer_id, find_spacer_id, defender_id)
        passer, target, defender = (
            env.get_player(passer_id),
            env.get_player(find_spacer_id),
            env.get_player(defender_id),
        )

        should_seek = [None]

        def action(env, offender):
            other_offender = list(filter(lambda o: o.id != offender.id, env.offense))[0]
            if offender.has_possession():
                if should_seek[0] == offender.id:
                    should_seek[0] = None
                confident = apply_pass_evaluate(
                    env, other_offender.id, defender_id, find_space.pass_evaluator
                )
                if confident:
                    offender.set_rotation(
                        get_beeline_orientation(
                            other_offender.position - offender.position
                        )
                    )
                    offender.shoot()
                    should_seek[0] = other_offender.id
                else:
                    offender.set_rotation(
                        apply_hold_ball(
                            env, offender.id, defender_id, find_space.holder
                        )
                    )
                    offender.run()
            else:
                if should_seek[0] == offender.id:
                    offender.set_rotation(
                        apply_seek(env, offender.id, find_space.seeker)
                    )
                    offender.run()
                else:
                    if other_offender.has_possession():
                        offender.set_rotation(
                            apply_find_space(
                                env,
                                other_offender.id,
                                offender.id,
                                defender_id,
                                net,
                            )
                        )
                        offender.run()

        controls = {}
        controls[passer_id] = action
        controls[find_spacer_id] = action
        env = with_offense_controls(env, controls)
        vis = BluelockEnvironmentVisualizer(env)
        while not defender.has_possession():
            env.update(5)
            vis.draw()


def watch_sequential():
    watch_find_space()
