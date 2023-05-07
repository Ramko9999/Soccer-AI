import numpy as np
import math
import neat
from environment.core import BluelockEnvironment
from util import Rect, get_beeline_orientation, get_unit_vector


def get_find_space_inputs(
    env: BluelockEnvironment, passer_id: int, defender_id: int, space_finder_id: int
):
    passer = env.get_player(passer_id)
    space_finder = env.get_player(space_finder_id)
    defender = env.get_player(defender_id)

    dx, dy = space_finder.position - passer.position
    ddx, ddy = space_finder.position - defender.position
    rect = Rect([0.0, 0.0], height=env.height, width=env.width)
    inputs = [dx, dy, ddx, ddy]
    for point in rect.get_vertices():
        pdx, pdy = np.array(point) - space_finder.position
        inputs.append(abs(pdx))
        inputs.append(abs(pdy))
    return inputs


def apply_find_space(
    env: BluelockEnvironment,
    passer_id: int,
    target_id: int,
    defender_id: int,
    find_space_model: neat.nn.FeedForwardNetwork,
):
    inputs = get_find_space_inputs(env, passer_id, target_id, defender_id)
    pass_confidence, vec_x, vec_y, run_confidence = find_space_model.activate(inputs)
    return (
        pass_confidence,
        get_beeline_orientation(np.array([vec_x, vec_y])),
        run_confidence,
    )


def get_hold_ball_inputs(env: BluelockEnvironment, holder_id: int, defender_id: int):
    offender, defender = env.get_player(holder_id), env.get_player(defender_id)
    dx, dy = offender.position - defender.position
    vx, vy = defender.speed * defender.tilt
    rect = Rect([0.0, 0.0], height=env.height, width=env.width)
    inputs = [dx, dy, vx, vy]
    for point in rect.get_vertices():
        pdx, pdy = np.array(point) - offender.position
        inputs.append(pdx)
        inputs.append(pdy)
    return inputs


def apply_hold_ball(
    env: BluelockEnvironment,
    holder_id: int,
    defender_id: int,
    holder_net: neat.nn.FeedForwardNetwork,
):
    vec_x, vec_y, move_confidence = holder_net.activate(
        get_hold_ball_inputs(env, holder_id, defender_id)
    )
    return get_beeline_orientation(np.array([vec_x, vec_y])), move_confidence


def get_seek_ball_inputs(env: BluelockEnvironment, seeker_id: int):
    offender = env.get_player(seeker_id)
    dx, dy = env.ball.position - offender.position
    vbx, vby = get_unit_vector(env.ball.direction) * env.ball.speed
    return [dx, dy, vbx, vby]


def apply_seek(
    env: BluelockEnvironment, seeker_id: int, seeker_model: neat.nn.FeedForwardNetwork
):
    outputs = seeker_model.activate(get_seek_ball_inputs(env, seeker_id))
    return get_beeline_orientation(np.array(outputs))


def get_pass_evaluate_inputs(
    env: BluelockEnvironment, target_id: int, defender_id: int
):
    passer = env.ball.possessor
    target = env.get_player(target_id)
    defender = env.get_player(defender_id)

    odx, ody = target.position - passer.position
    ddx, ddy = target.position - defender.position
    return [odx, ody, ddx, ddy]


def apply_pass_evaluate(
    env: BluelockEnvironment,
    target_id: int,
    defender_id: int,
    pass_eval_model: neat.nn.FeedForwardNetwork,
):
    inputs = get_pass_evaluate_inputs(env, target_id, defender_id)
    confidence = pass_eval_model.activate(inputs)[0]
    return confidence > 0.5
