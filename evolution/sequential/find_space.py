import numpy as np
import neat
import math
from environment.core import BluelockEnvironment
from evolution.util import scale_to_env_dims
from util import Rect, get_beeline_orientation


def get_find_space_inputs(
    env: BluelockEnvironment, possessor_id: int, defender_id: int, offballer_id: int
):
    possessor, defender, offballer = env.get_players_by_ids(
        possessor_id, defender_id, offballer_id
    )
    defender_disp_to_ball = scale_to_env_dims(
        env, possessor.position - defender.position
    )
    offballer_disp_to_ball = scale_to_env_dims(
        env, possessor.position - offballer.position
    )
    inputs = list(defender_disp_to_ball) + list(offballer_disp_to_ball)
    for point in Rect([0.0, 0.0], height=env.height, width=env.width).get_vertices():
        corner_displacement_to_ball = scale_to_env_dims(
            env, possessor.position - np.array(point)
        )
        inputs.extend(list(corner_displacement_to_ball))
    return inputs


def get_find_space_outputs(
    env: BluelockEnvironment,
    spacer_net: neat.nn.FeedForwardNetwork,
    possessor_id: int,
    defender_id: int,
    offballer_id: int,
):
    vx, vy = spacer_net.activate(
        get_find_space_inputs(env, possessor_id, defender_id, offballer_id)
    )
    speed_magnitude = math.sqrt(vx**2 + vy**2) / math.sqrt(2)
    orientation = get_beeline_orientation(np.array([vx, vy]))
    return speed_magnitude, orientation


def go_to_space(
    env: BluelockEnvironment,
    spacer_net: neat.nn.FeedForwardNetwork,
    possessor_id: int,
    defender_id: int,
    offballer_id: int,
):
    speed_mag, orientation = get_find_space_outputs(
        env, spacer_net, possessor_id, defender_id, offballer_id
    )
    offballer = env.get_player(offballer_id)
    offballer.set_rotation(orientation)
    offballer.run(speed_mag)
