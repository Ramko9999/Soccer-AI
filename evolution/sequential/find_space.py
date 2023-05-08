import numpy as np
import neat
import math
from environment.core import BluelockEnvironment, Offender, Defender, Ball
from environment.config import ENVIRONMENT_HEIGHT, ENVIRONMENT_WIDTH
from environment.defense.agent import with_policy_defense, naive_man_to_man
from evolution.util import scale_to_env_dims
from evolution.task import EvolutionTask
from evolution.sequential.seek import Seek, do_seek
from evolution.sequential.pass_ball import Pass, make_pass
from evolution.config import (
    CHECKPOINTS_PATH,
    MODELS_PATH,
    PLOTS_PATH,
    get_default_config,
)
from util import Rect, get_beeline_orientation, get_random_point, get_euclidean_dist
from visualization.visualizer import BluelockEnvironmentVisualizer


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


def with_offball_movement(
    env: BluelockEnvironment,
    spacer_net: neat.nn.FeedForwardNetwork,
    seeker_net: neat.nn.FeedForwardNetwork,
    possessor_id: int,
    defender_id: int,
    offballer_id: int,
):

    old_update = env.update

    def control():
        offballer = env.get_player(offballer_id)
        if env.does_offense_have_possession():
            if not offballer.has_possession():
                go_to_space(env, spacer_net, possessor_id, defender_id, offballer_id)
        else:
            do_seek(env, seeker_net, offballer_id)

    def new_update(*args, **kwargs):
        old_update(*args, **kwargs)
        control()

    env.update = new_update
    return env


TASK_NAME = "find_space"


class FindSpace(EvolutionTask):
    def __init__(
        self, seeker: neat.nn.FeedForwardNetwork, passer: neat.nn.FeedForwardNetwork
    ):
        super().__init__(
            CHECKPOINTS_PATH,
            MODELS_PATH,
            PLOTS_PATH,
            TASK_NAME,
            get_default_config(f"{TASK_NAME}.ini"),
        )
        self.seeker = seeker
        self.passer = passer
        self.possessor_id = 1
        self.offballer_id = 2
        self.defender_id = 3

    def get_episodes(self):
        envs = []
        for _ in range(40):
            possessor = Offender(
                self.possessor_id,
                get_random_point(x_max=ENVIRONMENT_WIDTH, y_max=ENVIRONMENT_HEIGHT),
            )
            offballer = Offender(
                self.offballer_id,
                get_random_point(x_max=ENVIRONMENT_WIDTH, y_max=ENVIRONMENT_HEIGHT),
            )
            defender = Defender(
                self.defender_id,
                get_random_point(x_max=ENVIRONMENT_WIDTH, y_max=ENVIRONMENT_HEIGHT),
            )
            ball = Ball((0, 0))
            possessor.possess(ball)
            envs.append(
                BluelockEnvironment(
                    dims=(ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT),
                    offense=[possessor, offballer],
                    defense=[defender],
                    ball=ball,
                ),
            )
        return envs

    def compute_fitness(self, genome, config) -> float:
        dt, allotted = 15, 6000
        find_space_alloted = 1500
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        episodes = self.get_episodes()
        fitness = 0
        for env in episodes:
            env = with_offball_movement(
                env,
                net,
                self.seeker,
                self.possessor_id,
                self.defender_id,
                self.offballer_id,
            )
            for _ in range(0, find_space_alloted, dt):
                env.update(dt)

            env = with_policy_defense(env, policy=naive_man_to_man)
            make_pass(env, self.passer, self.possessor_id, self.offballer_id)
            possessor, offballer, defender = env.get_players_by_ids(
                self.possessor_id, self.offballer_id, self.defender_id
            )
            initial_dist_to_possessor = get_euclidean_dist(
                possessor.position, offballer.position
            )
            initial_defender_pos = defender.position
            for _ in range(0, allotted, dt):
                if (
                    env.does_defense_have_possession()
                    or env.does_offense_have_possession()
                ):
                    break
                env.update(dt)

            # very very unlikely neither will have possession
            max_dist_possible = math.sqrt(env.width**2 + env.height**2)
            award = 0
            if env.does_defense_have_possession():
                award = (
                    get_euclidean_dist(defender.position, initial_defender_pos)
                    / max_dist_possible
                )
            else:
                award = 1
                if initial_dist_to_possessor > 0:
                    award += (
                        get_euclidean_dist(possessor.position, offballer.position)
                        / initial_dist_to_possessor
                    )
            fitness += award
        return fitness / len(episodes)


def evolve_find_space():
    seek = Seek()
    pass_ball = Pass(seek.get_best_model())
    find_space = FindSpace(seek.get_best_model(), pass_ball.get_best_model())
    for _ in find_space.evolve(100, 100):
        pass


def watch_find_space():
    dt, allotted = 3, 6000
    find_space_alloted = 1500
    seeker = Seek()
    passer = Pass(seeker.get_best_model())
    find_space = FindSpace(seeker.get_best_model(), passer.get_best_model())
    for env in find_space.get_episodes():
        env = with_offball_movement(
            env,
            find_space.get_best_model(),
            seeker.get_best_model(),
            find_space.possessor_id,
            find_space.defender_id,
            find_space.offballer_id,
        )
        vis = BluelockEnvironmentVisualizer(env)
        for _ in range(0, find_space_alloted, dt):
            env.update(dt)
            vis.draw()

        env = with_policy_defense(env, policy=naive_man_to_man)
        make_pass(
            env,
            passer.get_best_model(),
            find_space.possessor_id,
            find_space.offballer_id,
        )
        for _ in range(0, allotted, dt):
            env.update(dt)
            vis.draw()
