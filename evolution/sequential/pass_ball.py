import neat
import numpy as np
import math
from environment.core import BluelockEnvironment, Offender, Ball
from environment.config import ENVIRONMENT_HEIGHT, ENVIRONMENT_WIDTH, PLAYER_SHOT_SPEED
from evolution.config import (
    CHECKPOINTS_PATH,
    MODELS_PATH,
    get_default_config,
    PLOTS_PATH,
)
from evolution.task import EvolutionTask
from evolution.sequential.seek import with_seeker, Seek
from evolution.util import scale_to_env_dims
from visualization.visualizer import BluelockEnvironmentVisualizer
from util import (
    get_beeline_orientation,
    get_euclidean_dist,
    get_random_point,
)


def get_pass_inputs(env: BluelockEnvironment, possessor_id: int, target_id: int):
    possessor, target = env.get_players_by_ids(possessor_id, target_id)
    displacement_to_target = scale_to_env_dims(
        env, target.position - possessor.position
    )
    return [displacement_to_target[0], displacement_to_target[1]]


def get_pass_outputs(
    env: BluelockEnvironment,
    passer: neat.nn.FeedForwardNetwork,
    possessor_id: int,
    target_id: int,
):
    vx, vy = passer.activate(get_pass_inputs(env, possessor_id, target_id))
    power_magnitude = math.sqrt(vx**2 + vy**2) / math.sqrt(2)
    orientation = get_beeline_orientation(np.array([vx, vy]))
    return power_magnitude, orientation


def make_pass(
    env: BluelockEnvironment,
    passer_net: neat.nn.FeedForwardNetwork,
    possessor_id: int,
    target_id: int,
):
    # assumes the offense has possession
    possessor, target = env.get_players_by_ids(possessor_id, target_id)
    power_mag, orientation = get_pass_outputs(env, passer_net, possessor.id, target.id)
    possessor.set_rotation(orientation)
    possessor.shoot(power_mag * PLAYER_SHOT_SPEED)


TASK_NAME = "pass"


class Pass(EvolutionTask):
    def __init__(self, seeker: neat.nn.FeedForwardNetwork):
        config_file = get_default_config(f"{TASK_NAME}.ini")
        super().__init__(
            CHECKPOINTS_PATH,
            MODELS_PATH,
            PLOTS_PATH,
            TASK_NAME,
            config_file,
        )
        self.seeker = seeker
        self.possessor_id = 1
        self.offballer_id = 2

    def get_episodes(self) -> list[BluelockEnvironment]:
        envs = []
        for _ in range(40):
            possessor = Offender(
                self.possessor_id,
                get_random_point(ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT),
            )
            ball = Ball((0, 0))
            offballer = Offender(
                self.offballer_id,
                get_random_point(ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT),
            )
            possessor.possess(ball)
            envs.append(
                BluelockEnvironment(
                    (ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT),
                    [offballer, possessor],
                    [],
                    ball,
                )
            )
        return envs

    def compute_fitness(self, genome, config):
        dt, allotted = 15, 6000
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        episodes = self.get_episodes()
        fitness = 0
        for env in episodes:
            env = with_seeker(env, self.seeker, self.offballer_id)
            offender, possessor = env.get_players_by_ids(
                self.offballer_id, self.possessor_id
            )
            offender_pos = offender.position
            to_possessor_dist = get_euclidean_dist(
                offender.position, possessor.position
            )
            make_pass(env, net, self.possessor_id, self.offballer_id)
            moving_time = 0
            for elapsed in range(0, allotted, dt):
                if offender.has_possession():
                    break
                if offender.speed > 0:
                    moving_time += dt
                env.update(dt)

            seeker_movement_award = 0.1 / (
                0.1
                + (
                    get_euclidean_dist(offender_pos, offender.position)
                    / to_possessor_dist
                )
            )
            award = 0
            if offender.has_possession():
                award = 1 + seeker_movement_award
            else:
                award = seeker_movement_award
            fitness += award
        return fitness / len(episodes)


def evolve_pass():
    seek = Seek()
    pass_task = Pass(seek.get_best_model())
    for _ in pass_task.evolve(100, 100):
        pass


def watch_pass():
    seek = Seek()
    best_seeker = seek.get_best_model()
    pass_task = Pass(best_seeker)
    best_passer = pass_task.get_best_model()
    for env in pass_task.get_episodes():
        dt, allotted = 5, 6000
        env = with_seeker(env, best_seeker, pass_task.offballer_id)
        make_pass(env, best_passer, pass_task.possessor_id, pass_task.offballer_id)
        vis = BluelockEnvironmentVisualizer(env)
        for _ in range(0, allotted, dt):
            env.update(dt)
            vis.draw()
