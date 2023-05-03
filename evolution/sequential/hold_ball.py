import neat
import numpy as np
import os
import math
from environment.core import BluelockEnvironment, Offender, Defender, Ball
from environment.config import ENVIRONMENT_HEIGHT, ENVIRONMENT_WIDTH
from environment.defense.agent import with_policy_defense
from environment.defense.policy import naive_man_to_man
from evolution.config import CHECKPOINTS_PATH, MODELS_PATH, CONFIGS_PATH, PLOTS_PATH
from evolution.util import with_offense_controls
from evolution.sequential.task import SequentialEvolutionTask
from visualization.visualizer import BluelockEnvironmentVisualizer
from util import get_unit_vector, get_random_within_range, get_beeline_orientation, Rect


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
    outputs = holder_net.activate(get_hold_ball_inputs(env, holder_id, defender_id))
    return get_beeline_orientation(np.array(outputs))


class HoldBall(SequentialEvolutionTask):
    def __init__(self):
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            os.path.join(CONFIGS_PATH, "hold_ball.ini"),
        )
        super().__init__(CHECKPOINTS_PATH, MODELS_PATH, PLOTS_PATH, config, "hold_ball")

    def with_controls(
        self,
        env: BluelockEnvironment,
        offender_id: int,
        defender_id: int,
        net: neat.nn.FeedForwardNetwork,
    ):
        def action(env: BluelockEnvironment, offender: Offender):
            offender.set_rotation(apply_hold_ball(env, offender.id, defender_id, net))
            offender.run()

        controls = {}
        controls[offender_id] = action

        return with_offense_controls(env, controls)

    def get_env_factory(self):
        env_width, env_height = ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT
        offender_position = (env_width / 2, env_height / 2)
        angle = get_random_within_range(-math.pi, math.pi)
        defender_proximity = 125
        defender_position = tuple(
            defender_proximity * get_unit_vector(angle) + np.array(offender_position)
        )

        def factory(offender_id, defender_id):
            offender = Offender(offender_id, offender_position)
            defender = Defender(defender_id, defender_position)
            ball = Ball(offender_position)  # will be possessed by the offender
            env = BluelockEnvironment(
                (env_width, env_height), [offender], [defender], ball
            )
            return with_policy_defense(env, policy=naive_man_to_man)

        return factory

    def compute_fitness(self, genomes, config):
        defender_id = -1
        episodes = 10
        dt, alotted = 15, 12000
        factories = [self.get_env_factory() for _ in range(episodes)]
        for id, genome in genomes:
            genome.fitness = 0
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            for env in [
                self.with_controls(factory(id, defender_id), id, defender_id, net)
                for factory in factories
            ]:
                defender: Defender = env.get_player(defender_id)
                elapsed = 0
                while elapsed < alotted and not defender.has_possession():
                    env.update(dt)
                    elapsed += dt

                genome.fitness += elapsed
