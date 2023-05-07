import neat
import numpy as np
import os
import math
from environment.core import BluelockEnvironment, Offender, Defender, Ball
from environment.config import (
    ENVIRONMENT_HEIGHT,
    ENVIRONMENT_WIDTH,
    PLAYER_DEFENDER_SPEED,
)
from environment.defense.agent import with_policy_defense
from environment.defense.policy import naive_man_to_man
from evolution.config import CHECKPOINTS_PATH, MODELS_PATH, CONFIGS_PATH, PLOTS_PATH
from evolution.util import with_offense_controls
from evolution.sequential.inputs import apply_hold_ball
from evolution.sequential.task import SequentialEvolutionTask
from util import get_unit_vector, get_random_within_range, get_random_point


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
            orientation, move_confidence = apply_hold_ball(
                env, offender.id, defender_id, net
            )
            if move_confidence > 0:
                offender.set_rotation(orientation)
                offender.run()

        return with_offense_controls(env, [(offender_id, action)])

    def get_env_factory(self):
        env_width, env_height = ENVIRONMENT_WIDTH / 2, ENVIRONMENT_HEIGHT / 2
        offender_position = get_random_point(env_width, env_height)
        defender_position = get_random_point(env_width, env_height)

        def factory(offender_id, defender_id):
            offender = Offender(offender_id, offender_position)
            defender = Defender(defender_id, defender_position)
            defender.top_speed = PLAYER_DEFENDER_SPEED * 0.4
            ball = Ball(offender_position)  # will be possessed by the offender
            env = BluelockEnvironment(
                (env_width, env_height), [offender], [defender], ball
            )
            return with_policy_defense(env, policy=naive_man_to_man)

        return factory

    def compute_fitness(self, genomes, config):
        defender_id = -1
        episodes = 10
        dt, alotted = 25, 12000
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
                    x, y = env.get_player(id).position
                    size = env.get_player(id).size
                    if (
                        x == size
                        or x == env.width - size
                        or y == size
                        or y == env.height - size
                    ):
                        break
                    elapsed += dt

                genome.fitness += elapsed
