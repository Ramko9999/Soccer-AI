import neat
import numpy as np
import os
import math
from environment.core import BluelockEnvironment, Offender, Ball
from environment.config import ENVIRONMENT_HEIGHT, ENVIRONMENT_WIDTH, PLAYER_SHOT_SPEED
from evolution.config import CHECKPOINTS_PATH, MODELS_PATH, CONFIGS_PATH, PLOTS_PATH
from evolution.sequential.task import SequentialEvolutionTask
from evolution.util import with_offense_controls
from visualization.visualizer import BluelockEnvironmentVisualizer
from util import (
    get_unit_vector,
    get_random_within_range,
    get_euclidean_dist,
    get_beeline_orientation,
)


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


class SeekBall(SequentialEvolutionTask):
    def __init__(self):
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            os.path.join(CONFIGS_PATH, "seek_ball.ini"),
        )
        super().__init__(CHECKPOINTS_PATH, MODELS_PATH, PLOTS_PATH, config, "seek_ball")
        self.compute_fitness_call_count = 0

    def with_controls(
        self, env: BluelockEnvironment, player_id: int, net: neat.nn.FeedForwardNetwork
    ):
        def action(env: BluelockEnvironment, offender: Offender):
            offender.set_rotation(apply_seek(env, offender.id, net))
            offender.run()

        controls = {}
        controls[player_id] = action

        return with_offense_controls(env, controls)

    def get_env_factory(self):
        offender_position = (ENVIRONMENT_WIDTH / 2, ENVIRONMENT_HEIGHT / 2)
        angle = get_random_within_range(-math.pi, math.pi)
        ball_position = tuple(
            100 * get_unit_vector(angle) + np.array(offender_position)
        )
        ball_direction = -angle + get_random_within_range(math.pi / 4, -math.pi / 4)
        ball_speed = PLAYER_SHOT_SPEED

        def factory(offender_id):
            offender = Offender(offender_id, offender_position)
            ball = Ball(ball_position)
            ball.speed = ball_speed
            ball.direction = ball_direction
            return BluelockEnvironment(
                (ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT), [offender], [], ball
            )

        return factory

    def compute_fitness(self, genomes, config):
        episodes = 10
        dt, alotted = 15, 3500
        factories = [self.get_env_factory() for _ in range(episodes)]
        for id, genome in genomes:
            genome.fitness = 0
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            for env in [
                self.with_controls(factory(id), id, net) for factory in factories
            ]:
                offender: Offender = env.get_player(id)
                elapsed = 0
                dist_to_ball = float("inf")
                while elapsed < alotted and not offender.has_possession():
                    env.update(dt)
                    dist_to_ball = min(
                        dist_to_ball,
                        get_euclidean_dist(offender.position, env.ball.position),
                    )
                    elapsed += dt

                if not offender.has_possession():
                    genome.fitness -= dist_to_ball
                else:
                    genome.fitness += math.sqrt(alotted - elapsed)

        if self.should_visualize():
            dt = 7
            id, genome = max(genomes, key=lambda g: g[1].fitness)
            net = neat.nn.FeedForwardNetwork.create(genome, self.config)
            for factory in factories:
                env = self.with_controls(factory(id), id, net)
                vis = BluelockEnvironmentVisualizer(env)
                elapsed = 0
                while elapsed < alotted and not env.get_player(id).has_possession():
                    env.update(dt)
                    vis.draw()
                    elapsed += dt

        self.compute_fitness_call_count += 1

    def should_visualize(self):
        return self.compute_fitness_call_count % 2 == 0
