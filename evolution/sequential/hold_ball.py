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
        self.compute_fitness_call_count = 0

    def with_controls(
        self,
        env: BluelockEnvironment,
        offender_id: int,
        defender_id: int,
        net: neat.nn.FeedForwardNetwork,
    ):
        def action(env: BluelockEnvironment, offender: Offender):
            vx, vy = net.activate(self.get_inputs(env, offender_id, defender_id))
            orientation = get_beeline_orientation(np.array([vx, vy]))
            offender.set_rotation(orientation)
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

    def get_inputs(self, env: BluelockEnvironment, offender_id: int, defender_id: int):
        offender, defender = env.get_player(offender_id), env.get_player(defender_id)
        dx, dy = offender.position - defender.position
        vx, vy = defender.speed * defender.tilt
        rect = Rect([0.0, 0.0], height=env.height, width=env.width)
        inputs = [dx, dy, vx, vy]
        for point in rect.get_vertices():
            pdx, pdy = np.array(point) - offender.position
            inputs.append(pdx)
            inputs.append(pdy)
        return inputs

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

        if self.should_visualize():
            dt = 7
            id, genome = max(genomes, key=lambda g: g[1].fitness)
            net = neat.nn.FeedForwardNetwork.create(genome, self.config)
            for factory in factories:
                env = self.with_controls(factory(id, defender_id), id, defender_id, net)
                vis = BluelockEnvironmentVisualizer(env)
                elapsed = 0
                while (
                    elapsed < alotted
                    and not env.get_player(defender_id).has_possession()
                ):
                    env.update(dt)
                    vis.draw()
                    elapsed += dt

        self.compute_fitness_call_count += 1

    def should_visualize(self):
        return self.compute_fitness_call_count % 10 == 0
