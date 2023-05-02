import neat
import numpy as np
import evolution.visualize as visualize
import os
import math
from environment.core import BluelockEnvironment, Offender, Ball
from environment.config import ENVIRONMENT_HEIGHT, ENVIRONMENT_WIDTH, PLAYER_SHOT_SPEED
from evolution.config import CHECKPOINTS_PATH, MODELS_PATH, CONFIGS_PATH, PLOTS_PATH
from evolution.util import MostRecentHistoryRecorder, with_offense_controls
from visualization.visualizer import BluelockEnvironmentVisualizer
from util import (
    get_unit_vector,
    get_random_within_range,
    get_euclidean_dist,
    get_beeline_orientation,
)


class SequentialEvolutionTask:
    def __init__(
        self,
        checkpoint_dir: str,
        model_dir: str,
        plot_dir: str,
        config: neat.Config,
        task_name: str,
    ):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        self.checkpoint_path = os.path.join(checkpoint_dir, task_name)
        self.plot_path = os.path.join(plot_dir, task_name)
        self.model_path = os.path.join(model_dir, task_name)
        self.config = config

    def evolve(self, generations: int):
        population = neat.Population(self.config)
        if os.path.exists(self.checkpoint_path):
            population = MostRecentHistoryRecorder.restore_generation(
                self.checkpoint_path
            )
            generations -= population.generation
        population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)
        population.add_reporter(
            MostRecentHistoryRecorder(self.checkpoint_path, self.model_path)
        )
        winner = population.run(self.compute_fitness, n=generations)
        visualize.plot_stats(stats, filename=self.plot_path + "_fitness")
        visualize.draw_net(
            self.config, winner, fmt="png", filename=self.plot_path + "_net_arch"
        )

    def compute_fitness(self, genomes, config):
        pass

    def get_best_model(self):
        genome = MostRecentHistoryRecorder.load_best_genome(self.model_path)
        return neat.nn.FeedForwardNetwork.create(genome, self.config)


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
            vx, vy = net.activate(self.get_inputs(env, player_id))
            orientation = get_beeline_orientation(np.array([vx, vy]))
            offender.set_rotation(orientation)
            offender.run()

        controls = {}
        controls[player_id] = action

        return with_offense_controls(env, controls)

    def get_env_factory(self):
        offender_position = (ENVIRONMENT_WIDTH / 2, ENVIRONMENT_HEIGHT / 2)
        angle = get_random_within_range(-math.pi, math.pi)
        ball_position = tuple(80 * get_unit_vector(angle) + np.array(offender_position))
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

    def get_inputs(self, env: BluelockEnvironment, player_id: int):
        player = env.get_player(player_id)
        ball_velocity = get_unit_vector(env.ball.direction) * env.ball.speed
        return list(env.ball.position - player.position) + list(ball_velocity)

    def compute_fitness(self, genomes, config):
        episodes = 10
        dt, alotted = 15, 3000
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
        return self.compute_fitness_call_count % 10 == 0


def evolve_sequential():
    seek_ball = SeekBall()
    seek_ball.evolve(50)


def watch_sequential():
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
