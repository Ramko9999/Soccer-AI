import numpy as np
import random
import pickle
import gzip
import math
import neat
from environment.core import BluelockEnvironment, Offender
import evolution.visualize as visualize
from neat.population import Population
from neat.reporting import BaseReporter
from typing import Callable


class MostRecentHistoryRecorder(BaseReporter):
    def __init__(self, checkpoint_file_path: str, best_save_path: str):
        super().__init__()
        self.checkpoint_file_path = checkpoint_file_path
        self.best_save_path = best_save_path
        self.generation = 0

    def start_generation(self, generation):
        self.generation = generation

    def post_evaluate(self, config, population, species, best_genome):
        with open(self.best_save_path, "wb") as f:
            pickle.dump(best_genome, f)

    def end_generation(self, config, population, species_set):
        with open(self.checkpoint_file_path, "wb") as f:
            data = (self.generation, config, population, species_set, random.getstate())
            f.write(gzip.compress(pickle.dumps(data)))

    @staticmethod
    def load_best_genome(save_path: str):
        with open(save_path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def restore_generation(save_path: str):
        with open(save_path, "rb") as f:
            generation, config, population, species, random_state = pickle.loads(
                gzip.decompress(f.read())
            )
            random.setstate(random_state)
            return Population(config, (population, species, generation))


class EvolutionVisualizer(neat.StatisticsReporter):
    def __init__(self, output_prefix: str):
        super().__init__()
        self.output_prefix = output_prefix

    def post_evaluate(self, config, population, species, best_genome):
        super().post_evaluate(config, population, species, best_genome)
        net_image = f"{self.output_prefix}_net"
        visualize.draw_net(config, best_genome, filename=net_image, fmt="png")

    def end_generation(self, config, population, species_set):
        super().end_generation(config, population, species_set)
        fitness_plot = f"{self.output_prefix}_fitness"
        visualize.plot_stats(self, filename=fitness_plot)


def get_keepaway2v1_fitness(survival_time_ratio: float):
    return math.pow(5, 1 + survival_time_ratio) / 25


def scale_to_env_dims(env: BluelockEnvironment, displacement: np.ndarray):
    return np.array([displacement[0] / env.width, displacement[1] / env.height])


OffenseControl = Callable[[BluelockEnvironment, Offender], None]
OffenseControls = list[tuple[int, OffenseControl]]


def with_offense_controls(
    env: BluelockEnvironment, controls: OffenseControls
) -> BluelockEnvironment:
    control_map = {}
    for id, control in controls:
        control_map[id] = control
    old_update = env.update

    def new_update(*args, **kwargs):
        for player_id in control_map:
            control_map[player_id](env, env.get_player(player_id))
        old_update(*args, **kwargs)

    env.update = new_update
    return env
