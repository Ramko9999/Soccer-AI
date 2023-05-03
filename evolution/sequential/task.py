import neat
import os
from evolution.util import MostRecentHistoryRecorder, EvolutionVisualizer


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
        self.task_name = task_name

    def evolve(self, generations: int):
        population = neat.Population(self.config)
        if os.path.exists(self.checkpoint_path):
            population = MostRecentHistoryRecorder.restore_generation(
                self.checkpoint_path
            )
            generations -= population.generation + 1

        if generations <= 0:
            print(
                f"The desired # of generations have already been reached. Not training {self.task_name}"
            )
            return

        population.add_reporter(neat.StdOutReporter(True))
        population.add_reporter(EvolutionVisualizer(output_prefix=self.plot_path))
        population.add_reporter(
            MostRecentHistoryRecorder(self.checkpoint_path, self.model_path)
        )
        population.run(self.compute_fitness, n=generations)

    def compute_fitness(self, genomes, config):
        pass

    def get_best_model(self):
        genome = MostRecentHistoryRecorder.load_best_genome(self.model_path)
        return neat.nn.FeedForwardNetwork.create(genome, self.config)
