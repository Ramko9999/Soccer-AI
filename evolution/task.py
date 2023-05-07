import neat
import os
import multiprocessing
import random
import time
from evolution.util import MostRecentHistoryRecorder, EvolutionVisualizer


class EvolutionTask:
    def __init__(
        self,
        checkpoint_dir: str,
        model_dir: str,
        plot_dir: str,
        config: neat.Config,
        task_name: str,
        cpus: int = multiprocessing.cpu_count(),
        tag: str | None = None,
    ):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        self.tag = task_name
        if tag is not None:
            self.tag = tag
        self.checkpoint_path = os.path.join(checkpoint_dir, self.tag)
        self.plot_path = os.path.join(plot_dir, self.tag)
        self.model_path = os.path.join(model_dir, self.tag)
        self.config = config
        self.task_name = task_name
        self.cpus = cpus
        self.seed = self.get_seed()

    def evolve(self, generations: int, generation_step_size: int = 5):
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

        print(f"Evolving {self.task_name} with {self.cpus} cpus")

        population.add_reporter(neat.StdOutReporter(True))
        population.add_reporter(EvolutionVisualizer(output_prefix=self.plot_path))
        population.add_reporter(
            MostRecentHistoryRecorder(self.checkpoint_path, self.model_path)
        )
        pe = neat.ParallelEvaluator(self.cpus, self.eval_genome)

        def evaluate(genomes, config):
            self.seed = self.get_seed()
            pe.evaluate(genomes, config)

        while generations > 0:
            step_size = min(generations, generation_step_size)
            yield population.run(evaluate, n=step_size)
            generations -= step_size
        return

    # override
    def compute_fitness(self, genome, config) -> float:
        return float("-inf")

    def eval_genome(self, genome, config) -> float:
        random.seed(self.seed)
        return self.compute_fitness(genome, config)

    def get_best_model(self):
        genome = MostRecentHistoryRecorder.load_best_genome(self.model_path)
        return neat.nn.FeedForwardNetwork.create(genome, self.config)

    def get_seed(self):
        return int(time.time())
