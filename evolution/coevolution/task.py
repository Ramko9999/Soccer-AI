import neat
import random
import os
import multiprocessing
import time
import pickle
import gzip
import sys
from evolution.util import EvolutionVisualizer


class CoevolutionTask:
    def __init__(
        self,
        checkpoint_dir: str,
        model_dir: str,
        plot_dir: str,
        configs: list[neat.Config],
        task_name: str,
        population_tags: list[str],
        cpus: int = multiprocessing.cpu_count(),
    ):

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        self.task_name = task_name
        self.population_tags = population_tags
        self.checkpoint_path = os.path.join(checkpoint_dir, self.task_name)
        self.plot_path = os.path.join(plot_dir, self.task_name)
        self.model_path = os.path.join(model_dir, self.task_name)
        self.configs = configs
        self.cpus = cpus
        self.seed = self.get_seed()

    def evolve(self, generations: int, generation_step_size: int = 5):
        def noop_fitness(genome, config):
            pass

        populations = [neat.Population(config) for config in self.configs]
        start_generation = 0
        if os.path.exists(self.checkpoint_path):
            populations, start_generation = self.load_checkpoint()
            for tag, population in zip(self.population_tags, populations):
                print(f"{tag} {len(population.population)}")
            print(f"Loading checkpoint, evolving from generation {start_generation}")

        for tag, population in zip(self.population_tags, populations):
            population.add_reporter(EvolutionVisualizer(f"{self.plot_path}_{tag})"))

        for generation in range(start_generation, generations):
            self.checkpoint(populations, generation)
            for tag, population in zip(self.population_tags, populations):
                print(f"{generation}: {tag} {len(population.population)}")
            start = time.monotonic()
            teams = self.get_teams(populations)
            performances = self.evaluate_teams(teams, self.configs)
            best_performing_team, best_performance = teams[0], float("-inf")
            for team, performance in zip(teams, performances):
                for individual in team:
                    individual.fitness += performance
                if performance > best_performance:
                    best_performing_team, best_performance = team, performance

            self.save_best_team(best_performing_team)
            print(f"Fitness evaluation for {generation} generation finished in {round(time.monotonic() - start, 2)}s")
            print(
                f"The best performing team of {generation} has fitness {best_performance}"
            )
            for population in populations:
                population.run(noop_fitness, n=1)

            if generation > 0 and generation % generation_step_size == 0:
                yield best_performing_team

    def get_teams(self, populations: list[neat.Population], participations=5):
        def pick_random_individual(individuals):
            pick = random.randint(0, len(individuals) - 1)
            # swap pick with the end
            individuals[-1], individuals[pick] = individuals[pick], individuals[-1]
            return individuals.pop()

        # assumes all the populations have the same size
        min_pop_size = float("inf")
        for pop in populations:
            min_pop_size = min(len(pop.population), min_pop_size)
        pool = {}
        for population_id, population in enumerate(populations):
            pool[population_id] = []
            individuals = population.population
            count = 0
            for id in individuals:
                individuals[id].fitness = 0
                if count < min_pop_size:
                    for _ in range(participations):
                        pool[population_id].append(individuals[id])
                    count += 1


        total_teams = min_pop_size * participations
        teams = []
        for _ in range(total_teams):
            team = []
            for pop_id in pool:
                team.append(pick_random_individual(pool[pop_id]))
            teams.append(team)
    
        return teams

    def checkpoint(self, populations: list[neat.Population], generation: int):
        pop_data = []
        for population in populations:
            pop_data.append((population.population, population.config, population.species))
        
        data = (generation, pop_data, random.getstate())
        with open(self.checkpoint_path, "wb") as f:
            f.write(gzip.compress(pickle.dumps(data)))
        
    def load_checkpoint(self):
        with open(self.checkpoint_path, "rb") as f:
            generation, pop_data, random_state = pickle.loads(gzip.decompress(f.read()))
            populations = []
            for individuals, config, species in pop_data:
                populations.append(neat.Population(config, (individuals, species, 0)))
            random.setstate(random_state)
            return populations, generation

    def save_best_team(self, genomes):
        with open(self.model_path, "wb") as f:
            f.write(gzip.compress(pickle.dumps(genomes)))

    def load_best_team(self):
        with open(self.model_path, "rb") as f:
            genomes = pickle.loads(gzip.decompress(f.read()))
            nets = []
            for genome, config in zip(genomes, self.configs):
                nets.append(neat.nn.FeedForwardNetwork.create(genome, config))
            return nets

    def compute_fitness(self, genomes, configs) -> float:
        return float("-inf")

    def evaluate_teams(self, teams, configs) -> list[float]:
        self.seed = self.get_seed()
        performances = []
        with multiprocessing.Pool(processes=self.cpus) as pool:
            evals = []
            for team in teams:
                eval = pool.apply_async(self.evaluate_team, (team, configs))
                evals.append(eval)
            for _, eval in enumerate(evals):
                performances.append(eval.get(None))
        return performances

    def evaluate_team(self, team, configs) -> float:
        random.seed(self.seed)
        return self.compute_fitness(team, configs)

    def get_seed(self):
        return int(time.time())
