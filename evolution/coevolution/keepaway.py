import neat
import json
from environment.config import ENVIRONMENT_HEIGHT, ENVIRONMENT_WIDTH
from evolution.coevolution.task import CoevolutionTask
from evolution.config import (
    CHECKPOINTS_PATH,
    MODELS_PATH,
    PLOTS_PATH,
    get_default_config,
)
from evolution.util import get_keepaway2v1_fitness, get_keepaway2v1_env
from evolution.sequential.keepaway import with_fully_learned_behaviors
from util import get_random_point
from visualization.visualizer import BluelockEnvironmentVisualizer

TASK_NAME = "coevolved_keepaway"


class CoevolvedKeepaway(CoevolutionTask):
    def __init__(self, difficulty=0.5, is_dynamic=False):
        task_name = TASK_NAME
        if is_dynamic:
            task_name = f"{TASK_NAME}_dynamic"
        behaviors = ["seek", "pass", "find_space", "pass_evaluate"]
        configs = [get_default_config(f"{behavior}.ini") for behavior in behaviors]
        super().__init__(
            CHECKPOINTS_PATH, MODELS_PATH, PLOTS_PATH, configs, task_name, behaviors
        )
        self.difficulty = difficulty
        self.is_dynamic = is_dynamic

    def get_episodes(self):
        envs = []
        for _ in range(10):
            if self.is_dynamic:
                envs.append(
                    get_keepaway2v1_env(
                        self.difficulty,
                        defender_pos=get_random_point(
                            ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT
                        ),
                        possessor_pos=get_random_point(
                            ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT
                        ),
                    )
                )
            else:
                envs.append(get_keepaway2v1_env(self.difficulty))
        return envs

    def compute_fitness(self, genomes, configs) -> float:
        nets = []
        for genome, config in zip(genomes, configs):
            nets.append(neat.nn.FeedForwardNetwork.create(genome, config))

        seeker, passer, find_spacer, pass_evaluator = nets
        dt, allotted = 15, 24000
        episodes = self.get_episodes()
        fitness = 0
        for env in episodes:
            env = with_fully_learned_behaviors(
                env, seeker, passer, find_spacer, pass_evaluator
            )
            for elapsed in range(0, allotted, dt):
                if env.does_defense_have_possession():
                    break
                env.update(dt)

            fitness += get_keepaway2v1_fitness(elapsed / allotted)
        return fitness / len(episodes)


def coevolve_keepaway():
    stats = {"difficulty": {}, "fitness": {}}
    task = CoevolvedKeepaway()
    eval_count = 0
    for best_team in task.evolve(101, 5):
        eval_count += 1
        fitness = task.compute_fitness(best_team, task.configs)
        print(
            f"{eval_count} test at difficulty {task.difficulty} resulted in fitness of {fitness}"
        )
        if fitness > 0.8:
            task.difficulty += 0.05
        stats["difficulty"][eval_count] = task.difficulty
        stats["fitness"][eval_count] = fitness
        with open(f"coevolved_keepaway_stats.json", "w") as f:
            json.dump(stats, f, indent=2, sort_keys=True)


def watch_coevolved_keepaway():
    task = CoevolvedKeepaway()
    seeker, passer, find_spacer, pass_evaluator = task.load_best_team()
    dt = 5
    for env in task.get_episodes():
        env = with_fully_learned_behaviors(
            env, seeker, passer, find_spacer, pass_evaluator
        )
        vis = BluelockEnvironmentVisualizer(env)
        while not env.does_defense_have_possession():
            env.update(dt)
            vis.draw()
