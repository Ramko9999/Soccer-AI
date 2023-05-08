import neat
import json
from environment.config import ENVIRONMENT_HEIGHT, ENVIRONMENT_WIDTH
from environment.core import Offender, BluelockEnvironment
from evolution.config import (
    CHECKPOINTS_PATH,
    MODELS_PATH,
    PLOTS_PATH,
    get_default_config,
)
from evolution.sequential.seek import evolve_seek, watch_seek, do_seek, Seek
from evolution.sequential.pass_ball import evolve_pass, watch_pass, make_pass, Pass
from evolution.sequential.find_space import (
    evolve_find_space,
    watch_find_space,
    go_to_space,
    FindSpace,
)
from evolution.util import (
    scale_to_env_dims,
    get_keepaway2v1_env,
    get_keepaway2v1_fitness,
    get_random_point,
)
from evolution.task import EvolutionTask
from dataclasses import dataclass
from visualization.visualizer import BluelockEnvironmentVisualizer


def get_pass_evaluate_inputs(
    env: BluelockEnvironment, possessor_id: int, defender_id: int, offballer_id: int
):
    possessor, defender, offballer = env.get_players_by_ids(
        possessor_id, defender_id, offballer_id
    )
    defender_disp_to_ball = scale_to_env_dims(
        env, possessor.position - defender.position
    )
    offballer_disp_to_ball = scale_to_env_dims(
        env, possessor.position - offballer.position
    )
    return list(defender_disp_to_ball) + list(offballer_disp_to_ball)


def should_pass(
    env: BluelockEnvironment,
    evaluator_net: neat.nn.FeedForwardNetwork,
    possessor_id: int,
    defender_id: int,
    offballer_id: int,
):
    confidence = evaluator_net.activate(
        get_pass_evaluate_inputs(env, possessor_id, defender_id, offballer_id)
    )[0]
    return confidence > 0.5


@dataclass
class FullyLearnedBehaviorsControlState:
    who_should_seek: int | None = None

    def should_seek(self, offender: Offender):
        return self.who_should_seek is not None and self.who_should_seek == offender.id


def with_fully_learned_behaviors(
    env: BluelockEnvironment,
    seeker: neat.nn.FeedForwardNetwork,
    passer: neat.nn.FeedForwardNetwork,
    find_spacer: neat.nn.FeedForwardNetwork,
    pass_evaluator: neat.nn.FeedForwardNetwork,
):
    # hard coded 2 v 1
    def does_offense_have_possession():
        for offender in env.offense:
            if offender.has_possession():
                return True
        return False

    def get_roles():
        if env.offense[0].has_possession():
            return env.offense
        return env.offense[1], env.offense[0]

    old_update = env.update
    state = FullyLearnedBehaviorsControlState()

    def control():
        if does_offense_have_possession():
            possessor, offballer = get_roles()
            pass_confident = should_pass(
                env, pass_evaluator, possessor.id, env.defense[0].id, offballer.id
            )
            go_to_space(env, find_spacer, possessor.id, env.defense[0].id, offballer.id)
            if pass_confident:
                make_pass(env, passer, possessor.id, offballer.id)
                state.who_should_seek = offballer.id
        else:
            for offender in env.offense:
                if state.should_seek(offender):
                    do_seek(env, seeker, offender.id)

    def new_update(*args, **kwargs):
        control()
        old_update(*args, **kwargs)

    env.update = new_update
    return env


TASK_NAME = "pass_evaluate"


class SequentialKeepaway(EvolutionTask):
    def __init__(
        self,
        seeker: neat.nn.FeedForwardNetwork,
        passer: neat.nn.FeedForwardNetwork,
        spacer: neat.nn.FeedForwardNetwork,
        is_dynamic=False,
        difficulty=0.5,
    ):
        tag = TASK_NAME
        if is_dynamic:
            tag = f"{TASK_NAME}_dynamic"
        super().__init__(
            CHECKPOINTS_PATH,
            MODELS_PATH,
            PLOTS_PATH,
            tag,
            get_default_config(f"{TASK_NAME}.ini"),
        )
        self.seeker = seeker
        self.passer = passer
        self.spacer = spacer
        self.is_dynamic = is_dynamic
        self.difficulty = difficulty

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

    def compute_fitness(self, genome, config) -> float:
        dt, allotted = 15, 24000
        fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        episodes = self.get_episodes()
        for env in episodes:
            env = with_fully_learned_behaviors(
                env, self.seeker, self.passer, self.spacer, net
            )
            for elapsed in range(0, allotted, dt):
                if env.does_defense_have_possession():
                    break
                env.update(dt)

            fitness += get_keepaway2v1_fitness(elapsed / allotted)
        return fitness / len(episodes)


def evolve_pass_evaluator():
    stats = {"difficulty": {}, "fitness": {}}
    seek = Seek()
    pass_ball = Pass(seek.get_best_model())
    find_space = FindSpace(seek.get_best_model(), pass_ball.get_best_model())
    task = SequentialKeepaway(
        seek.get_best_model(),
        pass_ball.get_best_model(),
        find_space.get_best_model(),
        is_dynamic=True,
    )
    eval_count = 0
    for _, winner in enumerate(task.evolve(100, 5)):
        eval_count += 1
        fitness = task.compute_fitness(winner, task.config)
        print(
            f"{eval_count} test at difficulty {task.difficulty} resulted in fitness of {fitness}"
        )
        if fitness > 0.8:
            task.difficulty += 0.05
        stats["difficulty"][eval_count] = task.difficulty
        stats["fitness"][eval_count] = fitness
        with open(f"sequential_keepaway_dynamic_stats.json", "w") as f:
            json.dump(stats, f, indent=2, sort_keys=True)


def watch_pass_evaluator():
    seek = Seek()
    pass_ball = Pass(seek.get_best_model())
    find_space = FindSpace(seek.get_best_model(), pass_ball.get_best_model())
    pass_evaluator = SequentialKeepaway(
        seek.get_best_model(),
        pass_ball.get_best_model(),
        find_space.get_best_model(),
        is_dynamic=True,
    )
    dt = 5
    for env in pass_evaluator.get_episodes():
        env = with_fully_learned_behaviors(
            env,
            seek.get_best_model(),
            pass_ball.get_best_model(),
            find_space.get_best_model(),
            pass_evaluator.get_best_model(),
        )
        vis = BluelockEnvironmentVisualizer(env)
        while not env.does_defense_have_possession():
            env.update(dt)
            vis.draw()


def evolve_sequential_keepaway():
    evolve_seek()
    evolve_pass()
    evolve_find_space()
    evolve_pass_evaluator()


def watch_sequential_keepaway():
    # watch_seek()
    # watch_pass()
    # watch_find_space()
    watch_pass_evaluator()
