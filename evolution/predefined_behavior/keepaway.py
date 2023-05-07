import numpy as np
import math
import neat
import json
from environment.config import (
    ENVIRONMENT_HEIGHT,
    ENVIRONMENT_WIDTH,
    PLAYER_DEFENDER_SPEED,
)
from environment.core import BluelockEnvironment, Offender, Ball, Defender
from environment.defense.agent import with_policy_defense, naive_man_to_man
from evolution.task import EvolutionTask
from evolution.config import (
    CHECKPOINTS_PATH,
    MODELS_PATH,
    PLOTS_PATH,
    get_default_config,
)
from evolution.util import get_keepaway2v1_fitness, scale_to_env_dims
from visualization.visualizer import BluelockEnvironmentVisualizer
from util import Rect, get_random_point, get_beeline_orientation
from dataclasses import dataclass

# Predefined Behavior ANN's inputs
def get_passing_lane_creator_inputs(
    env: BluelockEnvironment, possessor_id: int, defender_id: int, offballer_id: int
):
    possessor, defender, offballer = env.get_players_by_ids(
        possessor_id, defender_id, offballer_id
    )
    defender_to_ball = scale_to_env_dims(env, possessor.position - defender.position)
    offballer_to_ball = scale_to_env_dims(env, possessor.position - offballer.position)
    inputs = [
        defender_to_ball[0],
        defender_to_ball[1],
        offballer_to_ball[0],
        offballer_to_ball[1],
    ]
    for point in Rect([0.0, 0.0], height=env.height, width=env.width).get_vertices():
        displacement_to_corner = scale_to_env_dims(
            env, np.array(point) - offballer.position
        )
        inputs.append(displacement_to_corner[0])
        inputs.append(displacement_to_corner[1])
    return inputs


def get_passing_lane_creator_recommendations(
    env: BluelockEnvironment,
    passing_lane_creator: neat.nn.FeedForwardNetwork,
    possessor_id: int,
    defender_id: int,
    offballer_id: int,
):
    inputs = get_passing_lane_creator_inputs(
        env, possessor_id, defender_id, offballer_id
    )
    pass_confidence, vx, vy = passing_lane_creator.activate(inputs)
    # clamped output is used so vx and vy will be [-1, 1]
    speed_magnitude = math.sqrt(vx**2 + vy**2) / math.sqrt(2)
    orientation = get_beeline_orientation([vx, vy])
    return pass_confidence > 0, speed_magnitude, orientation


def seek_ball(ball: Ball, seeker: Offender):
    angle_to_run_towards = get_beeline_orientation(ball.position - seeker.position)
    seeker.set_rotation(angle_to_run_towards)
    seeker.run()


def pass_ball(possessor: Offender, target: Offender):
    angle_to_pass = get_beeline_orientation(target.position - possessor.position)
    possessor.set_rotation(angle_to_pass)
    possessor.shoot()


@dataclass
class PredefinedBehaviorControlState:
    who_should_seek: int | None = None

    def should_seek(self, offender: Offender):
        return self.who_should_seek is not None and self.who_should_seek == offender.id


def with_predefined_pass_seek_behaviors(
    env: BluelockEnvironment, passing_lane_creator: neat.nn.FeedForwardNetwork
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
    state = PredefinedBehaviorControlState()

    def control():
        if does_offense_have_possession():
            possessor, offballer = get_roles()
            (
                should_pass,
                speed_magnitude,
                orientation,
            ) = get_passing_lane_creator_recommendations(
                env, passing_lane_creator, possessor.id, env.defense[0].id, offballer.id
            )
            offballer.set_rotation(orientation)
            offballer.run(speed_magnitude)
            if should_pass:
                pass_ball(possessor, offballer)
                state.who_should_seek = offballer.id
        else:
            for offender in env.offense:
                if state.should_seek(offender):
                    seek_ball(env.ball, offender)

    def new_update(*args, **kwargs):
        control()
        old_update(*args, **kwargs)

    env.update = new_update
    return env


TASK_NAME = "predefined_keepaway"


class PredefinedBehaviorKeepaway(EvolutionTask):
    def __init__(self, inital_difficulty: float = 0.5, is_dynamic: bool = False):
        tag = TASK_NAME
        if is_dynamic:
            tag = f"{TASK_NAME}_dynamic"
        config_file = get_default_config(f"{TASK_NAME}.ini")
        super().__init__(
            CHECKPOINTS_PATH, MODELS_PATH, PLOTS_PATH, config_file, TASK_NAME, tag=tag
        )
        self.is_dynamic = is_dynamic
        self.difficulty = inital_difficulty

    def get_episodes(self):
        envs = []
        for _ in range(10):
            defender_pos = (ENVIRONMENT_WIDTH // 2, 0)
            possessor_pos = (0, ENVIRONMENT_HEIGHT // 2)
            if self.is_dynamic:
                defender_pos = get_random_point(
                    x_max=ENVIRONMENT_WIDTH, y_max=ENVIRONMENT_HEIGHT
                )
                possessor_pos = get_random_point(
                    x_max=ENVIRONMENT_WIDTH, y_max=ENVIRONMENT_HEIGHT
                )

            possessor = Offender(1, possessor_pos)
            ball = Ball(possessor_pos)
            possessor.possess(ball)

            envs.append(
                with_policy_defense(
                    BluelockEnvironment(
                        (ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT),
                        [
                            possessor,
                            Offender(
                                2,
                                get_random_point(
                                    x_max=ENVIRONMENT_WIDTH, y_max=ENVIRONMENT_HEIGHT
                                ),
                            ),
                        ],
                        [
                            Defender(
                                3,
                                defender_pos,
                                top_speed=self.difficulty * PLAYER_DEFENDER_SPEED,
                            )
                        ],
                        ball,
                    ),
                    policy=naive_man_to_man,
                )
            )
        return envs

    def compute_fitness(self, genome, config) -> float:
        dt, allotted = 15, 24000
        fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        episodes = self.get_episodes()
        for env in episodes:
            env = with_predefined_pass_seek_behaviors(env, net)
            for elapsed in range(0, allotted, dt):
                if env.does_defense_have_possession():
                    break
                env.update(dt)

            fitness += get_keepaway2v1_fitness(elapsed / allotted)
        return fitness / len(episodes)


def evolve_predefined_behavior_keepaway():
    stats = {"difficulty": {}, "fitness": {}}
    task = PredefinedBehaviorKeepaway(is_dynamic=True)
    eval_count = 0
    for _, winner in enumerate(task.evolve(150, 5)):
        eval_count += 1
        fitness = task.compute_fitness(winner, task.config)
        print(
            f"{eval_count} test at difficulty {task.difficulty} resulted in fitness of {fitness}"
        )
        if fitness > 0.8:
            task.difficulty += 0.05
        stats["difficulty"][eval_count] = task.difficulty
        stats["fitness"][eval_count] = fitness
        with open(f"predefined_keepaway_dynamic_stats.json", "w") as f:
            json.dump(stats, f, indent=2, sort_keys=True)


def watch_predefined_behavior_keepaway():
    task = PredefinedBehaviorKeepaway(is_dynamic=True)
    best_passing_lane_creator = task.get_best_model()
    dt = 5
    for env in task.get_episodes():
        env = with_predefined_pass_seek_behaviors(env, best_passing_lane_creator)
        vis = BluelockEnvironmentVisualizer(env)
        while not env.does_defense_have_possession():
            env.update(dt)
            vis.draw()
