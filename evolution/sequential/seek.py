import neat
import numpy as np
import math
from environment.core import BluelockEnvironment, Offender, Ball
from environment.config import ENVIRONMENT_HEIGHT, ENVIRONMENT_WIDTH, PLAYER_SHOT_SPEED
from evolution.config import (
    CHECKPOINTS_PATH,
    MODELS_PATH,
    get_default_config,
    PLOTS_PATH,
)
from evolution.task import EvolutionTask
from evolution.util import  scale_to_env_dims
from visualization.visualizer import BluelockEnvironmentVisualizer
from util import (
    get_unit_vector,
    get_beeline_orientation,
    get_euclidean_dist,
    get_random_point,
    get_random_within_range
)

def get_seek_inputs(env: BluelockEnvironment, seeker_id: int):
    ball = env.ball
    seeker = env.get_player(seeker_id)
    displacement_to_ball = scale_to_env_dims(env, ball.position - seeker.position)
    ball_velocity = ball.speed * get_unit_vector(ball.direction)
    return [displacement_to_ball[0], displacement_to_ball[1], ball_velocity[0], ball_velocity[1]]

def get_seek_outputs(env: BluelockEnvironment, seeker: neat.nn.FeedForwardNetwork, seeker_id: int):
    vx, vy = seeker.activate(get_seek_inputs(env, seeker_id))
    speed_magnitude = math.sqrt(vx**2 + vy**2)/math.sqrt(2)
    orientation = get_beeline_orientation(np.array([vx, vy]))
    return speed_magnitude, orientation
    

def with_seeker(env: BluelockEnvironment, seeker_net: neat.nn.FeedForwardNetwork, seeker_id: int):
    old_update = env.update

    def new_update(*args, **kwargs):
        seeker = env.get_player(seeker_id)
        speed_magnitude, orientation = get_seek_outputs(env, seeker_net, seeker.id)
        seeker.set_rotation(orientation)
        seeker.run(speed_magnitude)
        old_update(*args, **kwargs)

    env.update = new_update
    return env

TASK_NAME = "seek"
class Seek(EvolutionTask):
    def __init__(self):
        config_file = get_default_config(f"{TASK_NAME}.ini")
        super().__init__(
            CHECKPOINTS_PATH, MODELS_PATH, PLOTS_PATH, config_file, TASK_NAME
        )
        self.offballer_id = 1

    def get_episodes(self) -> list[BluelockEnvironment]:
        envs = []
        for _ in range(20):
            offender = Offender(
                self.offballer_id,
                get_random_point(ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT),
            )
            ball = Ball(get_random_point(ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT))
            ball.direction = get_random_within_range(math.pi/20, -math.pi/20) + get_beeline_orientation(offender.position - ball.position)
            ball.speed = get_random_within_range(1, 0.5) * PLAYER_SHOT_SPEED
            envs.append(
                BluelockEnvironment(
                    (ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT), [offender], [], ball
                )
            )
        return envs

    def compute_fitness(self, genome, config):
        dt, allotted = 15, 6000
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        episodes = self.get_episodes()
        fitness = 0
        for env in episodes:
            env = with_seeker(env, net, self.offballer_id)
            offender = env.get_player(self.offballer_id)
            origin_position = offender.position
            dist_to_ball = float("inf")
            for _ in range(0, allotted, dt):
                if offender.has_possession():
                    break
                dist_to_ball = min(
                    dist_to_ball,
                    get_euclidean_dist(env.ball.position, offender.position),
                )
                env.update(dt)

            dist_traveled = get_euclidean_dist(offender.position, origin_position)
            award = 0
            max_dist_possible = math.sqrt(env.width**2 + env.height**2)
            if offender.has_possession():
                award = 2 - (dist_traveled / max_dist_possible)
            else:
                award = 1 - (dist_to_ball / max_dist_possible)
            fitness += award
        return fitness / len(episodes)

def evolve_seek():
    seek = Seek()
    for _ in seek.evolve(100, 100):
        pass

def watch_seek():
    seek = Seek()
    best_seeker = seek.get_best_model()
    for env in seek.get_episodes():
        dt, allotted = 5, 6000
        env = with_seeker(env, best_seeker, seek.offballer_id)
        vis = BluelockEnvironmentVisualizer(env)
        for _ in range(0, allotted, dt):
            env.update(dt)
            vis.draw()
