import os
import neat
import math
from environment.core import BluelockEnvironment, Offender, Defender, Ball, Player
from environment.config import (
    ENVIRONMENT_HEIGHT,
    ENVIRONMENT_WIDTH,
    PLAYER_DEFENDER_SPEED,
)
from environment.defense.agent import with_policy_defense, naive_man_to_man
from evolution.agent import with_sequentially_evolved_offense
from evolution.sequential.task import SequentialEvolutionTask
from evolution.config import CHECKPOINTS_PATH, CONFIGS_PATH, PLOTS_PATH, MODELS_PATH
from dataclasses import dataclass


@dataclass
class FindSpaceEpisodeDataTracker:
    posession_history: list[int]


def attach_possession_tracking(ball: Ball, tracker: FindSpaceEpisodeDataTracker):
    old_possessed_by = ball.possessed_by

    def new_possessed_by(player: Player):
        tracker.posession_history.append(player.id)
        old_possessed_by(player)

    ball.possessed_by = new_possessed_by


class FindSpace(SequentialEvolutionTask):
    def __init__(
        self,
        seeker: neat.nn.FeedForwardNetwork,
        holder: neat.nn.FeedForwardNetwork,
        pass_evaluator: neat.nn.FeedForwardNetwork,
    ):
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            os.path.join(CONFIGS_PATH, "find_space.ini"),
        )
        super().__init__(
            CHECKPOINTS_PATH, MODELS_PATH, PLOTS_PATH, config, "find_space"
        )
        self.seeker = seeker
        self.holder = holder
        self.pass_evaluator = pass_evaluator
        self.off1_id, self.off2_id, self.dfn1_id = 1, 2, 3

    def fitness_func(self, survival_time_ratio):
        return math.pow(10000, 1 + survival_time_ratio) / 10000

    def get_episodes_for_eval(self):
        env_width, env_height = 2 * ENVIRONMENT_WIDTH // 3, 2 * ENVIRONMENT_HEIGHT // 3
        offense_spawn_locations = [
            (0, 0),
            (env_width, 0),
            (env_width, env_height),
            (0, env_height),
        ]
        positions = []
        for i in range(len(offense_spawn_locations)):
            for j in range(len(offense_spawn_locations)):
                if i != j:
                    positions.append(
                        (offense_spawn_locations[i], offense_spawn_locations[j])
                    )
        envs = []
        for off1_pos, off2_pos in positions:
            off1 = Offender(self.off1_id, off1_pos)
            off2 = Offender(self.off2_id, off2_pos)
            dfn1 = Defender(self.dfn1_id, (env_width // 2, env_height // 2))
            dfn1.top_speed = PLAYER_DEFENDER_SPEED / 2
            ball = Ball(off1_pos)
            off1.possess(ball)
            env = with_policy_defense(
                BluelockEnvironment(
                    (env_width, env_height),
                    [off1, off2],
                    [dfn1],
                    ball,
                ),
                policy=naive_man_to_man,
            )
            envs.append(env)
        return envs

    def compute_fitness(self, genome, config):
        dt, alotted = 20, 18000
        fitness = 0
        episodes = self.get_episodes_for_eval()
        for env in episodes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            env, _ = with_sequentially_evolved_offense(
                env, self.seeker, self.holder, self.pass_evaluator, net
            )
            for elapsed in range(0, alotted, dt):
                if env.get_player(self.dfn1_id).has_possession():
                    break
                env.update(dt)
            fitness += self.fitness_func(elapsed / alotted)
        return fitness / len(episodes)
