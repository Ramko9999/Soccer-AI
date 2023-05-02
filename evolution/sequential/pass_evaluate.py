import os
import neat
import math
from environment.core import BluelockEnvironment, Offender, Defender, Ball
from environment.config import (
    PLAYER_SHOT_SPEED,
    BALL_FRICTION,
    ENVIRONMENT_HEIGHT,
    ENVIRONMENT_WIDTH,
)
from environment.defense.agent import with_policy_defense
from environment.defense.policy import naive_man_to_man
from evolution.util import with_offense_controls
from evolution.sequential.task import SequentialEvolutionTask
from evolution.sequential.seek_ball import apply_seek
from evolution.config import CHECKPOINTS_PATH, CONFIGS_PATH, PLOTS_PATH, MODELS_PATH
from util import (
    get_random_within_range,
    get_unit_vector,
    get_beeline_orientation,
    get_euclidean_dist,
)


def get_pass_evaluate_inputs(
    env: BluelockEnvironment, target_id: int, defender_id: int
):
    passer = env.ball.possessor
    target = env.get_player(target_id)
    defender = env.get_player(defender_id)

    odx, ody = target.position - passer.position
    ddx, ddy = target.position - defender.position
    vx, vy = PLAYER_SHOT_SPEED * passer.tilt
    return [odx, ody, ddx, ddy, vx, vy, BALL_FRICTION]


def apply_pass_evaluate(
    env: BluelockEnvironment,
    target_id: int,
    defender_id: int,
    pass_eval_model: neat.nn.FeedForwardNetwork,
):
    inputs = get_pass_evaluate_inputs(env, target_id, defender_id)
    confidence = pass_eval_model.activate(inputs)[0]
    return confidence > 0.5


class PassEvaluate(SequentialEvolutionTask):
    def __init__(self, seeker_model: neat.nn.FeedForwardNetwork):
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            os.path.join(CONFIGS_PATH, "pass_evaluate.ini"),
        )
        super().__init__(
            CHECKPOINTS_PATH, MODELS_PATH, PLOTS_PATH, config, "pass_evaluate"
        )
        self.seeker_model = seeker_model

    def get_env_factory(self):
        origin = (ENVIRONMENT_WIDTH / 2, ENVIRONMENT_HEIGHT / 2)
        pass_zone_radius = 200
        passer_pos = (
            pass_zone_radius
            * get_unit_vector(get_random_within_range(math.pi, -math.pi))
            + origin
        )
        target_pos = (
            pass_zone_radius
            * get_unit_vector(get_random_within_range(math.pi, -math.pi))
            + origin
        )
        defender_pos = (
            get_random_within_range(0, 0)
            * get_unit_vector(get_random_within_range(math.pi, -math.pi))
            + origin
        )

        def factory(passer_id, target_id, defender_id):
            passer = Offender(passer_id, passer_pos)
            target = Offender(target_id, target_pos)
            defender = Defender(defender_id, defender_pos)
            ball = Ball(passer_pos)
            passer.possess(ball)

            return with_policy_defense(
                BluelockEnvironment(
                    (ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT),
                    [passer, target],
                    [defender],
                    ball,
                ),
                policy=naive_man_to_man,
            )

        return factory

    def compute_fitness(self, genomes, config):
        episodes = 30
        passer_id, target_id, defender_id = 1, 2, 3
        factories = [self.get_env_factory() for _ in range(episodes)]
        dt, alotted = 15, 2000
        for _, genome in genomes:
            genome.fitness = 0
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            for factory in factories:
                env = factory(passer_id, target_id, defender_id)
                passer, target, defender = (
                    env.get_player(passer_id),
                    env.get_player(target_id),
                    env.get_player(defender_id),
                )

                def action(env: BluelockEnvironment, offender: Offender):
                    apply_seek(env, offender.id, self.seeker_model)

                controls = {}
                controls[target_id] = action
                env = with_offense_controls(env, controls)

                should_pass = apply_pass_evaluate(env, target_id, defender_id, net)

                # todo(): use passing model here
                passer.set_rotation(
                    get_beeline_orientation(target.position - passer.position)
                )
                passer.shoot()
                elapsed = 0
                target_dist_to_ball = defender_dist_to_ball = float("inf")
                while elapsed < alotted and not (
                    target.has_possession() or defender.has_possession()
                ):
                    env.update(dt)
                    elapsed += dt
                    target_dist_to_ball = min(
                        target_dist_to_ball,
                        get_euclidean_dist(target.position, env.ball.position),
                    )
                    defender_dist_to_ball = min(
                        defender_dist_to_ball,
                        get_euclidean_dist(defender.position, env.ball.position),
                    )

                if should_pass:
                    if target.has_possession():
                        genome.fitness += math.sqrt(defender_dist_to_ball)
                    elif defender.has_possession():
                        genome.fitness -= 2 * math.sqrt(target_dist_to_ball)
                else:
                    if defender.has_possession():
                        genome.fitness += math.sqrt(target_dist_to_ball)
                    elif target.has_possession():
                        genome.fitness -= 2 * math.sqrt(defender_dist_to_ball)
