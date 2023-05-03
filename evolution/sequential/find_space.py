import os
import neat
import numpy as np
from environment.core import BluelockEnvironment, Offender, Defender, Ball
from environment.config import (
    ENVIRONMENT_HEIGHT,
    ENVIRONMENT_WIDTH,
)
from environment.defense.agent import with_policy_defense
from environment.defense.policy import naive_man_to_man
from evolution.util import with_offense_controls
from evolution.sequential.task import SequentialEvolutionTask
from evolution.sequential.seek_ball import apply_seek
from evolution.sequential.hold_ball import apply_hold_ball
from evolution.sequential.pass_evaluate import apply_pass_evaluate
from evolution.config import CHECKPOINTS_PATH, CONFIGS_PATH, PLOTS_PATH, MODELS_PATH
from util import (
    get_random_point,
    get_random_within_range,
    get_beeline_orientation,
    Rect,
)
from visualization.visualizer import BluelockEnvironmentVisualizer


def get_find_space_inputs(
    env: BluelockEnvironment, passer_id: int, defender_id: int, space_finder_id: int
):
    passer = env.get_player(passer_id)
    space_finder = env.get_player(space_finder_id)
    defender = env.get_player(defender_id)

    dx, dy = space_finder.position - passer.position
    ddx, ddy = space_finder.position - defender.position
    rect = Rect([0.0, 0.0], height=env.height, width=env.width)
    inputs = [dx, dy, ddx, ddy]
    for point in rect.get_vertices():
        pdx, pdy = np.array(point) - space_finder.position
        inputs.append(pdx)
        inputs.append(pdy)
    return inputs


def apply_find_space(
    env: BluelockEnvironment,
    passer_id: int,
    target_id: int,
    defender_id: int,
    find_space_model: neat.nn.FeedForwardNetwork,
):
    inputs = get_find_space_inputs(env, passer_id, target_id, defender_id)
    orientation = get_beeline_orientation(find_space_model.activate(inputs))
    return orientation


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

    def get_env_factory(self):
        passer_pos = get_random_point(ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT)
        find_spacer_pos = get_random_point(ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT)
        defender_pos = get_random_within_range(ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT)

        def factory(passer_id, find_spacer_id, defender_id):
            passer = Offender(passer_id, passer_pos)
            space_finder = Offender(find_spacer_id, find_spacer_pos)
            defender = Defender(defender_id, defender_pos)
            ball = Ball(passer_pos)
            passer.possess(ball)

            return with_policy_defense(
                BluelockEnvironment(
                    (ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT),
                    [passer, space_finder],
                    [defender],
                    ball,
                ),
                policy=naive_man_to_man,
            )

        return factory

    def compute_fitness(self, genomes, config):
        episodes = 5
        passer_id, find_spacer_id, defender_id = 1, 2, 3
        factories = [self.get_env_factory() for _ in range(episodes)]
        dt, alotted = 5, 12000
        for _, genome in genomes:
            genome.fitness = 0
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            for factory in factories:
                env = factory(passer_id, find_spacer_id, defender_id)
                passer, target, defender = (
                    env.get_player(passer_id),
                    env.get_player(find_spacer_id),
                    env.get_player(defender_id),
                )

                should_seek = [None]

                def action(env: BluelockEnvironment, offender: Offender):
                    other_offender = list(
                        filter(lambda o: o.id != offender.id, env.offense)
                    )[0]
                    if offender.has_possession():
                        if should_seek[0] == offender.id:
                            should_seek[0] = None
                        confident = apply_pass_evaluate(
                            env, other_offender.id, defender_id, self.pass_evaluator
                        )
                        if confident:
                            offender.set_rotation(
                                get_beeline_orientation(
                                    other_offender.position - offender.position
                                )
                            )
                            offender.shoot()
                            should_seek[0] = other_offender.id
                        else:
                            offender.set_rotation(
                                apply_hold_ball(
                                    env, offender.id, defender_id, self.holder
                                )
                            )
                            offender.run()
                    else:
                        if should_seek[0] == offender.id:
                            offender.set_rotation(
                                apply_seek(env, offender.id, self.seeker)
                            )
                            offender.run()
                        else:
                            if other_offender.has_possession():
                                offender.set_rotation(
                                    apply_find_space(
                                        env,
                                        other_offender.id,
                                        offender.id,
                                        defender_id,
                                        net,
                                    )
                                )
                                offender.run()

                controls = {}
                controls[passer_id] = action
                controls[find_spacer_id] = action
                env = with_offense_controls(env, controls)
                vis = BluelockEnvironmentVisualizer(env)
                elapsed = 0
                while elapsed < alotted and not defender.has_possession():
                    env.update(dt)
                    vis.draw()
                    elapsed += dt
