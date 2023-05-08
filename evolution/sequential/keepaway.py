import neat
from environment.core import Offender, BluelockEnvironment
from evolution.sequential.seek import evolve_seek, watch_seek, do_seek
from evolution.sequential.pass_ball import evolve_pass, watch_pass, make_pass
from evolution.sequential.find_space import go_to_space
from evolution.util import scale_to_env_dims
from dataclasses import dataclass


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


def evolve_sequential_keepaway():
    evolve_seek()
    evolve_pass()


def watch_sequential_keepaway():
    # watch_seek()
    watch_pass()
