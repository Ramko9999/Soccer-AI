import neat
from environment.core import BluelockEnvironment, Offender, Ball
from evolution.predefined_behavior.keepaway import (
    get_passing_lane_creator_recommendations,
)
from evolution.sequential.inputs import apply_seek, apply_hold_ball, apply_find_space
from dataclasses import dataclass
from util import get_beeline_orientation


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


@dataclass
class SequentiallyEvolvedOffenseState:
    who_should_seek: int | None = None

    def should_seek(self, id: int):
        return self.who_should_seek is not None and self.who_should_seek == id


def with_sequentially_evolved_offense(
    env: BluelockEnvironment,
    seeker: neat.nn.FeedForwardNetwork,
    holder: neat.nn.FeedForwardNetwork,
    pass_evaluator: neat.nn.FeedForwardNetwork,
    find_spacer: neat.nn.FeedForwardNetwork,
):
    old_update = env.update
    state = SequentiallyEvolvedOffenseState()

    def clear_state():
        state.who_should_seek = None

    def action(offender: Offender):
        other_offender = list(filter(lambda o: o.id != offender.id, env.offense))[0]
        defender = env.defense[0]

        if offender.has_possession():
            if state.should_seek(offender.id):
                clear_state()

            pass_confidence = apply_find_space(
                env, offender.id, other_offender.id, defender.id, find_spacer
            )[0]
            if pass_confidence > 0:
                # todo(): replace with passing model
                pass_direction = get_beeline_orientation(
                    other_offender.position - offender.position
                )
                offender.set_rotation(pass_direction)
                offender.shoot()
                state.who_should_seek = other_offender.id
            else:
                """
                offender.set_rotation(
                    apply_hold_ball(env, offender.id, defender.id, holder)
                )
                offender.run()
                """
        else:
            if state.should_seek(offender.id):
                offender.set_rotation(apply_seek(env, offender.id, seeker))
                offender.run()
            else:
                _, orientation, run_confidence = apply_find_space(
                    env, other_offender.id, offender.id, defender.id, find_spacer
                )
                offender.set_rotation(orientation)
                if run_confidence > 0:
                    offender.run()

    def new_update(*args, **kwargs):
        for offender in env.offense:
            action(offender)
        old_update(*args, **kwargs)

    env.update = new_update
    return env, clear_state
