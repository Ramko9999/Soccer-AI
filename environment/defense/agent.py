from environment.core import BluelockEnvironment, Defender, Offender, Ball
from typing import Callable
from util import get_euclidean_dist, get_beeline_orientation
from dataclasses import dataclass
import heapq


@dataclass
class Assignment:
    defender_id: int
    orientation: float
    should_run: bool


DefensePolicy = Callable[[list[Defender], list[Offender], Ball], list[Assignment]]

# decorates over BluelockEnvironment by controlling the defense's movements according to some policy
def with_policy_defense(
    env: BluelockEnvironment, policy: DefensePolicy
) -> BluelockEnvironment:
    old_update = env.update

    def new_update(*args, **kwargs):
        assignments = policy(env.defense, env.offense, env.ball)
        for assignment in assignments:
            env.get_player(assignment.defender_id).set_rotation(assignment.orientation)
            if assignment.should_run:
                env.get_player(assignment.defender_id).run()
        old_update(*args, **kwargs)

    env.update = new_update
    return env


def naive_man_to_man(
    defenders: list[Defender], offenders: list[Offender], ball: Ball
) -> list[Assignment]:

    if len(defenders) == 0:
        return []

    assignments = []
    assigned_defenders = set([])
    if not ball.is_possessed():
        # try to steal the loose ball
        dist, closest_defender = float("inf"), defenders[0]
        for defender in defenders:
            defender_dist_to_ball = get_euclidean_dist(defender.position, ball.position)
            if defender_dist_to_ball < dist:
                dist, closest_defender = defender_dist_to_ball, defender

        assigned_defenders.add(closest_defender.id)
        assignments.append(
            Assignment(
                defender_id=closest_defender.id,
                orientation=get_beeline_orientation(
                    ball.position - closest_defender.position
                ),
                should_run=True,
            )
        )

    marked_offenders = set([])
    defender_to_offender_dists = []
    for defender in defenders:
        if defender.id not in assigned_defenders:
            for offender in offenders:
                # prioritize guarding the ball handler
                offender_priority = -1 if offender.has_possession() else 0
                defender_to_offender_dists.append(
                    (
                        offender_priority,
                        get_euclidean_dist(defender.position, offender.position),
                        get_beeline_orientation(offender.position - defender.position),
                        defender.id,
                        offender.id,
                    )
                )

    heapq.heapify(defender_to_offender_dists)
    while len(assignments) < len(defenders):
        _, dist, angle, defender_id, offender_id = heapq.heappop(
            defender_to_offender_dists
        )
        if defender_id in assigned_defenders or offender_id in marked_offenders:
            continue

        assigned_defenders.add(defender_id)
        assignments.append(
            Assignment(defender_id=defender_id, orientation=angle, should_run=dist > 3)
        )
        marked_offenders.add(offender_id)
    return assignments
