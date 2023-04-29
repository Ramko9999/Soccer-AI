from environment.core import Defender, Offender, Ball
from typing import Callable
from util import get_euclidean_dist, get_angle
import heapq

Assignments = dict[
    int, float, bool
]  # defender's id -> angle where to should go and whether it should stay put or run
DefensePolicy = Callable[[list[Defender], list[Offender], Ball], Assignments]


def naive_man_to_man(
    defenders: list[Defender], offenders: list[Offender], ball: Ball
) -> Assignments:

    assignments = {}
    if not ball.is_possessed():
        # try to steal the loose ball
        dist, closest_defender = float("inf"), None
        for defender in defenders:
            defender_dist_to_ball = get_euclidean_dist(defender.position, ball.position)
            if defender_dist_to_ball < dist:
                dist, closest_defender = defender_dist_to_ball, defender

        if closest_defender is not None:
            angle = get_angle(ball.position - closest_defender.position)
            assignments[closest_defender.id] = (angle, True)
            defenders = list(filter(lambda d: d.id != closest_defender.id, defenders))

    marked_offenders = set([])
    defender_to_offender_dists = []
    for defender in defenders:
        for offender in offenders:
            # prioritize guarding the ball handler
            offender_priority = -1 if offender.has_possession() else 0
            defender_to_offender_dists.append(
                (
                    offender_priority,
                    get_euclidean_dist(defender.position, offender.position),
                    get_angle(offender.position - defender.position),
                    defender.id,
                    offender.id,
                )
            )

    heapq.heapify(defender_to_offender_dists)
    while len(assignments) < len(defenders):
        _, dist, angle, defender_id, offender_id = heapq.heappop(
            defender_to_offender_dists
        )
        if defender_id in assignments or offender_id in marked_offenders:
            continue

        assignments[defender_id] = (angle, dist > 3)
        marked_offenders.add(offender_id)

    return assignments
