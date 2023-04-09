from core import Defender, Offender, Ball
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
    marked_offenders = set([])
    defender_to_offender_dists = []
    for defender in defenders:
        for offender in offenders:
            defender_to_offender_dists.append(
                (
                    get_euclidean_dist(defender.position, offender.position),
                    get_angle(offender.position - defender.position),
                    defender.id,
                    offender.id,
                )
            )

    heapq.heapify(defender_to_offender_dists)
    while len(assignments) < len(defenders):
        dist, angle, defender_id, offender_id = heapq.heappop(
            defender_to_offender_dists
        )
        if defender_id in assignments or offender_id in marked_offenders:
            continue

        assignments[defender_id] = (angle, dist > 3)
        marked_offenders.add(offender_id)

    return assignments
