from environment.core import BluelockEnvironment
from environment.defense.policy import DefensePolicy

# decorates over BluelockEnvironment by controlling the defense's movements according to some policy
def decorate_with_policy_defense(
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
