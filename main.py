import argparse
from environment.core import Offender, Defender, Ball
from environment.config import ENVIRONMENT_HEIGHT, ENVIRONMENT_WIDTH
from environment.core import BluelockEnvironment
from environment.defense.agent import with_policy_defense, naive_man_to_man
from evolution.predefined_behavior.keepaway import (
    evolve_predefined_behavior_keepaway,
    watch_predefined_behavior_keepaway,
)
from evolution.sequential.keepaway import (
    evolve_sequential_keepaway,
    watch_sequential_keepaway,
)
from evolution.coevolution.keepaway import coevolve_keepaway, watch_coevolved_keepaway
from visualization.visualizer import BluelockEnvironmentVisualizer
from util import get_random_point
from enum import Enum


class TrainingStyle(str, Enum):
    SEQUENTIAL = "sequential"
    PREDEFINED_BEHAVIOR = "predefined"
    COEVOLUTION = "coevolution"


def visualize(namespace: argparse.Namespace):
    offenders, defenders = [], []
    for i in range(namespace.offense):
        offenders.append(
            Offender(i, get_random_point(ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT))
        )
    for i in range(namespace.defense):
        defenders.append(
            Defender(
                i + namespace.offense,
                get_random_point(ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT),
            )
        )

    ball = Ball(position=get_random_point(ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT))
    offenders[0].possess(ball)

    env = with_policy_defense(
        BluelockEnvironment(
            dims=(ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT),
            offense=offenders,
            defense=defenders,
            ball=ball,
        ),
        policy=naive_man_to_man,
    )

    vis = BluelockEnvironmentVisualizer(env)
    vis.start()


def train(namespace: argparse.Namespace):
    style: TrainingStyle = namespace.style
    if style == TrainingStyle.SEQUENTIAL:
        evolve_sequential_keepaway()
    elif style == TrainingStyle.PREDEFINED_BEHAVIOR:
        evolve_predefined_behavior_keepaway()
    elif style == TrainingStyle.COEVOLUTION:
        coevolve_keepaway()


def watch(namespace: argparse.Namespace):
    style: TrainingStyle = namespace.style
    if style == TrainingStyle.SEQUENTIAL:
        watch_sequential_keepaway()
    elif style == TrainingStyle.PREDEFINED_BEHAVIOR:
        watch_predefined_behavior_keepaway()
    elif style == TrainingStyle.COEVOLUTION:
        watch_coevolved_keepaway()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A playmaking AI for soccer")
    subparsers = parser.add_subparsers()

    visualize_parser = subparsers.add_parser(
        name="visualize", description="Visualize the playmaking AI"
    )
    visualize_parser.set_defaults(func=visualize)
    visualize_parser.add_argument(
        "--offense", type=int, help="The number of offenders to play", default=2
    )
    visualize_parser.add_argument(
        "--defense", type=int, help="The number of defenders to play", default=2
    )

    train_parser = subparsers.add_parser(
        name="train", description="Train the playmaking AI"
    )
    train_parser.set_defaults(func=train)
    train_parser.add_argument(
        "--style",
        type=TrainingStyle,
        default=TrainingStyle.SEQUENTIAL,
        help="The methodology of training the playmaking AI. In 'sequential' training, the AI will learn each disjoint task of soccer and aggregate its learnings",
    )

    watch_parser = subparsers.add_parser(
        name="watch", description="Watch the result of training for the playmaking AI"
    )
    watch_parser.set_defaults(func=watch)
    watch_parser.add_argument(
        "--style",
        type=TrainingStyle,
        default=TrainingStyle.SEQUENTIAL,
        help="The type of training to watch the result of",
    )

    namespace = parser.parse_args()
    namespace.func(namespace)
