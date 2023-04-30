import argparse
from environment.core import Offender, Defender, Ball
from environment.config import ENVIRONMENT_HEIGHT, ENVIRONMENT_WIDTH
from environment.core import BluelockEnvironment
from environment.defense.policy import naive_man_to_man
from environment.defense.agent import decorate_with_policy_defense
from visualization.visualizer import BluelockEnvironmentVisualizer
from evolution.sequential import evolve_sequentially
from util import get_random_point
from enum import Enum


class TrainingStyle(str, Enum):
    SEQUENTIAL = "sequential"


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

    env = decorate_with_policy_defense(
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
        evolve_sequentially()


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

    namespace = parser.parse_args()
    namespace.func(namespace)
