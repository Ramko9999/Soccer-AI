import argparse
from environment.drill import BluelockDrill
from visualization import InteractiveBluelockDrillVisualization
from environment.defense_policy import naive_man_to_man
from enum import Enum
from evolution.sequential import evolve_sequentially


class TrainingStyle(str, Enum):
    SEQUENTIAL = "sequential"


def visualize(namespace: argparse.Namespace):
    offenders = [i for i in range(namespace.offense)]
    defenders = [i + namespace.offense for i in range(namespace.defense)]

    drill = BluelockDrill(
        640,
        480,
        offensive_players=offenders,
        defense_players=defenders,
        defense_policy=naive_man_to_man,
    )
    vis = InteractiveBluelockDrillVisualization(drill)
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
        help="The methodology of training the playmaking AI. In 'sequential' training, the AI will learn each disjoing task of soccer and aggregate its learnings",
    )

    namespace = parser.parse_args()
    namespace.func(namespace)
