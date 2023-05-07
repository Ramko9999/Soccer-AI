import os
import neat

output_path = os.path.join(os.path.dirname(__file__), "output")
CHECKPOINTS_PATH = os.path.join(output_path, "checkpoints")
MODELS_PATH = os.path.join(output_path, "models")
CONFIGS_PATH = os.path.join(os.path.dirname(__file__), "config")
PLOTS_PATH = os.path.join(output_path, "plots")


def get_default_config(config_file):
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        os.path.join(CONFIGS_PATH, config_file),
    )
