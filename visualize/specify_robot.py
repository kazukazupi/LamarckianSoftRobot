import argparse
import sys
from pathlib import Path

import torch

sys.path.append(".")

from visualize_base_class import Visualize

from alg import Config, Structure
from alg.ppo.run import ACTOR_CRITIC_FILE_NAME  # type: ignore


def load(exp_dir: Path, generation: int, id: int):

    saving_dir = exp_dir / f"generation{generation:02}" / f"id{id:02}"

    while not saving_dir.exists():
        generation -= 1
        assert generation >= 0
        saving_dir = exp_dir / f"generation{generation:02}" / f"id{id:02}"

    print(f"loading from {saving_dir}")

    structure = Structure.load(saving_dir)
    actor_critic_obs_rms = torch.load(
        saving_dir / ACTOR_CRITIC_FILE_NAME, map_location="cpu"
    )

    return structure, actor_critic_obs_rms


class SpecifyRobotVisualize(Visualize):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-e", "--exp-dir", type=str)
        parser.add_argument("-g", "--generation", type=int, required=True)
        parser.add_argument("-i", "--id", type=int, required=True)
        parser.add_argument("-m", "--movie-path", type=str, default=None)
        parser.add_argument("-n", "--num-episodes", type=int, default=1)

        args = parser.parse_args()

        exp_dir = Path(args.exp_dir)
        config = Config.load(exp_dir)

        structure, (actor_critic, obs_rms) = load(exp_dir, args.generation, args.id)

        super().__init__(
            structure=structure,
            actor_critic=actor_critic,
            env_name=config.env_name,
            movie_path=args.movie_path,
            num_episodes=args.num_episodes,
            envs=None,
            obs_rms=obs_rms,
        )


if __name__ == "__main__":

    SpecifyRobotVisualize().run()
