import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

sys.path.append(".")

from alg import Config, Structure
from alg.a2c_ppo_acktr.model import Policy  # type: ignore
from alg.ppo.run import ACTOR_CRITIC_FILE_NAME  # type: ignore
from visualize import visualize


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_dir", type=str)
    parser.add_argument("-m", "--movie_path", type=str, default=None)
    parser.add_argument("-n", "--num_episodes", type=int, default=1)

    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)

    config = Config.load(exp_dir)

    # with (exp_dir / "fitness.csv") as f:
    #     reader = csv.reader(f)
    #     for i, row in enumerate(reader):
    #         if i == 0:
    #             colname = row
    #         last_generation = row

    # generation = int(last_generation[0])
    # best_id = np.argmax([float(val) for val in last_generation[1:]])

    structure, (actor_critic, obs_rms) = load(exp_dir, 7, 1)

    visualize(
        structure=structure,
        actor_critic=actor_critic,
        env_name=config.env_name,
        num_episodes=args.num_episodes,
        obs_rm=obs_rms,
    )
