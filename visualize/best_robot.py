import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.append(".")
from specify_robot import SpecifyRobotVisualizer

from alg.globals import POP_CSV_FILE_NAME


class BestRobotVisualizer(SpecifyRobotVisualizer):
    def __init__(
        self,
        exp_dir: Path,
        movie_path: Optional[str],
        num_episodes: int = 1,
    ):

        with (exp_dir / POP_CSV_FILE_NAME).open() as fd:
            reader = csv.reader(fd)
            next(reader)

            last_row = next(reader)

            for row in reader:
                semi_last_row = last_row
                last_row = row

        generation = int(last_row[0])
        fitness_list = []
        for semi_fitness, fitness in zip(semi_last_row[1:], last_row[1:]):
            if fitness == "":
                fitness_list.append(float(semi_fitness))
            else:
                fitness_list.append(float(fitness))

        id_ = int(np.argmax(fitness_list))

        super().__init__(exp_dir, generation, id_, movie_path, num_episodes)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-dir", type=str)
    parser.add_argument("-m", "--movie-path", type=str, default=None)
    parser.add_argument("-n", "--num-episodes", type=int, default=1)

    args = parser.parse_args()

    visualizer = BestRobotVisualizer(
        exp_dir=Path(args.exp_dir),
        movie_path=args.movie_path,
        num_episodes=args.num_episodes,
    )

    visualizer.run()
