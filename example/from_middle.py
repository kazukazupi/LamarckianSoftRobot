import argparse
import sys
from pathlib import Path

sys.path.append(".")
from alg import Population

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-dir",
        type=str,
    )

    args = parser.parse_args()

    population = Population.load(Path(args.exp_dir))
    population.evolve()
