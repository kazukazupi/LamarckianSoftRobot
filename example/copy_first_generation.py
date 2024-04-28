import argparse
import sys
from pathlib import Path

sys.path.append(".")
from alg.utils import copy_first_generation

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--src-exp-dir",
        type=str,
        help="The directory of experiments already conducted",
    )
    parser.add_argument(
        "-d",
        "--dst-exp-dir",
        type=str,
        help="The directory for experiments conducted with altered conditions",
    )
    parser.add_argument(
        "-n",
        "--non-inherit",
        action="store_true",
        default=False,
        help="If True, do not inherit the controller",
    )
    parser.add_argument(
        "-c",
        "--crossover-rate",
        type=float,
        default=0.5,
        help="The ratio of offspring robots born through crossover (not yet implemented)",
    )

    args = parser.parse_args()

    copy_first_generation(
        src_exp_dir=Path(args.src_exp_dir),
        dst_exp_dir=Path(args.dst_exp_dir),
        non_inherit=args.non_inherit,
        crossover_rate=args.crossover_rate,
    )
