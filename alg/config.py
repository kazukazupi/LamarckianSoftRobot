import argparse
import json
from pathlib import Path
from typing import Tuple

import torch
from pydantic import BaseModel

CONFIG_FILE_NAME = "config.json"


class Config(BaseModel):

    exp_dir: Path
    shape: Tuple[int, int]
    max_evaluations: int
    population_size: int
    crossover_rate: float
    mutation_rate: float
    elite_rate_high: float
    elite_rate_low: float
    non_inherit: bool
    print_en: bool
    algo: str  # Literal["a2c", "ppo", "acktr"]
    gail: bool
    gail_experts_dir: str
    gail_batch_size: int
    gail_epoch: float
    lr: float
    eps: float
    alpha: float
    gamma: float
    use_gae: bool
    gae_lambda: float
    entropy_coef: float
    value_loss_coef: float
    max_grad_norm: float
    seed: int
    cuda_deterministic: bool
    max_iters: int
    num_processes: int
    num_steps: int
    ppo_epoch: int
    num_mini_batch: int
    clip_param: float
    log_interval: int
    save_interval: int
    num_evals: int
    eval_interval: int
    num_env_steps: int
    env_name: str
    log_dir: str
    save_dir: str
    use_proper_time_limits: bool
    recurrent_policy: bool
    use_linear_lr_decay: bool
    cuda: bool = torch.cuda.is_available()

    @classmethod
    def initialize(cls):

        parser = argparse.ArgumentParser()

        # TODO: encourage user to specify "result" as parent directory
        parser.add_argument(
            "--exp-dir",
            type=Path,
            default="result/experiment",
            help="directory to save log",
        )

        # GA params
        parser.add_argument(
            "--shape", type=Tuple[int, int], default=(5, 5), help="the shape of robots"
        ),
        parser.add_argument(
            "--max-evaluations", type=int, default=250, help="how many robots to train"
        )
        parser.add_argument(
            "--population-size",
            type=int,
            default=25,
            help="how many robots a population has",
        )
        parser.add_argument(
            "--crossover_rate",
            type=float,
            default=0.5,
            help="the probability of offspring generated through crossover",
        )
        parser.add_argument(
            "--mutation-rate",
            type=float,
            default=0.1,
            help="each voxel's probability to mutate",
        )
        parser.add_argument("--elite-rate-high", type=float, default=0.6)
        parser.add_argument("--elite-rate-low", type=float, default=0.1)
        parser.add_argument(
            "--non_inherit",
            action="store_false",
            default=True,
            help="do controller inheritance",
        )

        # PPO params
        parser.add_argument("--print-en", action="store_true", default=False)
        parser.add_argument(
            "--algo", default="ppo", help="algorithm to use: a2c | ppo | acktr"
        )
        parser.add_argument(
            "--gail",
            action="store_true",
            default=False,
            help="do imitation learning with gail",
        )
        parser.add_argument(
            "--gail-experts-dir",
            default="./gail_experts",
            help="directory that contains expert demonstrations for gail",
        )
        parser.add_argument(
            "--gail-batch-size",
            type=int,
            default=128,
            help="gail batch size (default: 128)",
        )
        parser.add_argument(
            "--gail-epoch", type=int, default=5, help="gail epochs (default: 5)"
        )
        parser.add_argument(
            "--lr", type=float, default=2.5e-4, help="learning rate (default: 2.5e-4)"
        )
        parser.add_argument(
            "--eps",
            type=float,
            default=1e-5,
            help="RMSprop optimizer epsilon (default: 1e-5)",
        )
        parser.add_argument(
            "--alpha",
            type=float,
            default=0.99,
            help="RMSprop optimizer apha (default: 0.99)",
        )
        parser.add_argument(
            "--gamma",
            type=float,
            default=0.99,
            help="discount factor for rewards (default: 0.99)",
        )
        parser.add_argument(
            "--use-gae",
            action="store_true",
            default=True,
            help="use generalized advantage estimation",
        )
        parser.add_argument(
            "--gae-lambda",
            type=float,
            default=0.95,
            help="gae lambda parameter (default: 0.95)",
        )
        parser.add_argument(
            "--entropy-coef",
            type=float,
            default=0.01,
            help="entropy term coefficient (default: 0.01)",
        )
        parser.add_argument(
            "--value-loss-coef",
            type=float,
            default=0.5,
            help="value loss coefficient (default: 0.5)",
        )
        parser.add_argument(
            "--max-grad-norm",
            type=float,
            default=0.5,
            help="max norm of gradients (default: 0.5)",
        )
        parser.add_argument(
            "--seed", type=int, default=1, help="random seed (default: 1)"
        )
        parser.add_argument(
            "--cuda-deterministic",
            action="store_true",
            default=False,
            help="sets flags for determinism when using CUDA (potentially slow!)",
        )
        parser.add_argument("--max-iters", type=int, default=1000)
        parser.add_argument(
            "--num-processes",
            type=int,
            # default=1,
            default=4,
            help="how many training CPU processes to use (default: 1)",
        )
        parser.add_argument(
            "--num-steps",
            type=int,
            default=128,
            help="number of forward steps in A2C / num steps to use in PPO (default: 128)",
        )
        parser.add_argument(
            "--ppo-epoch", type=int, default=4, help="number of ppo epochs (default: 4)"
        )
        parser.add_argument(
            "--num-mini-batch",
            type=int,
            default=4,
            help="number of batches for ppo (default: 4)",
        )
        parser.add_argument(
            "--clip-param",
            type=float,
            default=0.1,
            help="ppo clip parameter (default: 0.1)",
        )
        parser.add_argument(
            "--log-interval",
            type=int,
            default=10,
            help="log interval, one log per n updates (default: 10)",
        )
        parser.add_argument(
            "--save-interval",
            type=int,
            default=100,
            help="save interval, one save per n updates (default: 100)",
        )
        parser.add_argument(
            "--num-evals",
            type=int,
            default=1,
            help="number of times to evaluate each controller (for evaluation purposes not training). (default: 1) as most Evolution Gym environments are deterministic.",
        )
        parser.add_argument(
            "--eval-interval",
            type=int,
            # default=None,
            default=50,
            help="eval interval, one eval per n updates (default: None)",
        )
        parser.add_argument(
            "--num-env-steps",
            type=int,
            default=10e6,
            help="number of environment steps to train (default: 10e6)",
        )
        parser.add_argument(
            "--env-name",
            default="Walker-v0",
            help="environment to train on (default: Walker-v0)",
        )
        parser.add_argument(
            "--log-dir",
            default="/tmp/gym/",
            help="directory to save agent logs (default: /tmp/gym)",
        )
        parser.add_argument(
            "--save-dir",
            default="./trained_models/",
            help="directory to save agent logs (default: ./trained_models/)",
        )
        parser.add_argument(
            "--use-proper-time-limits",
            action="store_true",
            default=False,
            help="compute returns taking into account time limits",
        )
        parser.add_argument(
            "--recurrent-policy",
            action="store_true",
            default=False,
            help="use a recurrent policy",
        )
        parser.add_argument(
            "--use-linear-lr-decay",
            action="store_true",
            default=True,
            help="use a linear schedule on the learning rate",
        )
        args = parser.parse_args()
        args_dict = vars(args)

        config = cls(**args_dict)

        config.exp_dir.mkdir()

        # save parameter info
        with open(config.exp_dir / CONFIG_FILE_NAME, "w") as fp:
            fp.write(config.json(indent=3))

        return config

    @classmethod
    def load(cls, exp_dir: Path):

        with (exp_dir / CONFIG_FILE_NAME).open() as fd:
            json_dict = json.load(fd)

        config = cls(**json_dict)
        assert config.exp_dir == exp_dir

        return config
