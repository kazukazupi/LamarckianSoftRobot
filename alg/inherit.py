import numpy as np
import torch
from a2c_ppo_acktr.model import Policy  # type: ignore

"""
The observation information provided to the robot is given in vector form. 
The length of the vector varies depending on the task and the structure of the robot, and its components are arranged as 
[
    sub-vectors with lengths determined by the task,
    sub-vectors with lengths determined by the robot's structure,
    sub-vectors with lengths determined by the task
].
For detail, see https://arxiv.org/pdf/2201.09863.pdf
"""


def get_overhead(env_name: str) -> int:
    """
    return the length of the sub-vector with lengths determined by the task at the beginning of the observation vector.
    """

    if env_name in [
        "BridgeWalker-v0",
        "ObstacleTraverser-v0",
        "ObstacleTraverser-v1",
        "Climber-v2",
    ]:
        return 3
    elif env_name in [
        "Walker-v0",
        "BidirectionalWalker-v0",
        "Carrier-v0",
        "Carrier-v1",
        "Pusher-v0",
    ]:
        return 2
    else:
        raise NotImplementedError(
            f'function "get_over_head" does not support the environment {env_name}.'
        )


def get_overtail(env_name: str) -> int:
    """
    return the length of the sub-vector with lengths determined by the task at the end of the observation vector.
    """

    if env_name in ["Walker-v0", "BridgeWalker-v0"]:
        return 0
    elif env_name in ["BidirectionalWalker-v0"]:
        return 3
    elif env_name in ["Carrier-v0", "Carrier-v1", "Pusher-v0"]:
        return 4
    elif env_name in ["Climber-v2"]:
        return 7
    elif env_name in ["ObstacleTraverser-v0", "ObstacleTraverser-v1"]:
        return 11
    else:
        raise NotImplementedError(
            f'function "get_over_tail" does not support the environment {env_name}.'
        )
