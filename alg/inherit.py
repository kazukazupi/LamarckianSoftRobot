from typing import List, NamedTuple, Tuple

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


def get_mass_point_in_order(body: np.ndarray) -> List[Tuple[int, int]]:
    """
    Return the coordinates of the mass point of all voxels in order from top-left to bottom-right.

        top  left  mass  point  .___________.  top right mass point
                                │           │
                                │   voxel   │
                                │           │
        bottom left mass point  .___________.  bottom right mass point
    """

    contour = body != 0  # contour[h][w] = whether each voxel_{h,w} is empty ir not
    (H, W) = contour.shape

    mass_point_in_order = []

    for h in range(H):
        for w in range(W):
            if contour[h][w]:  # voxel_{h,w}

                mass_point = (h, w)  # top left mass point of this voxel
                if h == 0:  # at the top edge of the body
                    if w == 0:  # top left corner
                        mass_point_in_order.append(mass_point)
                    else:
                        if contour[h][w - 1]:  # if there is a voxel to the left
                            pass
                        else:
                            mass_point_in_order.append(mass_point)
                else:
                    if w == 0:  # at the left edge
                        if contour[h - 1][w]:  # if there is a voxel above
                            pass
                        else:
                            mass_point_in_order.append(mass_point)
                    else:
                        if contour[h - 1][w]:  # if there is a voxel above
                            pass
                        elif contour[h][w - 1]:  # if there is a voxel to the left
                            pass
                        else:
                            mass_point_in_order.append(mass_point)

                mass_point = (h, w + 1)  # top right mass point of this voxel
                if h == 0:  # at the top edge of the body
                    mass_point_in_order.append(mass_point)
                else:
                    if w == W - 1:  # at the right edge
                        if contour[h - 1][w]:  # if there is a voxel above
                            pass
                        else:
                            mass_point_in_order.append(mass_point)
                    else:
                        if contour[h - 1][w]:  # if there is a voxel above
                            pass
                        elif (
                            contour[h][w + 1] and contour[h - 1][w + 1]
                        ):  # If it is connected to the voxel at the top-right
                            pass
                        else:
                            mass_point_in_order.append(mass_point)

                mass_point = (h + 1, w)  # bottom left mass point of this voxel
                if w == 0:  # at the left edge
                    mass_point_in_order.append(mass_point)
                else:
                    if contour[h][w - 1]:  # if there is a voxel to the left
                        pass
                    else:
                        mass_point_in_order.append(mass_point)

                mass_point = (h + 1, w + 1)  # bottom right mass point of this voxel
                mass_point_in_order.append(mass_point)

    return mass_point_in_order


class MassPointWithCount(NamedTuple):
    mass_point: Tuple[int, int]  # coordinate
    count_: int  # Number of occurrences of this mass_point


def get_mass_point_in_order_with_count(body: np.ndarray) -> List[MassPointWithCount]:
    """
    convert 'mass point in order' to 'mass point with count in order'
    """

    mpio = get_mass_point_in_order(body)
    mpio_with_count = []

    contained: List[Tuple[int, int]] = []
    for mass_point in mpio:
        count = contained.count(mass_point)
        mpio_with_count.append(MassPointWithCount(mass_point, count))
        contained.append(mass_point)
        assert count <= 1

    return mpio_with_count
