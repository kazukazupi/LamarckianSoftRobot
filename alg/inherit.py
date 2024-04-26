from typing import List, NamedTuple, Tuple

import numpy as np
import torch

from alg.a2c_ppo_acktr.model import Policy  # type: ignore


def inherit_controller_mutation(
    parent_body: np.ndarray,
    parent_actor_critic: Policy,
    child_body: np.ndarray,
    child_observation_space_shape: tuple,
    child_action_space,
    env_name: str,
) -> Policy:

    # initialize neural network
    child_actor_critic = Policy(
        obs_shape=child_observation_space_shape, action_space=child_action_space
    )

    # load weight
    child_state_dict = child_actor_critic.state_dict()
    parent_state_dict = parent_actor_critic.state_dict()

    """
    i-th parent actor NN node corresponds to
    j-th child actor NN node (where j = mapping_table[i])
    """
    mapping_table_state = get_mapping_table_state(child_body, parent_body, env_name)
    mapping_table_action = get_mapping_table_action(parent_body, child_body)

    assert (
        len(mapping_table_state) == parent_state_dict["base.critic.0.weight"].shape[1]
    )
    assert (
        len(mapping_table_action) == parent_state_dict["dist.fc_mean.weight"].shape[0]
    )
    assert max(mapping_table_state) < child_state_dict["base.critic.0.weight"].shape[1]
    assert max(mapping_table_action) < child_state_dict["dist.fc_mean.weight"].shape[0]

    # ---------------------------
    #  copy weights of actor net
    # ---------------------------

    # copy weights of first layer
    child_state_dict["base.actor.0.weight"] = torch.transpose(
        child_state_dict["base.actor.0.weight"], 0, 1
    )
    parent_state_dict["base.actor.0.weight"] = torch.transpose(
        parent_state_dict["base.actor.0.weight"], 0, 1
    )
    for index_p, index_c in enumerate(mapping_table_state):
        if index_c == -1:
            continue
        child_state_dict["base.actor.0.weight"][index_c] = parent_state_dict[
            "base.actor.0.weight"
        ][index_p]
    child_state_dict["base.actor.0.weight"] = torch.transpose(
        child_state_dict["base.actor.0.weight"], 0, 1
    )

    # copy weights of the last layer
    for index_p, index_c in enumerate(mapping_table_action):
        if index_c == -1:
            continue
        child_state_dict["dist.fc_mean.weight"][index_c] = parent_state_dict[
            "dist.fc_mean.weight"
        ][index_p]
        child_state_dict["dist.fc_mean.bias"][index_c] = parent_state_dict[
            "dist.fc_mean.bias"
        ][index_p]
        child_state_dict["dist.logstd._bias"][index_c] = parent_state_dict[
            "dist.logstd._bias"
        ][index_p]

    # copy weights of the middle layer
    for key in ["base.actor.2.weight", "base.actor.2.bias", "base.actor.0.bias"]:
        child_state_dict[key] = parent_state_dict[key]

    # ---------------------------
    #  copy weights of critic net
    # ---------------------------

    # copy weights of first layer
    child_state_dict["base.critic.0.weight"] = torch.transpose(
        child_state_dict["base.critic.0.weight"], 0, 1
    )
    parent_state_dict["base.critic.0.weight"] = torch.transpose(
        parent_state_dict["base.critic.0.weight"], 0, 1
    )
    for index_p, index_c in enumerate(mapping_table_state):
        if index_c == -1:
            continue
        child_state_dict["base.critic.0.weight"][index_c] = parent_state_dict[
            "base.critic.0.weight"
        ][index_p]
    child_state_dict["base.critic.0.weight"] = torch.transpose(
        child_state_dict["base.critic.0.weight"], 0, 1
    )

    # copy weights of middle, last layer
    for key in parent_state_dict.keys():
        if ("critic" in key) and (key != "base.critic.0.weight"):
            child_state_dict[key] = parent_state_dict[key]

    child_actor_critic.load_state_dict(child_state_dict)

    return child_actor_critic


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


def get_mapping_table_state(
    child_body: np.ndarray, parent_body: np.ndarray, env_name: str
) -> List[int]:
    """
    Return: mapping_table
        child structure's input layer j-th node corresponds to
        parent structure's input layer i-th node
        where mp_mapping_table[i] = j
    """

    child_mpio_with_count = get_mass_point_in_order_with_count(child_body)
    parent_mpio_with_count = get_mass_point_in_order_with_count(parent_body)

    """mp_mapping_table
    child structure's j-th mass point corresponds to
    parent structure's i-th mass point.
    where mp_mapping_table[i] = j 
    """

    mp_mapping_table = []

    for mass_point_with_count in parent_mpio_with_count:
        if mass_point_with_count in child_mpio_with_count:
            mp_mapping_table.append(child_mpio_with_count.index(mass_point_with_count))
        else:
            mp_mapping_table.append(-1)

    overhead = get_overhead(env_name)
    overtail = get_overtail(env_name)

    mapping_table = [i for i in range(overhead)]
    mapping_table += list(
        map(lambda x: -1 if (x == -1) else overhead + x, mp_mapping_table)
    )
    mapping_table += list(
        map(
            lambda x: -1 if (x == -1) else overhead + x + len(child_mpio_with_count),
            mp_mapping_table,
        )
    )
    for i in range(overtail):
        mapping_table.append(i + overhead + 2 * len(child_mpio_with_count))

    return mapping_table


def get_mapping_table_action(body_s: np.ndarray, body_t: np.ndarray) -> List[int]:
    """
    Return: mapping_table
        child structure's output layer j-th node corresponds to
        parent structure's output layer i-th node
        where mp_mapping_table[i] = j
    """

    actuator_coordinates_s = np.stack(np.where(body_s >= 3), axis=-1)
    actuator_coordinates_t = np.stack(np.where(body_t >= 3), axis=-1)

    mapping_table = []

    for coordinate_s in actuator_coordinates_s:
        result = (
            np.transpose((actuator_coordinates_t - coordinate_s)) == 0
        )  # compare with each x,y coordinate
        result = result[0] & result[1]
        result = np.where(result == True)
        if len(result[0]) > 0:
            mapping_table.append(result[0][0])
        else:
            mapping_table.append(-1)

    return mapping_table
