import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from evogym import hashable, sample_robot  # type: ignore
from pydantic import BaseModel

from alg.a2c_ppo_acktr.model import Policy  # type: ignore
from alg.config import Config
from alg.globals import JSON_FILE_NAME
from alg.inherit import inherit_controller_mutation
from alg.ppo import run_ppo
from alg.ppo.envs import make_vec_envs
from alg.ppo.run import ACTOR_CRITIC_FILE_NAME
from alg.structure import Structure
from alg.utils import LogWriter


class CrossoverInfo(BaseModel):
    axis: int  # Literal[0, 1]
    mid: int


class IndividualInfo(BaseModel):
    id_: int
    generation: int
    parents_id: Optional[Union[Tuple[int], Tuple[int, int]]] = None
    fitness: Optional[float] = None
    crossover_info: Optional[CrossoverInfo] = None


class Individual:
    def __init__(
        self,
        structure: Structure,
        info: IndividualInfo,
        saving_dir: Path,
    ):

        assert saving_dir.exists()

        # initialize
        self.structure = structure
        self.info = info
        self.saving_dir = saving_dir

    def save(self):

        # save structure
        self.structure.save(self.saving_dir)

        # save robot info
        with open(self.saving_dir / JSON_FILE_NAME, "w") as fp:
            fp.write(self.info.json(indent=3))

    def train(
        self,
        config: Config,
        parents: Optional[List["Individual"]],
        log_writer: LogWriter,
    ):
        actor_critic = get_controller(
            self.structure,
            parents,
            config,
            self.crossover_info,
            log_writer,
        )
        self.info.fitness = run_ppo(
            self.structure, self.saving_dir, config, actor_critic
        )

    def reborn(self, robot_shape: Tuple[int, int]):
        self.structure = Structure(*sample_robot(robot_shape))

    @classmethod
    def init_random(
        cls,
        id_: int,
        generation: int,
        robot_shape: Tuple[int, int],
        generation_dir: Path,
    ):

        structure = Structure(*sample_robot(robot_shape))
        info = IndividualInfo(id_=id_, generation=generation)
        saving_dir = generation_dir / f"id{(id_):02}"

        saving_dir.mkdir(parents=False, exist_ok=False)

        return cls(structure, info, saving_dir)

    @classmethod
    def init_designated_structure(
        cls,
        structure: Structure,
        id_: int,
        generation: int,
        generation_dir: Path,
        parents_id: Union[None, Tuple[int], Tuple[int, int]],
    ):

        info = IndividualInfo(id_=id_, generation=generation, parents_id=parents_id)

        saving_dir = generation_dir / f"id{(id_):02}"
        saving_dir.mkdir(parents=False, exist_ok=False)

        return cls(structure, info, saving_dir)

    @classmethod
    def load(cls, saving_dir: Path):

        structure = Structure.load(saving_dir)

        with open(saving_dir / JSON_FILE_NAME, "r") as fp:
            json_data = json.load(fp)

        info = IndividualInfo(**json_data)

        return cls(structure, info, saving_dir)

    @property
    def id_(self):
        return self.info.id_

    @property
    def generation(self):
        return self.info.generation

    @property
    def parents_id(self):
        return self.info.parents_id

    @property
    def fitness(self):
        return self.info.fitness

    @property
    def crossover_info(self):
        return self.info.crossover_info

    @property
    def hash(self):
        return hashable(self.structure.body)


def get_controller(
    structure: Structure,
    parents: Optional[List[Individual]],
    config: Config,
    crossover_info: Optional[CrossoverInfo],
    log_writer: LogWriter,
) -> Policy:

    envs = make_vec_envs(
        env_name=config.env_name,
        robot_structure=structure.as_tuple(),
        seed=None,
        num_processes=1,
        gamma=config.gamma,
        log_dir=None,
        device=torch.device("cpu"),
        allow_early_resets=False,
    )

    # inherit is not allowed or has no parents
    if (config.non_inherit) or (parents is None):

        actor_critic = Policy(
            obs_shape=envs.observation_space.shape, action_space=envs.action_space
        )

    # inherit is allowed and configured from mutation
    elif len(parents) == 1:
        log_writer.print_and_write("\tinherited controller.")

        parent = parents[0]
        parent_actor_critic = torch.load(
            parent.saving_dir / ACTOR_CRITIC_FILE_NAME,
            map_location=lambda storage, loc: storage,
        )[0]

        actor_critic = inherit_controller_mutation(
            parent_body=parent.structure.body,
            parent_actor_critic=parent_actor_critic,
            child_body=structure.body,
            child_observation_space_shape=envs.observation_space.shape,
            child_action_space=envs.action_space,
            env_name=config.env_name,
        )

    # inherit is allowed and is emerged from crossover
    elif len(parents) == 2:
        log_writer.print_and_write("\tinherited controller.")
        raise NotImplementedError()
        # print("inherited controller.")

        # assert crossover_info is not None
        # axis = crossover_info["axis"]
        # mid = crossover_info["mid"]

        # parent1 = parents[0]
        # parent2 = parents[1]

        # parent1_actor_critic = torch.load(
        #     os.path.join(parent1.saving_dir, "actor_critic.pt"),
        #     map_location=lambda storage, loc: storage,
        # )[0]

        # parent2_actor_critic = torch.load(
        #     os.path.join(parent2.saving_dir, "actor_critic.pt"),
        #     map_location=lambda storage, loc: storage,
        # )[0]

        # actor_critic = inherit_controller_crossover(
        #     child_body=body,
        #     axis=axis,
        #     mid=mid,
        #     parent1_body=parent1.body,
        #     parent2_body=parent2.body,
        #     parent1_actor_critic=parent1_actor_critic,
        #     parent2_actor_critic=parent2_actor_critic,
        #     child_observation_space_shape=observation_space_shape,
        #     child_action_space=action_space,
        # )

    return actor_critic
