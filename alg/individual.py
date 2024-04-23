import json
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from evogym import hashable, sample_robot  # type: ignore
from pydantic import BaseModel

from alg.config import Config
from alg.ppo import run_ppo
from alg.structure import Structure

JSON_FILE_NAME = "robot_info.json"


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

    def train(self, config: Config):
        run_ppo(self.structure, self.saving_dir, config)

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
