import json
from pathlib import Path
from typing import Literal, NamedTuple, Optional, Tuple, Union

import numpy as np
from evogym import sample_robot  # type: ignore
from pydantic import BaseModel

BODY_FILE_NAME = "body.npy"
CONNECTIONS_FILE_NAME = "connections.npy"
JSON_FILE_NAME = "robot_info.json"


class CrossoverInfo(BaseModel):
    axis: Literal[0, 1]
    mid: int


class Structure(NamedTuple):
    body: np.ndarray
    connections: np.ndarray

    def save(self, saving_dir: Path):
        np.save(str(saving_dir / BODY_FILE_NAME), self.body)
        np.save(str(saving_dir / CONNECTIONS_FILE_NAME), self.connections)

    @staticmethod
    def load(saving_dir: Path):

        # load structure
        body = np.load(str(saving_dir / BODY_FILE_NAME))
        connections = np.load(str(saving_dir / CONNECTIONS_FILE_NAME))

        return Structure(body, connections)


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
            fp.write(self.info.model_dump_json(indent=3))

    @classmethod
    def init_random(
        cls,
        id_: int,
        generation: int,
        robot_shape: Tuple[int, int],
    ):

        structure = Structure(*sample_robot(robot_shape))
        info = IndividualInfo(id_=id_, generation=generation)
        saving_dir = Path(f"./experiment/generation{(generation):02}/id{(id_):02}")

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
