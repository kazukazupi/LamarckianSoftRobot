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


class IndividualInfo(BaseModel):
    id_: int
    generation: int
    parents_id: Optional[Union[Tuple[int], Tuple[int, int]]]
    fitness: Optional[float]
    crossover_info: Optional[CrossoverInfo]


class Individual:

    def __init__(
        self,
        id_: int,
        robot_shape: Tuple[int, int],
        generation: int,
        parents_id: Optional[Union[Tuple[int], Tuple[int, int]]] = None,
        fitness: Optional[float] = None,
        crossover_info: Optional[CrossoverInfo] = None,
    ):

        # initialize
        self.structure = Structure(*sample_robot(robot_shape))
        self.info = IndividualInfo(
            id_=id_,
            generation=generation,
            parents_id=parents_id,
            fitness=fitness,
            crossover_info=crossover_info,
        )

        # make directory to save information
        self.saving_dir = Path(
            f"./experiment/generation{(self.generation):02}/id{(self.id_):02}"
        )
        self.saving_dir.mkdir(parents=False, exist_ok=False)

    def save(self):

        # save structure
        self.structure.save(self.saving_dir)

        # save robot info
        with open(self.saving_dir / JSON_FILE_NAME, "w") as fp:
            fp.write(self.info.model_dump_json(indent=3))

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
