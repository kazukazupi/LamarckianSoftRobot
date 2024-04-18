from pathlib import Path
from typing import Literal, Optional, Tuple, TypedDict, Union

import numpy as np

BODY_FILE_NAME = "body.npy"
CONNECTIONS_FILE_NAME = "connections.npy"


class CrossoverInfo(TypedDict):
    axis: Literal[0, 1]
    mid: int


class Structure(TypedDict):
    body: np.ndarray
    connections: np.ndarray


class IndividualInfo(TypedDict):
    id_: int
    generation: int
    parents_id: Optional[Union[Tuple[int], Tuple[int, int]]]
    learning_en: bool
    fitness: Optional[float]
    crossover_info: CrossoverInfo


class Individual:

    def __init__(
        self,
        id_: int,
        structure: Structure,
        generation: int,
        parents_id: Optional[Union[Tuple[int], Tuple[int, int]]],
        learning_en: bool = True,
        fitness: Optional[float] = None,
        crossover_info: Optional[CrossoverInfo] = None,
    ):

        # initialize
        self.id_ = id_
        self.generation = generation
        self.structure = structure
        self.parents_id = parents_id
        self.learning_en = learning_en
        self.fitness = fitness
        self.crossover_info = crossover_info

        # make directory to save information
        self.saving_dir = Path(
            f"./experiment/generation{(self.generation + 1):02}/id{(self.id_ + 1):02}"
        )
        self.saving_dir.mkdir(parents=False, exist_ok=False)

    def save(self):

        # save structure
        np.save(str(self.saving_dir / BODY_FILE_NAME), self.structure.body)
        np.save(
            str(self.saving_dir / CONNECTIONS_FILE_NAME), self.structure.connections
        )

        # save information in robot_info.json
        individual_info = IndividualInfo(
            id_=self.id_,
            generation=self.generation,
            parents_id=self.parents_id,
            learning_en=self.learning_en,
            fitness=self.fitness,
            crossover_info=self.crossover_info,
        )
