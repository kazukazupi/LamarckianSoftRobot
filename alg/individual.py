from pathlib import Path
from typing import Literal, Optional, Tuple, TypedDict, Union

import numpy as np


class CrossoverInfo(TypedDict):
    axis: Literal[0, 1]
    mid: int


class Structure(TypedDict):
    body: np.ndarray
    connections: np.ndarray


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
        self.id = id_
        self.generation = generation
        self.structure = structure
        self.parents_id = parents_id
        self.learning_en = learning_en
        self.fitness = fitness
        self.crossover_info = crossover_info

        self.saving_dir = Path(
            f"./experiment/generation{(self.generation + 1):02}/id{(self.id + 1):02}"
        )
        self.saving_dir.mkdir(parents=False, exist_ok=False)
