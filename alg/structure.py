from pathlib import Path
from typing import Dict, Optional

import numpy as np
from evogym import has_actuator  # type: ignore
from evogym import draw, get_full_connectivity, hashable, is_connected

from alg.config import Config
from alg.globals import BODY_FILE_NAME, CONNECTIONS_FILE_NAME


class Structure:
    def __init__(self, body: np.ndarray, connections: Optional[np.ndarray] = None):

        self.body: np.ndarray = body
        self.connections: np.ndarray = (
            get_full_connectivity(self.body) if connections is None else connections
        )

    def save(self, saving_dir: Path):
        np.save(str(saving_dir / BODY_FILE_NAME), self.body)
        np.save(str(saving_dir / CONNECTIONS_FILE_NAME), self.connections)

    @staticmethod
    def load(saving_dir: Path):

        # load structure
        body = np.load(str(saving_dir / BODY_FILE_NAME))
        connections = np.load(str(saving_dir / CONNECTIONS_FILE_NAME))

        return Structure(body, connections)

    def as_tuple(self):
        return (self.body, self.connections)


def mutate_structure(
    parent_structure: Structure,
    config: Config,
    group_hashes: Dict[str, bool],
    num_attempts: int = 100,
) -> Optional[Structure]:

    parent_body = parent_structure.body

    # probability distribution
    voxel_pd = [0.6, 0.2, 0.2, 0.2, 0.2]
    mutation_pd = [config.mutation_rate, 1 - config.mutation_rate]

    for attempt in range(num_attempts):

        child_body = np.zeros_like(parent_body)

        # mutate
        for i in range(child_body.shape[0]):
            for j in range(child_body.shape[1]):
                if draw(mutation_pd) == 0:
                    child_body[i][j] = draw(voxel_pd)
                else:
                    child_body[i][j] = parent_body[i][j]

        if (
            is_connected(child_body)
            and has_actuator(child_body)
            and (not np.array_equal(child_body, parent_body))
            and (not hashable(child_body) in group_hashes)
        ):
            structure = Structure(child_body)
            group_hashes[hashable(child_body)] = True

            return structure

    return None
