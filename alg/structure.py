from pathlib import Path
from typing import NamedTuple

import numpy as np

BODY_FILE_NAME = "body.npy"
CONNECTIONS_FILE_NAME = "connections.npy"


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
