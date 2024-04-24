import csv
from collections import UserList
from pathlib import Path
from typing import List

from alg.individual import Individual


class IndexableList(UserList):
    def __setitem__(self, index, value):
        if len(self.data) == index:
            self.data.append(value)
        else:
            self.data[index] = value

    def append(self, value):
        raise NotImplementedError()


class LogWriter:
    """
    print message on console and save in txt file
    """

    def __init__(self, filepath: Path):
        self.filepath = filepath

    def print_and_write(self, message: str):

        print(message)

        if self.filepath.exists():
            mode = "a"
        else:
            mode = "w"

        with self.filepath.open(mode=mode, newline="\n") as fd:
            fd.write(message + "\n")


class FitnessWriter:
    """
    save fitness
    """

    def __init__(self, filepath: Path, pop_size: int):

        self.filepath = filepath
        self.pop_size = pop_size

        with self.filepath.open(mode="w") as fd:
            writer = csv.writer(fd)
            row = ["i_generation"] + [f"id{id_}" for id_ in range(self.pop_size)]
            writer.writerow(row)

    def write(self, generation: int, individual_list: IndexableList):

        assert len(individual_list) == self.pop_size

        with self.filepath.open(mode="a") as fd:
            writer = csv.writer(fd)
            row = [generation] + [
                individual.info.fitness for individual in individual_list
            ]
            writer.writerow(row)
