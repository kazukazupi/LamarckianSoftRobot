import csv
import shutil
from collections import UserList
from pathlib import Path

from alg.config import Config


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


def copy_first_generation(
    src_exp_dir: Path, dst_exp_dir: Path, non_inherit: bool, crossover_rate: float
):
    """
    To conduct experiments with altered conditions, copy only the first generation of experiment directory.
    """

    assert (src_exp_dir / "generation01").exists()

    dst_exp_dir.mkdir()

    shutil.copytree(
        src=(src_exp_dir / "generation00"), dst=(dst_exp_dir / "generation00")
    )

    config = Config.load(src_exp_dir)
    config.exp_dir = dst_exp_dir

    if config.non_inherit != non_inherit:
        print(f"changed config.non_inherit from {config.non_inherit} to {non_inherit}")
        config.non_inherit = non_inherit
    if config.crossover_rate != crossover_rate:
        print(
            f"changed config.non_inherit from {config.crossover_rate} to {crossover_rate}"
        )
        config.crossover_rate = crossover_rate

    config.save()
