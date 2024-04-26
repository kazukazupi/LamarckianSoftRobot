import glob
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, cast

import numpy as np

from alg.config import Config
from alg.individual import Individual
from alg.structure import Structure, mutate_structure
from alg.utils import FitnessWriter, IndexableList, LogWriter

POP_TXT_FILE_NAME = "log.txt"
POP_CSV_FILE_NAME = "fitness.csv"


class EvolutionState(Enum):
    CONTINUE = auto()
    END = auto()


class Population:
    def __init__(
        self,
        individual_list: IndexableList,
        group_hashes: Dict[str, bool],
        generation: int,
        num_evals: int,
        log_writer: LogWriter,
        fitness_writer: FitnessWriter,
        config: Config,
    ):
        """
        Args:
            individual_list: individuals in the population
            group_hashes: ensure robots with the same body are not born
            generation: generation number to start evolution
            num_evals: how many robots have been trained so far
        """

        self.individual_list = individual_list
        self.group_hashes = group_hashes
        self.generation = generation
        self.num_evals = num_evals
        self.log_writer = log_writer
        self.fitness_writer = fitness_writer
        self.config = config

    @classmethod
    def initialize(cls):
        """
        initialize population randomly from scratch
        """

        # read commandline args and make experiment dir
        config = Config.initialize()

        # setup log and fitness writer
        log_file_path = config.exp_dir / POP_TXT_FILE_NAME
        csv_file_path = config.exp_dir / POP_CSV_FILE_NAME

        log_writer = LogWriter(log_file_path)
        fitness_writer = FitnessWriter(csv_file_path, config.population_size)

        individual_list = IndexableList([])
        group_hashes = {}
        generation = 0
        num_evals = 0

        # make "generation00" directory
        generation_dir = config.exp_dir / f"generation{generation:02}"
        generation_dir.mkdir(parents=False, exist_ok=False)

        log_writer.print_and_write(f"started evolution.")
        log_writer.print_and_write(f"save results at: {str(config.exp_dir)}")

        for id_ in range(config.population_size):

            # sample a robot
            individual = Individual.init_random(
                id_=id_,
                generation=generation,
                robot_shape=config.shape,
                generation_dir=generation_dir,
            )

            # ensure that robots with the same body are not born
            while individual.hash in group_hashes:
                individual.reborn(config.shape)
            group_hashes[individual.hash] = True

            # save robot information
            individual.save()
            individual_list[id_] = individual

        log_writer.print_and_write(f"initialized {config.population_size} robots.")

        population = cls(
            individual_list,
            group_hashes,
            generation,
            num_evals,
            log_writer,
            fitness_writer,
            config,
        )

        return population

    @classmethod
    def load(cls, exp_dir: Path):
        """
        load interrupted experiment
        """

        # load configuration
        assert exp_dir.exists()
        config = Config.load(exp_dir)

        # setup log and fitness writer
        log_file_path = config.exp_dir / POP_TXT_FILE_NAME
        csv_file_path = config.exp_dir / POP_CSV_FILE_NAME
        log_writer = LogWriter(log_file_path)
        fitness_writer = FitnessWriter(csv_file_path, config.population_size)

        individual_list = IndexableList([])
        group_hashes = {}
        generation = 0
        num_evals = 0

        while True:

            generation_dir = config.exp_dir / f"generation{generation:02}"
            individual_dir_list = sorted(glob.glob(str(generation_dir / "id*")))
            path_list = [Path(val) for val in individual_dir_list]

            # retrieve information for each robot
            for path in path_list:
                individual = Individual.load(path)
                id_ = individual.id_
                individual_list[id_] = individual
                group_hashes[individual.hash] = True

                if individual.info.fitness is not None:
                    num_evals += 1

            # end retrieval
            if not (config.exp_dir / f"generation{(generation + 1):02}").exists():
                message = f"restardted evolution. (generation: {generation}, num_evals: {num_evals})"
                log_writer.print_and_write(message)
                break

            # go to next generation
            fitness_writer.write(generation, individual_list)
            generation += 1

        population = cls(
            individual_list,
            group_hashes,
            generation,
            num_evals,
            log_writer,
            fitness_writer,
            config,
        )

        return population

    def train(self) -> EvolutionState:
        """
        learning phase for the current generation (= self.generation)

        Return:
            whether the evolution has been finished or not
        """

        for individual in self.individual_list:

            # end evolution
            if self.num_evals > self.config.max_evaluations:
                self.fitness_writer.write(self.generation, self.individual_list)
                return EvolutionState.END

            id_ = individual.id_

            # skip training if robot has already learned
            if individual.fitness is not None:
                self.log_writer.print_and_write(
                    f"skip training robot {id_} (parents: {individual.parents_id})."
                )
                continue

            self.log_writer.print_and_write(
                f"training robot {id_} (parents: {individual.parents_id})..."
            )

            # set parents
            if individual.info.parents_id is None:
                parents = None
            else:
                parents = tuple(
                    [self.individual_list[id_] for id_ in individual.info.parents_id]
                )

            # train and set fitness value
            individual.train(self.config, parents)
            individual.save()

            self.log_writer.print_and_write(
                f"\tterminated. fitness: {individual.fitness}"
            )

            self.num_evals += 1

        # save fitness information
        self.fitness_writer.write(self.generation, self.individual_list)

        return EvolutionState.CONTINUE

    def reproduction(self, num_attempts: int = 100):
        """
        reproduction phase for the current generation (= self.generation)
        """

        # advance generation
        self.generation += 1
        generation_dir = self.config.exp_dir / f"generation{self.generation:02}"
        generation_dir.mkdir()

        # determine how many robots survive
        elite_rate = (self.config.max_evaluations - self.num_evals - 1) / (
            self.config.max_evaluations - 1
        ) * (
            self.config.elite_rate_high - self.config.elite_rate_low
        ) + self.config.elite_rate_low
        num_survivors = int(max([2, np.ceil(elite_rate * self.config.population_size)]))

        # make fitness ranking
        fitness_array = np.array(
            [individual.info.fitness for individual in self.individual_list]
        )
        assert all([f is not None for f in fitness_array])  # make sure training is done

        # select elite individuals
        id_ranking = np.argsort(-fitness_array)
        elite_id_list = id_ranking[0:num_survivors]

        for message in [
            "--------------------------------------------------------------------------------------------",
            f"genetic operation (elite_rate: {elite_rate}, num_survivors: {num_survivors})",
            "--------------------------------------------------------------------------------------------",
            f"elites: {elite_id_list}.",
        ]:
            self.log_writer.print_and_write(message)

        # configure child robots
        for id_, individual in enumerate(self.individual_list):

            # keep as it is if this robot is elite
            if id_ in elite_id_list:
                continue

            error_flag = True

            # mutation
            for attempt in range(num_attempts):

                parent_id = np.random.choice(elite_id_list)
                parent_structure = self.individual_list[parent_id].structure
                optional_child_structure = mutate_structure(
                    parent_structure,
                    self.config,
                    self.group_hashes,
                )

                if optional_child_structure is not None:
                    error_flag = False
                    break

            if error_flag:
                raise NotImplementedError()

            child_structure = cast(
                Structure, optional_child_structure
            )  # for static analysis

            # save child information
            child_individual = Individual.init_designated_structure(
                structure=child_structure,
                id_=id_,
                generation=self.generation,
                generation_dir=generation_dir,
                parents_id=(parent_id,),
            )

            self.group_hashes[child_individual.hash] = True

            child_individual.save()
            self.individual_list[id_] = child_individual

            self.log_writer.print_and_write(
                f"individual {id_} was reproduced from {child_individual.parents_id} by mutation."
            )

    def evolve(self):

        while True:
            for message in [
                "==============================================================================================",
                f"generation: {(self.generation):02}",
                "==============================================================================================",
            ]:
                self.log_writer.print_and_write(message)

            evolution_state = self.train()

            if evolution_state == EvolutionState.END:
                self.log_writer.print_and_write("end evolution.")
                return

            self.reproduction()
