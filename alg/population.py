from enum import Enum, auto
from typing import Dict, List

from alg.config import Config
from alg.individual import Individual
from alg.utils import FitnessWriter, LogWriter

POP_TXT_FILE_NAME = "log.txt"
POP_CSV_FILE_NAME = "fitness.csv"


class EvolutionState(Enum):
    CONTINUE = auto()
    END = auto()


class Population:
    def __init__(
        self,
        individual_list: List[Individual],
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
        config = Config()

        # setup log and fitness writer
        log_file_path = config.exp_dir / POP_TXT_FILE_NAME
        csv_file_path = config.exp_dir / POP_CSV_FILE_NAME

        log_writer = LogWriter(log_file_path)
        fitness_writer = FitnessWriter(csv_file_path, config.population_size)

        individual_list = []
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
            individual_list.append(individual)

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

    def train(self) -> EvolutionState:
        """
        learning phase for the current generation (= self.generation)

        Return:
            whether the evolution has been finished or not
        """

        for individual in self.individual_list:

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

            # train and set fitness value
            individual.train(self.config)
            individual.save()

            self.log_writer.print_and_write(
                f"\tterminated. fitness: {individual.fitness}"
            )

            self.num_evals += 1

            # end evolution
            if self.num_evals == self.config.max_evaluations:
                self.fitness_writer.write(self.generation, self.individual_list)
                return EvolutionState.END

        self.fitness_writer.write(self.generation, self.individual_list)

        return EvolutionState.CONTINUE
