from typing import Dict, List

from alg.config import Config
from alg.individual import Individual


class Population:
    def __init__(
        self,
        individual_list: List[Individual],
        group_hashes: Dict[str, bool],
        generation: int,
        num_evals: int,
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

    @classmethod
    def initialize(cls):
        """
        initialize population randomly from scratch
        """

        config = Config()

        individual_list = []
        group_hashes = {}
        generation = 0
        num_evals = 0

        generation_dir = config.exp_dir / f"generation{generation:02}"
        generation_dir.mkdir(parents=False, exist_ok=False)

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

        return cls(individual_list, group_hashes, generation, num_evals)
