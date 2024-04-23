from alg.config import Config
from alg.individual import Individual


class Population:
    def __init__(self, individual_list, group_hashes, generation, num_evals):

        self.individual_list = individual_list
        self.group_hashes = group_hashes
        self.generation = generation
        self.num_evals = num_evals

    @classmethod
    def initialize(cls):

        config = Config()

        individual_list = []
        group_hashes = {}
        generation = 0
        num_evals = 0

        generation_dir = config.exp_dir / f"generation{generation:02}"
        generation_dir.mkdir(parents=False, exist_ok=False)

        # sample robots
        for id_ in range(config.population_size):

            individual = Individual.init_random(
                id_=id_,
                generation=generation,
                robot_shape=config.shape,
                generation_dir=generation_dir,
            )

            while individual.hash in group_hashes:
                individual.reborn(config.shape)

            group_hashes[individual.hash] = True
            individual.save()
            individual_list.append(individual)

        return cls(individual_list, group_hashes, generation, num_evals)
