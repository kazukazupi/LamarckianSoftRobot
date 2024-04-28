import sys

sys.path.append(".")
from alg import Population

if __name__ == "__main__":

    population = Population.initialize()
    population.evolve()

# python ./example/run.py --max-iters 10 --population-size 5 --max-evaluations 10 --exp-dir ./result/experiment
