# Lamarckian Soft Robot

[日本語版のREADMEはこちら](./README-ja.md)

## Overview
This is the GitHub repository for the implementation of the paper "Lamarckian Co-design of Soft Robots via Transfer Learning," which was accepted at the [Genetic and Evolutionary Computation Conference (GECCO-2024)](http://gecco-2024.sigevo.org/HomePage). By sharing the learning results through deep reinforcement learning among individuals, it became possible to evolve soft robots more efficiently. For robot simulation engine, we uses [evogym](https://evolutiongym.github.io/).

## Environment Setup

### Setting up evogym Environment
Follow the Installation instructions in the [evogym GitHub repository](https://github.com/EvolutionGym/evogym) to set up the environment for running evogym. Since validation has only been done using conda, it is recommended to use conda.

### Setting up Environment for this Repository
Clone this repository:
```bash
git clone git@github.com:kazukazupi/LamarckianSoftRobot.git
```

Then enter the evogym environment:
```bash
conda activate evogym
```
Install the specific modules for this repository as follows:

```bash
conda install pydantic
```

## Making Pull Requests
If you're making a Pull Request, make sure to also install the formatter and linter:

```bash
conda install isort
conda install black
conda install mypy
```

Run the formatter and linter before making the Pull Request:
```bash
isort ./
black ./
mypy ./
```

## Tutorial
### Trying it out
You can evolve robots with the following code (it takes a few minutes to run):
```bash
python ./example/run.py --max-iters 10 --population-size 5 --max-evaluations 10 --exp-dir ./result/experiment
```
Refer to [./alg.config.py](./alg/config.py) for the command-line arguments and their meanings.

### Resuming interrupted experiments
If the experiment is interrupted due to an accident or other reason, you can resume the experiment by specifying the directory to store the experiment results:
```bash
python ./example/from_middle.py -e <directory of the experiment to resume>
```

### Performing experiments with different conditions
There may be times when you want to conduct experiments with different conditions while keeping the learning results of the initial generation intact. In such cases, first copy the initial generation of experiments to another experiment directory:
```bash
python example/copy_first_generation.py -s ./result/experiment -d ./result/experiment_non_inherit -n
```
Then, resume the evolution of the newly born experiment directory:
```bash
python ./example/from_middle.py --exp-dir ./result/experiment_non_inherit
```