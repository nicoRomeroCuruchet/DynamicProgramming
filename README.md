## Policy Iteration Algorithm for Continuous State Spaces

## Introduction
This project implements the Policy Iteration algorithm tailored for environments with continuous state spaces. By discretizing the state space, the algorithm efficiently finds optimal policies for a given Markov Decision Process (MDP).

## Features
- Implements Policy Iteration for continuous state spaces through discretization.
- Utilizes numerical stability techniques to ensure accurate computations.
- Provides a modular and scalable code structure for easy integration and experimentation.

## Dependencies Installation
- Create Conda environment with dependencies
	``` bash
	conda env create -f environment.yml
	```
- Activate environment
	``` bash
	conda activate DynamicProgramming
	```
- You can run Policy Iteration with the gymnasium's CartPoleEnv environment by doing:
	``` bash
	python3 PolicyIteration.py
	```
