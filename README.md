## Policy Iteration Algorithm for Continuous State Spaces

## Introduction
This project implements the Policy Iteration algorithm tailored for environments with continuous state spaces. By discretizing the state space, the algorithm efficiently finds optimal policies for a given Markov Decision Process (MDP).


## Dependencies Installation and Setting
- Create Conda environment with dependencies
	``` bash
	conda create --name DynamicProgramming python=3.11 ipykernel
	```
- Activate environment
	``` bash
	conda activate DynamicProgramming
	```
- Install the Policy Iteration class:
	``` bash
	pip install -e .
	```
 
## Features
- Implements Policy Iteration for continuous state spaces through discretization.
- Utilizes numerical stability techniques to ensure accurate computations.
- Provides a modular and scalable code structure for easy integration and experimentation.
