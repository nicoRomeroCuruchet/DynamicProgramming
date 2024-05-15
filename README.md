## Policy Iteration for Continuous Dynamics

This repository contains an implementation of **Policy Iteration (PI)** applied to environments with continuous dynamics.  The **Value Function (VF)** is approximated using linear interpolation within each simplex of the discretized state space. The interpolation coefficients act like probabilities in a stochastic process, which helps in approximating the continuous dynamics using a discrete Markov Decision Process (MDP). It was tested by the environments Cartpole and Mountain car provided by [Gymnasium](https://github.com/Farama-Foundation/Gymnasium).

The dynamic programming equation used to compute the VF [1] is given by: :

$$V(\xi) = \max_u \left[ \gamma^{\tau(\xi, u)} \sum_{i=0}^{d} p(\xi_i | \xi, u) V(\xi_i) + R(\xi, u) \right]$$ 

Here:

- $ V(\xi) $ is the value function at state $\xi.$
- $\gamma^{\tau(\xi, u)}$ is the discount factor.
- $ \tau(\xi, u) $ is the time taken to transition from state $ \xi $ under control $ u $.
- $ p(\xi_i | \xi, u) $ is the probability of transitioning to state $ \xi_i $ from $ \xi $ under control $ u $.
- $ R(\xi, u) $ is the immediate reward received from state $ \xi $ under control $ u $.

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
