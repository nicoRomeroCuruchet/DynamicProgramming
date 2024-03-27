from itertools import product

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from cartPole import CartPoleEnv


class PolicyIteration(object):
    """Policy Iteration Algorithm for gymnasium environment"""

    def __init__(self, env: gym.Env, gamma: float = 0.99, bins_space: dict = None):
        """Initializes the Policy Iteration.

        Parameters:
        - env (gym.Env): The environment in which the agent will interact.
        - gamma (float): The discount factor for future rewards. Default is 0.99.
        - bins_space (dict): A dictionary specifying the number of bins for each state variable. Default is None.

        Returns: None"""

        self.env = env
        self.gamma = gamma  # discaunt factor

        self.action_space = env.action_space
        self.bins_space = bins_space

        self.states_space = list(
            set(product(*bins_space.values()))
        )  # avoid repited states
        self.policy = {state: {0: 0.5, 1: 0.5} for state in self.states_space}
        self.value_function = {
            state: 0 for state in self.states_space
        }  # initialize value function

    def get_state(self, np_state: np.ndarray) -> tuple:
        """Discretizes the given state values based on the provided bins dictionary.

        Parameters:
        state (tuple): The state values to be discretized.
        bins_dict (dict): A dictionary containing the bins for each state value.

        Returns:
        tuple: The discretized states space values."""

        state = tuple(np_state)
        discretized_state = []
        for s_i, (_, bins) in zip(state, self.bins_space.items()):
            # # Digitize the value and adjust the index to be 0-based
            up_index = min(np.digitize(s_i, bins), len(bins) - 1)
            # discretized_value = bins[up_index]
            # find nearest bin
            up_abs = abs(bins[up_index] - s_i)
            down_abs = abs(bins[up_index - 1] - s_i)
            if down_abs < up_abs:
                discretized_value = bins[up_index - 1]
            else:
                discretized_value = bins[up_index]

            discretized_state.append(discretized_value)

        return tuple(discretized_state)

    def get_transition_reward_function(self) -> dict:
        """Generate a transition reward function table.

        Returns:
            dict: A dictionary representing the transition reward function table.
                The keys are tuples of (state, action), and the values are dictionaries
                with 'reward' and 'next_state' as keys."""

        table = {}
        for state in tqdm(self.states_space):
            for action in range(self.env.action_space.n):
                self.env.reset() # TODO: is this necessary? might be slow
                self.env.state = np.array(state, dtype=np.float64)  # set the state
                obs, _, terminated, done, info = self.env.step(action)
                obs = self.get_state(obs)
                reward = (
                    0 if (-0.2 < obs[2] < 0.2) and (-2.4 < obs[0] < 2.4) else -1
                )  # TODO remove this hardcoded reward
                table[(state, action)] = {"reward": reward, "next_state": obs}

        return table

    def get_value(self, state, value_function):
        """Retrieves the value of a given state from the value function.

        Parameters:
            state (any): The state for which the value needs to be retrieved.
            value_function (dict): A dictionary representing the value function.

        Returns:
            float: The value of the given state from the value function."""

        try:
            next_state_value = value_function[state]
        except (
            KeyError
        ):  # if next_state is not in value_function, assume it's a 'dead' state.
            next_state_value = -500
        return next_state_value

    def evaluate_policy(self, transition_and_reward_function: dict) -> dict:
        """Evaluates the given policy using the provided transition and reward function.

        Args:
            transition_and_reward_function (dict): A dictionary representing the transition and reward function.

        Returns:
            dict: A dictionary representing the new value function after evaluating the policy.
        """
        theta = 1e-2 # convergence threshold
        
        while True:
            delta = 0
            new_value_function = {}
            for state in self.states_space:
                new_val = 0
                for action in [0, 1]:
                    reward, next_state = transition_and_reward_function[
                        (state, action)
                    ].values()
                    next_state_value = self.get_value(next_state, self.value_function)
                    new_val += self.policy[state][action] * (
                        reward + self.gamma * next_state_value
                    )
                new_value_function[state] = new_val

            delta = max(delta, max(abs(new_value_function[state] - self.value_function[state]) for state in self.states_space))
            print(f"delta: {delta}")
            if delta < theta:
                break

            self.value_function = new_value_function
        return new_value_function

    def improve_policy(self, transition_and_reward_function: dict) -> dict:
        """Improves the current policy based on the given transition and reward function.

        Args:
            transition_and_reward_function (dict): A dictionary representing the transition and reward function.
                The keys are tuples of (state, action) and the values are dictionaries with 'reward' and 'next_state' keys.

        Returns:
            dict: The new policy after improvement."""
        
        policy_stable = True
        new_policy = {}

        for state in self.states_space:
            action_values = {}
            for action in [0, 1]:
                reward, next_state = transition_and_reward_function[
                    (state, action)
                ].values()
                action_values[action] = reward + self.gamma * self.get_value(
                    next_state, self.value_function
                )
            greedy_action, _ = max(action_values.items(), key=lambda pair: pair[1])
            new_policy[state] = {
                action: 1 if action is greedy_action else 0 for action in [0, 1]
            }
        if self.policy != new_policy:
            print(f"number of different actions: {sum([self.policy[state][0] != new_policy[state][0] for state in self.states_space])}")
            policy_stable = False

        self.policy = new_policy
        return policy_stable

    def run(self, nsteps):
        """Runs the policy iteration algorithm for a specified number of steps.

        Parameters:
        - nsteps (int): The number of steps to run the algorithm for. Default is 10 steps.
        """

        print("Generating transition and reward function table...")
        transition_and_reward_function = self.get_transition_reward_function()
        print("Running Policy Iteration algorithm...")
        for n in tqdm(range(nsteps)):
            print(f"solving step {n}")
            self.evaluate_policy(transition_and_reward_function)
            if self.improve_policy(transition_and_reward_function):
                break


def get_optimal_action(state, optimal_policy):
    """Returns the optimal action for a given state based on the optimal policy.

    Parameters:
    state (int): The current state.
    optimal_policy (dict): The optimal policy containing the action-value pairs for each state.

    Returns:
    int: The optimal action for the given state."""

    greedy_action, _ = max(optimal_policy[state].items(), key=lambda pair: pair[1])
    return greedy_action


if __name__ == "__main__":
    x_lim = 2.5
    x_dot_lim = 2.5
    theta_lim = 0.25
    theta_dot_lim = 2.5


    bins_space = {
        "x_space": np.linspace(-x_lim, x_lim, 20),
        "x_dot_space": np.linspace(-x_dot_lim, x_dot_lim, 20),
        "theta_space": np.linspace(-theta_lim, theta_lim, 20),
        "theta_dot_space": np.linspace(-theta_dot_lim, theta_dot_lim, 20),
    }

    pi = PolicyIteration(
        env=CartPoleEnv(sutton_barto_reward=False), bins_space=bins_space
    )

    STEPS = 10000
    # start the policy iteration algorithm
    pi.run(nsteps=STEPS)

    num_episodes = 10000
    cartpole = CartPoleEnv(render_mode="human")
    max_obs = np.array([0.0, 0.0, 0.0, 0.0])
    min_obs = np.array([0.0, 0.0, 0.0, 0.0])
    limits = np.array([x_lim, x_dot_lim, theta_lim, theta_dot_lim])
    for episode in range(0, num_episodes):
        observation, _ = cartpole.reset()
        for timestep in range(1, 1000):
            action = get_optimal_action(pi.get_state(observation), pi.policy)
            observation, reward, done, terminated, info = cartpole.step(action)
            max_obs = np.maximum(max_obs, observation)
            min_obs = np.minimum(min_obs, observation)
            if done:
                print(f"max_obs: {max_obs}")
                print(f"min_obs: {min_obs}")
                #check limits
                if np.all(max_obs <= limits) and np.all(min_obs >= -limits):
                    print(f"Episode {episode} finished after {timestep} timesteps")
                else:
                    print(f"Episode {episode} finished after {timestep} timesteps with out of limits observations")
                break
