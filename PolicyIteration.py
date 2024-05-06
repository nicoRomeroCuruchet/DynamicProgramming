import pickle
import numpy as np
from tqdm import tqdm
import gymnasium as gym
from loguru import logger
from itertools import product
from scipy.spatial import KDTree
from scipy.optimize import minimize

class PolicyIteration(object):
    """
    A class to perform Policy Iteration on discretized continuous environments.

    Attributes:
        env (gym.Env): The Gym environment to work with.
        bins_space (dict): A dictionary specifying the discretization bins for each state dimension.
        action_space (list): A list of all possible actions in the environment.
        gamma (float): The discount factor for future rewards.
        theta (float): The threshold for determining convergence in policy evaluation.
        states_space (list): The product of bins space values, representing all possible states.
        points (np.array): Numpy array of all states, used for KDTree construction.
        kd_tree (KDTree): A KDTree built from the points for efficient nearest neighbor queries.
        num_simplex_points (int): Number of points in a simplex for barycentric coordinate calculations.
        policy (dict): Current policy mapping from states to probabilities of selecting each action.
        value_function (dict): Current estimate of the value function.
        transition_reward_table (dict): A table storing transition probabilities and rewards for each state-action pair.

        Example:

            from classic_control.cartpole import CartPoleEnv 
            
            env = CartPoleEnv()
            bins_space = {
                "x_space": np.linspace(-x_lim, x_lim, 12), # position space (0)
                "x_dot_space": np.linspace(-x_dot_lim, x_dot_lim, 12), # velocity space (1)
                "theta_space": np.linspace(-theta_lim, theta_lim, 12), # angle space (2)
                "theta_dot_space": np.linspace(-theta_dot_lim, theta_dot_lim, 12), # angular velocity space (3)
            }
            action_space = [0, 1]
            pi = PolicyIteration(env, bins_space, action_space)
            pi.run()
    """
    def __init__(self, env: gym.Env,
                 bins_space: dict,
                 action_space,
                 gamma:float= 0.99,
                 theta:float= 5e-2):
        """ 
        Initializes the PolicyIteration object with the environment, state and action spaces, 
        and algorithm parameters.

        Parameters:
            env (gym.Env): The Gym environment to perform policy iteration on.
            bins_space (dict): The discretization of the state space.
            action_space (list): List of all possible actions.
            gamma (float): Discount factor for future rewards.
            theta (float): Small threshold for determining the convergence of the policy evaluation.
        
        Raises:
            ValueError: If action_space or bins_space is not provided or empty.
            TypeError: If action_space or bins_space is not of the correct type.
        """
        self.env   = env
        self.gamma = gamma  # discount factor
        self.theta = theta  # convergence threshold for policy evaluation

        # if action space is not provided, raise an error
        if action_space is None: 
            raise ValueError("Action space must be provided.")
        if not isinstance(action_space, list):
            raise TypeError("Action space must be a list.")
        if not action_space:
            raise ValueError("Action space cannot be empty.")
        
        # if bins_space is not provided, raise an error
        if bins_space is None:
            raise ValueError("Bins space must be provided.")
        if not isinstance(bins_space, dict):    
            raise TypeError("Bins space must be a dictionary.")
        if not bins_space:
            raise ValueError("Bins space cannot be empty.")

        self.action_space = action_space
        self.bins_space   = bins_space

        self.states_space = list(
            set(product(*bins_space.values())) # to avoid repeated states
        )
        self.points = np.array([np.array(e) for e in self.states_space])
        self.kd_tree = KDTree(self.points)
        self.num_simplex_points = int(self.points[0].shape[0] + 1)
        self.policy = {state: {action: 0.5 for action in self.action_space} for state in self.states_space}
        self.value_function = {state: 0 for state in self.states_space}
        self.transition_reward_table = None
        logger.info("Policy Iteration was correctly initialized.")
        logger.info(f"The enviroment name is: {self.env.__class__.__name__}")
        logger.info(f"The action space is: {self.action_space}")
        logger.info(f"Number of states: {len(self.states_space)}")

    def barycentric_coordinates(self, point:np.array, simplex:list)->np.array:
        """
        Calculates the barycentric coordinates of a point with respect to a given simplex.

        Parameters:
            point (np.array): The point for which to calculate the barycentric coordinates.
            simplex (list): The simplex as a list of points defining the simplex vertices.

        Returns:
            np.array: The barycentric coordinates of the point.
        """
        # Formulate the system of equations
        A = np.vstack([np.array(simplex).T, np.ones(len(simplex))])
        b = np.hstack([point, [1]])
        objective_function = lambda x: np.linalg.norm(A.dot(x) - b)
        # Define the constraint that the solution must be greater than zero
        constraints = ({'type': 'ineq', 'fun': lambda x: x})
        # Initial guess for the solution
        x0 = np.ones(len(simplex)) / self.num_simplex_points
        # Solve the optimization problem
        result = minimize(objective_function,
                          x0,
                          constraints=constraints,
                          tol=1e-3)
        # The approximate solution
        x_approx = result.x
        return x_approx

    def transition_reward_function(self):
        """
        Computes the transition and reward table for each state-action pair.

        Returns:
            dict: A dictionary containing the transition and reward information for each state-action pair.
                The keys are tuples of the form (state, action), and the values are dictionaries with the following keys:
                - "reward": The reward obtained when taking the specified action in the given state.
                - "next_state": The resulting state after taking the specified action in the given state.
                - "simplex": The simplex associated with the resulting state.
                - "barycentric_coordinates": The barycentric coordinates of the resulting state with respect to the simplex.
        """
        table = {}
        for state in tqdm(self.states_space):
            for action in self.action_space:
                self.env.reset()    # TODO: is this necessary? might be slow, to avoid warnings
                self.env.state = np.array(state, dtype=np.float64)  # set the state
                obs, reward, terminated, done, info = self.env.step(action)
                _, neighbors  = self.kd_tree.query([obs], k=self.num_simplex_points)
                simplex = self.points[neighbors[0]]
                lambdas = self.barycentric_coordinates(state, simplex)

                table[(state, action)] = {"reward": reward,
                                          "next_state": obs,
                                          "simplex": simplex,
                                          "barycentric_coordinates": lambdas}
                
        self.transition_reward_table = table
        

    def get_value(self, 
                  lambdas:np.array, 
                  simplex:list, 
                  value_function)->float:
        """
        Calculates the value for a state given the barycentric coordinates and simplex.

        Parameters:
            lambdas (np.array): Barycentric coordinates within the simplex.
            simplex (list): List of points defining the simplex.
            value_function (dict): The current value function.

        Returns:
            float: The calculated value of the state.

        Raises:
            Exception: If a state in the simplex is not found in the value function.
        """
        try:
            values = np.array([value_function[tuple(e)] for e in list(simplex)])
            next_state_value = np.dot(lambdas, values)
        except (
            KeyError
        ):
            logger.error(f"States in {simplex} not found in the value function.")
            raise Exception(f"States in {simplex} not found in the value function.")

        return next_state_value

    def policy_evaluation(self):
        """
        Performs the policy evaluation step of the Policy Iteration, updating the value function.
        """
        max_error = -1.0
        ii = 0 
        logger.info("Starting policy evaluation")
        while abs(max_error) > self.theta:
            new_value_function = {}
            errors = []
            for state in self.states_space:
                new_val = 0
                for action in self.action_space:
                    reward, next_state, simplex, bar_coor = self.transition_reward_table[(state, action)].values()
                    next_state_value = self.get_value(bar_coor, simplex, self.value_function)
                    # Checkout 'Variable Resolution Discretization in Optimal Control, eq 5'
                    new_val += self.policy[state][action] * (reward + self.gamma * next_state_value)
                new_value_function[state] = new_val
                # update the error: the maximum difference between the new and old value functions
                errors.append(abs(new_value_function[state] - self.value_function[state]))

            self.value_function = new_value_function # update the value function
            
            if ii % 20 == 0:    
                mean = np.round(np.mean(errors), 4)
                max_error = np.round(np.max(errors),4)
                errs = np.array(errors)
                indices = np.where(errs < self.theta)
                logger.info(f"Max Error: {max_error} | Avg Error: {mean} | {errs[indices].shape[0]}<{self.theta}")
            
            ii += 1
        logger.info("Policy evaluation finished.")

    def policy_improvement(self)->bool:
        """
        Performs the policy improvement step, updating the policy based on the current value function.

        Returns:
            bool: True if the policy is stable and no changes were made, False otherwise.
        """
        logger.info("Starting policy improvement")
        policy_stable = True
        new_policy = {}
        for state in tqdm(self.states_space):
            action_values = {}
            for action in self.action_space:
                reward, next_state, simplex, bar_coor = self.transition_reward_table[(state, action)].values()
                action_values[action] = reward + self.gamma * self.get_value(bar_coor, simplex, self.value_function)

            greedy_action, _ = max(action_values.items(), key=lambda pair: pair[1])
            
            new_policy[state] = {
                action: int(action == greedy_action) for action in self.action_space
            }
        if self.policy != new_policy:
            logger.info(f"The number of updated different actions:\
                        {sum([self.policy[state] != new_policy[state] for state in self.states_space])}")
            policy_stable = False

        logger.info("Policy improvement finished.")
        self.policy = new_policy
        return policy_stable
        
    def run(self, nsteps:int=100):
        """
        Executes the Policy Iteration algorithm for a specified number of steps or until convergence.

        Parameters:
            nsteps (int): Maximum number of iterations to run the policy iteration.
        """
        logger.info("Generating transition and reward function table...")
        self.transition_reward_function()
        logger.info("Transition and reward function table generated.")
        for n in tqdm(range(nsteps)):
            logger.info(f"solving step {n}")
            self.policy_evaluation()
            if self.policy_improvement():
                break
        
        self.save()
        self.env.close()

    def save(self):
        """
        Saves the policy and value function to files.
        """
        with open(self.env.__class__.__name__ + ".pkl", "wb") as f:
            pickle.dump(self, f)
            logger.info("Policy and value function saved.")        
