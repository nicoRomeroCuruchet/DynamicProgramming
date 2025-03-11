import os
import pickle
import numpy as np
import gymnasium as gym
from loguru import logger
from scipy.spatial import Delaunay
from utils.utils import plot_3D_value_function


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

    metadata = {"img_path": os.getcwd()+"/img/",}

    def __init__(self, env: gym.Env,
                 bins_space: dict,
                 action_space:np.array,
                 nsteps:int=100,
                 gamma:float= 0.99,
                 log:bool=False,
                 theta:float= 5e-2):
        
        """   Initializes the PolicyIteration object with the environment, state and action spaces, 
              and algorithm parameters.

        Parameters:
            env (gym.Env): The Gym environment to perform policy iteration on.
            bins_space (dict): The discretization of the state space.
            action_space (list): List of all possible actions.
            gamma (float): Discount factor for future rewards.
            theta (float): Small threshold for determining the convergence of the policy evaluation.
        
        Raises:
            ValueError: If action_space or bins_space is not provided or empty.
            TypeError: If action_space or bins_space is not of the correct type. """
        
        self.env:gym.Env = env       # working environment
        self.gamma:float = gamma     # discount factor
        self.theta:float = theta     # convergence threshold for policy evaluation
        self.nsteps:int = nsteps     # number of steps to run the policy iteration
        self.counter:int = 0         # counter for the number of steps
        self.log:bool = log          # log the progress of the policy iteration

        # if action space is not provided, raise an error
        if action_space is None: 
            raise ValueError("Action space must be provided.")
        if not isinstance(action_space, np.ndarray):
            raise TypeError("Action space must be a list.")
        if action_space.shape[0] == 0:
            raise ValueError("Action space cannot be empty.")
        
        # if bins_space is not provided, raise an error
        if bins_space is None:
            raise ValueError("Bins space must be provided.")
        if not isinstance(bins_space, dict):    
            raise TypeError("Bins space must be a dictionary.")
        if not bins_space:
            raise ValueError("Bins space cannot be empty.")

        self.action_space:np.ndarray = action_space
        self.bins_space:dict   = bins_space

        #get the minimum and maximum values for each dimension       
        self.cell_lower_bounds = np.array([min(v) for v in self.bins_space.values()], dtype=np.float32)
        self.cell_upper_bounds = np.array([max(v) for v in self.bins_space.values()], dtype=np.float32)
        logger.info(f"Lower bounds: {self.cell_lower_bounds}")
        logger.info(f"Upper bounds: {self.cell_upper_bounds}")
        # Generate the grid points for all dimensions
        self.grid = np.meshgrid(*self.bins_space.values(), indexing='ij')
        # Flatten and stack to create a list of points in the space
        self.states_space = np.vstack([g.ravel() for g in self.grid], dtype=np.float32).T
        # Get the terminal states
        self.terminal_states, self.terminal_reward = self.env.terminal(self.states_space)
        #
        self.num_simplex_points:int = int(self.states_space[0].shape[0] + 1) # number of points in a simplex one more than the dimension
        self.space_dim:int          = int(self.states_space[0].shape[0])
        self.action_space_size:int  = int(self.action_space.shape[0])   
        self.num_states:int         = int(self.states_space.shape[0])
        self.num_actions:int        = int(self.action_space.shape[0])

        logger.info(f"The action space is: {self.action_space}")
        logger.info(f"Number of states: {len(self.states_space)}")
        logger.info(f"Total states:{len(self.states_space)*len(self.action_space)}")
        # Initialize the transition and reward function table
        self.reward         = np.zeros((self.num_states, self.num_actions), dtype=np.float32)
        self.previous_state = np.zeros((self.num_states, self.num_actions, self.space_dim), dtype=np.float32)
        self.next_state     = np.zeros((self.num_states, self.num_actions, self.space_dim), dtype=np.float32)
        self.simplexes      = np.zeros((self.num_states, self.num_actions, self.num_simplex_points, self.space_dim), dtype=np.float32)
        self.lambdas        = np.zeros((self.num_states, self.num_actions, self.num_simplex_points, 1), dtype=np.float32)
        self.points_indexes = np.zeros((self.num_states, self.num_actions, self.num_simplex_points, 1), dtype=np.int32)
        # The policy is a mapping from states to probabilities of selecting each action
        self.policy = np.ones((self.num_states, self.num_actions), dtype=np.float32) / self.num_actions
        # The value function is an estimate of the enpected return from a given state
        self.value_function = np.zeros(self.num_states, dtype=np.float32)
        self.value_function[self.terminal_states] = self.terminal_reward
        logger.info("Policy Iteration was correctly initialized.")
        logger.info(f"The enviroment name is: {self.env.__class__.__name__}")
        

    def __in_cell__(self, obs: np.ndarray) -> np.ndarray:

        """ Check if the given observation is within the valid state bounds.

        Parameters:
            obs (np.ndarray): The observation array to check.

        Returns:
            np.ndarray: A boolean array indicating whether each observation is within the valid state bounds. """
        
        return np.all((obs >= self.cell_lower_bounds) & (obs <= self.cell_upper_bounds), axis=1)
  
    def barycentric_coordinates(self, points:np.ndarray)->tuple:

        """ Calculates the barycentric coordinates of a 2D point within a convex hull.
        Parameters:
            point (np.array): The 2D point for which to calculate the barycentric coordinates.
        Returns:
            result (np.array): The barycentric coordinates of the point.
            vertices_coordinates (np.array): The coordinates of the vertices of the simplex containing the point.
        Raises:
            ValueError: If the point is outside the convex hull. 
            ValueError: If the matrix A is singular. """

        assert points.shape == self.states_space.shape, f"point shape: {points.shape} and states_space shape: {self.states_space.shape}"
        # transform the points to a 3D space
        simplex_indexes = self.triangulation.find_simplex(points)
        # check if the point is outside the convex hull
        if np.any(simplex_indexes == -1):
            raise ValueError(f"The point {np.where(simplex_indexes == -1)[0]} is outside the convex hull.")
        
        points_indexes = self.triangulation.simplices[simplex_indexes]
        # get the simplexes
        simplexes = self.states_space[points_indexes]
        # Transpose the matrices in one go
        transposed_simplexes = simplexes.transpose(0, 2, 1)
        # Create the row of ones to be added, matching the shape (number of matrices, 1 row, number of columns)
        ones_row = np.ones((transposed_simplexes.shape[0], 1, transposed_simplexes.shape[2]))
        # Stack the transposed matrices with the row of ones along the second axis
        A = np.concatenate((transposed_simplexes, ones_row), axis=1)
        b = np.hstack([points,  np.ones((points.shape[0], 1))]).reshape(self.num_states,self.num_simplex_points,1)
        
        # Calculate the inverse of the resulting matrix
        try:
            inv_A = np.linalg.inv(A)
        except np.linalg.LinAlgError as e:
            raise ValueError(f"The matrix A is singular, using the pseudo-inverse instead:{e}.")
    
        assert inv_A.shape == (self.num_states, self.num_simplex_points, self.num_simplex_points), f"inv_A shape: {inv_A.shape}"
        assert b.shape == (self.num_states, self.num_simplex_points, 1), f"b shape: {b.shape}"

        lambdas = inv_A@b
        assert lambdas.shape == (self.num_states, self.num_simplex_points,1), f"lambdas shape: {lambdas.shape}"
        points_indexes = points_indexes.reshape(self.num_states, self.num_simplex_points,1)
        # to test recontruct the points:
        condition = np.linalg.norm(np.matmul(A, lambdas) - b, axis=1) < 1e-2
        assert np.all(condition) == True, f"condition: {condition}"
        
        return lambdas, simplexes, points_indexes
  
    def calculate_transition_reward_table(self):

        """ Computes the transition and reward table for each state-action pair.
            "next_state": The resulting state after taking the specified action in the given state.
            "reward": The reward obtained when taking the specified action in the given state.
            "previous_state": The state from which the action was taken.
            "lambdas": The barycentric coordinates of the resulting state with respect to the simplex.        
            "simplexes": The simplex associated with the resulting state.
            "points_indexes": The indexes of the points in the simplex. """   
           
        for j, action in enumerate(self.action_space):
            self.env.state = np.array(self.states_space, dtype=np.float32)  
            obs, reward, _, _, _ = self.env.step(action)
            # log if any state is outside the bounds of the environment
            states_outside = self.__in_cell__(obs)
            if bool(np.any(~states_outside)):
                reward = np.where(states_outside, reward, self.terminal_reward)
                logger.warning(f"Some states are outside the bounds of the environment.")
            # if any state is outside the bounds of the environment clip it to the bounds
            obs = np.clip(obs, self.cell_lower_bounds, self.cell_upper_bounds)
            # get the barycentric coordinates of the resulting state in npU for now.
            lambdas, simplexes, points_indexes = self.barycentric_coordinates(obs)
            # store the transition and reward information
            self.next_state[:,j]     = obs
            self.reward[:,j]         = reward
            self.previous_state[:,j] = self.states_space
            self.lambdas[:,j]        = lambdas
            self.simplexes[:,j]      = simplexes
            self.points_indexes[:,j] = points_indexes

    def get_value(self, lambdas:np.ndarray,  point_indexes:np.ndarray,  value_function:np.ndarray)->np.ndarray:

        """ Calculates the next state value based on the given lambdas, point indexes, and value function.
        Args:
            lambdas (np.ndarray): The lambdas array of shape (num_states, num_simplex_points,1).
            point_indexes (np.ndarray): The point indexes array of shape (num_states, num_simplex_points,1).
            value_function (np.ndarray): The value function.
        Returns:
            np.ndarray: The next state value.
        Raises:
            Exception: If states in point_indexes are not found in the value function. """
        
        assert lambdas.shape == (self.num_states, self.num_simplex_points,1), f"lambdas shape: {lambdas.shape}"
        assert point_indexes.shape == (self.num_states, self.num_simplex_points,1),  f"point_indexes shape: {point_indexes.shape}"

        try:
            values = value_function[point_indexes]
            next_state_value = np.einsum('ij,ij->i', lambdas.squeeze(-1), values. squeeze(-1))
        except (
            KeyError
        ):
            next_state_value = None
            raise Exception(f"States in {point_indexes} not found in the value function.")

        return next_state_value

    def policy_evaluation(self):

        """ Performs the policy evaluation step of the Policy Iteration, updating the value function.  """

        max_error = 2*self.theta
        i = 0 
        self.counter += 1
        logger.info("Starting policy evaluation")
        while np.abs(float(max_error)) > self.theta:
            # initialize the new value function to zeros
            new_value_function = np.zeros_like(self.value_function, dtype=np.float32)
            vf_next_state = np.zeros_like(self.value_function, dtype=np.float32)
            new_val = np.zeros_like(self.value_function, dtype=np.float32)
            new_value_function[self.terminal_states] = self.terminal_reward
            new_val[self.terminal_states] = self.terminal_reward

            for j, _ in enumerate(self.action_space):                
                # Checkout 'Variable Resolution Discretization in Optimal Control, eq 5'
                vf_next_state[~self.terminal_states] = self.get_value(self.lambdas[:, j], self.points_indexes[:, j], self.value_function)[~self.terminal_states]
                new_val[~self.terminal_states] += self.policy[~self.terminal_states,j] * (self.reward[~self.terminal_states,j] + self.gamma * vf_next_state[~self.terminal_states])

            new_value_function = new_val
            # update the error: the maximum difference between the new and old value functions
            errors = np.fabs(new_value_function[:] - self.value_function[:])
            self.value_function = new_value_function  # update the value function
            max_error = np.round(np.max(errors), 3)
            
            # log the progress
            if i % 150 == 0:
                mean      = np.round(np.mean(errors), 3)    
                indices   = np.where(errors<self.theta)
                logger.info(f"Max Error: {float(max_error)} | Avg Error: {float(mean)} | {errors[indices].shape[0]}<{self.theta}")
                # get date for the name of the image
                if self.log:
                    import time
                    timestamp = int(time.time())
                    img_name =  f"/3D_value_function_{timestamp}.png"
                    __path__ = PolicyIteration.metadata['img_path'] + img_name
                    # remove spaces in path
                    __path__ = __path__.replace(" ", "_")
                    vf_tmp = self.value_function
                    plot_3D_value_function(vf = vf_tmp,
                                        points = self.states_space,
                                        normalize=True,
                                        show=False,
                                        path=str(__path__))
            i += 1

        logger.info("Policy evaluation finished.")

    def policy_improvement(self)->bool:

        """ Performs the policy improvement step, updating the policy based on the current value function.

        Returns:
            bool: True if the policy is stable and no changes were made, False otherwise. """
        
        logger.info("Starting policy improvement")
        policy_stable = True
        new_policy = np.zeros_like(self.policy) # initialize the new policy to zeros
        action_values = np.zeros((self.states_space.shape[0],self.action_space.shape[0]), dtype=np.float32)
        for j, _ in enumerate(self.action_space):
            # element-wise multiplication of the policy and the result
            action_values_j = self.reward[:, j] + self.gamma * self.get_value(self.lambdas[:, j], self.points_indexes[:, j], self.value_function)
            action_values[:,j] = action_values_j 
        # update the policy to select the action with the highest value
        greedy_actions = np.argmax(action_values, axis=1)
        new_policy[np.arange(new_policy.shape[0]) ,greedy_actions] = 1 

        if not np.array_equal(self.policy, new_policy):
            logger.info(f"The number of updated different actions: {np.sum(self.policy != new_policy)}")
            policy_stable = False

        logger.info("Policy improvement finished.")
        self.policy = new_policy
        return policy_stable
        
    def run(self):

        """ Executes the Policy Iteration algorithm for a specified number of steps or until convergence. """

        # Create the Delaunay triangulation
        logger.info("Creating Delaunay triangulation over the state space...")
        self.triangulation = Delaunay(self.states_space)
        logger.info("Delaunay triangulation created.")
        
        #to plot delaunay triangulation:    
        #plt.plot(self.states_space[:, 0], self.states_space[:, 1], 'go', label='Data states_space', markersize=2)
        # plot the triangulation
        #plt.triplot(self.states_space[:, 0], self.states_space[:, 1], self.triangulation.simplices)
        #plt.scatter(-1.2,  0. , color='Red', s=10)
        #plt.show()
        
        # Generate the transition and reward function table
        logger.info("Generating transition and reward function table...")
        self.calculate_transition_reward_table()
        logger.info("Transition and reward function table generated.")
        for n in range(self.nsteps):
            logger.info(f"solving step {n}")
            self.policy_evaluation()
            if self.policy_improvement():
                break
        
        self.save()
        self.env.close()

    def save(self):

        """  Saves the policy and value function to files. """

        with open(self.env.__class__.__name__ + ".pkl", "wb") as f:
            pickle.dump(self, f)
            logger.info("Policy and value function saved.")        