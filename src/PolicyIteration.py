import os
import pickle
import gymnasium as gym
from loguru import logger
from scipy.spatial import Delaunay
from utils.utils import plot_3D_value_function


import numpy as np
try:
    import cupy as cp 
    if not cupy.cuda.runtime.is_available():
        raise ImportError("CUDA is not available. Falling back to NumPy.")
except (ImportError, AttributeError):
    import numpy as cp
    logger.warning("CUDA is not available. Falling back to NumPy.")
    def asarray(arr, *args, **kwargs):
        """In NumPy, this just ensures the object is a NumPy array, with support for additional arguments."""
        return np.array(arr, *args, **kwargs)
    
    def asnumpy(arr, *args, **kwargs):
        """In NumPy, this just ensures the object is a NumPy array."""
        return np.array(arr, *args, **kwargs) 
           
    np.asarray = asarray
    np.asnumpy = asnumpy
    cp = np


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
        self.cell_lower_bounds = cp.array([min(v) for v in self.bins_space.values()], dtype=cp.float32)
        self.cell_upper_bounds = cp.array([max(v) for v in self.bins_space.values()], dtype=cp.float32)
        logger.info(f"Lower bounds: {self.cell_lower_bounds}")
        logger.info(f"Upper bounds: {self.cell_upper_bounds}")
        # Generate the grid points for all dimensions
        self.grid = np.meshgrid(*self.bins_space.values(), indexing='ij')
        # Flatten and stack to create a list of points in the space
        self.states_space = np.vstack([g.ravel() for g in self.grid], dtype=np.float32).T
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
        self.reward         = cp.zeros((self.num_states, self.num_actions), dtype=cp.float32)
        self.previous_state = cp.zeros((self.num_states, self.num_actions, self.space_dim), dtype=cp.float32)
        self.next_state     = cp.zeros((self.num_states, self.num_actions, self.space_dim), dtype=cp.float32)
        self.simplexes      = cp.zeros((self.num_states, self.num_actions, self.num_simplex_points, self.space_dim), dtype=cp.float32)
        self.lambdas        = cp.zeros((self.num_states, self.num_actions, self.num_simplex_points, 1), dtype=cp.float32)
        self.points_indexes = cp.zeros((self.num_states, self.num_actions, self.num_simplex_points, 1), dtype=cp.int32)
        # The policy is a mapping from states to probabilities of selecting each action
        self.policy = cp.ones((self.num_states, self.num_actions), dtype=cp.float32) / self.num_actions
        # The value function is an estimate of the expected return from a given state
        self.value_function = cp.zeros(self.num_states, dtype=cp.float32)
        logger.info("Policy Iteration was correctly initialized.")
        logger.info(f"The enviroment name is: {self.env.__class__.__name__}")
        

    def __in_cell__(self, obs: cp.ndarray) -> cp.ndarray:

        """ Check if the given observation is within the valid state bounds.

        Parameters:
            obs (np.ndarray): The observation array to check.

        Returns:
            np.ndarray: A boolean array indicating whether each observation is within the valid state bounds. """
        
        return cp.all((obs >= self.cell_lower_bounds) & (obs <= self.cell_upper_bounds), axis=1)
  
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
        # transfer to gpu
        A_gpu = cp.asarray(A, dtype=np.float32)
        try:
            inv_A_gpu = cp.linalg.inv(A_gpu)
        except cp.linalg.LinAlgError as e:
            raise ValueError(f"The matrix A is singular, using the pseudo-inverse instead:{e}.")
    
        assert inv_A_gpu.shape == (self.num_states, self.num_simplex_points, self.num_simplex_points), f"inv_A shape: {inv_A_gpu.shape}"
        assert b.shape == (self.num_states, self.num_simplex_points, 1), f"b shape: {b.shape}"

        b_gpu       = cp.asarray(b, dtype=cp.float32)
        lambdas_gpu = cp.array(inv_A_gpu@b_gpu, dtype=cp.float32)

        assert lambdas_gpu.shape == (self.num_states, self.num_simplex_points,1), f"lambdas shape: {lambdas_gpu.shape}"
        points_indexes = points_indexes.reshape(self.num_states, self.num_simplex_points,1)
        # to test recontruct one point:
        #condition = cp.linalg.norm(np.matmul(A_gpu, lambdas_gpu) - b_gpu, axis=1) < 1e-2
        #assert cp.all(condition) == True, f"condition: {condition}"
        lambdas = cp.asnumpy(lambdas_gpu)
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
            self.env.state = cp.asarray(self.states_space, dtype=cp.float32)   
            obs_gpu, reward_gpu, _, _, _ = self.env.step(action)
            # log if any state is outside the bounds of the environment
            states_outside_gpu = self.__in_cell__(obs_gpu)
            if bool(cp.any(~states_outside_gpu)):
                # get the indexes of the states outside the bounds 
                reward_gpu = cp.where(states_outside_gpu, reward_gpu, -100)
                logger.warning(f"Some states are outside the bounds of the environment.")
            # if any state is outside the bounds of the environment clip it to the bounds
            obs_gpu = cp.clip(obs_gpu, self.cell_lower_bounds, self.cell_upper_bounds)
            # get the barycentric coordinates of the resulting state in CPU for now.
            obs_cpu = cp.asnumpy(obs_gpu)
            lambdas, simplexes, points_indexes = self.barycentric_coordinates(obs_cpu)
            # store the transition and reward information and transfer to gpu
            self.next_state[:,j]     = obs_gpu
            self.reward[:,j]         = reward_gpu
            self.previous_state[:,j] = cp.asarray(self.states_space, dtype=cp.float32)
            self.lambdas[:,j]        = cp.asarray(lambdas, dtype=cp.float32)
            self.simplexes[:,j]      = cp.asarray(simplexes, dtype=cp.float32)
            self.points_indexes[:,j] = cp.asarray(points_indexes, dtype=cp.int32) 

    def get_value(self, lambdas:cp.ndarray,  point_indexes:cp.ndarray,  value_function:cp.ndarray)->cp.ndarray:

        """ Calculates the next state value based on the given lambdas, point indexes, and value function.
        Args:
            lambdas (cp.ndarray): The lambdas array of shape (num_states, num_simplex_points,1).
            point_indexes (cp.ndarray): The point indexes array of shape (num_states, num_simplex_points,1).
            value_function (cp.ndarray): The value function.
        Returns:
            cp.ndarray: The next state value.
        Raises:
            Exception: If states in point_indexes are not found in the value function. """
        
        assert lambdas.shape == (self.num_states, self.num_simplex_points,1), f"lambdas shape: {lambdas.shape}"
        assert point_indexes.shape == (self.num_states, self.num_simplex_points,1),  f"point_indexes shape: {point_indexes.shape}"
        try:
            values = value_function[point_indexes]
            next_state_value = cp.einsum('ij,ij->i', lambdas.squeeze(-1), values. squeeze(-1))
        except (
            KeyError
        ):
            next_state_value = None
            raise Exception(f"States in {point_indexes} not found in the value function.")

        return next_state_value

    def policy_evaluation(self):

        """ Performs the policy evaluation step of the Policy Iteration, updating the value function.  """

        max_error = 2*self.theta
        ii = 0 
        self.counter += 1
        logger.info("Starting policy evaluation")
        while cp.abs(float(max_error)) > self.theta:
            # initialize the new value function to zeros
            new_value_function = cp.zeros_like(self.value_function, dtype=cp.float32) 
            new_val = cp.zeros_like(self.value_function, dtype=cp.float32)
            for j, _ in enumerate(self.action_space):                
                # Checkout 'Variable Resolution Discretization in Optimal Control, eq 5'
                next_state_value = self.get_value(self.lambdas[:, j], self.points_indexes[:, j], self.value_function)
                new_val += self.policy[:,j] * (self.reward[:,j] + self.gamma * next_state_value)
            new_value_function = new_val
            # update the error: the maximum difference between the new and old value functions
            errors = cp.fabs(new_value_function[:] - self.value_function[:])
            self.value_function = new_value_function  # update the value function
            # log the progress
            if ii % 150 == 0:
                mean      = cp.round(cp.mean(errors), 3)
                max_error = cp.round(cp.max(errors), 3)    
                indices   = cp.where(errors<self.theta)
                logger.info(f"Max Error: {float(max_error)} | Avg Error: {float(mean)} | {errors[indices].shape[0]}<{self.theta}")
                # get date for the name of the image
                if self.log:
                    import time
                    timestamp = int(time.time())
                    img_name =  f"/3D_value_function_{timestamp}.png"
                    __path__ = PolicyIteration.metadata['img_path'] + img_name
                    # remove spaces in path
                    __path__ = __path__.replace(" ", "_")
                    vf_tmp = cp.asnumpy(self.value_function)
                    plot_3D_value_function(vf = vf_tmp,
                                        points = self.states_space,
                                        normalize=True,
                                        show=False,
                                        path=str(__path__))
            ii += 1

        logger.info("Policy evaluation finished.")

    def policy_improvement(self)->bool:

        """ Performs the policy improvement step, updating the policy based on the current value function.

        Returns:
            bool: True if the policy is stable and no changes were made, False otherwise. """
        
        logger.info("Starting policy improvement")
        policy_stable = True
        new_policy = cp.zeros_like(self.policy) # initialize the new policy to zeros
        action_values = cp.zeros((self.states_space.shape[0],self.action_space.shape[0]), dtype=cp.float32)
        for j, _ in enumerate(self.action_space):
            # element-wise multiplication of the policy and the result
            action_values_j = self.reward[:, j] + self.gamma * self.get_value(self.lambdas[:, j], self.points_indexes[:, j], self.value_function)
            action_values[:,j] = action_values_j 
        # update the policy to select the action with the highest value
        greedy_actions = cp.argmax(action_values, axis=1)
        new_policy[cp.arange(new_policy.shape[0]) ,greedy_actions] = 1 

        if not cp.array_equal(self.policy, new_policy):
            logger.info(f"The number of updated different actions: {cp.sum(self.policy != new_policy)}")
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