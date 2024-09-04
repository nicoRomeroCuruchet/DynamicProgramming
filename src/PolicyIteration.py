import os
import pickle
import numpy as np
from tqdm import tqdm
import gymnasium as gym
from loguru import logger
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.optimize import minimize
from utils.utils import plot_2D_value_function,\
                         plot_3D_value_function


import jax.numpy as jnp
from jax import jit


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
        self.env:gym.Env = env    # working environment
        self.gamma:float = gamma  # discount factor
        self.theta:float = theta  # convergence threshold for policy evaluation
        self.nsteps:int = nsteps  # number of steps to run the policy iteration
        self.counter:int = 0      # counter for the number of steps

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
        self.cell_lower_bounds = np.array([min(v) for v in self.bins_space.values()], dtype=jnp.float32)
        self.cell_upper_bounds = np.array([max(v) for v in self.bins_space.values()], dtype=jnp.float32)
        logger.info(f"Lower bounds: {self.cell_lower_bounds}")
        logger.info(f"Upper bounds: {self.cell_upper_bounds}")
        # Generate the grid points for all dimensions
        self.grid = jnp.meshgrid(*self.bins_space.values(), indexing='ij')
        # Flatten and stack to create a list of points in the space
        self.states_space = jnp.vstack([g.ravel() for g in self.grid], dtype=np.float32).T
        # Create the Delaunay triangulation
        logger.info("Creating Delaunay triangulation...")
        self.triangulation = Delaunay(self.states_space)
        logger.info("Delaunay triangulation created.")
    
        #plt.plot(self.states_space[:, 0], self.states_space[:, 1], 'go', label='Data states_space', markersize=2)
        # plot the triangulation
        #plt.triplot(self.states_space[:, 0], self.states_space[:, 1], self.triangulation.simplices)
        #plt.scatter(-1.2,  0. , color='Red', s=10)
        #plt.show()

        self.num_simplex_points = int(self.states_space[0].shape[0] + 1) # number of points in a simplex one more than the dimension

        num_states = self.states_space.shape[0]
        num_actions = self.action_space.shape[0]

        dtype = [('reward', jnp.float32), 
                 ('previous_state', jnp.float32, self.states_space[0].shape), 
                 ('next_state', jnp.float32, self.states_space[0].shape), 
                 ('lambdas', jnp.float32, (1,self.num_simplex_points)),
                 ('simplex',  jnp.float32, (self.num_simplex_points, self.states_space[0].shape[0])),                
                 ('points_indexes', jnp.int32, (self.num_simplex_points,))] 

        # Initialize the transition and reward function table
        self.transition_reward_table = np.zeros((num_states, num_actions), dtype=dtype)
        # The policy is a mapping from states to probabilities of selecting each action
        self.policy = np.ones((num_states, num_actions), dtype=jnp.float32) / num_actions
        # The value function is an estimate of the expected return from a given state
        self.value_function = jnp.zeros(num_states, dtype=jnp.float32)
        logger.info("Policy Iteration was correctly initialized.")
        logger.info(f"The enviroment name is: {self.env.__class__.__name__}")
        logger.info(f"The action space is: {self.action_space}")
        logger.info(f"Number of states: {len(self.states_space)}")
        logger.info(f"Total states:{len(self.states_space)*len(self.action_space)}")

    def __check_state__(self, obs:np.array)->bool:
        """
        Checks if the given state is within the bounds of the environment.

        Parameters:
            obs (np.array): The obs to check.

        Returns:
            bool: True if the obs is within the bounds, False otherwise.
        """ 
        return np.all((obs >= self.cell_lower_bounds) & (obs <= self.cell_upper_bounds), axis=1)

    def barycentric_coordinates_2D(self, point:np.array)->tuple:
        """
        Calculates the barycentric coordinates of a 2D point within a convex hull.
        Parameters:
        - point: np.array
            The 2D point for which to calculate the barycentric coordinates.
        Returns:
        - result: np.array
            The barycentric coordinates of the point.
        - vertices_coordinates: np.array
            The coordinates of the vertices of the simplex containing the point.
        Raises:
        - ValueError: If the point is outside the convex hull.
        """
        simplex_index = self.triangulation.find_simplex(point)
        if simplex_index != -1:  # -1 indicates that the point is outside the convex hull
            simplex_vertices = self.triangulation.simplices[simplex_index]
            vertices_coordinates = self.points[simplex_vertices]
        else:
            logger.error(f"The point {point} is outside the convex hull.")
            raise ValueError(f"The point {point} is outside the convex hull.")
        
        vertices_coordinates = self.points[simplex_vertices]
        a, b, c = vertices_coordinates[0], vertices_coordinates[1],  vertices_coordinates[2]
        v0, v1, v2 = b - a, c - a, point - a
        # Compute the denominator
        den = v0[0] * v1[1] - v1[0] * v0[1]
        # Calculate barycentric coordinates
        v = (v2[0] * v1[1] - v1[0] * v2[1]) / den
        w = (v0[0] * v2[1] - v2[0] * v0[1]) / den
        u = 1.0 - v - w
        # Check if the point is inside the simplex
        result = np.array([u, v, w], dtype=np.float32)
        if np.any(result < -1.0e-2) and abs(np.sum(result) - 1.0) > 1.0e-2:
            logger.error(f"The point {point} is outside the convex hull.")
            raise ValueError(f"The point {point} is outside the convex hull.")
        
        return result, np.array(vertices_coordinates, dtype=np.float32)
    
    
    def barycentric_coordinates(self, points:np.array)->tuple:

        """
        Calculates the barycentric coordinates of a 2D point within a convex hull.
        Parameters:
        - point: np.array
            The 2D point for which to calculate the barycentric coordinates.
        Returns:
        - result: np.array
            The barycentric coordinates of the point.
        - vertices_coordinates: np.array
            The coordinates of the vertices of the simplex containing the point.
        Raises:
        - ValueError: If the point is outside the convex hull.
        """
        assert points.shape == self.states_space.shape, f"point shape: {points.shape} and states_space shape: {self.states_space.shape}"

        simplex_indexes = self.triangulation.find_simplex(points)
        # check if the point is outside the convex hull
        if jnp.any(simplex_indexes == -1):  # -1 indicates that the point is outside the convex hull
            raise ValueError(f"The point {jnp.where(simplex_indexes == -1)[0]} is outside the convex hull.")
        
        points_indexes = self.triangulation.simplices[simplex_indexes]
        
        simplexes = self.states_space[points_indexes]
        # Transpose the matrices in one go
        transposed_simplexes = simplexes.transpose(0, 2, 1)

        # Create the row of ones to be added, matching the shape (number of matrices, 1 row, number of columns)
        ones_row = np.ones((transposed_simplexes.shape[0], 1, transposed_simplexes.shape[2]))

        # Stack the transposed matrices with the row of ones along the second axis
        A = np.concatenate((transposed_simplexes, ones_row), axis=1)
        # Calculate the inverse of the resulting matrix
        try:
            inv_A = np.linalg.inv(A)
        except np.linalg.LinAlgError as e:
            #penrose-Moore pseudo inverse and log
            inv_A = np.linalg.pinv(A)
            logger.warning(f"The matrix A is singular, using the pseudo-inverse instead:{e}.")
        
        b = jnp.hstack([points,  np.ones((points.shape[0], 1))]).reshape(625,3,1)

        lambdas = np.array(inv_A@b, dtype=np.float32)
        print(f"lambdas shape: {lambdas.shape}")

        # recontruct the points
        print(A[0,:,:] @ lambdas[0,:,:], b[0,:,:])
        
        # check the correct shapes for the matrices multiplication
        assert inv_A.shape == (self.num_simplex_points, self.num_simplex_points), f"inv_A shape: {inv_A.shape}"
        assert b.shape == (self.num_simplex_points,), f"b shape: {b.shape}"
        # get barycentric coordinates: lambdas = A^-1 * b
        lambdas = jnp.array(inv_A@b.T,dtype=jnp.float32).reshape(1,self.num_simplex_points)
        # Check if the point is inside the simplex
        if jnp.any(lambdas < -1.0e-2) and abs(jnp.sum(lambdas) - 1.0) > 1.0e-2:
            logger.error(f"The point {point} is outside the convex hull.")
            raise ValueError(f"The point {point} is outside the convex hull.")
        
        assert lambdas.shape == (1,self.num_simplex_points), f"lambdas shape: {lambdas.shape}"
        return lambdas, (simplex, points_indexes)
    

    def step(self, state:np.array, action:float)->tuple:

        min_action = -1.0
        max_action = 1.0
        min_position = -1.2
        max_position = 0.6
        max_speed = 0.07
        goal_position = (
            0.45  # was 0.5 in gymnasium, 0.45 in Arnaud de Broissia's version
        )
        goal_velocity = (
            0.0  # was 0.0 in gymnasium, 0.0 in Arnaud de Broissia's version
        )
        power = 0.0008 # 0.0015

        position = state[:,0].copy()  # avoid modifying the original grid
        velocity = state[:,1].copy()  # avoid modifying the original grid

        force = min(max(action, min_action), max_action)
        velocity += force * power - 0.0025 * jnp.cos(3 * position)
        velocity = jnp.clip(velocity, -max_speed, max_speed)

        position += velocity
        position = jnp.clip(position, min_position, max_position)

        velocity = jnp.where((position == min_position) & (velocity < 0), 0, velocity)
        terminated = jnp.where((position >= goal_position) & (velocity >= goal_velocity), True, False)

        reward = jnp.zeros_like(terminated, dtype=np.float32)
        reward = jnp.where(terminated, 100.0, reward)
        reward -= jnp.pow(action, 2) * 0.1

        return jnp.vstack([position, velocity]).T, reward, terminated, False, {}

    def calculate_transition_reward_table(self):
        """
        Computes the transition and reward table for each state-action pair.
        
        Returns:
            dict: A dictionary containing the transition and reward information for each state-action pair.
                The keys are tuples of the form (state, action), and the values are dictionaries with the following keys:
                - "reward": The reward obtained when taking the specified action in the given state.
                - "previous_state": The state from which the action was taken.
                - "next_state": The resulting state after taking the specified action in the given state.
                - "simplex": The simplex associated with the resulting state.
                - "barycentric_coordinates": The barycentric coordinates of the resulting state with respect to the simplex.
        """
        for j, action in enumerate(self.action_space):
            self.env.state = jnp.array(self.states_space, dtype=jnp.float32)          
            obs, reward, _, _, _ = self.step(self.states_space, action)
            # log if any state is outside the bounds of the environment
            if not jnp.any(self.__check_state__(obs)):
                logger.warning(f"State {obs} is outside the bounds of the environment.")
            # if any state is outside the bounds of the environment, set the reward to -100
            reward = jnp.where(self.__check_state__(obs), reward, -100)
                
            lambdas, simplex_info = self.barycentric_coordinates(obs)
            simplex, points_indexes = simplex_info  # the points indexes for the state space and the simplex
            # assert shapes
            assert points_indexes.shape == (self.num_simplex_points,), f"points_indexes shape: {points_indexes.shape}"
            assert lambdas.shape == (1, self.num_simplex_points), f"lambdas shape: {lambdas.shape}"
            assert simplex.shape == (self.num_simplex_points, self.states_space.shape[0]),  f"simplex shape: {simplex.shape}"
            # store the transition and reward information
            self.transition_reward_table[:, j] = (reward, self.states_space, obs, lambdas, simplex, points_indexes)                

    def get_value(self, 
                  lambdas:np.array, 
                  point_indexes:np.array, 
                  value_function)->float:
        """
        Calculates the VF interpolation for a state given the barycentric coordinates and simplex.
        Doing this interpolation is thus mathematically equivalent to probabilistically jumping to
        a vertex: we approximate a deterministic continuous process by a stochastic discrete one

        Parameters:
            lambdas (np.array): Barycentric coordinates within the simplex.
            simplex (list): List of points defining the simplex.
            value_function (dict): The current value function.

        Returns:
            float: The calculated value of the state.
        Raises:
            Exception: If a state in the simplex is not found in the value function.
        """
        assert lambdas.shape == (1,self.num_simplex_points), f"lambdas shape: {lambdas.shape}"
        assert point_indexes.shape == (self.num_simplex_points,),  f"point_indexes shape: {point_indexes.shape}"

        try:
            values = np.array([value_function[i] for i in list(point_indexes)])
            next_state_value = lambdas@values
        except (
            KeyError
        ):
            next_state_value = 0
            logger.error(f"States in {point_indexes} not found in the value function.")
            raise Exception(f"States in {point_indexes} not found in the value function.")

        return next_state_value

    def policy_evaluation(self):
        """
        Performs the policy evaluation step of the Policy Iteration, updating the value function.
        """
        max_error = -1.0
        ii = 0 
        self.counter += 1
        logger.info("Starting policy evaluation")
        while abs(max_error) > self.theta:
            new_value_function = np.zeros_like(self.value_function) # initialize the new value function to zeros
            errors = []
            for i, state in enumerate(self.states_space):
                new_val = 0
                for j, action in enumerate(self.action_space):                
                    reward, _, _, lambdas, _, points_indexes = self.transition_reward_table[i, j]
                    # Checkout 'Variable Resolution Discretization in Optimal Control, eq 5'
                    next_state_value = self.get_value(lambdas, points_indexes, self.value_function)
                    new_val += self.policy[i,j] * (reward + self.gamma * next_state_value)
                new_value_function[i] = new_val
                # update the error: the maximum difference between the new and old value functions
                errors.append(abs(new_value_function[i] - self.value_function[i]))

            self.value_function = new_value_function # update the value function
            # log the progress
            if ii % 150 == 0:    
                mean = np.round(np.mean(errors), 4)
                max_error = np.round(np.max(errors),4)
                errs = np.array(errors)
                indices = np.where(errs < self.theta)
                logger.info(f"Max Error: {max_error} | Avg Error: {mean} | {errs[indices].shape[0]}<{self.theta}")
                plot_3D_value_function(self.value_function,
                                       self.states_space,
                                       show=False,
                                       number=int(self.counter*ii),
                                       path=f"{PolicyIteration.metadata['img_path']}/3D_value_function_{self.counter*ii}.png")
                #plot_2D_value_function(self.value_function,
                #                       show=False,
                #                       number=int(self.counter*ii), 
                #                       path=f"{PolicyIteration.metadata['img_path']}/2D_value_function_{self.counter*ii}.png")
                
            
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
        new_policy = jnp.zeros_like(self.policy) # initialize the new policy to zeros
        for i, state in enumerate(self.states_space):
            action_values = {}
            for j, action in enumerate(self.action_space):
                reward, _, _, lambdas, _, points_indexes = self.transition_reward_table[i, j]
                action_values[action] = reward + self.gamma * self.get_value(lambdas, points_indexes, self.value_function)

            greedy_action, _ = max(action_values.items(), key=lambda pair: pair[1])
            
            new_policy[i,:] = jnp.array([int(action == greedy_action) for action in self.action_space])

        if not jnp.array_equal(self.policy, new_policy):
            logger.info(f"The number of updated different actions: {sum(self.policy != new_policy)}")
            policy_stable = False

        logger.info("Policy improvement finished.")
        self.policy = new_policy
        return policy_stable
        
    def run(self):
        """
        Executes the Policy Iteration algorithm for a specified number of steps or until convergence..
        """
        logger.info("Generating transition and reward function table...")
        self.calculate_transition_reward_table()
        logger.info("Transition and reward function table generated.")
        for n in tqdm(range(self.nsteps)):
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