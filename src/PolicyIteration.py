import os
import pickle
import cupy as cp
import numpy as np
import gymnasium as gym
from loguru import logger
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from utils.utils import plot_2D_value_function,\
                        plot_3D_value_function


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
        self.env:gym.Env = env       # working environment
        self.gamma:float = gamma     # discount factor
        self.theta:float = theta     # convergence threshold for policy evaluation
        self.nsteps:int = nsteps     # number of steps to run the policy iteration
        self.counter:int = 0         # counter for the number of steps

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
        # 
        self.num_simplex_points:int = int(self.states_space[0].shape[0] + 1) # number of points in a simplex one more than the dimension
        self.space_dim:int          = int(self.states_space[0].shape[0])
        self.action_space_size:int  = int(self.action_space.shape[0])   
        self.num_states:int         = int(self.states_space.shape[0])
        self.num_actions:int        = int(self.action_space.shape[0])

        logger.info(f"The action space is: {self.action_space}")
        logger.info(f"Number of states: {len(self.states_space)}")
        logger.info(f"Total states:{len(self.states_space)*len(self.action_space)}")
        
        # Create the Delaunay triangulation
        logger.info("Creating Delaunay triangulation...")
        self.triangulation = Delaunay(self.states_space)
        logger.info("Delaunay triangulation created.")
        
        #to plot delaunay triangulation:    
        #plt.plot(self.states_space[:, 0], self.states_space[:, 1], 'go', label='Data states_space', markersize=2)
        # plot the triangulation
        #plt.triplot(self.states_space[:, 0], self.states_space[:, 1], self.triangulation.simplices)
        #plt.scatter(-1.2,  0. , color='Red', s=10)
        #plt.show()

        # Initialize the transition and reward function table
        self.reward         = cp.zeros((self.num_states, self.num_actions), dtype=cp.float32)
        self.previous_state = cp.zeros((self.num_states, self.num_actions, self.space_dim), dtype=cp.float32)
        self.next_state     = cp.zeros((self.num_states, self.num_actions, self.space_dim), dtype=cp.float32)
        self.simplexes      = cp.zeros((self.num_states, self.num_actions, self.num_simplex_points, self.space_dim), dtype=cp.float32)
        self.lambdas        = cp.zeros((self.num_states, self.num_actions, self.num_simplex_points, 1), dtype=cp.float32)
        self.points_indexes = cp.zeros((self.num_states, self.num_actions, self.num_simplex_points, 1), dtype=cp.int32)


        dtype = [('reward', np.float32), 
                 ('previous_state', np.float32, (self.space_dim,)), 
                 ('next_state', np.float32, (self.space_dim,)),
                 ('simplexes',  np.float32, (self.num_simplex_points, self.space_dim)), 
                 ('lambdas', np.float32, (self.num_simplex_points,1)),
                 ('points_indexes', np.int32, (self.num_simplex_points,1))] 
        


        # Initialize the transition and reward function table
        self.transition_reward_table = np.zeros((self.num_states, self.num_actions), dtype=dtype)
        # The policy is a mapping from states to probabilities of selecting each action
        self.policy = cp.ones((self.num_states, self.num_actions), dtype=cp.float32) / self.num_actions
        # The value function is an estimate of the expected return from a given state
        self.value_function = cp.zeros(self.num_states, dtype=cp.float32)
        logger.info("Policy Iteration was correctly initialized.")
        logger.info(f"The enviroment name is: {self.env.__class__.__name__}")
        

    def __in_cell__(self, obs: np.ndarray) -> np.ndarray:
        """ Check if the given observation is within the valid state bounds.

        Parameters:
        obs (np.ndarray): The observation to be checked.

        Returns:
        np.ndarray: A boolean array indicating whether each observation is within the valid state bounds. """
        return np.all((obs >= self.cell_lower_bounds) & (obs <= self.cell_upper_bounds), axis=1)
  
    def barycentric_coordinates(self, points:np.ndarray)->tuple:

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
    
    def step_1(self, state, action):

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self._sutton_barto_reward = True
        self.kinematics_integrator = "euler"
        self.steps_beyond_terminated = None
        self.theta_threshold_radians = 12 * 2 * np.pi / 360
        self.x_threshold = 2.4

        x, x_dot, theta, theta_dot = state[:,0], state[:,1], state[:,2], state[:,3]
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        state = np.vstack([x, x_dot, theta, theta_dot]).T
 
        terminated = np.where((x < -self.x_threshold) | 
                               (x >  self.x_threshold) | 
                               (theta < -self.theta_threshold_radians) | 
                               (theta > self.theta_threshold_radians), True, False)
        
        reward = np.zeros_like(terminated, dtype=np.float32)
        reward = np.where(terminated, -1, 0)
        states_outside = self.__in_cell__(state)
        reward = np.where(states_outside, reward, -100)

        return np.array(state, dtype=np.float32), reward, terminated, False, {}

    def step(self, state:np.ndarray, action:float)->tuple:

        min_action    = -1.0
        max_action    = +1.0
        min_position  = -1.2
        max_position  = +0.6
        max_speed     = +0.07

        goal_position = (
            0.45  # was 0.5 in gymnasium, 0.45 in Arnaud de Broissia's version
        )
        goal_velocity = (
            0.0  # was 0.0 in gymnasium, 0.0 in Arnaud de Broissia's version
        )
        power = 0.0008 # 0.0015

        position = state[:,0]  # avoid modifying the original grid
        velocity = state[:,1]  # avoid modifying the original grid

        # transfer to gpu
        position = cp.asarray(position)
        velocity = cp.asarray(velocity)
        action   = cp.asarray(action)

        force     = min(max(action, min_action), max_action)
        velocity += force * power - 0.0025 * cp.cos(3 * position)
        velocity  = cp.clip(velocity, -max_speed, max_speed)

        position += velocity
        position  = cp.clip(position, min_position, max_position)

        velocity   = cp.where((position == min_position) & (velocity < 0), 0, velocity)
        terminated = cp.where((position >= goal_position) & (velocity >= goal_velocity), True, False)

        reward  = cp.zeros_like(terminated, dtype=cp.float32)
        reward  = cp.where(terminated, 100.0, reward)
        reward -= cp.power(action, 2) * 0.1

        # transfer to cpu
        position   = cp.asnumpy(position)
        velocity   = cp.asnumpy(velocity)
        terminated = cp.asnumpy(terminated)
        reward     = cp.asnumpy(reward)

        return np.vstack([position, velocity]).T, reward, terminated, False, {}

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
            states = np.array(self.states_space, dtype=np.float32).copy()          
            obs, reward, _, _, _ = self.step(states, action)
            # log if any state is outside the bounds of the environment
            states_outside = self.__in_cell__(obs)
            if bool(np.any(~states_outside)):
                # get the indexes of the states outside the bounds
                #indexes = np.where(states_outside)
                logger.warning(f"Some states are outside the bounds of the environment.")
            # if any state is outside the bounds of the environment clip it to the bounds
            obs = np.clip(obs, self.cell_lower_bounds, self.cell_upper_bounds)
            # get the barycentric coordinates of the resulting state
            lambdas, simplexes, points_indexes = self.barycentric_coordinates(obs)
            # store the transition and reward information and transfer to gpu
            self.reward[:,j]         = cp.asarray(reward)
            self.previous_state[:,j] = cp.asarray(self.states_space)
            self.next_state[:,j]     = cp.asarray(obs)
            self.lambdas[:,j]        = cp.asarray(lambdas)
            self.simplexes[:,j]      = cp.asarray(simplexes)
            self.points_indexes[:,j] = cp.asarray(points_indexes) 

    
    def get_value(self, lambdas:cp.ndarray,  point_indexes:cp.ndarray,  value_function:cp.ndarray)->cp.ndarray:
        """
        Calculates the next state value based on the given lambdas, point indexes, and value function.
        Args:
            lambdas (np.ndarray): The lambdas array of shape (num_states, num_simplex_points,1).
            point_indexes (np.ndarray): The point indexes array of shape (num_states, num_simplex_points,1).
            value_function (np.ndarray): The value function.
        Returns:
            np.ndarray: The next state value.
        Raises:
            Exception: If states in point_indexes are not found in the value function.
        """
        assert lambdas.shape       == (self.num_states, self.num_simplex_points,1), f"lambdas shape: {lambdas.shape}"
        assert point_indexes.shape ==  (self.num_states, self.num_simplex_points,1),  f"point_indexes shape: {point_indexes.shape}"

        try:
            values = value_function[point_indexes.get()]
            next_state_value = cp.einsum('ij,ij->i', lambdas.squeeze(-1), values. squeeze(-1))
        except (
            KeyError
        ):
            next_state_value = 0
            raise Exception(f"States in {point_indexes} not found in the value function.")

        return next_state_value

    def policy_evaluation(self):
        """
        Performs the policy evaluation step of the Policy Iteration, updating the value function.
        """
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
            self.value_function = new_value_function    # update the value function
            
            # log the progress
            if ii % 200 == 0:
                mean      = cp.round(cp.mean(errors), 3)
                max_error = cp.round(cp.max(errors), 3)    
                indices   = cp.where(errors<self.theta)
                logger.info(f"Max Error: {float(max_error)} | Avg Error: {float(mean)} | {errors[indices].shape[0]}<{self.theta}")

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
        new_policy = cp.zeros_like(self.policy) # initialize the new policy to zeros
        #for i, state in enumerate(self.states_space):
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
        """
        Saves the policy and value function to files.
        """

        with open(self.env.__class__.__name__ + ".pkl", "wb") as f:
            pickle.dump(self, f)
            logger.info("Policy and value function saved.")        