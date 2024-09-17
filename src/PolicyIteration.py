import os
import pickle
import numpy as np
import gymnasium as gym
from loguru import logger
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from utils.utils import plot_2D_value_function,\
                        plot_3D_value_function

import jax.numpy as jnp
from jax import jit, vmap

@jit
def get_inversed_matrix(A:jnp.ndarray)->jnp.ndarray:

    try:
        inv_A = jnp.linalg.inv(A)
    except jnp.linalg.LinAlgError as e:
        #penrose-Moore pseudo inverse and log
        inv_A = jnp.linalg.pinv(A)
        logger.warning(f"The matrix A is singular, using the pseudo-inverse instead:{e}.")
    return inv_A

@jit
def elementwise_dot(a:jnp.ndarray, b:jnp.ndarray)->jnp.ndarray:
    return jnp.dot(a.T, b).squeeze()  # Squeeze to remove extra dimensions

@jit
def jax_get_value(lambdas: jnp.ndarray, values: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the next state value using Barycentric interpolation.

    Args:
        lambdas (jnp.ndarray): The lambdas array. dimension (num_states, num_simplex_points, 1).
        values (jnp.ndarray): The values array. dimension  (num_states, num_simplex_points, 1)

    Returns:
        jnp.ndarray: The next state value array.
    """ 
    return vmap(elementwise_dot)(lambdas, values)
 

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
        self.cell_lower_bounds = jnp.array([min(v) for v in self.bins_space.values()], dtype=jnp.float32)
        self.cell_upper_bounds = jnp.array([max(v) for v in self.bins_space.values()], dtype=jnp.float32)
        logger.info(f"Lower bounds: {self.cell_lower_bounds}")
        logger.info(f"Upper bounds: {self.cell_upper_bounds}")
        # Generate the grid points for all dimensions
        self.grid = jnp.meshgrid(*self.bins_space.values(), indexing='ij')
        # Flatten and stack to create a list of points in the space
        self.states_space = jnp.vstack([g.ravel() for g in self.grid], dtype=jnp.float32).T
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

        dtype = [('reward', jnp.float32), 
                 ('previous_state', jnp.float32, (self.space_dim,)), 
                 ('next_state', jnp.float32, (self.space_dim,)),
                 ('simplexes',  jnp.float32, (self.num_simplex_points, self.space_dim)), 
                 ('lambdas', jnp.float32, (self.num_simplex_points,1)),
                 ('points_indexes', jnp.int32, (self.num_simplex_points,1))] 

        # Initialize the transition and reward function table
        self.transition_reward_table = np.zeros((self.num_states, self.num_actions), dtype=dtype)
        # The policy is a mapping from states to probabilities of selecting each action
        self.policy = jnp.ones((self.num_states, self.num_actions), dtype=jnp.float32) / self.num_actions
        # The value function is an estimate of the expected return from a given state
        self.value_function = jnp.zeros(self.num_states, dtype=jnp.float32)
        logger.info("Policy Iteration was correctly initialized.")
        logger.info(f"The enviroment name is: {self.env.__class__.__name__}")
        

    def __in_cell__(self, obs: jnp.ndarray) -> jnp.ndarray:
        """ Check if the given observation is within the valid state bounds.

        Parameters:
        obs (jnp.ndarray): The observation to be checked.

        Returns:
        jnp.ndarray: A boolean array indicating whether each observation is within the valid state bounds. """
        return jnp.all((obs >= self.cell_lower_bounds) & (obs <= self.cell_upper_bounds), axis=1)
  
    def barycentric_coordinates(self, points:jnp.ndarray)->tuple:

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
        # get the simplexes
        simplexes = self.states_space[points_indexes]
        # Transpose the matrices in one go
        transposed_simplexes = simplexes.transpose(0, 2, 1)
        # Create the row of ones to be added, matching the shape (number of matrices, 1 row, number of columns)
        ones_row = jnp.ones((transposed_simplexes.shape[0], 1, transposed_simplexes.shape[2]))
        # Stack the transposed matrices with the row of ones along the second axis
        A = jnp.concatenate((transposed_simplexes, ones_row), axis=1)
        # Calculate the inverse of the resulting matrix
        inv_A = get_inversed_matrix(A)
    
        b = jnp.hstack([points,  jnp.ones((points.shape[0], 1))]).reshape(self.num_states,self.num_simplex_points,1)

        assert inv_A.shape == (self.num_states, self.num_simplex_points, self.num_simplex_points), \
                                                                        f"inv_A shape: {inv_A.shape}"
        assert b.shape == (self.num_states, self.num_simplex_points, 1), f"b shape: {b.shape}"
        lambdas = jnp.array(inv_A@b, dtype=jnp.float32)
        assert lambdas.shape == (self.num_states, self.num_simplex_points,1), f"lambdas shape: {lambdas.shape}"
        points_indexes = points_indexes.reshape(self.num_states, self.num_simplex_points,1)

        # to test recontruct one point:
        #condition = jnp.linalg.norm(jnp.matmul(A, lambdas) - b, axis=1) < 1e-2
        #assert jnp.all(condition) == True, f"condition: {condition}"

        return lambdas, simplexes, points_indexes
    
    def step(self, state, action):

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

        state = jnp.vstack([x, x_dot, theta, theta_dot]).T
 
        terminated = jnp.where((x < -self.x_threshold) | 
                               (x >  self.x_threshold) | 
                               (theta < -self.theta_threshold_radians) | 
                               (theta > self.theta_threshold_radians), True, False)
        
        reward = jnp.zeros_like(terminated, dtype=jnp.float32)
        reward = jnp.where(terminated, -1, 0)
        states_outside = self.__in_cell__(state)
        #reward = jnp.where(states_outside, reward, -100)
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return np.array(state, dtype=np.float32), reward, terminated, False, {}

    def step_1(self, state:jnp.ndarray, action:float)->tuple:

        min_action    = -1.0
        max_action    = 1.0
        min_position  = -1.2
        max_position  = 0.6
        max_speed     = 0.07
        goal_position = (
            0.45  # was 0.5 in gymnasium, 0.45 in Arnaud de Broissia's version
        )
        goal_velocity = (
            0.0  # was 0.0 in gymnasium, 0.0 in Arnaud de Broissia's version
        )
        power = 0.0008 # 0.0015

        position = state[:,0]  # avoid modifying the original grid
        velocity = state[:,1]  # avoid modifying the original grid

        force = min(max(action, min_action), max_action)
        velocity += force * power - 0.0025 * jnp.cos(3 * position)
        velocity = jnp.clip(velocity, -max_speed, max_speed)

        position += velocity
        position = jnp.clip(position, min_position, max_position)

        velocity = jnp.where((position == min_position) & (velocity < 0), 0, velocity)
        terminated = jnp.where((position >= goal_position) & (velocity >= goal_velocity), True, False)

        reward = jnp.zeros_like(terminated, dtype=jnp.float32)
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
            states = jnp.array(self.states_space, dtype=jnp.float32).copy()          
            obs, reward, _, _, _ = self.step(states, action)
            # log if any state is outside the bounds of the environment
            states_outside = self.__in_cell__(obs)
            if bool(jnp.any(~states_outside)):
                # get the indexes of the states outside the bounds
                #indexes = jnp.where(states_outside)
                logger.warning(f"Some states are outside the bounds of the environment.")

            # if any state is outside the bounds of the environment clip it to the bounds
            obs = jnp.clip(obs, self.cell_lower_bounds, self.cell_upper_bounds)
            # get the barycentric coordinates of the resulting state
            lambdas, simplexes, points_indexes = self.barycentric_coordinates(obs)
            # store the transition and reward information
            self.transition_reward_table['reward'][:, j] = reward
            self.transition_reward_table['previous_state'][:, j] = self.states_space
            self.transition_reward_table['next_state'][:, j] = obs
            self.transition_reward_table['lambdas'][:, j] = lambdas
            self.transition_reward_table['simplexes'][:, j] = simplexes
            self.transition_reward_table['points_indexes'][:, j] = points_indexes              
    
    def get_value(self, 
                  lambdas:jnp.ndarray, 
                  point_indexes:jnp.ndarray, 
                  value_function:jnp.ndarray)->jnp.ndarray:
        """
        Calculates the next state value based on the given lambdas, point indexes, and value function.
        Args:
            lambdas (jnp.ndarray): The lambdas array of shape (num_states, num_simplex_points).
            point_indexes (jnp.ndarray): The point indexes array of shape (num_states, num_simplex_points).
            value_function (jnp.ndarray): The value function.
        Returns:
            jnp.ndarray: The next state value.
        Raises:
            Exception: If states in point_indexes are not found in the value function.
        """
        assert lambdas.shape == (self.num_states, self.num_simplex_points,1), f"lambdas shape: {lambdas.shape}"
        assert point_indexes.shape ==  (self.num_states, self.num_simplex_points,1),  f"point_indexes shape: {point_indexes.shape}"
        
        try:
            values = value_function[point_indexes]
            next_state_value = jax_get_value(lambdas, values)
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
        while jnp.abs(float(max_error)) > self.theta:
            # initialize the new value function to zeros
            new_value_function = jnp.zeros_like(self.value_function, dtype=jnp.float32) 
            new_val = jnp.zeros_like(self.value_function, dtype=jnp.float32)
            for j, action in enumerate(self.action_space):                
                reward = self.transition_reward_table["reward"][:, j]
                lambdas = self.transition_reward_table["lambdas"][:, j]
                points_indexes = self.transition_reward_table["points_indexes"][:, j]
                # Checkout 'Variable Resolution Discretization in Optimal Control, eq 5'
                next_state_value = self.get_value(lambdas, points_indexes, self.value_function)
                new_val += self.policy[:,j] * (reward + self.gamma * next_state_value)
            new_value_function = new_value_function.at[:].set(new_val)
            # update the error: the maximum difference between the new and old value functions
            errors= abs(new_value_function[:] - self.value_function[:])

            self.value_function = new_value_function # update the value function
            
            # log the progress
            if ii % 250 == 0:
                mean = jnp.round(jnp.mean(errors), 5)
                max_error = jnp.round(jnp.max(errors),5)    
                errs = jnp.array(errors)
                indices = jnp.where(errs<self.theta)
                
                logger.info(f"Max Error: {float(max_error)} | Avg Error: {float(mean)} | {errs[indices].shape[0]}<{self.theta}")
                
                #plot_3D_value_function(self.value_function,
                #                       self.states_space,
                #                       show=False,
                #                       number=int(self.counter*ii),
                #                       path=f"{PolicyIteration.metadata['img_path']}/3D_value_function_{self.counter*ii}.png")
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
        #for i, state in enumerate(self.states_space):
        action_values = jnp.zeros((self.states_space.shape[0],self.action_space.shape[0]), dtype=jnp.float32)
        for j, action in enumerate(self.action_space):
            reward         = self.transition_reward_table["reward"][:, j]
            lambdas        = self.transition_reward_table["lambdas"][:, j]
            points_indexes = self.transition_reward_table["points_indexes"][:, j]
            # element-wise multiplication of the policy and the result
            action_values_j = reward + self.gamma * self.get_value(lambdas, points_indexes, self.value_function)
            action_values = action_values.at[:,j].set(action_values_j)
        # update the policy to select the action with the highest value
        greedy_actions = jnp.argmax(action_values, axis=1)
        new_policy = new_policy.at[jnp.arange(new_policy.shape[0]), greedy_actions].set(1)

        if not jnp.array_equal(self.policy, new_policy):
            logger.info(f"The number of updated different actions: {sum(self.policy != new_policy)}")
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