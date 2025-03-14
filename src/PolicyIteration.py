import time
import pickle
import numpy as np
import gymnasium as gym
from pathlib import Path
from loguru import logger
from typing import Optional, Tuple
from scipy.spatial import Delaunay
from dataclasses import dataclass, field
from utils.utils import plot_3D_value_function

@dataclass
class PolicyIterationConfig:
    gamma: float = 0.99
    theta: float = 5e-2
    n_steps: int = 100
    log: bool = False
    log_interval: int = 150
    img_path: Path = field(default_factory=lambda: Path.cwd() / "img")

@dataclass
class PolicyIteration(object):
    """
    Performs Policy Iteration on discretized continuous environments using barycentric coordinates.
    
    Implements parallelized policy evaluation and improvement with vectorized operations.
    
    Example:

        >>> import pickle
        >>> import numpy as np
        >>> from utils.utils import test_enviroment
        >>> from PolicyIteration import PolicyIteration
        >>> from classic_control.cartpole import CartPoleEnv
        
        >>> env =  CartPoleEnv(sutton_barto_reward=True)
        >>> # position thresholds:
        >>> x_lim         = 2.4
        >>> theta_lim     = 0.418 
        >>> # velocity thresholds:
        >>> x_dot_lim     = 3.1
        >>> theta_dot_lim = 3.1

        >>> bins_space = {
        ...     "x_space": np.linspace(-x_lim, x_lim, 10,  dtype=np.float32),                    
        ...     "x_dot_space": np.linspace(-x_dot_lim, x_dot_lim, 10,  dtype=np.float32),
        ...     "theta_space": np.linspace(-theta_lim, theta_lim, 10, dtype=np.float32),
        ...     "theta_dot_space": np.linspace(-theta_dot_lim, theta_dot_lim, 10, dtype=np.float32),
        ... }
        >>> pi = PolicyIteration(env, bins_space, action_space=np.array([0, 1]))
        >>> pi.run()
    """

    env: gym.Env
    bins_space: dict[str, np.ndarray]
    action_space: np.ndarray
    config: PolicyIterationConfig = field(default_factory=PolicyIterationConfig)
    
    # Derived attributes (computed in __post_init__)
    states_space: np.ndarray = field(init=False)
    triangulation: Delaunay = field(init=False)
    terminal_states: np.ndarray = field(init=False)
    terminal_reward: float = field(init=False)

    def __post_init__(self):
        """Validate inputs and initialize derived properties."""

        self._validate_inputs()
        self._initialize_state_space()
        self._initialize_components()
        
        logger.info("Initialized Policy Iteration for {}", self.env.__class__.__name__)
        logger.debug("State space shape: {}", self.states_space.shape)

    def _validate_inputs(self):
        """Validate input parameters with detailed error messages."""

        if not isinstance(self.action_space, np.ndarray):
            raise TypeError("Action space must be a numpy array.")
        if self.action_space.size == 0:
            raise ValueError("Action space cannot be empty.")
        if not self.bins_space:
            raise ValueError("Bins space must be non-empty.")

    def _initialize_state_space(self):
        """Create discretized state space grid."""

        grid = np.meshgrid(*self.bins_space.values(), indexing='ij')
        self.states_space = np.vstack([g.ravel() for g in grid]).astype(np.float32).T
        self.num_simplex_points:int = int(self.states_space[0].shape[0] + 1) # number of points in a simplex one more than the dimension
        self.space_dim:int          = int(self.states_space[0].shape[0])
        self.action_space_size:int  = int(self.action_space.shape[0])   
        self.num_states:int         = int(self.states_space.shape[0])
        self.num_actions:int        = int(self.action_space.shape[0])
        # Get terminal states information
        self.terminal_states, self.terminal_reward = self.env.terminal(self.states_space)
        # Calculate bounds
        self.cell_bounds = {
            'lower': np.array([min(v) for v in self.bins_space.values()], dtype=np.float32),
            'upper': np.array([max(v) for v in self.bins_space.values()], dtype=np.float32)
        }
        logger.info("Lower bounds: {}", self.cell_bounds['lower'])
        logger.info("Upper bounds: {}", self.cell_bounds['upper'])

    def _initialize_components(self):
        """Initialize algorithm components and data structures."""
        self.reward         = np.zeros((self.num_states, self.num_actions), dtype=np.float32)
        self.previous_state = np.zeros((self.num_states, self.num_actions, self.space_dim), dtype=np.float32)
        self.next_state     = np.zeros((self.num_states, self.num_actions, self.space_dim), dtype=np.float32)
        self.simplexes      = np.zeros((self.num_states, self.num_actions, self.num_simplex_points, self.space_dim), dtype=np.float32)
        self.lambdas        = np.zeros((self.num_states, self.num_actions, self.num_simplex_points, 1), dtype=np.float32)
        self.points_indexes = np.zeros((self.num_states, self.num_actions, self.num_simplex_points, 1), dtype=np.int32)
        # The policy is a mapping from states to probabilities of selecting each action
        # Initialize policy and value function: uniform policy and zero value function
        self.policy = np.full((self.num_states, self.num_actions), 1/self.num_actions, dtype=np.float32)
        self.value_function = np.where(
            self.terminal_states, self.terminal_reward, 0.0
        ).astype(np.float32)

    def in_bounds(self, obs: np.ndarray) -> np.ndarray:
        """ Check if the given observation is within the valid state bounds.

        Parameters:
            obs (np.ndarray): The observation array to check.

        Returns:
            np.ndarray: A boolean array indicating whether each observation is within the valid state bounds. """
        
        return np.all((obs >=  self.cell_bounds['lower']) & (obs <=  self.cell_bounds['upper']), axis=1)
  
    def _compute_barycentric_coordinates(self, points:np.ndarray)-> Tuple[np.ndarray, ...]:
        """ Vectorized computation of barycentric coordinates.
        
        Parameters:
            point (np.array): The 2D point for which to calculate the barycentric coordinates.
        Returns:
            result (np.array): The barycentric coordinates of the point.
            vertices_coordinates (np.array): The coordinates of the vertices of the simplex containing the point.
        Raises:
            ValueError: If the point is outside the convex hull. """

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
        
        assert A.shape == (self.num_states, self.num_simplex_points, self.space_dim+1), f"A shape: {A.shape}"
        assert b.shape == (self.num_states, self.num_simplex_points, 1), f"b shape: {b.shape}"
        # Calculate the baricentric coordinates
        try:
            lambdas = np.linalg.solve(A, b)
        except np.linalg.LinAlgError as e:
            lambdas = np.linalg.lstsq(A, b, rcond=None)[0]
            logger.warning(f"The matrix A is singular, using the pseudo-inverse instead:{e}.")
    
        assert lambdas.shape == (self.num_states, self.num_simplex_points,1), f"lambdas shape: {lambdas.shape}"
        points_indexes = points_indexes.reshape(self.num_states, self.num_simplex_points,1)
        # to test recontruct the points:
        #condition = np.linalg.norm(np.matmul(A, lambdas) - b, axis=1) < 1e-2
        #assert np.all(condition) == True, f"condition: {condition}"
        return lambdas, simplexes, points_indexes
  
    def calculate_transition_reward_table(self):
        """ Calculates the transition and reward table for each action in the action space.

        This method updates the `next_state`, `reward`, `previous_state`, `lambdas`, `simplexes`, and `points_indexes`
        attributes of the object based on the transition and reward information. """

        for j, action in enumerate(self.action_space):
            self.env.state = np.array(self.states_space, dtype=np.float32)  # TODO deepcopy the state space?
            obs, reward, _, _, _ = self.env.step(action)
            # log if any state is outside the bounds of the environment
            states_outside = self.in_bounds(obs)
            if bool(np.any(~states_outside)):
                reward = np.where(states_outside, reward, -100)
                logger.warning(f"Some states are outside the bounds of the environment.")
            # if any state is outside the bounds of the environment clip it to the bounds
            obs = np.clip(obs, self.cell_bounds['lower'], self.cell_bounds['upper'])
            # get the barycentric coordinates of the resulting states
            lambdas, simplexes, points_indexes = self._compute_barycentric_coordinates(obs)
            # store the transition and reward information
            self.previous_state[:,j] = self.states_space
            self.next_state[:,j] = obs
            self.lambdas[:,j] = lambdas         # The barycentric coordinates of the next state
            self.simplexes[:,j] = simplexes
            self.points_indexes[:,j] = points_indexes
            self.reward[:,j] = reward
            
    def get_value(self, lambdas:np.ndarray,  
                        point_indexes:np.ndarray,  
                        value_function:np.ndarray)->np.ndarray:
        """ Calculates the value of the next state based on the given lambdas, point indexes, and value function.

        Args:
            lambdas (np.ndarray): The lambdas array with shape (num_states, num_simplex_points, 1).
            point_indexes (np.ndarray): The point indexes array with shape (num_states, num_simplex_points, 1).
            value_function (np.ndarray): The value function array.

        Returns:
            np.ndarray: The value of the next state.

        Raises:
            Exception: If any of the states in point_indexes are not found in the value function. """
        
        assert lambdas.shape == (self.num_states, self.num_simplex_points,1), f"lambdas shape: {lambdas.shape}"
        assert point_indexes.shape == (self.num_states, self.num_simplex_points,1),  f"point_indexes shape: {point_indexes.shape}"

        lambdas = lambdas.reshape(lambdas.shape[0], lambdas.shape[1])  # shape: (n, d+1)
        point_indexes = point_indexes.reshape(point_indexes.shape[0], point_indexes.shape[1])  # shape: (n, d+1)

        try:
            values = value_function[point_indexes]
            next_state_value = np.sum(lambdas * values, axis=1)
        except (
            KeyError
        ):
            next_state_value = None
            raise Exception(f"States in {point_indexes} not found in the value function.")
        
        return next_state_value

    def policy_evaluation(self)->float:
        """ Performs the policy evaluation step of the Policy Iteration, updating the value function.  
        
        Returns:
            float: The error between the new and old value. """
        
        i:int = 0
        delta:float = float('inf')
        logger.info("Starting policy evaluation")
        while delta > self.config.theta:
            # initialize the new value function to zeros
            new_val = np.zeros_like(self.value_function, dtype=np.float32)
            new_val[self.terminal_states] = self.terminal_reward
            vf_next_state = np.zeros_like(self.value_function, dtype=np.float32)
            # iterate over the actions
            for action_idx, _ in enumerate(self.action_space):                
                # Checkout 'Variable Resolution Discretization in Optimal Control, eq 5'
                vf_next_state[~self.terminal_states] = self.get_value(self.lambdas[:, action_idx], self.points_indexes[:, action_idx], self.value_function)[~self.terminal_states]
                new_val[~self.terminal_states] += self.policy[~self.terminal_states, action_idx] * (self.reward[~self.terminal_states, action_idx] +\
                                                                                                        self.config.gamma * vf_next_state[~self.terminal_states])
            # update the value function
            new_value_function = new_val
            # update the error: the maximum difference between the new and old value functions
            errors = np.fabs(new_value_function[:] - self.value_function[:])

            self.value_function = new_value_function   # update the value function
            delta = np.round(np.max(errors), 3)        # update the error

            #just for logging the progress
            if i % self.config.log_interval == 0:
                
                mean = np.round(np.mean(errors), 3) 
                
                if not hasattr(self, 'policy_convergence'):
                    self.policy_convergence = []
                    self.delta = []
                self.policy_convergence.append(mean)
                self.delta.append(delta)

                indices = np.where(errors<self.config.theta)
                logger.info(f"Max Error: {float(delta)} | Avg Error: {float(mean)} | {errors[indices].shape[0]}<{self.config.theta}")
                # get date for the name of the image
                if self.config.log:
                    timestamp = int(time.time())
                    img_name =  f"/3D_value_function_{timestamp}.png"
                    __path__ = Path(f"{self.config.img_path}{img_name}")
                    plot_3D_value_function(vf=self.value_function, 
                                           points=self.states_space, 
                                           normalize=False, 
                                           show=False, 
                                           path=str(__path__))
            i += 1
                
        logger.success("Policy evaluation converged with Δ={:.2e}", delta)
        logger.success("Policy evaluation finished.")
        return delta

    def policy_improvement(self)->bool:
        """ Performs the policy improvement step, updating the policy based on the current value function.

        Returns:
            bool: True if the policy is stable and no changes were made, False otherwise. """
        
        logger.info("Starting policy improvement")
        policy_stable = True
        new_policy = np.zeros_like(self.policy) # initialize the new policy to zeros
        action_values = np.zeros((self.states_space.shape[0],self.action_space.shape[0]), dtype=np.float32)
        for action_idx, _ in enumerate(self.action_space):
            get_value = self.get_value(self.lambdas[:, action_idx], self.points_indexes[:, action_idx], self.value_function)
            action_values_j = self.reward[:, action_idx] + self.config.gamma * get_value
            action_values[:, action_idx] = action_values_j 
        # update the policy to select the action with the highest value
        new_policy = np.eye(self.action_space.size)[np.argmax(action_values, axis=1)]
        policy_stable = np.allclose(self.policy, new_policy)
        
        if not policy_stable:
            changes = np.sum(~np.isclose(self.policy, new_policy))
            logger.info("Policy updated: {} changes detected", changes)
            self.policy = new_policy
            
        return policy_stable
        
    def run(self):
        """ Executes the Policy Iteration algorithm for a specified number of steps or until convergence. """

        # Create the Delaunay triangulation
        logger.info("Creating Delaunay triangulation over the state space...")
        self.triangulation = Delaunay(self.states_space)
        logger.success("Delaunay triangulation created.")        
        # Generate the transition and reward function table
        logger.info("Generating transition and reward function table...")
        self.calculate_transition_reward_table()
        logger.success("Transition and reward function table generated.")
        for n in range(self.config.n_steps):
            logger.info(f"solving step {n}")
            self.policy_evaluation()
            if self.policy_improvement():
                break
        
        self.save()
        self.env.close()

    def save(self, path: Optional[Path] = None):
        """Save trained model with proper path handling."""
        path = path or Path(f"{self.env.__class__.__name__}_policy.pkl")
        with path.open('wb') as f:
            pickle.dump(self, f)
            
        logger.success("Saved policy to {}", path.resolve())

    @classmethod
    def load(cls, path: Path) -> "PolicyIteration":
        """Load saved policy instance."""
        with path.open('rb') as f:
            instance = pickle.load(f)
            
        if not isinstance(instance, cls):
            raise ValueError("Loaded object is not a PolicyIteration instance")
            
        return instance