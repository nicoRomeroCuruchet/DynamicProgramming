import torch
import pickle
import random
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import A2C

import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space



cartpole = gym.make("CartPole-v1")
import math
from typing import Optional, Tuple, Union

class CartPoleEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    ## Description

    This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in
    ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
    A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
    The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces
     in the left and right direction on the cart.

    ## Action Space

    The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction
     of the fixed force the cart is pushed with.

    - 0: Push cart to the left
    - 1: Push cart to the right

    **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle
     the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it

    ## Observation Space

    The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:

    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Cart Position         | -4.8                | 4.8               |
    | 1   | Cart Velocity         | -Inf                | Inf               |
    | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 3   | Pole Angular Velocity | -Inf                | Inf               |

    **Note:** While the ranges above denote the possible values for observation space of each element,
        it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
    -  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates
       if the cart leaves the `(-2.4, 2.4)` range.
    -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
       if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)

    ## Rewards
    Since the goal is to keep the pole upright for as long as possible, by default, a reward of `+1` is given for every step taken, including the termination step. The default reward threshold is 500 for v1 and 200 for v0 due to the time limit on the environment.

    If `sutton_barto_reward=True`, then a reward of `0` is awarded for every non-terminating step and `-1` for the terminating step. As a result, the reward threshold is 0 for v0 and v1.

    ## Starting State
    All observations are assigned a uniformly random value in `(-0.05, 0.05)`

    ## Episode End
    The episode ends if any one of the following occurs:

    1. Termination: Pole Angle is greater than ±12°
    2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    3. Truncation: Episode length is greater than 500 (200 for v0)

    ## Arguments

    Cartpole only has `render_mode` as a keyword for `gymnasium.make`.
    On reset, the `options` parameter allows the user to change the bounds used to determine the new random state.

    ```python
    >>> import gymnasium as gym
    >>> env = gym.make("CartPole-v1", render_mode="rgb_array")
    >>> env
    <TimeLimit<OrderEnforcing<PassiveEnvChecker<CartPoleEnv<CartPole-v1>>>>>
    >>> env.reset(seed=123, options={"low": -0.1, "high": 0.1})  # default low=-0.05, high=0.05
    (array([ 0.03647037, -0.0892358 , -0.05592803, -0.06312564], dtype=float32), {})

    ```

    | Parameter               | Type       | Default                 | Description                                                                                   |
    |-------------------------|------------|-------------------------|-----------------------------------------------------------------------------------------------|
    | `sutton_barto_reward`   | **bool**   | `False`                 | If `True` the reward function matches the original sutton barto implementation                |

    ## Vectorized environment

    To increase steps per seconds, users can use a custom vector environment or with an environment vectorizor.

    ```python
    >>> import gymnasium as gym
    >>> envs = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="vector_entry_point")
    >>> envs
    CartPoleVectorEnv(CartPole-v1, num_envs=3)
    >>> envs = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="sync")
    >>> envs
    SyncVectorEnv(CartPole-v1, num_envs=3)

    ```

    ## Version History
    * v1: `max_time_steps` raised to 500.
        - In Gymnasium `1.0.0a2` the `sutton_barto_reward` argument was added (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/790))
    * v0: Initial versions release.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(
        self, sutton_barto_reward: bool = False, render_mode: Optional[str] = None
    ):
        self._sutton_barto_reward = sutton_barto_reward

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_terminated = None

    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

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

        self.state = (x, x_dot, theta, theta_dot)
        print("state:", self.state)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not terminated:
            if self._sutton_barto_reward:
                reward = 0.0
            elif not self._sutton_barto_reward:
                reward = 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            if self._sutton_barto_reward:
                reward = -1.0
            elif not self._sutton_barto_reward:
                reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            if self._sutton_barto_reward:
                reward = -1.0
            elif not self._sutton_barto_reward:
                reward = 0.0

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(
            options, -0.05, 0.05  # default low
        )  # default high
        self.state = self.np_random.uniform(low=low, high=high, size=(4,))
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False



def approximate(raw_state):
    array = list(raw_state)
    return tuple([round(num, 2) + 0 for num in array])

def generate(lower_bound, upper_bound, interval):
    output = [lower_bound]
    while lower_bound <= upper_bound:
        lower_bound += interval
        output.append( float( round(lower_bound,1) + 0 ) )
    return output

def generate_states(x_approx, 
                    x_dot_approx, 
                    theta_approx, 
                    theta_dot_approx):
    states = []
    for x in x_approx:
        for x_dot in x_dot_approx:
            for theta in theta_approx:
                for theta_dot in theta_dot_approx:
                    states.append((x, x_dot, theta, theta_dot))
    return states

def get_value(state, value_function):

    try:
        print("state:", state)
        next_state_value = value_function[state]
    except KeyError: # if next_state is not in value_function, assume it's a 'dead' state.
        next_state_value = -500
    return next_state_value

def create_transition_reward_function(states, 
                                      actions, 
                                      ):

    env = CartPoleEnv()
    env.reset()
    table = {}
    for state in states:
        for action in actions:
            # np.array(self.state, dtype=np.float32), reward, terminated, False, {}
            env.reset()
            env.state = np.array((1,0,0.1,0), dtype=np.float32)
            print("state:", env.state)
            action = 0
            obs, reward, done, terminated, info = env.step(action)
            print("reward", reward, "action", action, "Next state:", np.array(env.state, dtype=np.float32), obs)
            1/0
            table[(state, action)] = {'reward':reward, 'next_state':approximate(obs)}

    return table

def evaluate_policy(state, 
                    actions, 
                    transition_and_reward_function, 
                    policy, 
                    value_function, 
                    gamma=1.0):

    new_val = 0
    for action in actions:
        reward, next_state = transition_and_reward_function[(state, action)].values()
        next_state_value = get_value(next_state, value_function)
        new_val += policy[state][action] * (reward + gamma*next_state_value)
        #new_val += (reward + gamma*next_state_value)
    return new_val

def improve_policy(states, 
                   actions, 
                   transition_and_reward_function, 
                   value_function, 
                   gamma=0.99):

    new_policy = {}
    for state in states:
        action_values = {}
        for action in actions:
            reward, next_state = transition_and_reward_function[(state, action)].values()
            action_values[action] = reward + gamma*get_value(next_state, value_function)
        greedy_action, value = max(action_values.items(), key= lambda pair: pair[1])
        new_policy[state] = {action:1 if action is greedy_action else 0 for action in actions}
    return new_policy

# Policy iteration
def policy_iteration(states, 
                     actions, 
                     transition_and_reward_function, 
                     policy, 
                     value_function, 
                     number_of_iterations=10):

    new_value_function = {}
    for _ in range(number_of_iterations):
        print("iteration_number:",_)
        # Evaluate every state under current policy
        for state in states:
            new_value_function[state] = evaluate_policy(state, actions, transition_and_reward_function, policy, value_function)
        # Policy improvement
        policy = improve_policy(states, actions, transition_and_reward_function, new_value_function)
        value_function = new_value_function
    return policy

def get_optimal_action(state, optimal_policy):
    try:
        greedy_action, prob = max(optimal_policy[state].items(), key= lambda pair: pair[1])
        return greedy_action
    except KeyError:
        return random.randint(0,1)

if __name__ == '__main__':
    # Setting up
    x_approx = generate(-0.5, 0.5, 0.01)
    x_dot_approx = generate(-2.0, 2.0, 0.1)
    theta_approx = generate(-0.1, 0.1, 0.01) # theta_thres is 24deg, i.e. 24*2*pi/360 radians
    theta_dot_approx = generate(-3.0, 3.0, 0.1)
    states = generate_states(x_approx, x_dot_approx, theta_approx, theta_dot_approx)
    actions = [0,1]
    #print("done generating states and actions")
    transition_and_reward_function = create_transition_reward_function(states, actions)
    starting_policy = {state:{0:0.5, 1:0.5} for state in states}
    value_function = {state:0 for state in states}
    optimal_policy = policy_iteration(states, actions, transition_and_reward_function, starting_policy, value_function)

    import pylab as plt
    import numpy as np

    X=[]
    Y=[]
    Z=[]
    # Sample data
    side = np.linspace(-2,2,15)
    for k in value_function.keys():
        X.append(k[0])
        Y.append(k[2])
        Z.append(value_function[k])
   

    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)

    # Define grid
    x_unique = np.sort(np.unique(X))
    y_unique = np.sort(np.unique(Y))

    # Create meshgrid
    X_grid, Y_grid = np.meshgrid(x_unique, y_unique)

    # Interpolate Z onto grid
    Z_grid = np.zeros_like(X_grid, dtype=float)
    for x, y, z in zip(X, Y, Z):
        x_index = np.where(x_unique == x)[0][0]
        y_index = np.where(y_unique == y)[0][0]
        Z_grid[y_index, x_index] = z
    # Plot the grid
    plt.pcolor(X_grid, Y_grid, Z_grid, cmap='gray')
    plt.colorbar(label='Z')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Grid Plot')
    plt.show()
   
    # Saving the optimal_policy on a file
    """optimal = open('optimal.pkl','wb')
    pickle.dump(optimal_policy, optimal)
    optimal.close()
    print("done finding optimal_policy")
    # Running test episodes with optimal_policy
    cartpole = gym.make("CartPole-v1", render_mode="human")
    total_steps = 0
    num_episodes = 10000
    for episode in range(0,num_episodes):
        observation, _ = cartpole.reset()
        for timestep in range(1,1000):
            obs_ = approximate(observation)
            action = get_optimal_action(obs_, optimal_policy)
            observation, reward, done, terminated, info = cartpole.step(action)
            #print("Observation:", observation, ' ', "Action:", action)
            if done:
                print("Episode {} finished after {} timesteps".format(episode,timestep))
                print(observation)
                total_steps += timestep
                break

    print("average time upright per episode:", total_steps/num_episodes)"""