import numpy as np
from gymnasium import spaces

from reduced_grumman import ReducedGrumman
from airplane_env import AirplaneEnv

class ReducedSymmetricGliderPullout(AirplaneEnv):

    def __init__(self, render_mode=None):
        self.airplane = ReducedGrumman()
        super().__init__(self.airplane)

        # Observation space: Flight Path Angle (γ), Air Speed (V)
        self.observation_space = spaces.Box(np.array([-np.pi, 0.9], np.float32), np.array([0, 4.0], np.float32), shape=(2,), dtype=np.float32)
        # Action space: Lift Coefficient
        self.action_space = spaces.Box(-0.5, 1.0, shape=(1,), dtype=np.float32)

    def _get_obs(self):
        return np.vstack([self.airplane.flight_path_angle, self.airplane.airspeed_norm]).T
        #return np.array([self.airplane.flight_path_angle, self.airplane.airspeed_norm], dtype=np.float32)

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):

        # Choose the initial agent's state uniformly
        [flight_path_angle, airspeed_norm] = np.random.uniform(self.observation_space.low, self.observation_space.high)
        self.airplane.reset(flight_path_angle, airspeed_norm, 0)

        observation = self._get_obs()

        return observation

    def step(self, action: list):
        # Update state
        c_lift = action[0]

        self.airplane.command_airplane(c_lift, 0, 0)

        # Calculate step reward: Height Loss
        reward = self.airplane.TIME_STEP * self.airplane.airspeed_norm * np.sin(self.airplane.flight_path_angle)
        terminated = self.termination()
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, info


    def termination(self,):
        terminate =  np.where(((self.airplane.flight_path_angle >= 0.0) | (self.airplane.flight_path_angle <= -np.pi)), True, False)
        return terminate