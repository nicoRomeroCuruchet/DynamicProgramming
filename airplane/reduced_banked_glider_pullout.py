import numpy as np
import gymnasium
from gymnasium import spaces
from matplotlib import pyplot as plt
from airplane.reduced_grumman import ReducedGrumman
from airplane.airplane_env import AirplaneEnv


class ReducedBankedGliderPullout(AirplaneEnv):

    def __init__(self, render_mode=None):

        self.airplane = ReducedGrumman()
        super().__init__(self.airplane)

        # Observation space: Flight Path Angle (γ), Air Speed (V), Bank Angle (μ)
        self.observation_space = spaces.Box(np.array([-np.pi, 0.9, np.deg2rad(-20)], np.float32), np.array([0, 4.0, np.deg2rad(200)], np.float32), shape=(3,), dtype=np.float32) 
        # Action space: Lift Coefficient (CL), Bank Rate (μ')
        self.action_space = spaces.Box(np.array([-0.5, np.deg2rad(-30)], np.float32), np.array([1.0, np.deg2rad(30)], np.float32), shape=(2,), dtype=np.float32) 

    def _get_obs(self):
        return np.vstack([self.airplane.flight_path_angle, self.airplane.airspeed_norm, self.airplane.bank_angle], dtype=np.float32).T

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):

        # Choose the initial agent's state uniformly
        [flight_path_angle, airspeed_norm, bank_angle] = np.random.uniform(self.observation_space.low, self.observation_space.high)
        self.airplane.reset(flight_path_angle, airspeed_norm, bank_angle)

        observation = self._get_obs(), {}

        return observation

    def step(self, action: list):
        # Update state
        action = np.clip(action, self.action_space.low, self.action_space.high)
        c_lift = action[0]
        bank_rate = action[1]

        self.airplane.command_airplane(c_lift, bank_rate, 0)

        # Calculate step reward: Height Loss
        # TODO: Analyze policy performance based on reward implementation.
        reward = self.airplane.TIME_STEP * self.airplane.airspeed_norm * np.sin(self.airplane.flight_path_angle)
        # reward = self.TIME_STEP * (self.airspeed_norm * self.STALL_AIRSPEED) * np.sin(self.Flight Path) - 0.01 * bank_rate ** 2
        terminated = self.termination()
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, False, info


    def termination(self,):
        terminate =  np.where((self.airplane.flight_path_angle >= 0.0) & (self.airplane.airspeed_norm >= 1) , True, False)
        return terminate