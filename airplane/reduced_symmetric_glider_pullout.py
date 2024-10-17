import numpy as np
from gymnasium import spaces

from airplane.reduced_grumman import ReducedGrumman
from airplane.airplane_env import AirplaneEnv

try:
    import cupy as xp 
    if not xp.cuda.is_available():
        raise ImportError("CUDA is not available. Falling back to NumPy.")
except (ImportError, AttributeError):
    xp = np

class ReducedSymmetricGliderPullout(AirplaneEnv):

    def __init__(self, render_mode=None):
        self.airplane = ReducedGrumman()
        super().__init__(self.airplane)

        # Observation space: Flight Path Angle (γ), Air Speed (V)
        self.observation_space = spaces.Box(np.array([-np.pi, 0.6], np.float32), 
                                            np.array([0, 4.0], np.float32), shape=(2,), dtype=np.float32)
        # Action space: Lift Coefficient
        self.action_space = spaces.Box(-0.5, 1.0, shape=(1,), dtype=np.float32)

    def _get_obs(self):
        return np.vstack([self.airplane.flight_path_angle, self.airplane.airspeed_norm], dtype=np.float32).T

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):

        # Choose the initial agent's state uniformly
        [flight_path_angle, airspeed_norm] = np.random.uniform(self.observation_space.low, self.observation_space.high)
        self.airplane.reset(flight_path_angle, airspeed_norm, 0)

        observation = self._get_obs()
        # clip the observation to the observation space
        observation = np.clip(observation, self.observation_space.low, self.observation_space.high).flatten()
        assert self.observation_space.contains(observation), "Observation is not within the observation space!"
        return observation, {}

    def step(self, action: list):
        # Update state
        c_lift = action #action[0]
        #self.airplane.command_airplane(c_lift, 0, 0)

        delta_throttle = 0
        bank_rate = 0

        init_terminal = self.termination() 

        self.airplane.last_c_lift = c_lift
        self.airplane.last_bank_rate = bank_rate
        self.airplane.last_throttle = delta_throttle
        
        c_drag = self.airplane._cd_from_cl(c_lift)

        # V_dot = - g sin γ - 0.5 * (ρ S V^2 CD / m) +  (thrust force / m) 
        airspeed_dot = - self.airplane.GRAVITY * np.sin(self.airplane.flight_path_angle) - 0.5 * self.airplane.AIR_DENSITY * (
                self.airplane.WING_SURFACE_AREA / self.airplane.MASS) * (self.airplane.airspeed_norm * self.airplane.STALL_AIRSPEED) ** 2 * c_drag \
                       + (self.airplane.THROTTLE_LINEAR_MAPPING * delta_throttle / self.airplane.MASS)

        # γ_dot = 0.5 * (ρ S V CL cos µ / m) - g cos γ / V
        flight_path_angle_dot = 0.5 * self.airplane.AIR_DENSITY * (self.airplane.WING_SURFACE_AREA / self.airplane.MASS) * (
                self.airplane.airspeed_norm * self.airplane.STALL_AIRSPEED) * c_lift * np.cos(self.airplane.bank_angle) \
                              - (self.airplane.GRAVITY / (self.airplane.airspeed_norm * self.airplane.STALL_AIRSPEED)) * np.cos(
            self.airplane.flight_path_angle)

        # μ_dot = μ_dot_commanded
        bank_angle_dot = bank_rate

        self.airplane.airspeed_norm     += self.airplane.TIME_STEP * (airspeed_dot / self.airplane.STALL_AIRSPEED)
        self.airplane.flight_path_angle += self.airplane.TIME_STEP * flight_path_angle_dot
        #clip the state to the observation space
        self.airplane.airspeed_norm = np.clip(self.airplane.airspeed_norm, self.observation_space.low[1], self.observation_space.high[1])
        self.airplane.flight_path_angle = np.clip(self.airplane.flight_path_angle, self.observation_space.low[0], self.observation_space.high[0])
        # Calculate step reward: Height Loss
        reward = self.airplane.TIME_STEP * self.airplane.airspeed_norm * np.sin(self.airplane.flight_path_angle)*27.331231856346
        
        # Get the next state
        info = self._get_info()
        terminated = self.termination() | init_terminal
        reward = np.where(init_terminal, 0, reward)

        return np.vstack([self.airplane.flight_path_angle, self.airplane.airspeed_norm], dtype=np.float32).T, reward, terminated, False, info


    def termination(self,):
        terminate =  np.where((self.airplane.flight_path_angle >= 0.0) & (self.airplane.airspeed_norm >= 1) , True, False)
        return terminate
