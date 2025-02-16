import pickle
import airplane
import numpy as np
import gymnasium as gym
from utils.utils import plot_3D_value_function   
from PolicyIteration import PolicyIteration

glider = gym.make('ReducedSymmetricGliderPullout-v0')

bins_space = {
    "flight_path_angle": np.linspace(-np.pi-0.01, 0.10, 100,      dtype=np.float32),    # Flight Path Angle (γ)    (0)
    "airspeed_norm":     np.linspace(0.7, 4,       100,      dtype=np.float32),    # Air Speed         (V)    (1)
}

pi = PolicyIteration(
    env=glider, 
    bins_space=bins_space,
    action_space=np.linspace(-0.4, 1.0, 15, dtype=np.float32),
    gamma=0.99,
    theta=1e-3,
)


#pi.run()

with open(glider.__class__.__name__ + ".pkl", "rb") as f:
    pi: PolicyIteration = pickle.load(f)

plot_3D_value_function(vf = pi.value_function,
                       points = pi.states_space,
                       normalize=False,
                       show=True,
                       path="./test_vf.png")