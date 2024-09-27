import pickle
import numpy as np
from utils.utils import plot_3D_value_function   
from PolicyIteration import PolicyIteration
from classic_control.continuous_mountain_car import Continuous_MountainCarEnv

env=Continuous_MountainCarEnv()

bins_space = {
    "x_space":     np.linspace(env.min_position, env.max_position, 100,      dtype=np.float32),    # position space    (0)
    "x_dot_space": np.linspace(-abs(env.max_speed), abs(env.max_speed), 100, dtype=np.float32),    # velocity space    (1)
}

pi = PolicyIteration(
    env=env, 
    bins_space=bins_space,
    action_space=np.linspace(-1.0, +1.0,9, dtype=np.float32),
    gamma=0.99,
    theta=1e-3,
)
pi.run()

with open(env.__class__.__name__ + ".pkl", "rb") as f:
    pi: PolicyIteration = pickle.load(f)

plot_3D_value_function(vf = pi.value_function,
                       points = pi.states_space,
                       normalize=True,
                       show=True,
                       path="./test_vf.png")